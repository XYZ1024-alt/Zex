use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{
    agent::Message,
    provider::ThinkingLevel,
    secure::{Cipher, EncryptedContent, EncryptionKey},
};

#[derive(Debug, Clone)]
pub struct SessionStore {
    directory: PathBuf,
    encryption_key: Option<EncryptionKey>,
    ciphers: Arc<Mutex<HashMap<String, Arc<Cipher>>>>,
}

#[derive(Debug, Clone)]
pub struct LoadedSession {
    pub id: String,
    pub messages: Vec<Message>,
    pub thinking_level: Option<ThinkingLevel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummary {
    pub id: String,
    pub updated_at: OffsetDateTime,
    pub message_count: usize,
    pub preview: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionHeader {
    format: u8,
    id: String,
    created_at: String,
    updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thinking_level: Option<ThinkingLevel>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionRecord {
    Session { session: SessionHeader },
    Message { message: Message },
}

#[derive(Debug, Serialize, Deserialize)]
struct EncryptedSessionFile {
    format: u8,
    encryption: EncryptedContent,
    content: String,
}

impl SessionStore {
    pub fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            encryption_key: None,
            ciphers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_encryption(directory: PathBuf, encryption_key: Option<EncryptionKey>) -> Self {
        Self {
            directory,
            encryption_key,
            ciphers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn allocate_id(&self) -> Result<String> {
        new_session_id(OffsetDateTime::now_utc())
    }

    pub fn memory_directory(&self, id: &str) -> Result<PathBuf> {
        Ok(self.directory.join(validate_session_id(id)?).join("memory"))
    }

    pub async fn delete(&self, id: &str) -> Result<bool> {
        let id = validate_session_id(id)?;
        let mut deleted = false;
        for path in [
            self.path_for(id),
            self.directory.join(format!("{id}.jsonl.tmp")),
        ] {
            match tokio::fs::remove_file(&path).await {
                Ok(()) => deleted = true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("failed to delete session {}", path.display()));
                }
            }
        }
        let memory_root = self.directory.join(id);
        match tokio::fs::remove_dir_all(&memory_root).await {
            Ok(()) => deleted = true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to delete session memory {}", memory_root.display())
                });
            }
        }
        Ok(deleted)
    }

    pub async fn save(
        &self,
        session_id: Option<&str>,
        model: &str,
        thinking_level: ThinkingLevel,
        messages: &[Message],
    ) -> Result<String> {
        tokio::fs::create_dir_all(&self.directory)
            .await
            .with_context(|| {
                format!(
                    "failed to create session directory {}",
                    self.directory.display()
                )
            })?;

        let now = OffsetDateTime::now_utc();
        let id = match session_id {
            Some(id) => validate_session_id(id)?.to_owned(),
            None => new_session_id(now)?,
        };
        let path = self.path_for(&id);
        let created_at = match self.read_header(&path).await? {
            Some(header) => header.created_at,
            None => format_timestamp(now)?,
        };
        let header = SessionHeader {
            format: 1,
            id: id.clone(),
            created_at,
            updated_at: format_timestamp(now)?,
            model: Some(model.to_owned()),
            thinking_level: Some(thinking_level),
        };
        let content = self.encode_session(&id, serialize_session(header, messages)?)?;
        atomic_write(&path, &content)
            .await
            .with_context(|| format!("failed to save session {}", path.display()))?;
        Ok(id)
    }

    pub async fn load(&self, session_id: Option<&str>) -> Result<Option<LoadedSession>> {
        let id = match session_id {
            Some(id) => validate_session_id(id)?.to_owned(),
            None => match self.list().await?.into_iter().next() {
                Some(session) => session.id,
                None => return Ok(None),
            },
        };
        let path = self.path_for(&id);
        let cipher = self.cipher(&id)?;
        match read_session(&path, cipher.as_deref()).await {
            Ok((header, messages)) => Ok(Some(LoadedSession {
                id: header.id,
                messages,
                thinking_level: header.thinking_level,
            })),
            Err(error)
                if error
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|io_error| io_error.kind() == std::io::ErrorKind::NotFound) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    pub async fn list(&self) -> Result<Vec<SessionSummary>> {
        let mut entries = match tokio::fs::read_dir(&self.directory).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to read session directory {}",
                        self.directory.display()
                    )
                });
            }
        };
        let mut sessions = BTreeMap::new();

        while let Some(entry) = entries
            .next_entry()
            .await
            .context("failed to inspect saved sessions")?
        {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl") {
                continue;
            }
            let id = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .with_context(|| format!("session path {} has no UTF-8 ID", path.display()))?;
            let cipher = self.cipher(id)?;
            let (header, messages) = read_session(&path, cipher.as_deref()).await?;
            let updated_at = OffsetDateTime::parse(&header.updated_at, &Rfc3339)
                .with_context(|| format!("invalid updated_at in {}", path.display()))?;
            sessions.insert(
                (std::cmp::Reverse(updated_at), header.id.clone()),
                SessionSummary {
                    id: header.id,
                    updated_at,
                    message_count: messages.len(),
                    preview: preview(&messages),
                },
            );
        }

        Ok(sessions.into_values().collect())
    }

    async fn read_header(&self, path: &Path) -> Result<Option<SessionHeader>> {
        let id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .with_context(|| format!("session path {} has no UTF-8 ID", path.display()))?;
        let cipher = self.cipher(id)?;
        match read_session(path, cipher.as_deref()).await {
            Ok((header, _)) => Ok(Some(header)),
            Err(error)
                if error
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|io_error| io_error.kind() == std::io::ErrorKind::NotFound) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn encode_session(&self, id: &str, content: Vec<u8>) -> Result<Vec<u8>> {
        if self.encryption_key.is_none() {
            return Ok(content);
        }
        let content = String::from_utf8(content).context("serialized session is not UTF-8")?;
        let cipher = self
            .cipher(id)?
            .context("session encryption key disappeared")?;
        let (content, encryption) = cipher.encrypt(id, &content)?;
        serde_json::to_vec(&EncryptedSessionFile {
            format: 1,
            encryption,
            content,
        })
        .context("failed to serialize encrypted session")
    }

    fn cipher(&self, id: &str) -> Result<Option<Arc<Cipher>>> {
        let Some(key) = &self.encryption_key else {
            return Ok(None);
        };
        if let Some(cipher) = self
            .ciphers
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(id)
            .cloned()
        {
            return Ok(Some(cipher));
        }
        let derived = Arc::new(Cipher::new(key, &format!("session:{id}"))?);
        let mut ciphers = self
            .ciphers
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(cipher) = ciphers.get(id) {
            return Ok(Some(Arc::clone(cipher)));
        }
        ciphers.insert(id.to_owned(), Arc::clone(&derived));
        Ok(Some(derived))
    }

    fn path_for(&self, id: &str) -> PathBuf {
        self.directory.join(format!("{id}.jsonl"))
    }
}

pub(crate) async fn atomic_write(path: &Path, content: &[u8]) -> Result<()> {
    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    tokio::fs::create_dir_all(parent)
        .await
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut temporary_name = path
        .file_name()
        .context("atomic write target has no file name")?
        .to_os_string();
    temporary_name.push(format!(".tmp-{}-{sequence}", std::process::id()));
    let temporary_path = parent.join(temporary_name);
    let result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary_path)
            .await
            .with_context(|| format!("failed to create {}", temporary_path.display()))?;
        tokio::io::AsyncWriteExt::write_all(&mut file, content)
            .await
            .with_context(|| format!("failed to write {}", temporary_path.display()))?;
        tokio::io::AsyncWriteExt::flush(&mut file)
            .await
            .with_context(|| format!("failed to flush {}", temporary_path.display()))?;
        file.sync_all()
            .await
            .with_context(|| format!("failed to sync {}", temporary_path.display()))?;
        drop(file);
        atomic_replace(&temporary_path, path).await?;
        sync_parent_directory(parent).await
    }
    .await;
    if let Err(error) = result {
        return match tokio::fs::remove_file(&temporary_path).await {
            Ok(()) => Err(error),
            Err(cleanup) if cleanup.kind() == std::io::ErrorKind::NotFound => Err(error),
            Err(cleanup) => Err(error.context(format!(
                "also failed to remove temporary file {}: {cleanup}",
                temporary_path.display()
            ))),
        };
    }
    Ok(())
}

#[cfg(windows)]
async fn atomic_replace(source: &Path, destination: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    tokio::task::spawn_blocking(move || {
        let replaced = unsafe {
            MoveFileExW(
                source.as_ptr(),
                destination.as_ptr(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
            )
        };
        if replaced == 0 {
            return Err(std::io::Error::last_os_error()).context("MoveFileExW failed");
        }
        Ok(())
    })
    .await
    .context("atomic replacement task failed")?
}

#[cfg(not(windows))]
async fn atomic_replace(source: &Path, destination: &Path) -> Result<()> {
    tokio::fs::rename(source, destination)
        .await
        .with_context(|| {
            format!(
                "failed to replace {} with {}",
                destination.display(),
                source.display()
            )
        })
}

#[cfg(windows)]
async fn sync_parent_directory(_parent: &Path) -> Result<()> {
    Ok(())
}

#[cfg(not(windows))]
async fn sync_parent_directory(parent: &Path) -> Result<()> {
    let parent = parent.to_path_buf();
    tokio::task::spawn_blocking(move || {
        std::fs::File::open(&parent)
            .and_then(|directory| directory.sync_all())
            .with_context(|| format!("failed to sync {}", parent.display()))
    })
    .await
    .context("directory sync task failed")?
}

pub fn format_session_summaries(sessions: &[SessionSummary]) -> Result<String> {
    if sessions.is_empty() {
        return Ok("No saved sessions.".to_owned());
    }

    let mut records = Vec::with_capacity(sessions.len());
    let local_offset = time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC);
    for session in sessions {
        let updated_at = session
            .updated_at
            .to_offset(local_offset)
            .format(&Rfc3339)
            .context("failed to format session timestamp")?;
        records.push(format!(
            "{}\n  Updated: {updated_at} · Messages: {}\n  {}",
            session.id, session.message_count, session.preview
        ));
    }
    Ok(records.join("\n\n"))
}

fn validate_session_id(id: &str) -> Result<&str> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        bail!("session ID may contain only ASCII letters, digits, and '-'");
    }
    Ok(id)
}

fn new_session_id(now: OffsetDateTime) -> Result<String> {
    let timestamp = now
        .format(time::macros::format_description!(
            "[year][month][day]-[hour][minute][second]"
        ))
        .context("failed to format session ID timestamp")?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .subsec_nanos();
    Ok(format!("{timestamp}-{nonce:08x}"))
}

fn format_timestamp(timestamp: OffsetDateTime) -> Result<String> {
    timestamp
        .format(&Rfc3339)
        .context("failed to format session timestamp")
}

fn serialize_session(header: SessionHeader, messages: &[Message]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    serde_json::to_writer(&mut output, &SessionRecord::Session { session: header })
        .context("failed to serialize the session header")?;
    output.push(b'\n');
    for message in messages {
        serde_json::to_writer(
            &mut output,
            &SessionRecord::Message {
                message: message.clone(),
            },
        )
        .context("failed to serialize a session message")?;
        output.push(b'\n');
    }
    Ok(output)
}

async fn read_session(
    path: &Path,
    cipher: Option<&Cipher>,
) -> Result<(SessionHeader, Vec<Message>)> {
    let content = tokio::fs::read(path)
        .await
        .with_context(|| format!("failed to read session {}", path.display()))?;
    let content = if let Ok(encrypted) = serde_json::from_slice::<EncryptedSessionFile>(&content) {
        if encrypted.format != 1 {
            bail!(
                "unsupported encrypted session format {} in {}",
                encrypted.format,
                path.display()
            );
        }
        let id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .context("encrypted session path has no UTF-8 ID")?;
        let cipher =
            cipher.context("session is encrypted; set ZEX_MEMORY_ENCRYPTION_KEY to unlock it")?;
        cipher.decrypt(id, &encrypted.content, &encrypted.encryption)?
    } else {
        String::from_utf8(content).context("session file is not UTF-8")?
    };
    let mut lines = content.lines();
    let header = parse_header_line(
        lines
            .next()
            .with_context(|| format!("session {} is empty", path.display()))?,
    )
    .with_context(|| format!("failed to parse session header {}", path.display()))?;
    let mut messages = Vec::new();
    for (index, line) in lines.enumerate() {
        let record: SessionRecord = serde_json::from_str(line).with_context(|| {
            format!(
                "failed to parse session {} line {}",
                path.display(),
                index + 2
            )
        })?;
        match record {
            SessionRecord::Message { message } => messages.push(message),
            SessionRecord::Session { .. } => {
                bail!(
                    "session {} contains an unexpected header at line {}",
                    path.display(),
                    index + 2
                );
            }
        }
    }
    Ok((header, messages))
}

fn parse_header_line(line: &str) -> Result<SessionHeader> {
    match serde_json::from_str(line).context("invalid session header JSON")? {
        SessionRecord::Session { session } if session.format == 1 => Ok(session),
        SessionRecord::Session { session } => {
            bail!("unsupported session format {}", session.format)
        }
        SessionRecord::Message { .. } => bail!("first session record must be a header"),
    }
}

fn preview(messages: &[Message]) -> String {
    messages
        .iter()
        .find_map(|message| match message {
            Message::User { content } => Some(content),
            _ => None,
        })
        .map(|content| {
            let mut preview: String = content.chars().take(80).collect();
            if content.chars().count() > 80 {
                preview.push('…');
            }
            preview.replace(['\r', '\n'], " ")
        })
        .unwrap_or_else(|| "(no user prompt)".to_owned())
}

#[cfg(test)]
mod tests {
    use std::{
        process,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{agent::Message, memory::MemoryEncryptionKey};

    use super::{SessionStore, SessionSummary, format_session_summaries};

    fn temp_directory(label: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("zex-{label}-{}-{unique}", process::id()))
    }

    #[tokio::test]
    async fn saves_lists_and_resumes_a_session() {
        let directory = temp_directory("sessions");
        let store = SessionStore::new(directory.clone());
        let first = vec![Message::User {
            content: "first task".to_owned(),
        }];
        let second = vec![
            Message::User {
                content: "first task".to_owned(),
            },
            Message::Assistant {
                content: "done".to_owned(),
                thinking: Some("Completed the saved task.".to_owned()),
                tool_calls: Vec::new(),
                provider_state: None,
            },
        ];

        let id = store
            .save(
                None,
                "model-a",
                crate::provider::ThinkingLevel::Medium,
                &first,
            )
            .await
            .unwrap();
        let resumed = store.load(Some(&id)).await.unwrap().unwrap();
        assert_eq!(resumed.id, id);
        assert_eq!(resumed.messages, first);
        assert_eq!(
            resumed.thinking_level,
            Some(crate::provider::ThinkingLevel::Medium)
        );

        store
            .save(
                Some(&id),
                "model-b",
                crate::provider::ThinkingLevel::High,
                &second,
            )
            .await
            .unwrap();
        let sessions = store.list().await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, id);
        assert_eq!(sessions[0].message_count, 2);
        assert_eq!(sessions[0].preview, "first task");
        let loaded = store.load(None).await.unwrap().unwrap();
        assert_eq!(loaded.messages, second);
        assert_eq!(
            loaded.thinking_level,
            Some(crate::provider::ThinkingLevel::High)
        );
        let persisted = tokio::fs::read_to_string(directory.join(format!("{id}.jsonl")))
            .await
            .unwrap();
        assert!(persisted.contains("\"thinking\":\"Completed the saved task.\""));
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn encrypted_sessions_hide_transcript_content_and_require_the_key() {
        let directory = temp_directory("encrypted-session");
        let key = MemoryEncryptionKey::new("session encryption key".to_owned()).unwrap();
        let store = SessionStore::with_encryption(directory.clone(), Some(key.clone()));
        let secret = "private user prompt and reasoning";
        let id = store
            .save(
                None,
                "model-a",
                crate::provider::ThinkingLevel::High,
                &[Message::User {
                    content: secret.to_owned(),
                }],
            )
            .await
            .unwrap();
        let stored = tokio::fs::read_to_string(directory.join(format!("{id}.jsonl")))
            .await
            .unwrap();
        assert!(!stored.contains(secret));
        assert!(stored.contains("\"encryption\""));
        assert_eq!(
            store.load(Some(&id)).await.unwrap().unwrap().messages,
            vec![Message::User {
                content: secret.to_owned()
            }]
        );

        let locked = SessionStore::new(directory.clone());
        let error = locked.load(Some(&id)).await.unwrap_err();
        assert!(error.to_string().contains("ZEX_MEMORY_ENCRYPTION_KEY"));
        let wrong = SessionStore::with_encryption(
            directory.clone(),
            MemoryEncryptionKey::new("wrong key".to_owned()),
        );
        let error = wrong.load(Some(&id)).await.unwrap_err();
        assert!(error.to_string().contains("authentication failed"));

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn persisted_session_contains_messages_but_no_config_secret() {
        let directory = temp_directory("secret");
        let store = SessionStore::new(directory.clone());
        let secret = "secret-that-must-not-be-persisted";
        let id = store
            .save(
                None,
                "model",
                crate::provider::ThinkingLevel::Medium,
                &[Message::User {
                    content: "hello".to_owned(),
                }],
            )
            .await
            .unwrap();
        let content = tokio::fs::read_to_string(directory.join(format!("{id}.jsonl")))
            .await
            .unwrap();

        assert!(content.contains("\"hello\""));
        assert!(!content.contains(secret));
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn loads_version_one_sessions_without_a_saved_model() {
        let directory = temp_directory("legacy-model");
        tokio::fs::create_dir_all(&directory).await.unwrap();
        let id = "20260812-120000-deadbeef";
        let content = concat!(
            "{\"type\":\"session\",\"session\":{\"format\":1,\"id\":\"20260812-120000-deadbeef\",\"created_at\":\"2026-08-12T12:00:00Z\",\"updated_at\":\"2026-08-12T12:00:00Z\"}}\n",
            "{\"type\":\"message\",\"message\":{\"role\":\"user\",\"content\":\"legacy\"}}\n"
        );
        tokio::fs::write(directory.join(format!("{id}.jsonl")), content)
            .await
            .unwrap();
        let store = SessionStore::new(directory.clone());

        let loaded = store.load(Some(id)).await.unwrap().unwrap();

        assert!(matches!(
            &loaded.messages[0],
            Message::User { content } if content == "legacy"
        ));
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn loads_legacy_assistant_messages_without_thinking() {
        let directory = temp_directory("legacy-thinking");
        tokio::fs::create_dir_all(&directory).await.unwrap();
        let id = "20260812-120000-feedface";
        let content = concat!(
            "{\"type\":\"session\",\"session\":{\"format\":1,\"id\":\"20260812-120000-feedface\",\"created_at\":\"2026-08-12T12:00:00Z\",\"updated_at\":\"2026-08-12T12:00:00Z\"}}\n",
            "{\"type\":\"message\",\"message\":{\"role\":\"assistant\",\"content\":\"legacy answer\"}}\n"
        );
        tokio::fs::write(directory.join(format!("{id}.jsonl")), content)
            .await
            .unwrap();
        let store = SessionStore::new(directory.clone());

        let loaded = store.load(Some(id)).await.unwrap().unwrap();

        assert!(matches!(
            &loaded.messages[0],
            Message::Assistant {
                content,
                thinking: None,
                tool_calls,
                provider_state: None,
            } if content == "legacy answer" && tool_calls.is_empty()
        ));
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn list_orders_sessions_by_most_recent_update() {
        let directory = temp_directory("recent-first");
        let store = SessionStore::new(directory.clone());
        let older = store
            .save(
                None,
                "model",
                crate::provider::ThinkingLevel::Medium,
                &[Message::User {
                    content: "older".to_owned(),
                }],
            )
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let newer = store
            .save(
                None,
                "model",
                crate::provider::ThinkingLevel::Medium,
                &[Message::User {
                    content: "newer".to_owned(),
                }],
            )
            .await
            .unwrap();

        let sessions = store.list().await.unwrap();

        assert_eq!(
            sessions
                .iter()
                .map(|session| session.id.as_str())
                .collect::<Vec<_>>(),
            vec![newer.as_str(), older.as_str()]
        );
        assert!(sessions[0].updated_at >= sessions[1].updated_at);
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn delete_removes_session_and_addressable_memory() {
        let directory = temp_directory("delete");
        let store = SessionStore::new(directory.clone());
        let id = store
            .save(
                None,
                "model",
                crate::provider::ThinkingLevel::Medium,
                &[Message::User {
                    content: "delete me".to_owned(),
                }],
            )
            .await
            .unwrap();
        let memory_directory = store.memory_directory(&id).unwrap();
        tokio::fs::create_dir_all(&memory_directory).await.unwrap();
        tokio::fs::write(memory_directory.join("records.jsonl"), b"memory")
            .await
            .unwrap();

        assert!(store.delete(&id).await.unwrap());
        assert!(store.load(Some(&id)).await.unwrap().is_none());
        assert!(!tokio::fs::try_exists(&memory_directory).await.unwrap());
        assert!(!store.delete(&id).await.unwrap());

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[test]
    fn session_summary_formatter_uses_record_blocks_instead_of_fixed_columns() {
        let output = format_session_summaries(&[
            SessionSummary {
                id: "session-one".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH,
                message_count: 2,
                preview: "first task".to_owned(),
            },
            SessionSummary {
                id: "session-two".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH,
                message_count: 1,
                preview: "second task".to_owned(),
            },
        ])
        .unwrap();

        assert!(output.contains("session-one\n  Updated:"));
        assert!(output.contains("Messages: 2\n  first task"));
        assert!(output.contains("\n\nsession-two\n  Updated:"));
        assert!(!output.contains("ID                          Updated"));
    }
}
