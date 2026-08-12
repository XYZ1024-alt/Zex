use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::agent::Message;

#[derive(Debug, Clone)]
pub struct SessionStore {
    directory: PathBuf,
}

#[derive(Debug, Clone)]
pub struct LoadedSession {
    pub id: String,
    pub model: Option<String>,
    pub messages: Vec<Message>,
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
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionRecord {
    Session { session: SessionHeader },
    Message { message: Message },
}

impl SessionStore {
    pub fn new(directory: PathBuf) -> Self {
        Self { directory }
    }

    pub async fn save(
        &self,
        session_id: Option<&str>,
        model: &str,
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
        };
        let content = serialize_session(header, messages)?;
        let temporary_path = self.directory.join(format!("{id}.jsonl.tmp"));
        tokio::fs::write(&temporary_path, content)
            .await
            .with_context(|| {
                format!(
                    "failed to write temporary session {}",
                    temporary_path.display()
                )
            })?;
        if let Err(error) = tokio::fs::rename(&temporary_path, &path).await {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                tokio::fs::remove_file(&path)
                    .await
                    .with_context(|| format!("failed to replace session {}", path.display()))?;
                tokio::fs::rename(&temporary_path, &path)
                    .await
                    .with_context(|| format!("failed to save session {}", path.display()))?;
            } else {
                return Err(error)
                    .with_context(|| format!("failed to save session {}", path.display()));
            }
        }
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
        match read_session(&path).await {
            Ok((header, messages)) => Ok(Some(LoadedSession {
                id: header.id,
                model: header.model,
                messages,
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
            let (header, messages) = read_session(&path).await?;
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
        match tokio::fs::read_to_string(path).await {
            Ok(content) => parse_header(&content).map(Some),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => {
                Err(error).with_context(|| format!("failed to read session {}", path.display()))
            }
        }
    }

    fn path_for(&self, id: &str) -> PathBuf {
        self.directory.join(format!("{id}.jsonl"))
    }
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

async fn read_session(path: &Path) -> Result<(SessionHeader, Vec<Message>)> {
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read session {}", path.display()))?;
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

fn parse_header(content: &str) -> Result<SessionHeader> {
    parse_header_line(content.lines().next().context("session is empty")?)
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

    use crate::agent::Message;

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

        let id = store.save(None, "model-a", &first).await.unwrap();
        let resumed = store.load(Some(&id)).await.unwrap().unwrap();
        assert_eq!(resumed.id, id);
        assert_eq!(resumed.model.as_deref(), Some("model-a"));
        assert_eq!(resumed.messages, first);

        store.save(Some(&id), "model-b", &second).await.unwrap();
        let sessions = store.list().await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, id);
        assert_eq!(sessions[0].message_count, 2);
        assert_eq!(sessions[0].preview, "first task");
        assert_eq!(store.load(None).await.unwrap().unwrap().messages, second);
        let persisted = tokio::fs::read_to_string(directory.join(format!("{id}.jsonl")))
            .await
            .unwrap();
        assert!(persisted.contains("\"thinking\":\"Completed the saved task.\""));
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

        assert!(loaded.model.is_none());
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
