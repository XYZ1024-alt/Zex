use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    path::PathBuf,
    str::FromStr,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use tokio::{
    fs::{File, OpenOptions},
    io::{AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader, SeekFrom},
    sync::Mutex as AsyncMutex,
};

use crate::{
    agent::{estimate_tokens, token_window},
    secure::{Cipher as MemoryCipher, EncryptedContent as ContentEncryption},
    session::atomic_write,
};

const RECORDS_FILE: &str = "records.jsonl";
const AUDIT_FILE: &str = "audit.jsonl";
const PIN_STATE_FILE: &str = "pin-state.json";
const PIN_STATE_FORMAT: u8 = 1;
const AUDIT_ARCHIVE_LIMIT: usize = 4;
#[cfg(not(test))]
const AUDIT_ROTATE_BYTES: u64 = 4 * 1024 * 1024;
#[cfg(test)]
const AUDIT_ROTATE_BYTES: u64 = 4 * 1024;
const MEMORY_ID_HEX_LEN: usize = 24;
const ACTIVE_POINTER_LIMIT: usize = 64;
const SYSTEM_POINTER_LIMIT: usize = 24;
const MAX_INLINE_TOOL_TOKENS: usize = 1_024;
const RECALL_WINDOW: Duration = Duration::from_secs(60);
const PER_MODEL_TURN_RECALL_LIMIT: usize = 3;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryMode {
    PointerPriority,
    Summary,
    #[default]
    Hybrid,
}

impl FromStr for MemoryMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pointer_priority" | "pointer-priority" => Ok(Self::PointerPriority),
            "summary" => Ok(Self::Summary),
            "hybrid" => Ok(Self::Hybrid),
            _ => bail!("memory.mode must be pointer_priority, summary, or hybrid"),
        }
    }
}

pub use crate::secure::EncryptionKey as MemoryEncryptionKey;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryConfig {
    pub enabled: bool,
    pub mode: MemoryMode,
    pub recall_rate_limit: usize,
    pub max_recall_tokens: usize,
    pub hot_cache_size: usize,
    pub auto_pin_important: bool,
    pub max_auto_pins: usize,
    pub max_records: usize,
    pub max_store_bytes: u64,
    pub retention_days: u64,
    pub encryption_key: Option<MemoryEncryptionKey>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: MemoryMode::Hybrid,
            recall_rate_limit: 5,
            max_recall_tokens: 2_048,
            hot_cache_size: 32,
            auto_pin_important: true,
            max_auto_pins: 32,
            max_records: 10_000,
            max_store_bytes: 64 * 1024 * 1024,
            retention_days: 30,
            encryption_key: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    ToolResult,
    Message,
    FileSnapshot,
    Decision,
    Summary,
}

impl MemoryKind {
    fn id_prefix(self) -> &'static str {
        match self {
            Self::ToolResult | Self::FileSnapshot => "obs",
            Self::Message => "turn",
            Self::Decision => "decision",
            Self::Summary => "summary",
        }
    }

    fn citation_label(self) -> &'static str {
        match self {
            Self::ToolResult => "tool result",
            Self::Message => "message",
            Self::FileSnapshot => "file snapshot",
            Self::Decision => "decision",
            Self::Summary => "summary",
        }
    }

    fn filter_name(self) -> &'static str {
        match self {
            Self::ToolResult => "tool_result",
            Self::Message => "message",
            Self::FileSnapshot => "file_snapshot",
            Self::Decision => "decision",
            Self::Summary => "summary",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPointer {
    pub id: String,
    pub kind: MemoryKind,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_tool: Option<String>,
    pub token_estimate: usize,
    pub importance: u8,
    pub pinned: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryItem {
    #[serde(flatten)]
    pointer: MemoryPointer,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    encryption: Option<ContentEncryption>,
    content: String,
}

struct NewMemoryItem {
    kind: MemoryKind,
    content: String,
    source_tool: Option<String>,
    importance: u8,
    pinned: bool,
    parent_id: Option<String>,
    metadata: BTreeMap<String, String>,
    turn_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AuditAction {
    Recall,
    RecallMissing,
    RecallRateLimited,
    Pin,
    Unpin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditEvent {
    action: AuditAction,
    id: String,
    created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    returned_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EncryptedAuditEvent {
    format: u8,
    encryption: ContentEncryption,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PinState {
    format: u8,
    overrides: BTreeMap<String, bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PointerFilter {
    Text(String),
    Field { key: String, value: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MemoryTurnState {
    Committed,
    Aborted,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "record_type", rename_all = "snake_case")]
enum StoreRecord {
    Item {
        item: MemoryItem,
    },
    Turn {
        turn_id: String,
        state: MemoryTurnState,
    },
    // Existing stores may contain inline audit records. New audit events are
    // written to audit.jsonl so rebuilding the content index does not scan
    // operational history.
    Audit {
        audit: AuditEvent,
    },
}

#[derive(Debug, Clone)]
struct RecordLocation {
    offset: u64,
    length: usize,
    pointer: MemoryPointer,
    turn_id: Option<String>,
}

#[derive(Debug, Default)]
struct StoreIndex {
    entries: HashMap<String, RecordLocation>,
    order: Vec<String>,
    message_fingerprints: HashMap<String, String>,
    tool_fingerprints: HashMap<String, String>,
    turn_states: HashMap<String, MemoryTurnState>,
}

#[derive(Debug)]
struct HotCache {
    capacity: usize,
    values: HashMap<String, Arc<String>>,
    order: VecDeque<String>,
}

impl HotCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            values: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, id: &str) -> Option<Arc<String>> {
        let value = self.values.get(id)?.clone();
        self.touch(id);
        Some(value)
    }

    fn insert(&mut self, id: String, content: String) {
        if self.capacity == 0 {
            return;
        }
        self.values.insert(id.clone(), Arc::new(content));
        self.touch(&id);
        while self.values.len() > self.capacity {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.values.remove(&oldest);
        }
    }

    fn touch(&mut self, id: &str) {
        if let Some(index) = self.order.iter().position(|candidate| candidate == id) {
            self.order.remove(index);
        }
        self.order.push_back(id.to_owned());
    }
}

#[derive(Debug, Default)]
struct RecallLimiter {
    current_model_turn: usize,
    recent: VecDeque<Instant>,
}

impl RecallLimiter {
    fn begin_model_turn(&mut self) {
        self.current_model_turn = 0;
    }

    fn acquire(&mut self, configured_limit: usize) -> Result<()> {
        let now = Instant::now();
        while self
            .recent
            .front()
            .is_some_and(|instant| now.duration_since(*instant) >= RECALL_WINDOW)
        {
            self.recent.pop_front();
        }
        let per_turn_limit = configured_limit.min(PER_MODEL_TURN_RECALL_LIMIT);
        if self.current_model_turn >= per_turn_limit {
            bail!(
                "recall rate limit exceeded: at most {per_turn_limit} recall calls are allowed per model turn"
            );
        }
        if self.recent.len() >= configured_limit {
            bail!(
                "recall rate limit exceeded: at most {configured_limit} recall calls are allowed per minute"
            );
        }
        self.current_model_turn += 1;
        self.recent.push_back(now);
        Ok(())
    }
}

#[derive(Debug)]
struct SessionMemory {
    session_id: String,
    directory: PathBuf,
    records_path: PathBuf,
    audit_path: PathBuf,
    pin_state_path: PathBuf,
    session_hash: u64,
    next_sequence: AtomicU64,
    next_turn_sequence: AtomicU64,
    append_lock: AsyncMutex<()>,
    audit_append_lock: AsyncMutex<()>,
    dedupe_lock: AsyncMutex<()>,
    pin_lock: AsyncMutex<()>,
    index: Mutex<StoreIndex>,
    pin_overrides: Mutex<BTreeMap<String, bool>>,
    cache: Mutex<HotCache>,
    cipher: Option<MemoryCipher>,
    needs_encryption_migration: bool,
}

impl SessionMemory {
    async fn open(
        session_id: String,
        directory: PathBuf,
        cache_size: usize,
        encryption_key: Option<&MemoryEncryptionKey>,
    ) -> Result<Self> {
        let records_path = directory.join(RECORDS_FILE);
        let audit_path = directory.join(AUDIT_FILE);
        let pin_state_path = directory.join(PIN_STATE_FILE);
        let cipher = encryption_key
            .map(|key| MemoryCipher::new(key, &session_id))
            .transpose()?;
        let mut index = StoreIndex::default();
        let mut item_count = 0u64;
        let mut turn_count = 0u64;
        let mut verified_encryption = false;
        let mut has_plaintext_records = false;

        match File::open(&records_path).await {
            Ok(file) => {
                let mut reader = BufReader::new(file);
                let mut line = Vec::new();
                let mut offset = 0u64;
                let mut line_number = 0usize;
                loop {
                    line.clear();
                    let length = reader
                        .read_until(b'\n', &mut line)
                        .await
                        .with_context(|| format!("failed to scan {}", records_path.display()))?;
                    if length == 0 {
                        break;
                    }
                    line_number += 1;
                    let complete = line.ends_with(b"\n");
                    let payload = line
                        .strip_suffix(b"\n")
                        .unwrap_or(&line)
                        .strip_suffix(b"\r")
                        .unwrap_or_else(|| line.strip_suffix(b"\n").unwrap_or(&line));
                    let record = match serde_json::from_slice::<StoreRecord>(payload) {
                        Ok(record) => record,
                        Err(_) if !complete => {
                            drop(reader);
                            OpenOptions::new()
                                .write(true)
                                .open(&records_path)
                                .await
                                .with_context(|| {
                                    format!(
                                        "failed to recover incomplete memory record {}",
                                        records_path.display()
                                    )
                                })?
                                .set_len(offset)
                                .await
                                .with_context(|| {
                                    format!(
                                        "failed to truncate incomplete memory record {}",
                                        records_path.display()
                                    )
                                })?;
                            break;
                        }
                        Err(error) => {
                            return Err(error).with_context(|| {
                                format!(
                                    "failed to parse memory store {} line {line_number}",
                                    records_path.display()
                                )
                            });
                        }
                    };
                    match record {
                        StoreRecord::Item { item } => {
                            has_plaintext_records |= item.encryption.is_none();
                            if let Some(encryption) = &item.encryption
                                && !verified_encryption
                            {
                                cipher
                                    .as_ref()
                                    .context(
                                        "memory content is encrypted; set ZEX_MEMORY_ENCRYPTION_KEY to unlock it",
                                    )?
                                    .decrypt(&item.pointer.id, &item.content, encryption)?;
                                verified_encryption = true;
                            }
                            validate_memory_id(&item.pointer.id)?;
                            if index.entries.contains_key(&item.pointer.id) {
                                bail!(
                                    "memory store {} reuses ID {}",
                                    records_path.display(),
                                    item.pointer.id
                                );
                            }
                            item_count += 1;
                            if let Some(turn_id) = &item.turn_id {
                                index
                                    .turn_states
                                    .entry(turn_id.clone())
                                    .or_insert(MemoryTurnState::Aborted);
                            }
                            index.order.push(item.pointer.id.clone());
                            if let Some(fingerprint) = item.pointer.metadata.get("fingerprint") {
                                match item.pointer.kind {
                                    MemoryKind::ToolResult | MemoryKind::FileSnapshot => {
                                        index
                                            .tool_fingerprints
                                            .insert(fingerprint.clone(), item.pointer.id.clone());
                                    }
                                    MemoryKind::Message
                                    | MemoryKind::Decision
                                    | MemoryKind::Summary => {
                                        index
                                            .message_fingerprints
                                            .insert(fingerprint.clone(), item.pointer.id.clone());
                                    }
                                }
                            }
                            index.entries.insert(
                                item.pointer.id.clone(),
                                RecordLocation {
                                    offset,
                                    length,
                                    pointer: item.pointer,
                                    turn_id: item.turn_id,
                                },
                            );
                        }
                        StoreRecord::Turn { turn_id, state } => {
                            turn_count += 1;
                            index.turn_states.insert(turn_id, state);
                        }
                        StoreRecord::Audit { audit } => match audit.action {
                            AuditAction::Pin => {
                                if let Some(entry) = index.entries.get_mut(&audit.id) {
                                    entry.pointer.pinned = true;
                                }
                            }
                            AuditAction::Unpin => {
                                if let Some(entry) = index.entries.get_mut(&audit.id) {
                                    entry.pointer.pinned = false;
                                }
                            }
                            AuditAction::Recall
                            | AuditAction::RecallMissing
                            | AuditAction::RecallRateLimited => {}
                        },
                    }
                    offset = offset.saturating_add(length as u64);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to open {}", records_path.display()));
            }
        }

        let pin_overrides = match tokio::fs::read(&pin_state_path).await {
            Ok(content) => {
                let state: PinState = serde_json::from_slice(&content).with_context(|| {
                    format!("failed to parse pin state {}", pin_state_path.display())
                })?;
                if state.format != PIN_STATE_FORMAT {
                    bail!(
                        "unsupported pin state format {} in {}",
                        state.format,
                        pin_state_path.display()
                    );
                }
                for (id, pinned) in &state.overrides {
                    validate_memory_id(id)?;
                    let entry = index.entries.get_mut(id).with_context(|| {
                        format!(
                            "pin state {} references missing memory ID {id}",
                            pin_state_path.display()
                        )
                    })?;
                    entry.pointer.pinned = *pinned;
                }
                state.overrides
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to read {}", pin_state_path.display()));
            }
        };

        Ok(Self {
            session_hash: fnv1a64(session_id.as_bytes()),
            session_id,
            directory,
            records_path,
            audit_path,
            pin_state_path,
            next_sequence: AtomicU64::new(item_count),
            next_turn_sequence: AtomicU64::new(turn_count),
            append_lock: AsyncMutex::new(()),
            audit_append_lock: AsyncMutex::new(()),
            dedupe_lock: AsyncMutex::new(()),
            pin_lock: AsyncMutex::new(()),
            index: Mutex::new(index),
            pin_overrides: Mutex::new(pin_overrides),
            cache: Mutex::new(HotCache::new(cache_size)),
            needs_encryption_migration: cipher.is_some() && has_plaintext_records,
            cipher,
        })
    }

    async fn store(&self, item: NewMemoryItem) -> Result<MemoryPointer> {
        let NewMemoryItem {
            kind,
            content,
            source_tool,
            importance,
            pinned,
            parent_id,
            metadata,
            turn_id,
        } = item;
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let id = format!(
            "§{}_{:016x}{:08x}",
            kind.id_prefix(),
            self.session_hash,
            sequence
        );
        let pointer = MemoryPointer {
            id: id.clone(),
            kind,
            created_at: timestamp()?,
            source_tool,
            token_estimate: estimate_tokens(&content),
            importance,
            pinned,
            parent_id,
            metadata,
        };
        let item = self.encode_item(pointer.clone(), turn_id.clone(), &content)?;
        let (offset, length) = self.append(&StoreRecord::Item { item }).await?;

        let mut index = self.index.lock().unwrap_or_else(|error| error.into_inner());
        if index.entries.contains_key(&id) {
            bail!("memory ID {id} was generated more than once");
        }
        index.order.push(id.clone());
        if let Some(fingerprint) = pointer.metadata.get("fingerprint") {
            match pointer.kind {
                MemoryKind::ToolResult | MemoryKind::FileSnapshot => {
                    index
                        .tool_fingerprints
                        .insert(fingerprint.clone(), id.clone());
                }
                MemoryKind::Message | MemoryKind::Decision | MemoryKind::Summary => {
                    index
                        .message_fingerprints
                        .insert(fingerprint.clone(), id.clone());
                }
            }
        }
        index.entries.insert(
            id.clone(),
            RecordLocation {
                offset,
                length,
                pointer: pointer.clone(),
                turn_id,
            },
        );
        drop(index);
        self.cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(id, content);
        Ok(pointer)
    }

    fn encode_item(
        &self,
        pointer: MemoryPointer,
        turn_id: Option<String>,
        content: &str,
    ) -> Result<MemoryItem> {
        let (content, encryption) = match &self.cipher {
            Some(cipher) => {
                let (content, encryption) = cipher.encrypt(&pointer.id, content)?;
                (content, Some(encryption))
            }
            None => (content.to_owned(), None),
        };
        Ok(MemoryItem {
            pointer,
            turn_id,
            encryption,
            content,
        })
    }

    async fn enforce_retention(
        &self,
        max_records: usize,
        max_store_bytes: u64,
        retention_days: u64,
        protected: &HashSet<String>,
    ) -> Result<bool> {
        let visible = self.pointers(None);
        let indexed_records = self
            .index
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entries
            .len();
        let store_bytes = match tokio::fs::metadata(&self.records_path).await {
            Ok(metadata) => metadata.len(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to inspect {}", self.records_path.display()));
            }
        };
        let retention_days = i64::try_from(retention_days).unwrap_or(i64::MAX);
        let cutoff = OffsetDateTime::now_utc()
            .checked_sub(time::Duration::days(retention_days))
            .unwrap_or(OffsetDateTime::UNIX_EPOCH);
        let is_expired = |pointer: &MemoryPointer| {
            !pointer.pinned
                && !protected.contains(&pointer.id)
                && OffsetDateTime::parse(&pointer.created_at, &Rfc3339)
                    .is_ok_and(|created_at| created_at < cutoff)
        };
        let needs_rewrite = indexed_records != visible.len()
            || visible.len() > max_records
            || store_bytes > max_store_bytes
            || visible.iter().any(is_expired)
            || self.needs_encryption_migration;
        if !needs_rewrite {
            return Ok(false);
        }

        let retained_by_age = visible
            .into_iter()
            .filter(|pointer| !is_expired(pointer))
            .collect::<Vec<_>>();
        let protected_count = retained_by_age
            .iter()
            .filter(|pointer| pointer.pinned || protected.contains(&pointer.id))
            .count();
        let unprotected_limit = max_records.saturating_sub(protected_count);
        let retained_unpinned = retained_by_age
            .iter()
            .rev()
            .filter(|pointer| !pointer.pinned && !protected.contains(&pointer.id))
            .take(unprotected_limit)
            .map(|pointer| pointer.id.clone())
            .collect::<HashSet<_>>();
        let retained_by_count = retained_by_age
            .into_iter()
            .filter(|pointer| {
                pointer.pinned
                    || protected.contains(&pointer.id)
                    || retained_unpinned.contains(&pointer.id)
            })
            .collect::<Vec<_>>();

        let mut encoded = Vec::with_capacity(retained_by_count.len());
        for mut pointer in retained_by_count {
            let (_, content) = self
                .read(&pointer.id, None)
                .await?
                .with_context(|| format!("retained memory {} disappeared", pointer.id))?;
            if self.cipher.is_some() {
                pointer.metadata.remove("preview");
            }
            let item = self.encode_item(pointer.clone(), None, &content)?;
            let mut line = serde_json::to_vec(&StoreRecord::Item { item })
                .context("failed to serialize retained memory record")?;
            line.push(b'\n');
            encoded.push((pointer, line));
        }

        let protected_bytes = encoded
            .iter()
            .filter(|(pointer, _)| pointer.pinned || protected.contains(&pointer.id))
            .map(|(_, line)| line.len() as u64)
            .sum::<u64>();
        let mut available = max_store_bytes.saturating_sub(protected_bytes);
        let retained_unpinned = encoded
            .iter()
            .rev()
            .filter(|(pointer, _)| !pointer.pinned && !protected.contains(&pointer.id))
            .filter_map(|(pointer, line)| {
                let length = line.len() as u64;
                if length > available {
                    return None;
                }
                available -= length;
                Some(pointer.id.clone())
            })
            .collect::<HashSet<_>>();
        let mut content = Vec::new();
        let mut retained_ids = HashSet::new();
        for (pointer, line) in encoded {
            if pointer.pinned
                || protected.contains(&pointer.id)
                || retained_unpinned.contains(&pointer.id)
            {
                retained_ids.insert(pointer.id);
                content.extend_from_slice(&line);
            }
        }
        let overrides = {
            let overrides = self
                .pin_overrides
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            overrides
                .iter()
                .filter(|(id, _)| retained_ids.contains(*id))
                .map(|(id, pinned)| (id.clone(), *pinned))
                .collect::<BTreeMap<_, _>>()
        };
        self.persist_pin_state(&overrides).await?;
        atomic_write(&self.records_path, &content)
            .await
            .with_context(|| format!("failed to compact {}", self.records_path.display()))?;
        *self
            .pin_overrides
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = overrides;
        Ok(true)
    }

    fn allocate_turn_id(&self) -> String {
        let sequence = self.next_turn_sequence.fetch_add(1, Ordering::Relaxed);
        format!("tx-{:016x}-{sequence:08x}", self.session_hash)
    }

    async fn finish_turn(&self, turn_id: &str, state: MemoryTurnState) -> Result<()> {
        self.append(&StoreRecord::Turn {
            turn_id: turn_id.to_owned(),
            state,
        })
        .await?;
        self.index
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .turn_states
            .insert(turn_id.to_owned(), state);
        Ok(())
    }

    async fn append(&self, record: &StoreRecord) -> Result<(u64, usize)> {
        let mut line = serde_json::to_vec(record).context("failed to serialize memory record")?;
        line.push(b'\n');
        let _append = self.append_lock.lock().await;
        tokio::fs::create_dir_all(&self.directory)
            .await
            .with_context(|| {
                format!(
                    "failed to create memory directory {}",
                    self.directory.display()
                )
            })?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&self.records_path)
            .await
            .with_context(|| format!("failed to open {}", self.records_path.display()))?;
        let offset = file
            .metadata()
            .await
            .with_context(|| format!("failed to inspect {}", self.records_path.display()))?
            .len();
        file.write_all(&line)
            .await
            .with_context(|| format!("failed to append {}", self.records_path.display()))?;
        file.flush()
            .await
            .with_context(|| format!("failed to flush {}", self.records_path.display()))?;
        file.sync_data()
            .await
            .with_context(|| format!("failed to sync {}", self.records_path.display()))?;
        Ok((offset, line.len()))
    }

    async fn append_audit(&self, audit: AuditEvent) -> Result<()> {
        let mut line = match &self.cipher {
            Some(cipher) => {
                let audit =
                    serde_json::to_string(&audit).context("failed to serialize memory audit")?;
                let (content, encryption) = cipher.encrypt("audit", &audit)?;
                serde_json::to_vec(&EncryptedAuditEvent {
                    format: 1,
                    encryption,
                    content,
                })
                .context("failed to serialize encrypted memory audit")?
            }
            None => serde_json::to_vec(&audit).context("failed to serialize memory audit")?,
        };
        line.push(b'\n');
        let _append = self.audit_append_lock.lock().await;
        tokio::fs::create_dir_all(&self.directory)
            .await
            .with_context(|| {
                format!(
                    "failed to create memory directory {}",
                    self.directory.display()
                )
            })?;
        self.rotate_audit_if_needed(line.len()).await?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.audit_path)
            .await
            .with_context(|| format!("failed to open {}", self.audit_path.display()))?;
        file.write_all(&line)
            .await
            .with_context(|| format!("failed to append {}", self.audit_path.display()))?;
        file.flush()
            .await
            .with_context(|| format!("failed to flush {}", self.audit_path.display()))?;
        file.sync_data()
            .await
            .with_context(|| format!("failed to sync {}", self.audit_path.display()))?;
        Ok(())
    }

    async fn rotate_audit_if_needed(&self, next_record_bytes: usize) -> Result<()> {
        let current_bytes = match tokio::fs::metadata(&self.audit_path).await {
            Ok(metadata) => metadata.len(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to inspect {}", self.audit_path.display()));
            }
        };
        if current_bytes == 0
            || current_bytes.saturating_add(next_record_bytes as u64) <= AUDIT_ROTATE_BYTES
        {
            return Ok(());
        }

        let archive = self.directory.join(format!(
            "audit-{}-{:08x}.jsonl",
            OffsetDateTime::now_utc().unix_timestamp_nanos(),
            self.next_sequence.load(Ordering::Relaxed)
        ));
        tokio::fs::rename(&self.audit_path, &archive)
            .await
            .with_context(|| {
                format!(
                    "failed to rotate {} to {}",
                    self.audit_path.display(),
                    archive.display()
                )
            })?;
        self.prune_audit_archives().await
    }

    async fn prune_audit_archives(&self) -> Result<()> {
        let mut entries = tokio::fs::read_dir(&self.directory)
            .await
            .with_context(|| {
                format!(
                    "failed to inspect memory directory {}",
                    self.directory.display()
                )
            })?;
        let mut archives = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .context("failed to inspect memory audit archives")?
        {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("audit-") && name.ends_with(".jsonl") {
                archives.push(entry.path());
            }
        }
        archives.sort();
        let remove_count = archives.len().saturating_sub(AUDIT_ARCHIVE_LIMIT);
        for archive in archives.into_iter().take(remove_count) {
            tokio::fs::remove_file(&archive)
                .await
                .with_context(|| format!("failed to remove {}", archive.display()))?;
        }
        Ok(())
    }

    async fn read(
        &self,
        id: &str,
        active_turn: Option<&str>,
    ) -> Result<Option<(MemoryPointer, Arc<String>)>> {
        let pointer = {
            let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
            let Some(location) = index.entries.get(id) else {
                return Ok(None);
            };
            if !Self::location_is_visible(&index, location, active_turn) {
                return Ok(None);
            }
            location.pointer.clone()
        };
        if let Some(content) = self
            .cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(id)
        {
            return Ok(Some((pointer, content)));
        }
        let location = {
            let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
            index
                .entries
                .get(id)
                .cloned()
                .context("memory index changed during recall")?
        };
        let mut file = File::open(&self.records_path)
            .await
            .with_context(|| format!("failed to open {}", self.records_path.display()))?;
        file.seek(SeekFrom::Start(location.offset))
            .await
            .with_context(|| format!("failed to seek {}", self.records_path.display()))?;
        let mut line = vec![0u8; location.length];
        file.read_exact(&mut line)
            .await
            .with_context(|| format!("failed to read {}", self.records_path.display()))?;
        let item = match serde_json::from_slice::<StoreRecord>(trim_line_ending(&line))
            .context("failed to parse indexed memory record")?
        {
            StoreRecord::Item { item } => item,
            StoreRecord::Turn { .. } | StoreRecord::Audit { .. } => {
                bail!("memory index points to a non-item record")
            }
        };
        if item.pointer.id != id {
            bail!("memory index for {id} points to {}", item.pointer.id);
        }
        let content = Arc::new(match item.encryption {
            Some(encryption) => self
                .cipher
                .as_ref()
                .context("memory content is encrypted; set ZEX_MEMORY_ENCRYPTION_KEY to unlock it")?
                .decrypt(id, &item.content, &encryption)?,
            None => item.content,
        });
        self.cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(id.to_owned(), (*content).clone());
        Ok(Some((pointer, content)))
    }

    fn pointer(&self, id: &str, active_turn: Option<&str>) -> Option<MemoryPointer> {
        let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
        let location = index.entries.get(id)?;
        Self::location_is_visible(&index, location, active_turn).then(|| location.pointer.clone())
    }

    fn pointer_by_message_fingerprint(
        &self,
        fingerprint: &str,
        active_turn: Option<&str>,
    ) -> Option<MemoryPointer> {
        let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
        let id = index.message_fingerprints.get(fingerprint)?;
        let location = index.entries.get(id)?;
        Self::location_is_visible(&index, location, active_turn).then(|| location.pointer.clone())
    }

    fn pointer_by_tool_fingerprint(
        &self,
        fingerprint: &str,
        active_turn: Option<&str>,
    ) -> Option<MemoryPointer> {
        let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
        let id = index.tool_fingerprints.get(fingerprint)?;
        let location = index.entries.get(id)?;
        Self::location_is_visible(&index, location, active_turn).then(|| location.pointer.clone())
    }

    fn pointers(&self, active_turn: Option<&str>) -> Vec<MemoryPointer> {
        let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
        index
            .order
            .iter()
            .filter_map(|id| index.entries.get(id))
            .filter(|location| Self::location_is_visible(&index, location, active_turn))
            .map(|entry| entry.pointer.clone())
            .collect()
    }

    fn location_is_visible(
        index: &StoreIndex,
        location: &RecordLocation,
        active_turn: Option<&str>,
    ) -> bool {
        let Some(turn_id) = location.turn_id.as_deref() else {
            return true;
        };
        match index.turn_states.get(turn_id) {
            Some(MemoryTurnState::Committed) => true,
            Some(MemoryTurnState::Aborted) => false,
            None => active_turn == Some(turn_id),
        }
    }

    async fn set_pinned(
        &self,
        id: &str,
        pinned: bool,
        active_turn: Option<&str>,
    ) -> Result<MemoryPointer> {
        let _pin = self.pin_lock.lock().await;
        let action = if pinned {
            AuditAction::Pin
        } else {
            AuditAction::Unpin
        };
        let Some(pointer) = self.pointer(id, active_turn) else {
            let message = missing_id_error(id);
            self.append_audit(AuditEvent {
                action,
                id: id.to_owned(),
                created_at: timestamp()?,
                reason: None,
                returned_tokens: None,
                truncated: false,
                error: Some(message.clone()),
            })
            .await?;
            bail!(message);
        };

        let previous = pointer.pinned;
        let previous_override = {
            let mut overrides = self
                .pin_overrides
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            overrides.insert(id.to_owned(), pinned)
        };
        {
            let mut index = self.index.lock().unwrap_or_else(|error| error.into_inner());
            let entry = index
                .entries
                .get_mut(id)
                .context("memory record disappeared while updating pin state")?;
            entry.pointer.pinned = pinned;
        }
        let overrides = self
            .pin_overrides
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        if let Err(error) = self.persist_pin_state(&overrides).await {
            self.restore_pin_state(id, previous, previous_override)
                .await?;
            return Err(error);
        }

        if let Err(error) = self
            .append_audit(AuditEvent {
                action,
                id: id.to_owned(),
                created_at: timestamp()?,
                reason: None,
                returned_tokens: None,
                truncated: false,
                error: None,
            })
            .await
        {
            self.restore_pin_state(id, previous, previous_override)
                .await?;
            return Err(error);
        }

        let mut updated = pointer;
        updated.pinned = pinned;
        Ok(updated)
    }

    async fn enforce_auto_pin_limit(&self, limit: usize, active_turn: Option<&str>) -> Result<()> {
        let explicit = self
            .pin_overrides
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .keys()
            .cloned()
            .collect::<HashSet<_>>();
        let excess = {
            let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
            let automatic = index
                .order
                .iter()
                .filter_map(|id| index.entries.get(id))
                .filter(|location| Self::location_is_visible(&index, location, active_turn))
                .filter(|location| {
                    location.pointer.pinned && !explicit.contains(&location.pointer.id)
                })
                .map(|location| location.pointer.id.clone())
                .collect::<Vec<_>>();
            let count = automatic.len().saturating_sub(limit);
            automatic.into_iter().take(count).collect::<Vec<_>>()
        };
        for id in excess {
            self.set_pinned(&id, false, active_turn).await?;
        }
        Ok(())
    }

    async fn restore_pin_state(
        &self,
        id: &str,
        pinned: bool,
        previous_override: Option<bool>,
    ) -> Result<()> {
        {
            let mut index = self.index.lock().unwrap_or_else(|error| error.into_inner());
            let entry = index
                .entries
                .get_mut(id)
                .context("memory record disappeared while restoring pin state")?;
            entry.pointer.pinned = pinned;
        }
        let overrides = {
            let mut overrides = self
                .pin_overrides
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            match previous_override {
                Some(previous) => {
                    overrides.insert(id.to_owned(), previous);
                }
                None => {
                    overrides.remove(id);
                }
            }
            overrides.clone()
        };
        self.persist_pin_state(&overrides)
            .await
            .context("failed to restore pin state after persistence error")
    }

    async fn persist_pin_state(&self, overrides: &BTreeMap<String, bool>) -> Result<()> {
        tokio::fs::create_dir_all(&self.directory)
            .await
            .with_context(|| {
                format!(
                    "failed to create memory directory {}",
                    self.directory.display()
                )
            })?;
        let content = serde_json::to_vec(&PinState {
            format: PIN_STATE_FORMAT,
            overrides: overrides.clone(),
        })
        .context("failed to serialize memory pin state")?;
        atomic_write(&self.pin_state_path, &content)
            .await
            .with_context(|| format!("failed to save {}", self.pin_state_path.display()))
    }
}

#[derive(Debug)]
pub struct MemoryRuntime {
    config: MemoryConfig,
    session: RwLock<Option<Arc<SessionMemory>>>,
    active_turn: Mutex<Option<String>>,
    active_pointers: Mutex<Vec<String>>,
    recall_limiter: Mutex<RecallLimiter>,
}

impl MemoryRuntime {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            session: RwLock::new(None),
            active_turn: Mutex::new(None),
            active_pointers: Mutex::new(Vec::new()),
            recall_limiter: Mutex::new(RecallLimiter::default()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn mode(&self) -> MemoryMode {
        self.config.mode
    }

    pub fn max_recall_tokens(&self) -> usize {
        self.config.max_recall_tokens
    }

    pub async fn activate(&self, session_id: &str, directory: PathBuf) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        let session = SessionMemory::open(
            session_id.to_owned(),
            directory,
            self.config.hot_cache_size,
            self.config.encryption_key.as_ref(),
        )
        .await?;
        session
            .enforce_auto_pin_limit(self.config.max_auto_pins, None)
            .await?;
        *self
            .session
            .write()
            .unwrap_or_else(|error| error.into_inner()) = Some(Arc::new(session));
        *self
            .active_turn
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
        self.active_pointers
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        self.recall_limiter
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .begin_model_turn();
        Ok(())
    }

    pub async fn maintain(&self) -> Result<()> {
        let session = self.active_session()?;
        session
            .enforce_auto_pin_limit(self.config.max_auto_pins, None)
            .await?;
        let protected = self
            .active_pointers
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        if !session
            .enforce_retention(
                self.config.max_records,
                self.config.max_store_bytes,
                self.config.retention_days,
                &protected,
            )
            .await?
        {
            return Ok(());
        }
        let reopened = SessionMemory::open(
            session.session_id.clone(),
            session.directory.clone(),
            self.config.hot_cache_size,
            self.config.encryption_key.as_ref(),
        )
        .await?;
        *self
            .session
            .write()
            .unwrap_or_else(|error| error.into_inner()) = Some(Arc::new(reopened));
        Ok(())
    }

    pub fn begin_turn(&self) -> Result<()> {
        let session = self.active_session()?;
        let mut active_turn = self
            .active_turn
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if active_turn.is_some() {
            bail!("addressable memory already has an active turn");
        }
        *active_turn = Some(session.allocate_turn_id());
        Ok(())
    }

    pub async fn commit_turn(&self) -> Result<()> {
        self.finish_active_turn(MemoryTurnState::Committed).await
    }

    pub async fn abort_turn(&self) -> Result<()> {
        self.finish_active_turn(MemoryTurnState::Aborted).await
    }

    async fn finish_active_turn(&self, state: MemoryTurnState) -> Result<()> {
        let Some(turn_id) = self.active_turn_id() else {
            return Ok(());
        };
        self.active_session()?.finish_turn(&turn_id, state).await?;
        let mut active_turn = self
            .active_turn
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if active_turn.as_deref() == Some(&turn_id) {
            *active_turn = None;
        }
        Ok(())
    }

    pub fn begin_model_turn(&self) {
        self.recall_limiter
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .begin_model_turn();
    }

    pub async fn store_tool_result(
        &self,
        tool: &str,
        arguments: &Value,
        content: String,
    ) -> Result<MemoryPointer> {
        let session = self.active_session()?;
        let active_turn = self.active_turn_id();
        let fingerprint = tool_result_fingerprint(tool, arguments, &content)?;
        let _dedupe = session.dedupe_lock.lock().await;
        if let Some(pointer) =
            session.pointer_by_tool_fingerprint(&fingerprint, active_turn.as_deref())
        {
            return Ok(pointer);
        }
        let importance = tool_importance(tool, &content);
        let mut metadata = tool_metadata(arguments);
        metadata.insert("fingerprint".to_owned(), fingerprint);
        let pointer = session
            .store(NewMemoryItem {
                kind: if tool == "read" {
                    MemoryKind::FileSnapshot
                } else {
                    MemoryKind::ToolResult
                },
                content,
                source_tool: Some(tool.to_owned()),
                importance,
                pinned: self.config.auto_pin_important && importance >= 85,
                parent_id: None,
                metadata,
                turn_id: active_turn.clone(),
            })
            .await
            .with_context(|| format!("failed to persist {tool} result in addressable memory"))?;
        if pointer.pinned && active_turn.is_none() {
            session
                .enforce_auto_pin_limit(self.config.max_auto_pins, active_turn.as_deref())
                .await
                .context("failed to enforce automatic memory pin limit")?;
        }
        Ok(pointer)
    }

    pub async fn store_message(
        &self,
        kind: MemoryKind,
        role: &str,
        content: String,
    ) -> Result<MemoryPointer> {
        let session = self.active_session()?;
        let active_turn = self.active_turn_id();
        let fingerprint = message_fingerprint(role, &content);
        let _dedupe = session.dedupe_lock.lock().await;
        if let Some(pointer) =
            session.pointer_by_message_fingerprint(&fingerprint, active_turn.as_deref())
        {
            return Ok(pointer);
        }
        let mut metadata = BTreeMap::new();
        metadata.insert("role".to_owned(), role.to_owned());
        if self.config.encryption_key.is_none() {
            metadata.insert("preview".to_owned(), one_line_preview(&content, 96));
        }
        metadata.insert("fingerprint".to_owned(), fingerprint);
        session
            .store(NewMemoryItem {
                kind,
                content,
                source_tool: None,
                importance: 70,
                pinned: false,
                parent_id: None,
                metadata,
                turn_id: active_turn,
            })
            .await
            .context("failed to persist compacted conversation in addressable memory")
    }

    pub fn render_tool_result(&self, pointer: &MemoryPointer, visible_output: String) -> String {
        match self.config.mode {
            MemoryMode::Summary => format!(
                "{visible_output}\n\n[addressable copy] {}",
                self.citation(pointer)
            ),
            MemoryMode::PointerPriority | MemoryMode::Hybrid
                if pointer.token_estimate
                    > self.config.max_recall_tokens.min(MAX_INLINE_TOOL_TOKENS) =>
            {
                self.citation(pointer)
            }
            MemoryMode::PointerPriority | MemoryMode::Hybrid => {
                format!(
                    "{visible_output}\n\n[addressable copy] {}",
                    self.citation(pointer)
                )
            }
        }
    }

    pub fn citation_for_id(&self, id: &str) -> Option<String> {
        let active_turn = self.active_turn_id();
        self.active_session()
            .ok()?
            .pointer(id, active_turn.as_deref())
            .map(|pointer| self.citation(&pointer))
    }

    pub fn pointer_for_id(&self, id: &str) -> Option<MemoryPointer> {
        let active_turn = self.active_turn_id();
        self.active_session()
            .ok()?
            .pointer(id, active_turn.as_deref())
    }

    pub fn contains(&self, id: &str) -> bool {
        self.pointer_for_id(id).is_some()
    }

    pub fn set_active_pointers(&self, ids: impl IntoIterator<Item = String>) {
        let Ok(session) = self.active_session() else {
            return;
        };
        let active_turn = self.active_turn_id();
        let mut seen = HashSet::new();
        let active = ids
            .into_iter()
            .filter(|id| {
                seen.insert(id.clone()) && session.pointer(id, active_turn.as_deref()).is_some()
            })
            .collect();
        *self
            .active_pointers
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = active;
    }

    pub fn system_pointer_manifest(&self) -> Vec<String> {
        self.pointer_manifest(
            self.active_pointers
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .iter()
                .cloned(),
            SYSTEM_POINTER_LIMIT,
            true,
        )
    }

    pub fn compaction_pointer_manifest(
        &self,
        ids: impl IntoIterator<Item = String>,
    ) -> Vec<String> {
        self.pointer_manifest(ids, ACTIVE_POINTER_LIMIT, true)
    }

    pub async fn recall(
        &self,
        id: &str,
        reason: Option<String>,
        offset_tokens: Option<usize>,
        max_tokens: Option<usize>,
    ) -> Result<String> {
        let session = self.active_session()?;
        let active_turn = self.active_turn_id();
        let offset_tokens = offset_tokens.unwrap_or(0);
        let max_tokens = max_tokens
            .unwrap_or(self.config.max_recall_tokens)
            .min(self.config.max_recall_tokens);
        if max_tokens == 0 {
            bail!("recall max_tokens must be greater than zero");
        }
        let rate_result = self
            .recall_limiter
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .acquire(self.config.recall_rate_limit);
        if let Err(error) = rate_result {
            let message = error.to_string();
            session
                .append_audit(AuditEvent {
                    action: AuditAction::RecallRateLimited,
                    id: id.to_owned(),
                    created_at: timestamp()?,
                    reason,
                    returned_tokens: None,
                    truncated: false,
                    error: Some(message.clone()),
                })
                .await?;
            bail!(message);
        }
        if let Err(error) = validate_memory_id(id) {
            let message = format!("invalid memory ID {id:?}: {error}");
            session
                .append_audit(AuditEvent {
                    action: AuditAction::RecallMissing,
                    id: id.to_owned(),
                    created_at: timestamp()?,
                    reason,
                    returned_tokens: None,
                    truncated: false,
                    error: Some(message.clone()),
                })
                .await?;
            bail!("{message}; do not infer or invent its contents");
        }
        let Some((pointer, content)) = session.read(id, active_turn.as_deref()).await? else {
            let message = missing_id_error(id);
            session
                .append_audit(AuditEvent {
                    action: AuditAction::RecallMissing,
                    id: id.to_owned(),
                    created_at: timestamp()?,
                    reason,
                    returned_tokens: None,
                    truncated: false,
                    error: Some(message.clone()),
                })
                .await?;
            bail!(message);
        };
        if offset_tokens >= pointer.token_estimate && pointer.token_estimate > 0 {
            let message = format!(
                "recall offset {offset_tokens} is outside {id}, which contains approximately {} tokens",
                pointer.token_estimate
            );
            session
                .append_audit(AuditEvent {
                    action: AuditAction::Recall,
                    id: id.to_owned(),
                    created_at: timestamp()?,
                    reason,
                    returned_tokens: None,
                    truncated: false,
                    error: Some(message.clone()),
                })
                .await?;
            bail!(message);
        }
        let window = token_window(&content, offset_tokens, max_tokens);
        let truncated = window.start > 0 || window.end < window.total;
        let returned_tokens = window.end.saturating_sub(window.start);
        session
            .append_audit(AuditEvent {
                action: AuditAction::Recall,
                id: id.to_owned(),
                created_at: timestamp()?,
                reason,
                returned_tokens: Some(returned_tokens),
                truncated,
                error: None,
            })
            .await?;
        if truncated {
            let continuation = if window.end < window.total {
                format!("; next_offset={}", window.end)
            } else {
                "; end of content".to_owned()
            };
            Ok(format!(
                "[recalled {id}: token window {}..{} of ~{} tokens{continuation}]\n{}",
                window.start, window.end, window.total, window.content
            ))
        } else {
            Ok(format!(
                "[recalled {id}: exact content, ~{} tokens]\n{}",
                pointer.token_estimate, window.content
            ))
        }
    }

    pub async fn pin(&self, id: &str) -> Result<String> {
        let pointer = self.update_pin(id, true).await?;
        Ok(format!("Pinned {}", self.citation(&pointer)))
    }

    pub async fn unpin(&self, id: &str) -> Result<String> {
        let pointer = self.update_pin(id, false).await?;
        Ok(format!("Unpinned {}", self.citation(&pointer)))
    }

    pub async fn list_pointers(&self, filter: Option<&str>) -> Result<String> {
        let session = self.active_session()?;
        let active_turn = self.active_turn_id();
        let filter = filter.map(str::trim).filter(|filter| !filter.is_empty());
        let parsed_filter = filter.map(parse_pointer_filters).transpose()?;
        let active = self
            .active_pointers
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        let candidates = if filter.is_some() {
            session.pointers(active_turn.as_deref())
        } else {
            let active_set = active.iter().cloned().collect::<HashSet<_>>();
            let mut pointers = session
                .pointers(active_turn.as_deref())
                .into_iter()
                .filter(|pointer| pointer.pinned && !active_set.contains(&pointer.id))
                .collect::<Vec<_>>();
            pointers.extend(
                active
                    .into_iter()
                    .filter_map(|id| session.pointer(&id, active_turn.as_deref())),
            );
            pointers
        };
        let mut matches = Vec::new();
        for pointer in candidates {
            let mut content = None;
            let mut matched = true;
            for filter in parsed_filter.as_deref().unwrap_or_default() {
                if self.pointer_matches_filter(&pointer, filter) {
                    continue;
                }
                let PointerFilter::Text(value) = filter else {
                    matched = false;
                    break;
                };
                if content.is_none() {
                    content = session
                        .read(&pointer.id, active_turn.as_deref())
                        .await?
                        .map(|(_, content)| content);
                }
                if !content
                    .as_deref()
                    .is_some_and(|content| contains_case_insensitive(content, value))
                {
                    matched = false;
                    break;
                }
            }
            if matched {
                matches.push(pointer);
            }
        }
        matches.sort_by(|left, right| {
            right
                .pinned
                .cmp(&left.pinned)
                .then_with(|| right.created_at.cmp(&left.created_at))
        });
        let total = matches.len();
        matches.truncate(ACTIVE_POINTER_LIMIT);
        if matches.is_empty() {
            return Ok(match filter {
                Some(filter) => format!("No addressable pointers match {filter:?}."),
                None => "No active or pinned addressable pointers.".to_owned(),
            });
        }
        let mut lines = Vec::with_capacity(matches.len());
        for pointer in &matches {
            let preview = session
                .read(&pointer.id, active_turn.as_deref())
                .await?
                .map(|(_, content)| one_line_preview(&content, 160))
                .unwrap_or_default();
            let suffix = if preview.is_empty() {
                String::new()
            } else {
                format!(" — {preview}")
            };
            lines.push(format!("- {}{suffix}", self.citation(pointer)));
        }
        let mut output = lines.join("\n");
        if total > matches.len() {
            output.push_str(&format!(
                "\n- [{} more pointers omitted; pass a narrower filter]",
                total - matches.len()
            ));
        }
        Ok(output)
    }

    fn pointer_matches_filter(&self, pointer: &MemoryPointer, filter: &PointerFilter) -> bool {
        match filter {
            PointerFilter::Text(value) => {
                contains_case_insensitive(&self.citation(pointer), value)
                    || pointer
                        .source_tool
                        .as_deref()
                        .is_some_and(|tool| contains_case_insensitive(tool, value))
                    || pointer
                        .metadata
                        .values()
                        .any(|candidate| contains_case_insensitive(candidate, value))
            }
            PointerFilter::Field { key, value } => match key.as_str() {
                "id" => contains_case_insensitive(&pointer.id, value),
                "kind" => pointer.kind.filter_name().eq_ignore_ascii_case(value),
                "tool" => pointer
                    .source_tool
                    .as_deref()
                    .is_some_and(|tool| tool.eq_ignore_ascii_case(value)),
                "role" | "path" | "pattern" | "file_glob" | "command" => pointer
                    .metadata
                    .get(key)
                    .is_some_and(|candidate| contains_case_insensitive(candidate, value)),
                "pinned" => pointer.pinned == (value == "true"),
                _ => unreachable!("pointer filter keys are validated during parsing"),
            },
        }
    }

    pub fn records_path(&self) -> Option<PathBuf> {
        self.session
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .map(|session| session.records_path.clone())
    }

    pub fn active_session_id(&self) -> Option<String> {
        self.session
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .map(|session| session.session_id.clone())
    }

    pub fn is_control_tool(name: &str) -> bool {
        matches!(name, "recall" | "pin" | "unpin" | "list_pointers")
    }

    async fn update_pin(&self, id: &str, pinned: bool) -> Result<MemoryPointer> {
        let session = self.active_session()?;
        let active_turn = self.active_turn_id();
        if let Err(error) = validate_memory_id(id) {
            let message = format!("invalid memory ID {id:?}: {error}");
            session
                .append_audit(AuditEvent {
                    action: if pinned {
                        AuditAction::Pin
                    } else {
                        AuditAction::Unpin
                    },
                    id: id.to_owned(),
                    created_at: timestamp()?,
                    reason: None,
                    returned_tokens: None,
                    truncated: false,
                    error: Some(message.clone()),
                })
                .await?;
            bail!(message);
        }
        session.set_pinned(id, pinned, active_turn.as_deref()).await
    }

    fn active_session(&self) -> Result<Arc<SessionMemory>> {
        if !self.config.enabled {
            bail!("addressable memory is disabled by configuration");
        }
        self.session
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
            .context("addressable memory has no active session")
    }

    fn active_turn_id(&self) -> Option<String> {
        self.active_turn
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    fn pointer_manifest(
        &self,
        ids: impl IntoIterator<Item = String>,
        limit: usize,
        include_pinned: bool,
    ) -> Vec<String> {
        let Ok(session) = self.active_session() else {
            return Vec::new();
        };
        let active_turn = self.active_turn_id();
        let mut seen = HashSet::new();
        let mut pinned = if include_pinned {
            session
                .pointers(active_turn.as_deref())
                .into_iter()
                .filter(|pointer| pointer.pinned && seen.insert(pointer.id.clone()))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if pinned.len() > limit {
            let split = pinned.len() - limit;
            pinned.drain(0..split);
        }
        let remaining = limit.saturating_sub(pinned.len());
        let mut recent = ids
            .into_iter()
            .filter(|id| seen.insert(id.clone()))
            .filter_map(|id| session.pointer(&id, active_turn.as_deref()))
            .collect::<Vec<_>>();
        if recent.len() > remaining {
            let split = recent.len() - remaining;
            recent.drain(0..split);
        }
        pinned.extend(recent);
        let pointers = pinned;
        pointers
            .iter()
            .map(|pointer| self.citation(pointer))
            .collect()
    }

    fn citation(&self, pointer: &MemoryPointer) -> String {
        let source = pointer
            .source_tool
            .as_deref()
            .or_else(|| pointer.metadata.get("role").map(String::as_str))
            .unwrap_or(pointer.kind.citation_label());
        let details = pointer
            .metadata
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "preview" | "fingerprint" | "role"))
            .take(2)
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(", ");
        let pinned = if pointer.pinned { ", pinned" } else { "" };
        if details.is_empty() {
            format!(
                "[{}] {source} → {} (~{} tokens{pinned}; recall available)",
                pointer.kind.citation_label(),
                pointer.id,
                compact_number(pointer.token_estimate)
            )
        } else {
            format!(
                "[{}] {source} → {} (~{} tokens, {details}{pinned}; recall available)",
                pointer.kind.citation_label(),
                pointer.id,
                compact_number(pointer.token_estimate)
            )
        }
    }
}

pub fn extract_memory_ids(content: &str) -> Vec<String> {
    let mut ids = Vec::new();
    for (offset, _) in content.match_indices('§') {
        let candidate = &content[offset..];
        for prefix in ["§obs_", "§turn_", "§decision_", "§summary_"] {
            let Some(rest) = candidate.strip_prefix(prefix) else {
                continue;
            };
            let hex = rest
                .bytes()
                .take_while(u8::is_ascii_hexdigit)
                .take(MEMORY_ID_HEX_LEN + 1)
                .collect::<Vec<_>>();
            if hex.len() == MEMORY_ID_HEX_LEN {
                let id = format!("{prefix}{}", String::from_utf8_lossy(&hex));
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
    }
    ids
}

pub fn validate_memory_id(id: &str) -> Result<&str> {
    let Some((prefix, suffix)) = id.split_once('_') else {
        bail!("expected §obs_<hex>, §turn_<hex>, §decision_<hex>, or §summary_<hex>");
    };
    if !matches!(prefix, "§obs" | "§turn" | "§decision" | "§summary")
        || suffix.len() != MEMORY_ID_HEX_LEN
        || !suffix.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("ID has an unsupported prefix or malformed hexadecimal suffix");
    }
    Ok(id)
}

fn parse_pointer_filters(filter: &str) -> Result<Vec<PointerFilter>> {
    filter
        .split_whitespace()
        .map(|term| {
            let Some((key, value)) = term.split_once('=') else {
                return Ok(PointerFilter::Text(term.to_ascii_lowercase()));
            };
            let key = key.to_ascii_lowercase();
            if !matches!(
                key.as_str(),
                "id"
                    | "kind"
                    | "tool"
                    | "role"
                    | "path"
                    | "pattern"
                    | "file_glob"
                    | "command"
                    | "pinned"
            ) {
                bail!(
                    "unsupported pointer filter field {key:?}; use id, kind, tool, role, path, pattern, file_glob, command, or pinned"
                );
            }
            if value.is_empty() {
                bail!("pointer filter field {key:?} requires a value");
            }
            let value = value.to_ascii_lowercase();
            if key == "pinned" && !matches!(value.as_str(), "true" | "false") {
                bail!("pointer filter pinned must be true or false");
            }
            Ok(PointerFilter::Field { key, value })
        })
        .collect()
}

fn contains_case_insensitive(candidate: &str, value: &str) -> bool {
    let value = value.as_bytes();
    if value.is_empty() {
        return true;
    }
    candidate
        .as_bytes()
        .windows(value.len())
        .any(|window| window.eq_ignore_ascii_case(value))
}

fn tool_metadata(arguments: &Value) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    for key in ["path", "pattern", "file_glob"] {
        if let Some(value) = arguments.get(key).and_then(Value::as_str) {
            metadata.insert(key.to_owned(), one_line_preview(value, 120));
        }
    }
    if let Some(command) = arguments.get("command").and_then(Value::as_str) {
        metadata.insert("command".to_owned(), one_line_preview(command, 120));
    }
    metadata
}

fn tool_result_fingerprint(tool: &str, arguments: &Value, content: &str) -> Result<String> {
    let arguments =
        serde_json::to_vec(arguments).context("failed to serialize tool arguments for memory")?;
    let mut bytes = Vec::with_capacity(tool.len() + arguments.len() + content.len() + 2);
    bytes.extend_from_slice(tool.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&arguments);
    bytes.push(0);
    bytes.extend_from_slice(content.as_bytes());
    Ok(fingerprint_bytes(&bytes))
}

fn tool_importance(tool: &str, content: &str) -> u8 {
    if matches!(tool, "write" | "edit") {
        90
    } else if content.starts_with("tool error:") {
        80
    } else if tool == "read" {
        70
    } else {
        60
    }
}

fn missing_id_error(id: &str) -> String {
    format!(
        "memory ID {id:?} does not exist in the active session; do not infer or invent its contents"
    )
}

fn timestamp() -> Result<String> {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .context("failed to format memory timestamp")
}

fn trim_line_ending(line: &[u8]) -> &[u8] {
    line.strip_suffix(b"\n")
        .unwrap_or(line)
        .strip_suffix(b"\r")
        .unwrap_or_else(|| line.strip_suffix(b"\n").unwrap_or(line))
}

fn one_line_preview(content: &str, max_chars: usize) -> String {
    let content = content.replace(['\r', '\n'], " ");
    let mut preview = content.chars().take(max_chars).collect::<String>();
    if content.chars().count() > max_chars {
        preview.push('…');
    }
    preview
}

fn compact_number(tokens: usize) -> String {
    if tokens < 1_000 {
        return tokens.to_string();
    }
    format!("{:.1}k", tokens as f64 / 1_000.0)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_seed(bytes, 0xcbf2_9ce4_8422_2325)
}

fn fnv1a64_seed(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn message_fingerprint(role: &str, content: &str) -> String {
    let mut bytes = Vec::with_capacity(role.len() + content.len() + 1);
    bytes.extend_from_slice(role.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(content.as_bytes());
    fingerprint_bytes(&bytes)
}

fn fingerprint_bytes(bytes: &[u8]) -> String {
    format!(
        "{:016x}{:016x}",
        fnv1a64(bytes),
        fnv1a64_seed(bytes, 0x8422_2325_cbf2_9ce4)
    )
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, process, time::SystemTime};

    use serde_json::json;

    use super::{
        AUDIT_ARCHIVE_LIMIT, AUDIT_FILE, MemoryConfig, MemoryEncryptionKey, MemoryKind,
        MemoryRuntime, PIN_STATE_FILE, extract_memory_ids,
    };

    fn temporary_directory(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("zex-memory-{label}-{}-{unique}", process::id()))
    }

    #[tokio::test]
    async fn persists_recalls_pins_limits_and_reopens_records() {
        let directory = temporary_directory("store");
        let config = MemoryConfig {
            recall_rate_limit: 5,
            max_recall_tokens: 64,
            hot_cache_size: 2,
            ..MemoryConfig::default()
        };
        let runtime = MemoryRuntime::new(config.clone());
        runtime
            .activate("20260815-120000-deadbeef", directory.clone())
            .await
            .unwrap();
        let content = "precise observation".repeat(4);
        let pointer = runtime
            .store_tool_result("read", &json!({"path": "src/main.rs"}), content.clone())
            .await
            .unwrap();

        assert!(pointer.id.starts_with("§obs_"));
        assert_eq!(
            extract_memory_ids(&runtime.citation(&pointer)),
            vec![pointer.id.clone()]
        );
        let recalled = runtime
            .recall(
                &pointer.id,
                Some("verify exact source".to_owned()),
                None,
                None,
            )
            .await
            .unwrap();
        assert!(recalled.ends_with(&content));
        assert!(runtime.pin(&pointer.id).await.unwrap().contains("Pinned"));
        assert!(
            runtime
                .list_pointers(None)
                .await
                .unwrap()
                .contains(&pointer.id)
        );
        assert!(
            runtime
                .unpin(&pointer.id)
                .await
                .unwrap()
                .contains("Unpinned")
        );
        let missing_pin = runtime
            .pin("§obs_000000000000000000000000")
            .await
            .unwrap_err();
        assert!(missing_pin.to_string().contains("does not exist"));

        let missing = runtime
            .recall("§obs_000000000000000000000000", None, None, None)
            .await
            .unwrap_err();
        assert!(missing.to_string().contains("does not exist"));

        runtime.begin_model_turn();
        for _ in 0..3 {
            runtime.recall(&pointer.id, None, None, None).await.unwrap();
        }
        let limited = runtime
            .recall(&pointer.id, None, None, None)
            .await
            .unwrap_err();
        assert!(limited.to_string().contains("rate limit exceeded"));

        let records_path = runtime.records_path().unwrap();
        let log = tokio::fs::read_to_string(&records_path).await.unwrap();
        assert!(log.contains("\"record_type\":\"item\""));
        assert!(!log.contains("\"action\":"));
        let audit = tokio::fs::read_to_string(directory.join(AUDIT_FILE))
            .await
            .unwrap();
        assert!(audit.contains("\"action\":\"recall\""));
        assert!(audit.contains("\"action\":\"pin\""));
        assert!(audit.contains("\"error\":\"memory ID"));
        let pin_state = tokio::fs::read_to_string(directory.join(PIN_STATE_FILE))
            .await
            .unwrap();
        assert!(pin_state.contains(&pointer.id));
        assert!(pin_state.contains("false"));

        drop(runtime);
        let reopened = MemoryRuntime::new(config);
        reopened
            .activate("20260815-120000-deadbeef", directory.clone())
            .await
            .unwrap();
        let recalled = reopened
            .recall(&pointer.id, None, None, None)
            .await
            .unwrap();
        assert!(recalled.ends_with(&content));
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn recall_returns_a_bounded_excerpt_for_large_content() {
        let directory = temporary_directory("bounded");
        let runtime = MemoryRuntime::new(MemoryConfig {
            max_recall_tokens: 32,
            ..MemoryConfig::default()
        });
        runtime
            .activate("20260815-120000-feedface", directory.clone())
            .await
            .unwrap();
        let pointer = runtime
            .store_tool_result(
                "read",
                &json!({"path": "large.txt"}),
                "large ".repeat(2_000),
            )
            .await
            .unwrap();

        let recalled = runtime.recall(&pointer.id, None, None, None).await.unwrap();

        assert!(recalled.contains("token window"));
        assert!(crate::agent::estimate_tokens(&recalled) < 96);
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn recall_pages_to_middle_and_tail_windows() {
        let directory = temporary_directory("paged");
        let runtime = MemoryRuntime::new(MemoryConfig {
            recall_rate_limit: 10,
            max_recall_tokens: 16,
            ..MemoryConfig::default()
        });
        runtime
            .activate("20260815-120000-abcdef12", directory.clone())
            .await
            .unwrap();
        let content = format!("{}TAIL_SENTINEL", "pageable-content ".repeat(200));
        let pointer = runtime
            .store_tool_result("read", &json!({"path": "paged.txt"}), content)
            .await
            .unwrap();

        let first = runtime
            .recall(&pointer.id, None, Some(0), Some(8))
            .await
            .unwrap();
        let middle = runtime
            .recall(&pointer.id, None, Some(8), Some(8))
            .await
            .unwrap();
        let tail = runtime
            .recall(
                &pointer.id,
                None,
                Some(pointer.token_estimate.saturating_sub(16)),
                Some(16),
            )
            .await
            .unwrap();

        assert!(first.contains("next_offset="));
        assert_ne!(first, middle);
        assert!(tail.contains("TAIL_SENTINEL"));

        runtime.begin_model_turn();
        let error = runtime
            .recall(&pointer.id, None, Some(pointer.token_estimate), Some(8))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("outside"));

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn deduplicates_tool_results_and_supports_structured_filters() {
        let directory = temporary_directory("dedupe-filter");
        let runtime = MemoryRuntime::new(MemoryConfig::default());
        runtime
            .activate("20260815-120000-01020304", directory.clone())
            .await
            .unwrap();

        let first = runtime
            .store_tool_result(
                "read",
                &json!({"path": "src/main.rs"}),
                "same source".to_owned(),
            )
            .await
            .unwrap();
        let duplicate = runtime
            .store_tool_result(
                "read",
                &json!({"path": "src/main.rs"}),
                "same source".to_owned(),
            )
            .await
            .unwrap();
        let search = runtime
            .store_tool_result(
                "grep",
                &json!({"pattern": "needle", "path": "src"}),
                "src/main.rs:1:needle".to_owned(),
            )
            .await
            .unwrap();

        assert_eq!(duplicate.id, first.id);
        assert_ne!(search.id, first.id);
        let records = tokio::fs::read_to_string(runtime.records_path().unwrap())
            .await
            .unwrap();
        assert_eq!(records.lines().count(), 2);

        let read_matches = runtime
            .list_pointers(Some("tool=read path=src/main.rs kind=file_snapshot"))
            .await
            .unwrap();
        assert!(read_matches.contains(&first.id));
        assert!(!read_matches.contains(&search.id));
        let grep_matches = runtime
            .list_pointers(Some("tool=grep pattern=needle"))
            .await
            .unwrap();
        assert!(grep_matches.contains(&search.id));
        let content_matches = runtime.list_pointers(Some("same source")).await.unwrap();
        assert!(content_matches.contains(&first.id));
        assert!(content_matches.contains("same source"));
        let invalid = runtime
            .list_pointers(Some("unknown=value"))
            .await
            .unwrap_err();
        assert!(
            invalid
                .to_string()
                .contains("unsupported pointer filter field")
        );

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn automatic_pins_are_bounded_without_evicting_manual_pins() {
        let directory = temporary_directory("pin-lifecycle");
        let runtime = MemoryRuntime::new(MemoryConfig {
            max_auto_pins: 1,
            ..MemoryConfig::default()
        });
        runtime
            .activate("20260815-120000-31323334", directory.clone())
            .await
            .unwrap();
        let first = runtime
            .store_tool_result(
                "write",
                &json!({"path": "first.txt"}),
                "first write".to_owned(),
            )
            .await
            .unwrap();
        let second = runtime
            .store_tool_result(
                "edit",
                &json!({"path": "second.txt"}),
                "second edit".to_owned(),
            )
            .await
            .unwrap();
        assert!(!runtime.pointer_for_id(&first.id).unwrap().pinned);
        assert!(runtime.pointer_for_id(&second.id).unwrap().pinned);

        runtime.pin(&first.id).await.unwrap();
        let third = runtime
            .store_tool_result(
                "write",
                &json!({"path": "third.txt"}),
                "third write".to_owned(),
            )
            .await
            .unwrap();
        assert!(runtime.pointer_for_id(&first.id).unwrap().pinned);
        assert!(!runtime.pointer_for_id(&second.id).unwrap().pinned);
        assert!(runtime.pointer_for_id(&third.id).unwrap().pinned);

        runtime.begin_turn().unwrap();
        let aborted = runtime
            .store_tool_result(
                "edit",
                &json!({"path": "aborted.txt"}),
                "aborted edit".to_owned(),
            )
            .await
            .unwrap();
        runtime.abort_turn().await.unwrap();
        runtime.maintain().await.unwrap();
        assert!(runtime.pointer_for_id(&aborted.id).is_none());
        assert!(runtime.pointer_for_id(&third.id).unwrap().pinned);

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn retention_compacts_records_but_preserves_active_pointers() {
        let directory = temporary_directory("retention");
        let runtime = MemoryRuntime::new(MemoryConfig {
            max_records: 10,
            max_store_bytes: 2_500,
            ..MemoryConfig::default()
        });
        runtime
            .activate("20260815-120000-41424344", directory.clone())
            .await
            .unwrap();
        let mut pointers = Vec::new();
        for index in 0..4 {
            pointers.push(
                runtime
                    .store_tool_result(
                        "read",
                        &json!({"path": format!("{index}.txt")}),
                        format!("record {index} {}", "x".repeat(1_600)),
                    )
                    .await
                    .unwrap(),
            );
        }
        runtime.set_active_pointers([pointers[0].id.clone(), pointers[3].id.clone()]);
        runtime.maintain().await.unwrap();

        assert!(runtime.contains(&pointers[0].id));
        assert!(runtime.contains(&pointers[3].id));
        assert!(!runtime.contains(&pointers[1].id));
        assert!(!runtime.contains(&pointers[2].id));
        let records = tokio::fs::read_to_string(runtime.records_path().unwrap())
            .await
            .unwrap();
        assert_eq!(records.lines().count(), 2);

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn encrypted_memory_never_writes_plaintext_and_requires_the_key() {
        let directory = temporary_directory("encrypted");
        let session_id = "20260815-120000-51525354";
        let key = MemoryEncryptionKey::new("correct horse battery staple".to_owned()).unwrap();
        let config = MemoryConfig {
            encryption_key: Some(key.clone()),
            ..MemoryConfig::default()
        };
        let runtime = MemoryRuntime::new(config.clone());
        runtime
            .activate(session_id, directory.clone())
            .await
            .unwrap();
        let secret = "private observation that must stay encrypted";
        let pointer = runtime
            .store_tool_result("read", &json!({"path": "secret.txt"}), secret.to_owned())
            .await
            .unwrap();
        let stored = tokio::fs::read_to_string(runtime.records_path().unwrap())
            .await
            .unwrap();
        assert!(!stored.contains(secret));
        assert!(stored.contains("\"encryption\""));
        drop(runtime);

        let reopened = MemoryRuntime::new(config);
        reopened
            .activate(session_id, directory.clone())
            .await
            .unwrap();
        assert!(
            reopened
                .recall(
                    &pointer.id,
                    Some("private audit reason".to_owned()),
                    None,
                    None,
                )
                .await
                .unwrap()
                .ends_with(secret)
        );
        let audit = tokio::fs::read_to_string(directory.join(AUDIT_FILE))
            .await
            .unwrap();
        assert!(!audit.contains("private audit reason"));
        assert!(audit.contains("\"encryption\""));
        drop(reopened);

        let locked = MemoryRuntime::new(MemoryConfig::default());
        let error = locked
            .activate(session_id, directory.clone())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("ZEX_MEMORY_ENCRYPTION_KEY"));

        let wrong_key = MemoryRuntime::new(MemoryConfig {
            encryption_key: MemoryEncryptionKey::new("wrong key".to_owned()),
            ..MemoryConfig::default()
        });
        let error = wrong_key
            .activate(session_id, directory.clone())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("authentication failed"));

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn enabling_encryption_migrates_existing_memory_records() {
        let directory = temporary_directory("encryption-migration");
        let session_id = "20260815-120000-61626364";
        let secret = "legacy plaintext memory";
        let plain = MemoryRuntime::new(MemoryConfig::default());
        plain.activate(session_id, directory.clone()).await.unwrap();
        let pointer = plain
            .store_message(MemoryKind::Message, "user", secret.to_owned())
            .await
            .unwrap();
        drop(plain);

        let encrypted = MemoryRuntime::new(MemoryConfig {
            encryption_key: MemoryEncryptionKey::new("migration key".to_owned()),
            ..MemoryConfig::default()
        });
        encrypted
            .activate(session_id, directory.clone())
            .await
            .unwrap();
        encrypted.maintain().await.unwrap();
        let stored = tokio::fs::read_to_string(encrypted.records_path().unwrap())
            .await
            .unwrap();
        assert!(!stored.contains(secret));
        assert!(
            encrypted
                .recall(&pointer.id, None, None, None)
                .await
                .unwrap()
                .ends_with(secret)
        );

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn pin_snapshot_survives_reopen_without_masking_new_automatic_pins() {
        let directory = temporary_directory("pin-state");
        let config = MemoryConfig::default();
        let runtime = MemoryRuntime::new(config.clone());
        runtime
            .activate("20260815-120000-11121314", directory.clone())
            .await
            .unwrap();
        let manually_unpinned = runtime
            .store_tool_result(
                "write",
                &json!({"path": "first.txt"}),
                "wrote first".to_owned(),
            )
            .await
            .unwrap();
        let manually_pinned = runtime
            .store_tool_result("read", &json!({"path": "source.txt"}), "source".to_owned())
            .await
            .unwrap();
        runtime.unpin(&manually_unpinned.id).await.unwrap();
        runtime.pin(&manually_pinned.id).await.unwrap();
        drop(runtime);

        let reopened = MemoryRuntime::new(config.clone());
        reopened
            .activate("20260815-120000-11121314", directory.clone())
            .await
            .unwrap();
        assert!(
            !reopened
                .pointer_for_id(&manually_unpinned.id)
                .unwrap()
                .pinned
        );
        assert!(reopened.pointer_for_id(&manually_pinned.id).unwrap().pinned);
        let automatically_pinned = reopened
            .store_tool_result(
                "edit",
                &json!({"path": "second.txt"}),
                "edited second".to_owned(),
            )
            .await
            .unwrap();
        assert!(automatically_pinned.pinned);
        drop(reopened);

        let reopened_again = MemoryRuntime::new(config);
        reopened_again
            .activate("20260815-120000-11121314", directory.clone())
            .await
            .unwrap();
        assert!(
            reopened_again
                .pointer_for_id(&automatically_pinned.id)
                .unwrap()
                .pinned
        );

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn rotates_audit_logs_and_keeps_records_free_of_operational_events() {
        let directory = temporary_directory("audit-rotation");
        let runtime = MemoryRuntime::new(MemoryConfig::default());
        runtime
            .activate("20260815-120000-21222324", directory.clone())
            .await
            .unwrap();
        let pointer = runtime
            .store_tool_result(
                "read",
                &json!({"path": "rotate.txt"}),
                "rotation source".to_owned(),
            )
            .await
            .unwrap();

        for index in 0..240 {
            if index % 2 == 0 {
                runtime.pin(&pointer.id).await.unwrap();
            } else {
                runtime.unpin(&pointer.id).await.unwrap();
            }
        }

        let records = tokio::fs::read_to_string(runtime.records_path().unwrap())
            .await
            .unwrap();
        assert!(!records.contains("\"action\":"));
        let mut entries = tokio::fs::read_dir(&directory).await.unwrap();
        let mut archives = 0usize;
        let mut current = 0usize;
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("audit-") && name.ends_with(".jsonl") {
                archives += 1;
            } else if name == AUDIT_FILE {
                current += 1;
            }
        }
        assert!(archives > 0);
        assert!(archives <= AUDIT_ARCHIVE_LIMIT);
        assert_eq!(current, 1);

        tokio::fs::remove_dir_all(directory).await.unwrap();
    }
}
