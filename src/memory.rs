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

use crate::agent::{estimate_tokens, truncate_to_token_budget};

const RECORDS_FILE: &str = "records.jsonl";
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryConfig {
    pub enabled: bool,
    pub mode: MemoryMode,
    pub recall_rate_limit: usize,
    pub max_recall_tokens: usize,
    pub hot_cache_size: usize,
    pub auto_pin_important: bool,
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
#[serde(tag = "record_type", rename_all = "snake_case")]
enum StoreRecord {
    Item { item: MemoryItem },
    Audit { audit: AuditEvent },
}

#[derive(Debug, Clone)]
struct RecordLocation {
    offset: u64,
    length: usize,
    pointer: MemoryPointer,
}

#[derive(Debug, Default)]
struct StoreIndex {
    entries: HashMap<String, RecordLocation>,
    order: Vec<String>,
    message_fingerprints: HashMap<String, String>,
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
    session_hash: u64,
    next_sequence: AtomicU64,
    append_lock: AsyncMutex<()>,
    index: Mutex<StoreIndex>,
    cache: Mutex<HotCache>,
}

impl SessionMemory {
    async fn open(session_id: String, directory: PathBuf, cache_size: usize) -> Result<Self> {
        let records_path = directory.join(RECORDS_FILE);
        let mut index = StoreIndex::default();
        let mut item_count = 0u64;

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
                            validate_memory_id(&item.pointer.id)?;
                            if index.entries.contains_key(&item.pointer.id) {
                                bail!(
                                    "memory store {} reuses ID {}",
                                    records_path.display(),
                                    item.pointer.id
                                );
                            }
                            item_count += 1;
                            index.order.push(item.pointer.id.clone());
                            if let Some(fingerprint) = item.pointer.metadata.get("fingerprint") {
                                index
                                    .message_fingerprints
                                    .entry(fingerprint.clone())
                                    .or_insert_with(|| item.pointer.id.clone());
                            }
                            index.entries.insert(
                                item.pointer.id.clone(),
                                RecordLocation {
                                    offset,
                                    length,
                                    pointer: item.pointer,
                                },
                            );
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

        Ok(Self {
            session_hash: fnv1a64(session_id.as_bytes()),
            session_id,
            directory,
            records_path,
            next_sequence: AtomicU64::new(item_count),
            append_lock: AsyncMutex::new(()),
            index: Mutex::new(index),
            cache: Mutex::new(HotCache::new(cache_size)),
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
        let item = MemoryItem {
            pointer: pointer.clone(),
            content: content.clone(),
        };
        let (offset, length) = self.append(&StoreRecord::Item { item }).await?;

        let mut index = self.index.lock().unwrap_or_else(|error| error.into_inner());
        if index.entries.contains_key(&id) {
            bail!("memory ID {id} was generated more than once");
        }
        index.order.push(id.clone());
        if let Some(fingerprint) = pointer.metadata.get("fingerprint") {
            index
                .message_fingerprints
                .entry(fingerprint.clone())
                .or_insert_with(|| id.clone());
        }
        index.entries.insert(
            id.clone(),
            RecordLocation {
                offset,
                length,
                pointer: pointer.clone(),
            },
        );
        drop(index);
        self.cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(id, content);
        Ok(pointer)
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
        Ok((offset, line.len()))
    }

    async fn append_audit(&self, audit: AuditEvent) -> Result<()> {
        self.append(&StoreRecord::Audit { audit }).await?;
        Ok(())
    }

    async fn read(&self, id: &str) -> Result<Option<(MemoryPointer, Arc<String>)>> {
        let pointer = {
            let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
            let Some(location) = index.entries.get(id) else {
                return Ok(None);
            };
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
            StoreRecord::Audit { .. } => bail!("memory index points to an audit record"),
        };
        if item.pointer.id != id {
            bail!("memory index for {id} points to {}", item.pointer.id);
        }
        let content = Arc::new(item.content);
        self.cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(id.to_owned(), (*content).clone());
        Ok(Some((pointer, content)))
    }

    fn pointer(&self, id: &str) -> Option<MemoryPointer> {
        self.index
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entries
            .get(id)
            .map(|entry| entry.pointer.clone())
    }

    fn pointer_by_message_fingerprint(&self, fingerprint: &str) -> Option<MemoryPointer> {
        let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
        let id = index.message_fingerprints.get(fingerprint)?;
        index.entries.get(id).map(|entry| entry.pointer.clone())
    }

    fn pointers(&self) -> Vec<MemoryPointer> {
        let index = self.index.lock().unwrap_or_else(|error| error.into_inner());
        index
            .order
            .iter()
            .filter_map(|id| index.entries.get(id))
            .map(|entry| entry.pointer.clone())
            .collect()
    }

    async fn set_pinned(&self, id: &str, pinned: bool) -> Result<MemoryPointer> {
        let action = if pinned {
            AuditAction::Pin
        } else {
            AuditAction::Unpin
        };
        let Some(pointer) = self.pointer(id) else {
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
        self.append_audit(AuditEvent {
            action,
            id: id.to_owned(),
            created_at: timestamp()?,
            reason: None,
            returned_tokens: None,
            truncated: false,
            error: None,
        })
        .await?;
        let mut index = self.index.lock().unwrap_or_else(|error| error.into_inner());
        let entry = index
            .entries
            .get_mut(id)
            .context("memory record disappeared while updating pin state")?;
        entry.pointer.pinned = pinned;
        let mut updated = pointer;
        updated.pinned = pinned;
        Ok(updated)
    }
}

#[derive(Debug)]
pub struct MemoryRuntime {
    config: MemoryConfig,
    session: RwLock<Option<Arc<SessionMemory>>>,
    active_pointers: Mutex<Vec<String>>,
    recall_limiter: Mutex<RecallLimiter>,
}

impl MemoryRuntime {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            session: RwLock::new(None),
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
        let session =
            SessionMemory::open(session_id.to_owned(), directory, self.config.hot_cache_size)
                .await?;
        *self
            .session
            .write()
            .unwrap_or_else(|error| error.into_inner()) = Some(Arc::new(session));
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
        let importance = tool_importance(tool, &content);
        session
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
                metadata: tool_metadata(arguments),
            })
            .await
            .with_context(|| format!("failed to persist {tool} result in addressable memory"))
    }

    pub async fn store_message(
        &self,
        kind: MemoryKind,
        role: &str,
        content: String,
    ) -> Result<MemoryPointer> {
        let session = self.active_session()?;
        let fingerprint = message_fingerprint(role, &content);
        if let Some(pointer) = session.pointer_by_message_fingerprint(&fingerprint) {
            return Ok(pointer);
        }
        let mut metadata = BTreeMap::new();
        metadata.insert("role".to_owned(), role.to_owned());
        metadata.insert("preview".to_owned(), one_line_preview(&content, 96));
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
        self.active_session()
            .ok()?
            .pointer(id)
            .map(|pointer| self.citation(&pointer))
    }

    pub fn pointer_for_id(&self, id: &str) -> Option<MemoryPointer> {
        self.active_session().ok()?.pointer(id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.pointer_for_id(id).is_some()
    }

    pub fn set_active_pointers(&self, ids: impl IntoIterator<Item = String>) {
        let Ok(session) = self.active_session() else {
            return;
        };
        let mut seen = HashSet::new();
        let active = ids
            .into_iter()
            .filter(|id| seen.insert(id.clone()) && session.pointer(id).is_some())
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

    pub async fn recall(&self, id: &str, reason: Option<String>) -> Result<String> {
        let session = self.active_session()?;
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
        let Some((pointer, content)) = session.read(id).await? else {
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
        let (returned, truncated, returned_tokens) =
            truncate_to_token_budget(&content, self.config.max_recall_tokens);
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
            Ok(format!(
                "[recalled {id}: controlled excerpt ~{returned_tokens} of ~{} tokens; the complete content remains stored at the same ID]\n{returned}",
                pointer.token_estimate
            ))
        } else {
            Ok(format!(
                "[recalled {id}: exact content, ~{} tokens]\n{returned}",
                pointer.token_estimate
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

    pub fn list_pointers(&self, filter: Option<&str>) -> Result<String> {
        let session = self.active_session()?;
        let filter = filter.map(str::trim).filter(|filter| !filter.is_empty());
        let active = self
            .active_pointers
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        let candidates = if filter.is_some() {
            session.pointers()
        } else {
            let active_set = active.iter().cloned().collect::<HashSet<_>>();
            let mut pointers = session
                .pointers()
                .into_iter()
                .filter(|pointer| pointer.pinned && !active_set.contains(&pointer.id))
                .collect::<Vec<_>>();
            pointers.extend(active.into_iter().filter_map(|id| session.pointer(&id)));
            pointers
        };
        let mut matches = candidates
            .into_iter()
            .filter(|pointer| {
                filter.is_none_or(|filter| {
                    let filter = filter.to_ascii_lowercase();
                    self.citation(pointer)
                        .to_ascii_lowercase()
                        .contains(&filter)
                        || pointer
                            .metadata
                            .values()
                            .any(|value| value.to_ascii_lowercase().contains(&filter))
                })
            })
            .collect::<Vec<_>>();
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
        let mut output = matches
            .iter()
            .map(|pointer| format!("- {}", self.citation(pointer)))
            .collect::<Vec<_>>()
            .join("\n");
        if total > matches.len() {
            output.push_str(&format!(
                "\n- [{} more pointers omitted; pass a narrower filter]",
                total - matches.len()
            ));
        }
        Ok(output)
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
        session.set_pinned(id, pinned).await
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

    fn pointer_manifest(
        &self,
        ids: impl IntoIterator<Item = String>,
        limit: usize,
        include_pinned: bool,
    ) -> Vec<String> {
        let Ok(session) = self.active_session() else {
            return Vec::new();
        };
        let mut seen = HashSet::new();
        let mut pinned = if include_pinned {
            session
                .pointers()
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
            .filter_map(|id| session.pointer(&id))
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
    format!(
        "{:016x}{:016x}",
        fnv1a64(&bytes),
        fnv1a64_seed(&bytes, 0x8422_2325_cbf2_9ce4)
    )
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, process, time::SystemTime};

    use serde_json::json;

    use super::{MemoryConfig, MemoryRuntime, extract_memory_ids};

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
            .recall(&pointer.id, Some("verify exact source".to_owned()))
            .await
            .unwrap();
        assert!(recalled.ends_with(&content));
        assert!(runtime.pin(&pointer.id).await.unwrap().contains("Pinned"));
        assert!(runtime.list_pointers(None).unwrap().contains(&pointer.id));
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
            .recall("§obs_000000000000000000000000", None)
            .await
            .unwrap_err();
        assert!(missing.to_string().contains("does not exist"));

        runtime.begin_model_turn();
        for _ in 0..3 {
            runtime.recall(&pointer.id, None).await.unwrap();
        }
        let limited = runtime.recall(&pointer.id, None).await.unwrap_err();
        assert!(limited.to_string().contains("rate limit exceeded"));

        let records_path = runtime.records_path().unwrap();
        let log = tokio::fs::read_to_string(&records_path).await.unwrap();
        assert!(log.contains("\"record_type\":\"item\""));
        assert!(log.contains("\"action\":\"recall\""));
        assert!(log.contains("\"action\":\"pin\""));
        assert!(log.contains("\"error\":\"memory ID"));

        drop(runtime);
        let reopened = MemoryRuntime::new(config);
        reopened
            .activate("20260815-120000-deadbeef", directory.clone())
            .await
            .unwrap();
        let recalled = reopened.recall(&pointer.id, None).await.unwrap();
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

        let recalled = runtime.recall(&pointer.id, None).await.unwrap();

        assert!(recalled.contains("controlled excerpt"));
        assert!(crate::agent::estimate_tokens(&recalled) < 96);
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }
}
