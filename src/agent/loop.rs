use std::{
    collections::HashMap,
    future::Future,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::{
    agent::{
        AgentEvent, AssistantMessage, EventSender, Message, MessageRole, PromptOutcome,
        estimate_tokens, token_window,
    },
    memory::{MemoryKind, MemoryMode, MemoryPointer, MemoryRuntime, extract_memory_ids},
    provider::{PreparedRequest, Provider, ThinkingLevel, ToolDefinition},
    tools::ToolRegistry,
};

const SYSTEM_PROMPT: &str = "You are Zex, a minimal AI agent core. Be concise and accurate. Use grep to search file contents, glob to find files, and bash only for other system commands. Use read, write, and edit for file operations. Use tool results to finish the task.";
const MEMORY_POLICY_MARKER: &str = "\n\n[Addressable memory policy]";
const POINTER_MANIFEST_MARKER: &str = "[Current valid addressable pointers]";
const MEMORY_POLICY: &str = "[Addressable memory policy]\n\
- Large observations may appear only as §id citations. Call recall with an exact visible ID only when missing details directly affect the current decision.\n\
- Never invent, alter, or guess an ID. If an ID is not visible, use list_pointers with a narrow filter. Prefer pinned and recently listed pointers.\n\
- Recall is rate-limited. Do not recall information already present in the active context, and do not batch speculative recalls.\n\
- A recall failure means the content is unavailable. State that fact and never reconstruct or hallucinate it.\n\
- pin preserves high-value evidence or decisions; unpin it when no longer central.\n\
Good: a compiler error cites §obs_abcd... and the exact diagnostic affects the fix, so recall that exact ID once.\n\
Bad: invent §obs_latest, repeatedly recall every pointer, or guess file contents after recall reports a missing ID.";
const AUTO_COMPACT_PERCENT: usize = 85;
const OUTPUT_RESERVE_TOKENS: u64 = 8_192;
const SUMMARY_ITEM_CHARS: usize = 480;
const POINTER_SUMMARY_ITEM_CHARS: usize = 180;
const SUMMARY_MAX_TOKENS: usize = 3_072;
const TOOL_SUMMARY_EDGE_CHARS: usize = 180;
const ANCHOR_MAX_CHARS: usize = 4_000;
const PRUNE_KEEP_TOOL_RESULTS: usize = 4;
const PRUNE_MIN_CHARS: usize = 2_000;
/// Bounds on the estimate-to-billed ratio we are willing to trust. The JSON
/// body we measure is not the wire format a provider tokenizes, so a modest
/// offset is expected; a sample far outside this band means the provider is
/// reporting something else entirely — cache-excluded input tokens, most
/// often — and budgeting against it would be worse than not calibrating.
const CALIBRATION_MIN: f64 = 0.7;
const CALIBRATION_MAX: f64 = 1.5;
const CALIBRATION_WEIGHT: f64 = 0.5;

pub struct Agent<P> {
    provider: P,
    tools: ToolRegistry,
    model: String,
    messages: Vec<Message>,
    events: EventSender,
    turn_timeout: Duration,
    max_turns: usize,
    fallback_context_tokens: usize,
    compact_keep_turns: usize,
    thinking_level: ThinkingLevel,
    memory: Option<Arc<MemoryRuntime>>,
    calibration: Option<f64>,
}

pub struct AgentOptions {
    pub model: String,
    pub turn_timeout: Duration,
    pub max_turns: usize,
    pub max_context_tokens: usize,
    pub compact_keep_turns: usize,
    pub thinking_level: ThinkingLevel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactStats {
    pub before_tokens: usize,
    pub after_tokens: usize,
    pub freed_tokens: usize,
    pub kept_turns: usize,
    pub summarized_turns: usize,
    pub summarized_tool_outputs: usize,
    pub pruned_tool_outputs: usize,
}

#[derive(Debug, Clone, Copy)]
struct CompactionResult {
    kept_turns: usize,
    summarized_turns: usize,
    summarized_tool_outputs: usize,
}

impl<P> Agent<P>
where
    P: Provider,
{
    pub fn new(
        provider: P,
        tools: ToolRegistry,
        events: EventSender,
        options: AgentOptions,
        messages: Option<Vec<Message>>,
    ) -> Self {
        let memory = tools.memory().cloned();
        let messages = provider.sanitize_history(
            &options.model,
            &normalize_messages(messages.unwrap_or_default()),
        );
        let mut agent = Self {
            provider,
            tools,
            model: options.model,
            messages,
            events,
            turn_timeout: options.turn_timeout,
            max_turns: options.max_turns,
            fallback_context_tokens: options.max_context_tokens,
            compact_keep_turns: options.compact_keep_turns,
            thinking_level: options.thinking_level,
            memory,
            calibration: None,
        };
        agent.sync_memory_context();
        agent
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
        // Calibration is per model and per request shape; carrying it across a
        // switch would budget the new model against the old one's billing.
        self.calibration = None;
        self.messages = self.provider.sanitize_history(&self.model, &self.messages);
        self.sync_memory_context();
    }

    pub fn thinking_level(&self) -> ThinkingLevel {
        crate::provider::normalize_thinking_level(
            &self.provider.thinking_capabilities(&self.model),
            self.thinking_level,
        )
        .effective
    }

    pub fn thinking_preference(&self) -> ThinkingLevel {
        self.thinking_level
    }

    pub fn thinking_capabilities(&self) -> crate::provider::ThinkingCapabilities {
        self.provider.thinking_capabilities(&self.model)
    }

    pub fn set_thinking_level(&mut self, thinking_level: ThinkingLevel) {
        self.thinking_level = thinking_level;
    }

    /// Effective token budget for the active model: the model's context
    /// window minus an output reserve, falling back to the configured
    /// `max_context_tokens` when no limit is known.
    pub fn context_budget(&self) -> usize {
        self.request_budgets().0
    }

    fn request_budgets(&self) -> (usize, usize) {
        self.provider
            .context_limit(&self.model)
            .and_then(|limit| {
                let output_tokens = limit
                    .output
                    .unwrap_or(OUTPUT_RESERVE_TOKENS)
                    .min(OUTPUT_RESERVE_TOKENS)
                    .min(limit.context / 4);
                Some((
                    usize::try_from(limit.context.saturating_sub(output_tokens)).ok()?,
                    usize::try_from(output_tokens).ok()?,
                ))
            })
            .filter(|(input_tokens, output_tokens)| *input_tokens > 0 && *output_tokens > 0)
            .unwrap_or((
                self.fallback_context_tokens,
                usize::try_from(OUTPUT_RESERVE_TOKENS)
                    .expect("the fixed output reserve must fit usize"),
            ))
    }

    fn prepare_request(
        &self,
        definitions: &[ToolDefinition],
    ) -> Result<PreparedRequest<P::Request>> {
        let (_, max_output_tokens) = self.request_budgets();
        // The pointer manifest is wire-only state: it changes on every tool
        // result, so keeping it in `self.messages[0]` would rewrite the system
        // prompt each turn and defeat provider prefix caching. Appending it
        // last leaves the whole prefix byte-identical between turns.
        let Some(manifest) = self.pointer_manifest_message() else {
            return self.provider.prepare_request(
                &self.model,
                self.thinking_level,
                &self.messages,
                definitions,
                max_output_tokens,
            );
        };
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        messages.extend_from_slice(&self.messages);
        messages.push(manifest);
        self.provider.prepare_request(
            &self.model,
            self.thinking_level,
            &messages,
            definitions,
            max_output_tokens,
        )
    }

    fn pointer_manifest_message(&self) -> Option<Message> {
        let manifest = self.memory.as_ref()?.system_pointer_manifest();
        if manifest.is_empty() {
            return None;
        }
        let pointers = manifest
            .into_iter()
            .map(|citation| format!("- {citation}"))
            .collect::<Vec<_>>()
            .join("\n");
        Some(Message::System {
            content: format!("{POINTER_MANIFEST_MARKER}\n{pointers}"),
        })
    }

    /// Input tokens for a prepared request, scaled by what the provider
    /// actually billed for earlier requests on this model. Every budget
    /// decision goes through here so the 85% threshold lands on the real
    /// number rather than on the raw JSON-body estimate.
    fn measured_tokens(&self, request: &PreparedRequest<P::Request>) -> usize {
        self.scaled_tokens(request.input_tokens())
    }

    fn scaled_tokens(&self, tokens: usize) -> usize {
        let Some(calibration) = self.calibration else {
            return tokens;
        };
        ((tokens as f64) * calibration).round() as usize
    }

    /// Folds one billed request into the running calibration.
    fn record_usage(&mut self, estimated: usize, usage: Option<crate::agent::CompletionUsage>) {
        let Some(actual) = usage.and_then(|usage| usage.input_tokens) else {
            return;
        };
        if estimated == 0 || actual == 0 {
            return;
        }
        let ratio = actual as f64 / estimated as f64;
        if !(CALIBRATION_MIN..=CALIBRATION_MAX).contains(&ratio) {
            return;
        }
        self.calibration = Some(match self.calibration {
            Some(previous) => previous + (ratio - previous) * CALIBRATION_WEIGHT,
            None => ratio,
        });
    }

    pub fn default_tool_timeout(&self) -> Duration {
        self.tools.default_timeout()
    }

    pub fn clear(&mut self) {
        self.messages = fresh_messages();
        self.sync_memory_context();
    }

    pub async fn replace_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        self.messages = self
            .provider
            .sanitize_history(&self.model, &normalize_messages(messages));
        self.initialize_memory().await?;
        let definitions = self.tools.definitions();
        let _ = self.prepare_with_compaction(&definitions).await?;
        Ok(())
    }

    pub fn memory_enabled(&self) -> bool {
        self.memory.as_ref().is_some_and(|memory| memory.enabled())
    }

    pub async fn activate_memory(&mut self, session_id: &str, directory: PathBuf) -> Result<()> {
        if let Some(memory) = &self.memory {
            memory.activate(session_id, directory).await?;
        }
        self.sync_memory_context();
        Ok(())
    }

    pub async fn initialize_memory(&mut self) -> Result<()> {
        let Some(memory) = self.memory.clone() else {
            self.sync_memory_context();
            return Ok(());
        };
        for message in &self.messages {
            match message {
                Message::User { content } => {
                    memory
                        .store_message(MemoryKind::Message, "user", content.clone())
                        .await?;
                }
                Message::Assistant { .. } => {
                    memory
                        .store_message(
                            MemoryKind::Message,
                            "assistant",
                            serde_json::to_string_pretty(message)
                                .context("failed to serialize assistant history for memory")?,
                        )
                        .await?;
                }
                Message::System { .. } | Message::Tool { .. } => {}
            }
        }
        let tool_calls = tool_calls_by_id(&self.messages);
        let pending = self
            .messages
            .iter()
            .enumerate()
            .filter_map(|(index, message)| {
                let Message::Tool {
                    tool_call_id,
                    content,
                } = message
                else {
                    return None;
                };
                if extract_memory_ids(content)
                    .iter()
                    .any(|id| memory.contains(id))
                    || content.starts_with("[tool output cleared")
                {
                    return None;
                }
                let (name, arguments) = tool_calls.get(tool_call_id)?;
                (!MemoryRuntime::is_control_tool(name))
                    .then(|| (index, name.clone(), arguments.clone(), content.clone()))
            })
            .collect::<Vec<_>>();
        for (index, name, arguments, content) in pending {
            let pointer = memory
                .store_tool_result(&name, &arguments, content.clone())
                .await?;
            if let Some(Message::Tool {
                content: current, ..
            }) = self.messages.get_mut(index)
                && *current == content
            {
                *current = memory.render_tool_result(&pointer, content);
            }
        }
        self.sync_memory_context();
        memory
            .maintain()
            .await
            .context("failed to maintain addressable memory")?;
        self.sync_memory_context();
        Ok(())
    }

    /// Estimated input tokens for the exact request shape that the active
    /// provider would send, including sanitized reasoning state and tool
    /// definitions.
    pub fn context_tokens(&self) -> usize {
        let definitions = self.tools.definitions();
        let tokens = self
            .prepare_request(&definitions)
            .map(|request| request.input_tokens())
            .unwrap_or_else(|_| fallback_request_tokens(&self.messages, &definitions));
        self.scaled_tokens(tokens)
    }

    pub async fn compact(&mut self) -> Result<CompactStats> {
        let definitions = self.tools.definitions();
        self.sync_memory_context();
        let request = self.prepare_request(&definitions)?;
        let (stats, request) = self
            .reduce_prepared_context(&definitions, request, true)
            .await?;
        self.ensure_request_fits(&request)?;
        Ok(stats)
    }

    async fn reduce_prepared_context(
        &mut self,
        definitions: &[ToolDefinition],
        mut request: PreparedRequest<P::Request>,
        force_compaction: bool,
    ) -> Result<(CompactStats, PreparedRequest<P::Request>)> {
        let before_tokens = self.measured_tokens(&request);
        let threshold = self.context_budget().saturating_mul(AUTO_COMPACT_PERCENT) / 100;
        let pruned_tool_outputs = prune_tool_outputs(&mut self.messages, self.memory.as_deref());
        if pruned_tool_outputs > 0 {
            self.sync_memory_context();
            request = self.prepare_request(definitions)?;
        }
        let (mut stats, request) =
            if force_compaction || self.measured_tokens(&request) >= threshold {
                self.compact_prepared(definitions, request).await?
            } else {
                let after_tokens = self.measured_tokens(&request);
                (
                    CompactStats {
                        before_tokens,
                        after_tokens,
                        freed_tokens: before_tokens.saturating_sub(after_tokens),
                        kept_turns: 0,
                        summarized_turns: 0,
                        summarized_tool_outputs: 0,
                        pruned_tool_outputs: 0,
                    },
                    request,
                )
            };
        stats.before_tokens = before_tokens;
        stats.freed_tokens = before_tokens.saturating_sub(stats.after_tokens);
        stats.pruned_tool_outputs = pruned_tool_outputs;
        Ok((stats, request))
    }

    async fn compact_prepared(
        &mut self,
        definitions: &[ToolDefinition],
        mut request: PreparedRequest<P::Request>,
    ) -> Result<(CompactStats, PreparedRequest<P::Request>)> {
        let budget = self.context_budget();
        let before_tokens = self.measured_tokens(&request);
        let mut current_tokens = before_tokens;
        let mut keep_turns = self.compact_keep_turns;
        let summary_token_budget = (budget / 4).clamp(1, SUMMARY_MAX_TOKENS);
        let mut stats = CompactStats {
            before_tokens,
            after_tokens: before_tokens,
            freed_tokens: 0,
            kept_turns: 0,
            summarized_turns: 0,
            summarized_tool_outputs: 0,
            pruned_tool_outputs: 0,
        };
        loop {
            let archived = self.archive_compacted_messages(keep_turns).await?;
            let next = compact_messages(
                &mut self.messages,
                keep_turns,
                summary_token_budget,
                self.memory.as_deref(),
                &archived,
            );
            self.sync_memory_context();
            let next_request = self.prepare_request(definitions)?;
            stats.after_tokens = self.measured_tokens(&next_request);
            stats.kept_turns = next.kept_turns;
            stats.summarized_turns += next.summarized_turns;
            stats.summarized_tool_outputs += next.summarized_tool_outputs;
            let made_progress = stats.after_tokens < current_tokens;
            request = next_request;
            if !made_progress || stats.after_tokens <= budget || next.kept_turns <= 1 {
                break;
            }
            current_tokens = stats.after_tokens;
            keep_turns = next.kept_turns - 1;
        }
        stats.freed_tokens = before_tokens.saturating_sub(stats.after_tokens);
        Ok((stats, request))
    }

    async fn archive_compacted_messages(
        &self,
        keep_turns: usize,
    ) -> Result<HashMap<usize, MemoryPointer>> {
        let Some(memory) = self.memory.clone() else {
            return Ok(HashMap::new());
        };
        let user_indices = user_indices(&self.messages);
        let summarized_turns = user_indices
            .len()
            .saturating_sub(user_indices.len().min(keep_turns));
        if summarized_turns == 0 {
            return Ok(HashMap::new());
        }
        let keep_start = user_indices[summarized_turns];
        let tool_calls = tool_calls_by_id(&self.messages);
        let mut archived = HashMap::new();
        for (index, message) in self.messages[1..keep_start].iter().enumerate() {
            let index = index + 1;
            if let Some(pointer) = extract_memory_ids(message_content(message))
                .into_iter()
                .find_map(|id| memory.pointer_for_id(&id))
            {
                archived.insert(index, pointer);
                continue;
            }
            let stored = match message {
                Message::System { .. } => None,
                Message::User { content } => Some(
                    memory
                        .store_message(MemoryKind::Message, "user", content.clone())
                        .await?,
                ),
                Message::Assistant { .. } => Some(
                    memory
                        .store_message(
                            MemoryKind::Message,
                            "assistant",
                            serde_json::to_string_pretty(message)
                                .context("failed to serialize assistant history for memory")?,
                        )
                        .await?,
                ),
                Message::Tool {
                    tool_call_id,
                    content,
                } => {
                    let Some((name, arguments)) = tool_calls.get(tool_call_id) else {
                        continue;
                    };
                    if MemoryRuntime::is_control_tool(name)
                        || content.starts_with("[tool output cleared")
                    {
                        None
                    } else {
                        Some(
                            memory
                                .store_tool_result(name, arguments, content.clone())
                                .await?,
                        )
                    }
                }
            };
            if let Some(pointer) = stored {
                archived.insert(index, pointer);
            }
        }
        Ok(archived)
    }

    /// Refreshes which citations the memory runtime treats as active. The
    /// system prompt only ever receives the static policy text: everything
    /// that varies per turn is appended at request time instead, so
    /// `messages[0]` stays byte-stable for the life of the session.
    fn sync_memory_context(&mut self) {
        let Some(Message::System { content }) = self.messages.first() else {
            return;
        };
        let base = content
            .split_once(MEMORY_POLICY_MARKER)
            .map(|(base, _)| base)
            .unwrap_or(content)
            .to_owned();
        let Some(memory) = &self.memory else {
            if let Some(Message::System { content }) = self.messages.first_mut() {
                *content = base;
            }
            return;
        };
        let ids = self
            .messages
            .iter()
            .skip(1)
            .flat_map(|message| extract_memory_ids(message_content(message)))
            .collect::<Vec<_>>();
        memory.set_active_pointers(ids);
        if let Some(Message::System { content }) = self.messages.first_mut() {
            *content = format!("{base}\n\n{MEMORY_POLICY}");
        }
    }

    pub fn has_conversation(&self) -> bool {
        self.messages
            .iter()
            .any(|message| !matches!(message, Message::System { .. }))
    }

    pub async fn prompt(&mut self, prompt: impl Into<String>) -> Result<AssistantMessage> {
        match self
            .prompt_with_cancellation(prompt, std::future::pending())
            .await?
        {
            PromptOutcome::Completed(message) => Ok(message),
            PromptOutcome::Cancelled => {
                unreachable!("a pending cancellation future cannot resolve")
            }
        }
    }

    pub async fn prompt_cancellable(
        &mut self,
        prompt: impl Into<String>,
        mut cancellation: tokio::sync::watch::Receiver<bool>,
    ) -> Result<PromptOutcome> {
        self.prompt_with_cancellation(prompt, async move {
            while !*cancellation.borrow() {
                if cancellation.changed().await.is_err() {
                    std::future::pending::<()>().await;
                }
            }
        })
        .await
    }

    async fn prompt_with_cancellation<F>(
        &mut self,
        prompt: impl Into<String>,
        cancellation: F,
    ) -> Result<PromptOutcome>
    where
        F: Future<Output = ()>,
    {
        let checkpoint = self.messages.len();
        let prompt = prompt.into();
        if let Some(memory) = &self.memory {
            memory.begin_turn()?;
            if let Err(error) = memory
                .store_message(MemoryKind::Message, "user", prompt.clone())
                .await
            {
                memory
                    .abort_turn()
                    .await
                    .context("failed to abort memory turn after storing the user prompt failed")?;
                return Err(error);
            }
        }
        self.messages.push(Message::User {
            content: prompt.clone(),
        });
        let _ = self.events.send(AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: prompt,
        });

        let resolution = {
            let turn = tokio::time::timeout(self.turn_timeout, self.run_loop());
            tokio::pin!(turn);
            tokio::pin!(cancellation);

            tokio::select! {
                biased;
                _ = &mut cancellation => None,
                result = &mut turn => Some(result),
            }
        };

        match resolution {
            None => {
                if let Some(memory) = &self.memory {
                    memory
                        .abort_turn()
                        .await
                        .context("failed to abort the cancelled memory turn")?;
                }
                retain_user_prompt(&mut self.messages, checkpoint);
                let _ = self.events.send(AgentEvent::TurnCancelled);
                Ok(PromptOutcome::Cancelled)
            }
            Some(result) => match result {
                Ok(Ok(message)) => {
                    if let Some(memory) = &self.memory
                        && let Err(error) = memory.commit_turn().await
                    {
                        let abort_error = memory.abort_turn().await.err();
                        retain_user_prompt(&mut self.messages, checkpoint);
                        return Err(match abort_error {
                            Some(abort_error) => error.context(format!(
                                "failed to commit memory turn; abort also failed: {abort_error:#}"
                            )),
                            None => error.context("failed to commit memory turn"),
                        });
                    }
                    if let Some(memory) = &self.memory
                        && let Err(error) = memory.maintain().await
                    {
                        let _ = self.events.send(AgentEvent::Error {
                            message: format!("addressable memory maintenance failed: {error:#}"),
                        });
                    }
                    Ok(PromptOutcome::Completed(message))
                }
                Ok(Err(error)) => {
                    if let Some(memory) = &self.memory {
                        memory
                            .abort_turn()
                            .await
                            .context("failed to abort memory turn after agent error")?;
                    }
                    retain_user_prompt(&mut self.messages, checkpoint);
                    Err(error)
                }
                Err(_) => {
                    if let Some(memory) = &self.memory {
                        memory
                            .abort_turn()
                            .await
                            .context("failed to abort timed-out memory turn")?;
                    }
                    retain_user_prompt(&mut self.messages, checkpoint);
                    let message = format!(
                        "agent turn exceeded its {} second timeout",
                        self.turn_timeout.as_secs()
                    );
                    let _ = self.events.send(AgentEvent::Error {
                        message: message.clone(),
                    });
                    bail!(message);
                }
            },
        }
    }

    /// Repeats provider completion and tool execution until the model returns text only.
    async fn run_loop(&mut self) -> Result<AssistantMessage> {
        let definitions = self.tools.definitions();

        for _ in 0..self.max_turns {
            let request = self.prepare_with_compaction(&definitions).await?;
            let estimated_tokens = request.input_tokens();
            if let Some(memory) = &self.memory {
                memory.begin_model_turn();
            }
            let assistant = match self
                .provider
                .complete(request.into_payload(), &self.events)
                .await
            {
                Ok(assistant) => assistant,
                Err(error) => {
                    let _ = self.events.send(AgentEvent::Error {
                        message: format!("{error:#}"),
                    });
                    return Err(error);
                }
            };
            self.record_usage(estimated_tokens, assistant.usage);

            let assistant_message = Message::Assistant {
                content: assistant.content.clone(),
                thinking: assistant.thinking.clone(),
                tool_calls: assistant.tool_calls.clone(),
                provider_state: assistant
                    .provider_state
                    .clone()
                    .and_then(|state| self.provider.encode_provider_state(&self.model, state)),
            };
            if let Some(memory) = &self.memory {
                memory
                    .store_message(
                        MemoryKind::Message,
                        "assistant",
                        serde_json::to_string_pretty(&assistant_message)
                            .context("failed to serialize assistant message for memory")?,
                    )
                    .await?;
            }
            self.messages.push(assistant_message);

            if assistant.tool_calls.is_empty() {
                let _ = self.events.send(AgentEvent::TurnEnd);
                return Ok(assistant);
            }

            for tool_call in assistant.tool_calls {
                let name = tool_call.name;
                let call_id = tool_call.id;
                let arguments = tool_call.arguments;
                let parsed_arguments = serde_json::from_str::<Value>(&arguments);
                let memory_arguments = parsed_arguments
                    .as_ref()
                    .ok()
                    .cloned()
                    .unwrap_or(Value::Null);
                let timeout = parsed_arguments
                    .as_ref()
                    .ok()
                    .and_then(|arguments| self.tools.execution_timeout(arguments).ok())
                    .unwrap_or_else(|| self.tools.default_timeout());
                let _ = self.events.send(AgentEvent::ToolStart {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                    timeout,
                });

                let started = Instant::now();
                let result = match parsed_arguments {
                    Ok(arguments) => self.tools.execute(&name, arguments).await,
                    Err(error) => Err(anyhow::Error::from(error)),
                };
                let elapsed = started.elapsed();
                let (event_output, content, is_error, change) = match result {
                    Ok(outcome) => {
                        let content = match (&self.memory, outcome.memory.as_ref()) {
                            (Some(memory), Some(pointer)) => {
                                memory.render_tool_result(pointer, outcome.output.clone())
                            }
                            _ => outcome.output.clone(),
                        };
                        (outcome.output, content, false, outcome.change)
                    }
                    Err(error) => {
                        let output = format!("tool error: {error:#}");
                        let content = if let Some(memory) = &self.memory
                            && !MemoryRuntime::is_control_tool(&name)
                        {
                            let pointer = memory
                                .store_tool_result(&name, &memory_arguments, output.clone())
                                .await?;
                            memory.render_tool_result(&pointer, output.clone())
                        } else {
                            output.clone()
                        };
                        (output, content, true, None)
                    }
                };

                let _ = self.events.send(AgentEvent::ToolEnd {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    output: event_output,
                    is_error,
                    elapsed,
                    change,
                });
                self.messages.push(Message::Tool {
                    tool_call_id: call_id,
                    content,
                });
                self.sync_memory_context();
            }
        }

        let message = format!(
            "agent reached the configured limit of {} provider turns",
            self.max_turns
        );
        let _ = self.events.send(AgentEvent::Error {
            message: message.clone(),
        });
        bail!(message)
    }

    async fn prepare_with_compaction(
        &mut self,
        definitions: &[ToolDefinition],
    ) -> Result<PreparedRequest<P::Request>> {
        self.sync_memory_context();
        let request = self.prepare_request(definitions)?;
        let threshold = self.context_budget().saturating_mul(AUTO_COMPACT_PERCENT) / 100;
        if self.measured_tokens(&request) < threshold {
            self.ensure_request_fits(&request)?;
            return Ok(request);
        }
        let (stats, request) = self
            .reduce_prepared_context(definitions, request, false)
            .await?;
        if stats.freed_tokens > 0 {
            self.sync_memory_context();
            let _ = self.events.send(AgentEvent::ContextCompacted {
                stats: stats.clone(),
            });
        }
        self.ensure_request_fits(&request)?;
        Ok(request)
    }

    fn ensure_request_fits(&self, request: &PreparedRequest<P::Request>) -> Result<()> {
        let budget = self.context_budget();
        let tokens = self.measured_tokens(request);
        if tokens <= budget {
            return Ok(());
        }
        bail!(
            "prepared request requires approximately {} input tokens, exceeding the {} token input budget with {} output tokens reserved; shorten the current prompt or use a model with a larger context window",
            tokens,
            budget,
            request.max_output_tokens()
        )
    }
}

fn retain_user_prompt(messages: &mut Vec<Message>, checkpoint: usize) {
    let user_prompt = messages.get(checkpoint).cloned();
    messages.truncate(checkpoint);
    if let Some(user_prompt) = user_prompt {
        messages.push(user_prompt);
    }
}

fn fresh_messages() -> Vec<Message> {
    vec![Message::System {
        content: SYSTEM_PROMPT.to_owned(),
    }]
}

fn normalize_messages(messages: Vec<Message>) -> Vec<Message> {
    if messages.is_empty() {
        return fresh_messages();
    }
    if matches!(messages.first(), Some(Message::System { .. })) {
        messages
    } else {
        let mut normalized = fresh_messages();
        normalized.extend(messages);
        normalized
    }
}

fn fallback_request_tokens(messages: &[Message], tools: &[ToolDefinition]) -> usize {
    serde_json::to_string(&(messages, tools))
        .map(|request| estimate_tokens(&request))
        .unwrap_or_else(|_| messages.iter().map(Message::token_estimate).sum())
}

fn message_content(message: &Message) -> &str {
    match message {
        Message::System { content }
        | Message::User { content }
        | Message::Assistant { content, .. }
        | Message::Tool { content, .. } => content,
    }
}

fn user_indices(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| matches!(message, Message::User { .. }).then_some(index))
        .collect()
}

fn tool_calls_by_id(messages: &[Message]) -> HashMap<String, (String, Value)> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant { tool_calls, .. } => Some(tool_calls),
            _ => None,
        })
        .flatten()
        .map(|call| {
            (
                call.id.clone(),
                (
                    call.name.clone(),
                    serde_json::from_str(&call.arguments).unwrap_or(Value::Null),
                ),
            )
        })
        .collect()
}

/// Replaces older, large tool outputs with a placeholder, keeping the most
/// recent `PRUNE_KEEP_TOOL_RESULTS` tool results intact. Returns how many
/// tool outputs were pruned.
fn prune_tool_outputs(messages: &mut [Message], memory: Option<&MemoryRuntime>) -> usize {
    let tool_indices = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| matches!(message, Message::Tool { .. }).then_some(index))
        .collect::<Vec<_>>();
    let prunable = tool_indices.len().saturating_sub(PRUNE_KEEP_TOOL_RESULTS);
    let mut pruned = 0;
    for &index in &tool_indices[..prunable] {
        if let Message::Tool { content, .. } = &mut messages[index] {
            let chars = content.chars().count();
            if chars > PRUNE_MIN_CHARS && !content.starts_with("[tool output cleared") {
                let citation = memory.and_then(|memory| {
                    extract_memory_ids(content)
                        .into_iter()
                        .find_map(|id| memory.citation_for_id(&id))
                });
                let pointer_mode = matches!(
                    memory.map(MemoryRuntime::mode),
                    Some(MemoryMode::PointerPriority | MemoryMode::Hybrid)
                );
                match citation {
                    // The body is still in the store in every mode, so name it
                    // rather than stranding a record the model can no longer
                    // address.
                    Some(citation) => {
                        *content = citation;
                        pruned += 1;
                    }
                    None if !pointer_mode => {
                        *content = format!("[tool output cleared to free context: {chars} chars]");
                        pruned += 1;
                    }
                    // Unaddressable output under a pointer mode is the only
                    // copy left; keep it rather than destroy it.
                    None => {}
                }
            }
        }
    }
    pruned
}

fn compact_messages(
    messages: &mut Vec<Message>,
    keep_turns: usize,
    summary_token_budget: usize,
    memory: Option<&MemoryRuntime>,
    archived: &HashMap<usize, MemoryPointer>,
) -> CompactionResult {
    let user_indices = user_indices(messages);
    let existing_summary = matches!(
        messages.get(1),
        Some(Message::System { content })
            if content.starts_with("[Compacted earlier conversation:")
    );
    let kept_turns = user_indices.len().min(keep_turns);
    let summarized_turns = user_indices.len().saturating_sub(kept_turns);
    if summarized_turns == 0 {
        if existing_summary
            && let Some(Message::System { content }) = messages.get_mut(1)
            && let Some((header, body)) = content.split_once('\n')
        {
            let bounded = bounded_summary_lines(
                body.lines().map(str::to_owned).collect(),
                summary_token_budget,
            )
            .join("\n");
            *content = format!("{header}\n{bounded}");
        }
        return CompactionResult {
            kept_turns,
            summarized_turns: 0,
            summarized_tool_outputs: 0,
        };
    }

    let anchor_index = (!existing_summary).then_some(user_indices[0]);
    let keep_start = user_indices[summarized_turns];
    let mode = memory.map(MemoryRuntime::mode);
    let pointer_mode = matches!(mode, Some(MemoryMode::PointerPriority | MemoryMode::Hybrid));
    let mut tool_names = HashMap::new();
    let mut summary_lines = Vec::new();
    let mut pointer_ids = Vec::new();
    // Keep the original task statement visible across compactions instead of
    // reducing it to a summary line.
    if let Some(anchor_index) = anchor_index
        && let Message::User { content } = &messages[anchor_index]
    {
        let pointer = archived.get(&anchor_index);
        if let Some(pointer) = pointer {
            pointer_ids.push(pointer.id.clone());
        }
        let id = pointer
            .filter(|_| pointer_mode)
            .map(|pointer| format!(" {}", pointer.id))
            .unwrap_or_default();
        summary_lines.push(format!("Original request{id}:\n{}", anchor_text(content)));
    }
    if existing_summary && let Message::System { content } = &messages[1] {
        pointer_ids.extend(extract_memory_ids(content));
        let prior_body = content
            .split_once('\n')
            .map(|(_, body)| body)
            .unwrap_or(content)
            .split("\n[Available addressable pointers]")
            .next()
            .unwrap_or_default();
        summary_lines.extend(prior_body.lines().map(str::to_owned));
    }
    let mut summarized_tool_outputs = 0usize;
    for (index, message) in messages[1..keep_start].iter().enumerate() {
        let index = index + 1;
        match message {
            Message::System { content }
                if !content.starts_with("[Compacted earlier conversation:") =>
            {
                summary_lines.push(format!("Prior context: {}", summarize_text(content)));
            }
            Message::System { .. } => {}
            Message::User { content } if Some(index) != anchor_index => {
                let pointer = archived.get(&index);
                if let Some(pointer) = pointer {
                    pointer_ids.push(pointer.id.clone());
                }
                let id = pointer
                    .filter(|_| pointer_mode)
                    .map(|pointer| format!(" {}", pointer.id))
                    .unwrap_or_default();
                let summary = if mode == Some(MemoryMode::PointerPriority) {
                    summarize_text_to(content, POINTER_SUMMARY_ITEM_CHARS)
                } else {
                    summarize_text(content)
                };
                summary_lines.push(format!("User{id}: {summary}"));
            }
            Message::User { .. } => {}
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let pointer = archived.get(&index);
                if let Some(pointer) = pointer {
                    pointer_ids.push(pointer.id.clone());
                }
                let id = pointer
                    .filter(|_| pointer_mode)
                    .map(|pointer| format!(" {}", pointer.id))
                    .unwrap_or_default();
                if mode == Some(MemoryMode::PointerPriority) {
                    let summary = if content.trim().is_empty() {
                        "assistant tool turn".to_owned()
                    } else {
                        summarize_text_to(content, POINTER_SUMMARY_ITEM_CHARS)
                    };
                    summary_lines.push(format!("Assistant{id}: {summary}"));
                } else {
                    if !content.trim().is_empty() {
                        summary_lines.push(format!("Assistant{id}: {}", summarize_text(content)));
                    }
                }
                for call in tool_calls {
                    tool_names.insert(call.id.clone(), call.name.clone());
                    if mode != Some(MemoryMode::PointerPriority) {
                        summary_lines.push(format!(
                            "Tool call {}: {}",
                            call.name,
                            summarize_text(&call.arguments)
                        ));
                    }
                }
            }
            Message::Tool {
                tool_call_id,
                content,
            } => {
                summarized_tool_outputs += 1;
                let pointer = archived.get(&index).cloned().or_else(|| {
                    memory.and_then(|memory| {
                        extract_memory_ids(content)
                            .into_iter()
                            .find_map(|id| memory.pointer_for_id(&id))
                    })
                });
                if pointer_mode && let (Some(memory), Some(pointer)) = (memory, pointer) {
                    pointer_ids.push(pointer.id.clone());
                    summary_lines.push(memory.citation_for_id(&pointer.id).unwrap_or_else(|| {
                        format!(
                            "Tool result {}: {}",
                            tool_names
                                .get(tool_call_id)
                                .map(String::as_str)
                                .unwrap_or("unknown"),
                            pointer.id
                        )
                    }));
                } else {
                    summary_lines.push(format!(
                        "Tool result {}: {}",
                        tool_names
                            .get(tool_call_id)
                            .map(String::as_str)
                            .unwrap_or("unknown"),
                        summarize_tool_output(content)
                    ));
                }
            }
        }
    }

    let total_summarized_turns = prior_summary_turns(messages).saturating_add(summarized_turns);
    if pointer_mode && let Some(memory) = memory {
        let manifest = memory.compaction_pointer_manifest(pointer_ids);
        if !manifest.is_empty() {
            summary_lines.push("[Available addressable pointers]".to_owned());
            summary_lines.extend(manifest.into_iter().map(|citation| format!("- {citation}")));
        }
    }
    let summary_body = bounded_summary_lines(summary_lines, summary_token_budget).join("\n");
    let summary = Message::System {
        content: format!(
            "[Compacted earlier conversation: {} turn(s)]\n{}",
            total_summarized_turns, summary_body
        ),
    };
    let mut compacted = Vec::with_capacity(messages.len() - keep_start + 2);
    compacted.push(messages[0].clone());
    compacted.push(summary);
    compacted.extend(messages[keep_start..].iter().cloned());
    *messages = compacted;

    CompactionResult {
        kept_turns,
        summarized_turns,
        summarized_tool_outputs,
    }
}

fn prior_summary_turns(messages: &[Message]) -> usize {
    let Some(Message::System { content }) = messages.get(1) else {
        return 0;
    };
    content
        .strip_prefix("[Compacted earlier conversation: ")
        .and_then(|content| content.split_once(" turn(s)]"))
        .and_then(|(turns, _)| turns.parse().ok())
        .unwrap_or(0)
}

fn anchor_text(content: &str) -> String {
    truncate_middle(content.trim(), ANCHOR_MAX_CHARS)
}

fn summarize_text(content: &str) -> String {
    summarize_text_to(content, SUMMARY_ITEM_CHARS)
}

fn summarize_text_to(content: &str, max_chars: usize) -> String {
    truncate_middle(&content.replace(['\r', '\n'], " "), max_chars)
}

fn bounded_summary_lines(lines: Vec<String>, max_tokens: usize) -> Vec<String> {
    let total_tokens = lines
        .iter()
        .map(|line| estimate_tokens(line).saturating_add(1))
        .sum::<usize>();
    if total_tokens <= max_tokens {
        return lines;
    }
    let Some(first) = lines.first().cloned() else {
        return lines;
    };
    const OMITTED_MARKER: &str =
        "[older structured summary entries omitted; exact records remain addressable]";
    let marker_tokens = estimate_tokens(OMITTED_MARKER).saturating_add(1);
    if max_tokens <= marker_tokens {
        return vec![token_window(&first, 0, max_tokens).content];
    }
    let content_budget = max_tokens.saturating_sub(marker_tokens);
    let first_budget = (content_budget / 2).max(1);
    let first = token_window(&first, 0, first_budget).content;
    let mut remaining = max_tokens.saturating_sub(estimate_tokens(&first).saturating_add(1));
    remaining = remaining.saturating_sub(marker_tokens);
    let mut recent = Vec::new();
    for line in lines.into_iter().skip(1).rev() {
        let tokens = estimate_tokens(&line).saturating_add(1);
        if tokens > remaining {
            if recent.is_empty() && remaining > 1 {
                recent.push(token_window(&line, 0, remaining.saturating_sub(1)).content);
                remaining = 0;
            }
            continue;
        }
        remaining -= tokens;
        recent.push(line);
    }
    recent.reverse();
    let mut bounded = Vec::with_capacity(recent.len() + 2);
    bounded.push(first);
    bounded.push(OMITTED_MARKER.to_owned());
    bounded.extend(recent);
    bounded
}

fn summarize_tool_output(content: &str) -> String {
    let count = content.chars().count();
    if count <= TOOL_SUMMARY_EDGE_CHARS * 2 {
        return content.replace(['\r', '\n'], " ");
    }
    let head = content
        .chars()
        .take(TOOL_SUMMARY_EDGE_CHARS)
        .collect::<String>();
    let tail = content
        .chars()
        .rev()
        .take(TOOL_SUMMARY_EDGE_CHARS)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!(
        "{} … [{} chars omitted] … {}",
        head.replace(['\r', '\n'], " "),
        count.saturating_sub(TOOL_SUMMARY_EDGE_CHARS * 2),
        tail.replace(['\r', '\n'], " ")
    )
}

fn truncate_middle(content: &str, max_chars: usize) -> String {
    let count = content.chars().count();
    if count <= max_chars {
        return content.to_owned();
    }
    let edge = max_chars / 2;
    let head = content.chars().take(edge).collect::<String>();
    let tail = content
        .chars()
        .rev()
        .take(edge)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{head} … {tail}")
}

#[cfg(test)]
mod tests;
