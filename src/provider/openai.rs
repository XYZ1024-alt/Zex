use std::{
    collections::{BTreeMap, VecDeque},
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    agent::{
        AgentEvent, AssistantMessage, CompletionUsage, EventSender, Message, MessageRole, ToolCall,
        estimate_tokens,
    },
    provider::{
        NormalizedThinking, OpenAiApi, PreparedRequest, Provider, ThinkingLevel, ToolDefinition,
    },
};

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    endpoint: String,
    api: OpenAiApi,
    /// Memo for the whole-body token count. Preparing a request happens
    /// several times per turn on byte-identical input — status refresh,
    /// budget check, each compaction retry — and BPE over a full context is
    /// by far the most expensive part of it.
    token_cache: Arc<Mutex<VecDeque<(BodyKey, usize)>>>,
}

/// Body hash plus length. The length makes an already negligible hash
/// collision unable to hand back another request's token count.
type BodyKey = (u64, usize);

const TOKEN_CACHE_ENTRIES: usize = 4;

fn body_key(body: &[u8]) -> BodyKey {
    let mut hasher = DefaultHasher::new();
    body.hash(&mut hasher);
    (hasher.finish(), body.len())
}

#[derive(Debug)]
pub struct OpenAiPreparedRequest {
    body: Vec<u8>,
}

impl OpenAiProvider {
    pub fn new(
        base_url: &str,
        api_key: String,
        api: OpenAiApi,
        request_timeout: Duration,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(request_timeout)
            .build()
            .context("failed to build the HTTP client")?;
        let endpoint = format!("{}/{}", base_url.trim_end_matches('/'), api.endpoint());

        Ok(Self {
            client,
            api_key,
            endpoint,
            api,
            token_cache: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    pub(crate) fn api(&self) -> OpenAiApi {
        self.api
    }

    pub(crate) fn prepare_normalized(
        &self,
        model: &str,
        thinking: &NormalizedThinking,
        messages: &[Message],
        tools: &[ToolDefinition],
        max_output_tokens: usize,
    ) -> Result<PreparedRequest<OpenAiPreparedRequest>> {
        let body = match self.api {
            OpenAiApi::ChatCompletions => serde_json::to_vec(&ChatRequest::new(
                model,
                thinking,
                messages,
                tools,
                max_output_tokens,
            )),
            OpenAiApi::Responses => serde_json::to_vec(&ResponsesRequest::new(
                model,
                thinking,
                messages,
                tools,
                max_output_tokens,
            )),
        }
        .context("failed to serialize the prepared provider request")?;
        let input_tokens = self.count_body_tokens(&body)?;
        Ok(PreparedRequest::new(
            input_tokens,
            max_output_tokens,
            OpenAiPreparedRequest { body },
        ))
    }

    fn count_body_tokens(&self, body: &[u8]) -> Result<usize> {
        let key = body_key(body);
        if let Some(tokens) = self
            .token_cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .find_map(|(cached, tokens)| (*cached == key).then_some(*tokens))
        {
            return Ok(tokens);
        }
        let serialized = std::str::from_utf8(body)
            .context("prepared provider request was not valid UTF-8 JSON")?;
        let tokens = estimate_tokens(serialized);
        let mut cache = self
            .token_cache
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        cache.push_back((key, tokens));
        while cache.len() > TOKEN_CACHE_ENTRIES {
            cache.pop_front();
        }
        Ok(tokens)
    }

    async fn send_request(&self, request: OpenAiPreparedRequest) -> Result<reqwest::Response> {
        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(request.body)
            .send()
            .await
            .with_context(|| format!("failed to call {}", self.endpoint))?;

        if response.status().is_success() {
            return Ok(response);
        }

        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|error| format!("<failed to read response body: {error}>"));
        bail!("OpenAI-compatible provider returned {status}: {body}");
    }

    pub(crate) async fn complete_prepared(
        &self,
        request: OpenAiPreparedRequest,
        events: &EventSender,
    ) -> Result<AssistantMessage> {
        let started = Instant::now();
        let response = self.send_request(request).await?;
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let (body, mut message) =
            read_response_body(response, &content_type, self.api, events).await?;
        let elapsed = started.elapsed();
        message.usage = response_usage(&body, &content_type, self.api);
        if let Some(output_tokens) = message
            .usage
            .and_then(|usage| usage.output_tokens)
            .filter(|tokens| *tokens > 0)
        {
            let _ = events.send(AgentEvent::ProviderUsage {
                output_tokens,
                elapsed,
            });
        }
        Ok(message)
    }

    pub async fn list_models(
        base_url: &str,
        api_key: &str,
        request_timeout: Duration,
    ) -> Result<Vec<String>> {
        if api_key.trim().is_empty() {
            bail!("Provider API key is required before fetching models");
        }
        let endpoint = format!("{}/models", base_url.trim_end_matches('/'));
        let client = Client::builder()
            .timeout(request_timeout)
            .build()
            .context("failed to build the model discovery HTTP client")?;
        let response = client
            .get(&endpoint)
            .bearer_auth(api_key)
            .send()
            .await
            .with_context(|| format!("failed to fetch models from {endpoint}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|error| format!("<failed to read response body: {error}>"));
            bail!("Provider model discovery returned {status}: {body}");
        }
        let body = response
            .bytes()
            .await
            .context("failed to read Provider model discovery response")?;
        parse_models_response(&body)
    }
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelsResponseItem>,
}

#[derive(Debug, Deserialize)]
struct ModelsResponseItem {
    id: String,
}

fn parse_models_response(body: &[u8]) -> Result<Vec<String>> {
    let response: ModelsResponse =
        serde_json::from_slice(body).context("Provider model discovery returned invalid JSON")?;
    let mut models = response
        .data
        .into_iter()
        .map(|model| model.id.trim().to_owned())
        .filter(|id| !id.is_empty())
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    Ok(models)
}

impl Provider for OpenAiProvider {
    type Request = OpenAiPreparedRequest;

    fn sanitize_history(&self, model: &str, messages: &[Message]) -> Vec<Message> {
        let source = crate::provider::ProviderStateSource::new(
            self.endpoint.clone(),
            model.to_owned(),
            self.api,
        );
        crate::provider::sanitize_history_provider_states(messages, Some(&source))
    }

    fn encode_provider_state(&self, model: &str, state: Value) -> Option<Value> {
        Some(crate::provider::encode_provider_state(
            &crate::provider::ProviderStateSource::new(
                self.endpoint.clone(),
                model.to_owned(),
                self.api,
            ),
            state,
        ))
    }

    fn prepare_request(
        &self,
        model: &str,
        thinking_level: ThinkingLevel,
        messages: &[Message],
        tools: &[ToolDefinition],
        max_output_tokens: usize,
    ) -> Result<PreparedRequest<Self::Request>> {
        let capabilities = self.thinking_capabilities(model);
        let thinking = crate::provider::normalize_thinking_level(&capabilities, thinking_level);
        let messages = crate::provider::sanitize_messages(
            messages,
            capabilities.supports_interleaved_thinking,
            Some(&crate::provider::ProviderStateSource::new(
                self.endpoint.clone(),
                model.to_owned(),
                self.api,
            )),
        );
        self.prepare_normalized(model, &thinking, &messages, tools, max_output_tokens)
    }

    async fn complete(
        &self,
        request: Self::Request,
        events: &EventSender,
    ) -> Result<AssistantMessage> {
        self.complete_prepared(request, events).await
    }
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    stream: bool,
    stream_options: ChatStreamOptions,
    max_completion_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

impl<'a> ChatRequest<'a> {
    fn new(
        model: &'a str,
        thinking: &'a NormalizedThinking,
        messages: &'a [Message],
        tools: &'a [ToolDefinition],
        max_output_tokens: usize,
    ) -> Self {
        Self {
            model,
            messages: messages.iter().map(WireMessage::from).collect(),
            stream: true,
            stream_options: ChatStreamOptions {
                include_usage: true,
            },
            max_completion_tokens: max_output_tokens,
            reasoning_effort: thinking.provider_value.as_deref(),
            tools: tools.iter().map(WireTool::from).collect(),
            tool_choice: (!tools.is_empty()).then_some("auto"),
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatStreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
enum WireMessage<'a> {
    System {
        content: &'a str,
    },
    User {
        content: &'a str,
    },
    Assistant {
        content: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_details: Option<&'a Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking_blocks: Option<&'a Value>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<WireToolCall<'a>>,
    },
    Tool {
        tool_call_id: &'a str,
        content: &'a str,
    },
}

impl<'a> From<&'a Message> for WireMessage<'a> {
    fn from(message: &'a Message) -> Self {
        match message {
            Message::System { content } => Self::System { content },
            Message::User { content } => Self::User { content },
            Message::Assistant {
                content,
                thinking,
                tool_calls,
                provider_state,
            } => Self::Assistant {
                content: (!content.is_empty()).then_some(content),
                reasoning_content: chat_reasoning_content(thinking, provider_state),
                reasoning_details: provider_state_field(provider_state, "reasoning_details"),
                thinking_blocks: provider_state_field(provider_state, "thinking_blocks"),
                tool_calls: tool_calls.iter().map(WireToolCall::from).collect(),
            },
            Message::Tool {
                tool_call_id,
                content,
            } => Self::Tool {
                tool_call_id,
                content,
            },
        }
    }
}

fn chat_reasoning_content<'a>(
    thinking: &'a Option<String>,
    provider_state: &'a Option<Value>,
) -> Option<&'a str> {
    provider_state_field(provider_state, "reasoning_content")
        .and_then(Value::as_str)
        .or_else(|| provider_state_field(provider_state, "reasoning").and_then(Value::as_str))
        .or(thinking.as_deref())
}

fn provider_state_field<'a>(provider_state: &'a Option<Value>, field: &str) -> Option<&'a Value> {
    provider_state.as_ref()?.as_object()?.get(field)
}

#[derive(Debug, Serialize)]
struct WireToolCall<'a> {
    id: &'a str,
    r#type: &'static str,
    function: WireFunctionCall<'a>,
}

impl<'a> From<&'a ToolCall> for WireToolCall<'a> {
    fn from(tool_call: &'a ToolCall) -> Self {
        Self {
            id: &tool_call.id,
            r#type: "function",
            function: WireFunctionCall {
                name: &tool_call.name,
                arguments: &tool_call.arguments,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct WireFunctionCall<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Debug, Serialize)]
struct WireTool<'a> {
    r#type: &'static str,
    function: WireToolDefinition<'a>,
}

impl<'a> From<&'a ToolDefinition> for WireTool<'a> {
    fn from(tool: &'a ToolDefinition) -> Self {
        Self {
            r#type: "function",
            function: WireToolDefinition {
                name: &tool.name,
                description: &tool.description,
                parameters: &tool.parameters,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct ResponsesRequest<'a> {
    model: &'a str,
    input: Vec<ResponsesInputItem<'a>>,
    stream: bool,
    store: bool,
    max_output_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoning<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ResponsesTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

impl<'a> ResponsesRequest<'a> {
    fn new(
        model: &'a str,
        thinking: &'a NormalizedThinking,
        messages: &'a [Message],
        tools: &'a [ToolDefinition],
        max_output_tokens: usize,
    ) -> Self {
        Self {
            model,
            input: messages
                .iter()
                .flat_map(Vec::<ResponsesInputItem<'_>>::from)
                .collect(),
            stream: true,
            store: false,
            max_output_tokens,
            reasoning: thinking
                .provider_value
                .as_deref()
                .map(|effort| ResponsesReasoning { effort }),
            tools: tools.iter().map(ResponsesTool::from).collect(),
            tool_choice: (!tools.is_empty()).then_some("auto"),
        }
    }
}

#[derive(Debug, Serialize)]
struct ResponsesReasoning<'a> {
    effort: &'a str,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ResponsesInputItem<'a> {
    Message(ResponsesMessage<'a>),
    FunctionCall(ResponsesFunctionCall<'a>),
    FunctionCallOutput(ResponsesFunctionCallOutput<'a>),
    ProviderOutput(&'a Value),
}

#[derive(Debug, Serialize)]
struct ResponsesMessage<'a> {
    r#type: &'static str,
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ResponsesFunctionCall<'a> {
    r#type: &'static str,
    call_id: &'a str,
    name: &'a str,
    arguments: &'a str,
}

#[derive(Debug, Serialize)]
struct ResponsesFunctionCallOutput<'a> {
    r#type: &'static str,
    call_id: &'a str,
    output: &'a str,
}

impl<'a> ResponsesInputItem<'a> {
    fn message(role: &'static str, content: &'a str) -> Self {
        Self::Message(ResponsesMessage {
            r#type: "message",
            role,
            content,
        })
    }

    fn function_call(call_id: &'a str, name: &'a str, arguments: &'a str) -> Self {
        Self::FunctionCall(ResponsesFunctionCall {
            r#type: "function_call",
            call_id,
            name,
            arguments,
        })
    }

    fn function_call_output(call_id: &'a str, output: &'a str) -> Self {
        Self::FunctionCallOutput(ResponsesFunctionCallOutput {
            r#type: "function_call_output",
            call_id,
            output,
        })
    }
}

impl<'a> From<&'a Message> for Vec<ResponsesInputItem<'a>> {
    fn from(message: &'a Message) -> Self {
        match message {
            Message::System { content } => {
                vec![ResponsesInputItem::message("system", content)]
            }
            Message::User { content } => vec![ResponsesInputItem::message("user", content)],
            Message::Assistant {
                content,
                tool_calls,
                provider_state,
                ..
            } => {
                if let Some(output) = provider_state.as_ref().and_then(Value::as_array) {
                    return output
                        .iter()
                        .map(ResponsesInputItem::ProviderOutput)
                        .collect();
                }
                let mut items =
                    Vec::with_capacity(usize::from(!content.is_empty()) + tool_calls.len());
                if !content.is_empty() {
                    items.push(ResponsesInputItem::message("assistant", content));
                }
                items.extend(tool_calls.iter().map(|tool_call| {
                    ResponsesInputItem::function_call(
                        &tool_call.id,
                        &tool_call.name,
                        &tool_call.arguments,
                    )
                }));
                items
            }
            Message::Tool {
                tool_call_id,
                content,
            } => vec![ResponsesInputItem::function_call_output(
                tool_call_id,
                content,
            )],
        }
    }
}

#[derive(Debug, Serialize)]
struct ResponsesTool<'a> {
    r#type: &'static str,
    name: &'a str,
    description: &'a str,
    parameters: &'a serde_json::Value,
}

impl<'a> From<&'a ToolDefinition> for ResponsesTool<'a> {
    fn from(tool: &'a ToolDefinition) -> Self {
        Self {
            r#type: "function",
            name: &tool.name,
            description: &tool.description,
            parameters: &tool.parameters,
        }
    }
}

#[derive(Debug, Serialize)]
struct WireToolDefinition<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Debug, Default, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    thinking: Option<String>,
    reasoning_details: Option<Vec<Value>>,
    thinking_blocks: Option<Vec<Value>>,
    provider_specific_fields: Option<Value>,
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCall {
    index: usize,
    id: Option<String>,
    function: Option<StreamFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct StreamFunctionCall {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

/// A stream that goes silent for this long is treated as wedged: relays keep
/// SSE connections alive with ping/keep-alive bytes, so a complete stall means
/// the upstream will never finish. Aborting here beats waiting out the (much
/// longer) overall request timeout with a dead-looking UI.
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

enum ResponseMode {
    /// Nothing classifiable received yet (only whitespace so far).
    Undecided,
    /// SSE stream parsed incrementally, deltas forwarded live.
    Stream(SseParser),
    /// Non-SSE body; buffered whole and parsed at EOF.
    Buffered,
}

/// Read the provider response. SSE bodies are parsed line by line as chunks
/// arrive so thinking and text deltas reach the UI immediately; other bodies
/// keep the old buffer-then-parse behavior.
async fn read_response_body(
    mut response: reqwest::Response,
    content_type: &str,
    api: OpenAiApi,
    events: &EventSender,
) -> Result<(Vec<u8>, AssistantMessage)> {
    let mut body = Vec::new();
    let mut carry = Vec::new();
    let mut mode = if content_type.contains("text/event-stream") {
        ResponseMode::Stream(SseParser::new(api))
    } else {
        ResponseMode::Undecided
    };

    loop {
        let chunk = match tokio::time::timeout(STREAM_IDLE_TIMEOUT, response.chunk()).await {
            Ok(Ok(Some(chunk))) => chunk,
            Ok(Ok(None)) => break,
            Ok(Err(error)) => {
                return Err(error).context("failed to read provider response body");
            }
            Err(_) => bail!(
                "provider stream stalled: no data for {} seconds",
                STREAM_IDLE_TIMEOUT.as_secs()
            ),
        };
        body.extend_from_slice(&chunk);
        match &mut mode {
            ResponseMode::Undecided => {
                carry.extend_from_slice(&chunk);
                if body_looks_like_sse(&body) {
                    mode = ResponseMode::Stream(SseParser::new(api));
                } else if body.iter().any(|byte| !byte.is_ascii_whitespace()) {
                    mode = ResponseMode::Buffered;
                }
                if let ResponseMode::Stream(parser) = &mut mode {
                    feed_complete_lines(&mut carry, parser, events)?;
                }
            }
            ResponseMode::Stream(parser) => {
                carry.extend_from_slice(&chunk);
                feed_complete_lines(&mut carry, parser, events)?;
            }
            ResponseMode::Buffered => {}
        }
    }

    match mode {
        ResponseMode::Stream(mut parser) => {
            if !carry.is_empty() {
                parser.feed_line(&carry, events)?;
            }
            let message = parser.finish(events)?;
            Ok((body, message))
        }
        ResponseMode::Undecided | ResponseMode::Buffered => {
            let message = parse_response_body(&body, content_type, api, events)?;
            Ok((body, message))
        }
    }
}

fn body_looks_like_sse(body: &[u8]) -> bool {
    body.split(|byte| *byte == b'\n')
        .any(|line| line.trim_ascii_start().starts_with(b"data:"))
}

/// Feed every newline-terminated line currently in the carry buffer to the
/// parser. A multibyte UTF-8 character split across chunks is reassembled
/// inside the buffer before its line is parsed.
fn feed_complete_lines(
    carry: &mut Vec<u8>,
    parser: &mut SseParser,
    events: &EventSender,
) -> Result<()> {
    while let Some(pos) = carry.iter().position(|byte| *byte == b'\n') {
        let line: Vec<u8> = carry.drain(..=pos).collect();
        parser.feed_line(&line[..line.len() - 1], events)?;
    }
    Ok(())
}

fn parse_response_body(
    body: &[u8],
    content_type: &str,
    api: OpenAiApi,
    events: &EventSender,
) -> Result<AssistantMessage> {
    let body = strip_utf8_bom(body);
    if body.iter().all(u8::is_ascii_whitespace) {
        bail!(
            "provider returned an empty response body; expected OpenAI JSON or SSE from the configured /{} endpoint",
            api.endpoint()
        );
    }

    let first_byte = body
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace());
    let looks_like_json = matches!(first_byte, Some(b'{') | Some(b'['));
    let looks_like_sse = body_looks_like_sse(body);

    if content_type.contains("text/event-stream") || looks_like_sse {
        return match api {
            OpenAiApi::ChatCompletions => parse_chat_stream_body(body, events),
            OpenAiApi::Responses => parse_responses_stream_body(body, events),
        };
    }
    if content_type.contains("json") || looks_like_json {
        return match api {
            OpenAiApi::ChatCompletions => parse_chat_non_stream_body(body, events),
            OpenAiApi::Responses => parse_responses_non_stream_body(body, events),
        };
    }

    bail!(
        "provider returned an unsupported response (Content-Type: {}): {}",
        display_content_type(content_type),
        body_preview(body)
    )
}

fn response_usage(body: &[u8], content_type: &str, api: OpenAiApi) -> Option<CompletionUsage> {
    let body = strip_utf8_bom(body);
    let looks_like_sse = content_type.contains("text/event-stream")
        || body
            .split(|byte| *byte == b'\n')
            .any(|line| line.trim_ascii_start().starts_with(b"data:"));
    if looks_like_sse {
        return body
            .split(|byte| *byte == b'\n')
            .rev()
            .filter_map(|line| {
                let line = line.strip_suffix(b"\r").unwrap_or(line);
                let data = line.strip_prefix(b"data:")?;
                let data = data.strip_prefix(b" ").unwrap_or(data);
                (data != b"[DONE]" && !data.is_empty())
                    .then(|| serde_json::from_slice::<Value>(data).ok())
                    .flatten()
            })
            .find_map(|event| usage_from_value(&event, api));
    }

    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|response| usage_from_value(&response, api))
}

fn usage_from_value(value: &Value, api: OpenAiApi) -> Option<CompletionUsage> {
    let usage = match api {
        OpenAiApi::ChatCompletions => value.get("usage"),
        OpenAiApi::Responses => value.get("usage").or_else(|| {
            value
                .get("response")
                .and_then(|response| response.get("usage"))
        }),
    }?;
    Some(CompletionUsage {
        input_tokens: usage
            .get("input_tokens")
            .or_else(|| usage.get("prompt_tokens"))
            .and_then(Value::as_u64),
        output_tokens: usage
            .get("output_tokens")
            .or_else(|| usage.get("completion_tokens"))
            .and_then(Value::as_u64),
    })
}

fn parse_chat_stream_body(body: &[u8], events: &EventSender) -> Result<AssistantMessage> {
    let mut parser = ChatStreamParser::new();
    for line in body.split(|byte| *byte == b'\n') {
        parser.feed_line(line, events)?;
    }
    parser.finish(events)
}

enum SseParser {
    Chat(ChatStreamParser),
    Responses(ResponsesStreamParser),
}

impl SseParser {
    fn new(api: OpenAiApi) -> Self {
        match api {
            OpenAiApi::ChatCompletions => Self::Chat(ChatStreamParser::new()),
            OpenAiApi::Responses => Self::Responses(ResponsesStreamParser::default()),
        }
    }

    fn feed_line(&mut self, line: &[u8], events: &EventSender) -> Result<()> {
        match self {
            Self::Chat(parser) => parser.feed_line(line, events),
            Self::Responses(parser) => parser.feed_line(line, events),
        }
    }

    fn finish(self, events: &EventSender) -> Result<AssistantMessage> {
        match self {
            Self::Chat(parser) => parser.finish(events),
            Self::Responses(parser) => parser.finish(events),
        }
    }
}

/// Incremental Chat Completions SSE parser. Thinking deltas are forwarded to
/// the UI as they arrive (receiving one guarantees the final message has
/// non-empty thinking); content deltas stay buffered until `finish` so
/// `finish_message`'s `<think>` tag splitting still decides what is shown.
struct ChatStreamParser {
    content: String,
    thinking: String,
    provider_state: ChatProviderState,
    tool_calls: BTreeMap<usize, ToolCallAccumulator>,
    parsed_events: tokio::sync::mpsc::UnboundedSender<AgentEvent>,
    parsed_event_receiver: tokio::sync::mpsc::UnboundedReceiver<AgentEvent>,
    pending_content: Vec<AgentEvent>,
    thinking_emitted: bool,
}

impl ChatStreamParser {
    fn new() -> Self {
        let (parsed_events, parsed_event_receiver) = tokio::sync::mpsc::unbounded_channel();
        Self {
            content: String::new(),
            thinking: String::new(),
            provider_state: ChatProviderState::default(),
            tool_calls: BTreeMap::new(),
            parsed_events,
            parsed_event_receiver,
            pending_content: Vec::new(),
            thinking_emitted: false,
        }
    }

    fn feed_line(&mut self, line: &[u8], events: &EventSender) -> Result<()> {
        consume_sse_line(
            line,
            &mut self.content,
            &mut self.thinking,
            &mut self.provider_state,
            &mut self.tool_calls,
            &self.parsed_events,
        )?;
        while let Ok(event) = self.parsed_event_receiver.try_recv() {
            match event {
                AgentEvent::ThinkingDelta { .. } => {
                    self.thinking_emitted = true;
                    let _ = events.send(event);
                }
                event => self.pending_content.push(event),
            }
        }
        Ok(())
    }

    fn finish(mut self, events: &EventSender) -> Result<AssistantMessage> {
        drop(self.parsed_events);
        while let Ok(event) = self.parsed_event_receiver.try_recv() {
            self.pending_content.push(event);
        }
        let message = finish_message(
            self.content,
            self.thinking,
            self.provider_state.finish(),
            self.tool_calls,
        );
        if let Some(thinking) = &message.thinking {
            if !self.thinking_emitted {
                let _ = events.send(AgentEvent::ThinkingDelta {
                    delta: thinking.clone(),
                });
            }
            if !message.content.is_empty() {
                let _ = events.send(AgentEvent::MessageDelta {
                    role: MessageRole::Assistant,
                    delta: message.content.clone(),
                });
            }
        } else {
            for event in std::mem::take(&mut self.pending_content) {
                let _ = events.send(event);
            }
        }
        Ok(message)
    }
}

fn consume_sse_line(
    line: &[u8],
    content: &mut String,
    thinking: &mut String,
    provider_state: &mut ChatProviderState,
    tool_calls: &mut BTreeMap<usize, ToolCallAccumulator>,
    events: &EventSender,
) -> Result<()> {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let Some(data) = line.strip_prefix(b"data:") else {
        return Ok(());
    };
    let data = data.strip_prefix(b" ").unwrap_or(data);

    if data == b"[DONE]" || data.is_empty() {
        return Ok(());
    }

    let chunk: StreamChunk =
        serde_json::from_slice(data).context("provider returned an invalid streaming chunk")?;
    for choice in chunk.choices {
        let StreamDelta {
            content: content_delta,
            reasoning_content,
            reasoning,
            thinking: direct_thinking,
            reasoning_details,
            thinking_blocks,
            provider_specific_fields,
            tool_calls: delta_tool_calls,
        } = choice.delta;
        let reasoning_delta = reasoning_content
            .as_deref()
            .or(reasoning.as_deref())
            .or(direct_thinking.as_deref());
        let direct_thinking_blocks = thinking_blocks.as_deref();
        let provider_thinking_blocks = provider_specific_fields
            .as_ref()
            .and_then(|fields| fields.get("thinking_blocks"))
            .and_then(Value::as_array)
            .map(Vec::as_slice);
        if let Some(reasoning_details) = reasoning_details {
            provider_state.reasoning_details.extend(reasoning_details);
        }
        if let Some(blocks) = direct_thinking_blocks {
            provider_state
                .thinking_blocks
                .extend(blocks.iter().cloned());
        } else if let Some(blocks) = provider_thinking_blocks {
            provider_state
                .thinking_blocks
                .extend(blocks.iter().cloned());
        }
        let thinking_delta = reasoning_delta
            .map(ToOwned::to_owned)
            .or_else(|| thinking_text_from_blocks(direct_thinking_blocks))
            .or_else(|| thinking_text_from_blocks(provider_thinking_blocks));
        if let Some(delta) = thinking_delta.filter(|delta| !delta.is_empty()) {
            if reasoning_delta.is_some() {
                provider_state.reasoning_content.push_str(&delta);
            }
            thinking.push_str(&delta);
            let _ = events.send(AgentEvent::ThinkingDelta { delta });
        }
        if let Some(delta) = content_delta {
            content.push_str(&delta);
            let _ = events.send(AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta,
            });
        }

        for tool_call in delta_tool_calls.unwrap_or_default() {
            let entry = tool_calls.entry(tool_call.index).or_default();
            if let Some(id) = tool_call.id {
                entry.id.push_str(&id);
            }
            if let Some(function) = tool_call.function {
                if let Some(name) = function.name {
                    entry.name.push_str(&name);
                }
                if let Some(arguments) = function.arguments {
                    entry.arguments.push_str(&arguments);
                }
            }
        }
    }

    Ok(())
}

fn finish_message(
    content: String,
    thinking: String,
    provider_state: Option<Value>,
    tool_calls: BTreeMap<usize, ToolCallAccumulator>,
) -> AssistantMessage {
    let (thinking, content) = finalize_thinking(thinking, content);
    AssistantMessage {
        content,
        thinking,
        tool_calls: tool_calls
            .into_values()
            .map(|call| ToolCall {
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            })
            .collect(),
        provider_state,
        usage: None,
    }
}

#[derive(Debug, Default)]
struct ChatProviderState {
    reasoning_content: String,
    reasoning_details: Vec<Value>,
    thinking_blocks: Vec<Value>,
}

impl ChatProviderState {
    fn finish(self) -> Option<Value> {
        let mut fields = serde_json::Map::new();
        if !self.reasoning_content.is_empty() {
            fields.insert(
                "reasoning_content".to_owned(),
                Value::String(self.reasoning_content),
            );
        }
        if !self.reasoning_details.is_empty() {
            fields.insert(
                "reasoning_details".to_owned(),
                Value::Array(self.reasoning_details),
            );
        }
        if !self.thinking_blocks.is_empty() {
            fields.insert(
                "thinking_blocks".to_owned(),
                Value::Array(self.thinking_blocks),
            );
        }
        (!fields.is_empty()).then_some(Value::Object(fields))
    }
}

fn thinking_text_from_blocks(blocks: Option<&[Value]>) -> Option<String> {
    let mut text = String::new();
    for block in blocks.unwrap_or_default() {
        if let Some(value) = block
            .get("thinking")
            .or_else(|| block.get("text"))
            .or_else(|| block.get("summary"))
            .and_then(Value::as_str)
        {
            text.push_str(value);
        }
    }
    (!text.is_empty()).then_some(text)
}

fn finalize_thinking(explicit: String, content: String) -> (Option<String>, String) {
    if !explicit.is_empty() {
        let (_, content) = split_think_tags(&content);
        return (Some(explicit), content);
    }
    split_think_tags(&content)
}

fn split_think_tags(content: &str) -> (Option<String>, String) {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";

    let Some(open) = content.find(OPEN) else {
        return (None, content.to_owned());
    };
    let after_open = open + OPEN.len();
    let Some(close_offset) = content[after_open..].find(CLOSE) else {
        return (None, content.to_owned());
    };
    let close = after_open + close_offset;
    let after_close = close + CLOSE.len();
    let thinking = content[after_open..close].trim().to_owned();
    let answer = format!("{}{}", &content[..open], &content[after_close..])
        .trim()
        .to_owned();
    ((!thinking.is_empty()).then_some(thinking), answer)
}

#[derive(Debug, Deserialize)]
struct NonStreamResponse {
    choices: Vec<NonStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct NonStreamChoice {
    message: NonStreamMessage,
}

#[derive(Debug, Deserialize)]
struct NonStreamMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    thinking: Option<String>,
    reasoning_details: Option<Value>,
    thinking_blocks: Option<Value>,
    #[serde(default)]
    tool_calls: Vec<NonStreamToolCall>,
}

#[derive(Debug, Deserialize)]
struct NonStreamToolCall {
    id: String,
    function: NonStreamFunctionCall,
}

#[derive(Debug, Deserialize)]
struct NonStreamFunctionCall {
    name: String,
    arguments: String,
}

fn parse_chat_non_stream_body(body: &[u8], events: &EventSender) -> Result<AssistantMessage> {
    let response: NonStreamResponse = serde_json::from_slice(body).with_context(|| {
        format!(
            "provider returned invalid OpenAI JSON: {}",
            body_preview(body)
        )
    })?;
    let message = response
        .choices
        .into_iter()
        .next()
        .context("provider response contained no choices")?
        .message;
    let NonStreamMessage {
        content,
        reasoning_content,
        reasoning,
        thinking: direct_thinking,
        reasoning_details,
        thinking_blocks,
        tool_calls,
    } = message;
    let content = content.unwrap_or_default();
    let raw_reasoning = reasoning_content.or(reasoning).or(direct_thinking);
    let explicit_thinking = raw_reasoning
        .clone()
        .or_else(|| {
            thinking_blocks
                .as_ref()
                .and_then(Value::as_array)
                .and_then(|blocks| thinking_text_from_blocks(Some(blocks)))
        })
        .unwrap_or_default();
    let (thinking, content) = finalize_thinking(explicit_thinking, content);

    if let Some(thinking) = &thinking {
        let _ = events.send(AgentEvent::ThinkingDelta {
            delta: thinking.clone(),
        });
    }
    if !content.is_empty() {
        let _ = events.send(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: content.clone(),
        });
    }

    let mut provider_state = serde_json::Map::new();
    if let Some(reasoning_content) = raw_reasoning {
        provider_state.insert(
            "reasoning_content".to_owned(),
            Value::String(reasoning_content),
        );
    }
    if let Some(reasoning_details) = reasoning_details {
        provider_state.insert("reasoning_details".to_owned(), reasoning_details);
    }
    if let Some(thinking_blocks) = thinking_blocks {
        provider_state.insert("thinking_blocks".to_owned(), thinking_blocks);
    }

    Ok(AssistantMessage {
        content,
        thinking,
        tool_calls: tool_calls
            .into_iter()
            .map(|call| ToolCall {
                id: call.id,
                name: call.function.name,
                arguments: call.function.arguments,
            })
            .collect(),
        provider_state: (!provider_state.is_empty()).then_some(Value::Object(provider_state)),
        usage: None,
    })
}

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    #[serde(default)]
    output: Vec<Value>,
    error: Option<ResponsesError>,
}

#[derive(Debug, Deserialize)]
struct ResponsesError {
    message: String,
}

fn parse_responses_non_stream_body(body: &[u8], events: &EventSender) -> Result<AssistantMessage> {
    let response: ResponsesResponse = serde_json::from_slice(body).with_context(|| {
        format!(
            "provider returned invalid Responses API JSON: {}",
            body_preview(body)
        )
    })?;
    if let Some(error) = response.error {
        bail!("Responses API returned an error: {}", error.message);
    }

    finish_responses_message(response.output, events, false)
}

fn parse_responses_stream_body(body: &[u8], events: &EventSender) -> Result<AssistantMessage> {
    let mut parser = ResponsesStreamParser::default();
    for line in body.split(|byte| *byte == b'\n') {
        parser.feed_line(line, events)?;
    }
    parser.finish(events)
}

/// Incremental Responses API SSE parser. Text and reasoning-summary deltas are
/// forwarded to the UI as they arrive; the final message is assembled from the
/// completed response output in `finish`.
#[derive(Default)]
struct ResponsesStreamParser {
    content: String,
    thinking: String,
    output: Vec<Value>,
    completed_output: Option<Vec<Value>>,
    response_error: Option<String>,
}

impl ResponsesStreamParser {
    fn feed_line(&mut self, line: &[u8], events: &EventSender) -> Result<()> {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(data) = line.strip_prefix(b"data:") else {
            return Ok(());
        };
        let data = data.strip_prefix(b" ").unwrap_or(data);
        if data == b"[DONE]" || data.is_empty() {
            return Ok(());
        }

        let event: Value = serde_json::from_slice(data)
            .context("provider returned an invalid Responses API streaming event")?;
        match event.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    self.content.push_str(delta);
                    let _ = events.send(AgentEvent::MessageDelta {
                        role: MessageRole::Assistant,
                        delta: delta.to_owned(),
                    });
                }
            }
            Some("response.reasoning_summary_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    self.thinking.push_str(delta);
                    let _ = events.send(AgentEvent::ThinkingDelta {
                        delta: delta.to_owned(),
                    });
                }
            }
            Some("response.output_item.done") => {
                if let Some(item) = event.get("item") {
                    self.output.push(item.clone());
                }
            }
            Some("response.completed") => {
                self.completed_output = event
                    .get("response")
                    .and_then(|response| response.get("output"))
                    .and_then(Value::as_array)
                    .cloned();
            }
            Some("response.failed") | Some("error") => {
                self.response_error = response_stream_error(&event);
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(self, events: &EventSender) -> Result<AssistantMessage> {
        if let Some(error) = self.response_error {
            bail!("Responses API stream failed: {error}");
        }
        let output = self.completed_output.unwrap_or(self.output);

        let (completion_events, mut completion_event_receiver) =
            tokio::sync::mpsc::unbounded_channel();
        let mut message =
            finish_responses_message(output, &completion_events, !self.content.is_empty())?;
        drop(completion_events);
        let completion_events =
            std::iter::from_fn(|| completion_event_receiver.try_recv().ok()).collect::<Vec<_>>();
        let mut thinking = self.thinking;
        if !self.content.is_empty() {
            let (fallback_thinking, stripped_content) = split_think_tags(&self.content);
            message.content = stripped_content;
            if thinking.is_empty()
                && message.thinking.is_none()
                && let Some(fallback_thinking) = fallback_thinking
            {
                thinking = fallback_thinking;
            }
        }
        if !thinking.is_empty() {
            message.thinking = Some(thinking);
        }
        for event in completion_events {
            if matches!(event, AgentEvent::ThinkingDelta { .. }) && message.thinking.is_some() {
                continue;
            }
            let _ = events.send(event);
        }
        Ok(message)
    }
}

fn finish_responses_message(
    output: Vec<Value>,
    events: &EventSender,
    text_already_emitted: bool,
) -> Result<AssistantMessage> {
    let mut content = String::new();
    let mut thinking = String::new();
    let mut tool_calls = Vec::new();

    for item in &output {
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => {
                for part in item
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        thinking.push_str(text);
                    }
                }
            }
            Some("message") => {
                for part in item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if part.get("type").and_then(Value::as_str) == Some("output_text")
                        && let Some(text) = part.get("text").and_then(Value::as_str)
                    {
                        content.push_str(text);
                    }
                }
            }
            Some("function_call") => {
                let call_id = required_response_string(item, "call_id", "function call")?;
                let name = required_response_string(item, "name", "function call")?;
                let arguments = required_response_string(item, "arguments", "function call")?;
                tool_calls.push(ToolCall {
                    id: call_id.to_owned(),
                    name: name.to_owned(),
                    arguments: arguments.to_owned(),
                });
            }
            _ => {}
        }
    }
    let (fallback_thinking, content) = split_think_tags(&content);
    if thinking.is_empty()
        && let Some(fallback_thinking) = fallback_thinking
    {
        thinking = fallback_thinking;
    }

    if !thinking.is_empty() {
        let _ = events.send(AgentEvent::ThinkingDelta {
            delta: thinking.clone(),
        });
    }
    if !text_already_emitted && !content.is_empty() {
        let _ = events.send(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: content.clone(),
        });
    }

    Ok(AssistantMessage {
        content,
        thinking: (!thinking.is_empty()).then_some(thinking),
        tool_calls,
        provider_state: Some(Value::Array(output)),
        usage: None,
    })
}

fn required_response_string<'a>(item: &'a Value, field: &str, item_kind: &str) -> Result<&'a str> {
    item.get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("Responses API {item_kind} omitted {field}"))
}

fn response_stream_error(event: &Value) -> Option<String> {
    event
        .get("error")
        .and_then(|error| error.get("message").or(Some(error)))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| {
            event
                .get("response")
                .and_then(|response| response.get("error"))
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
}

fn strip_utf8_bom(body: &[u8]) -> &[u8] {
    body.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(body)
}

fn display_content_type(content_type: &str) -> &str {
    if content_type.is_empty() {
        "<missing>"
    } else {
        content_type
    }
}

fn body_preview(body: &[u8]) -> String {
    const MAX_PREVIEW_CHARS: usize = 500;

    let text = String::from_utf8_lossy(body);
    let mut preview = text.chars().take(MAX_PREVIEW_CHARS).collect::<String>();
    preview = preview.replace('\r', "\\r").replace('\n', "\\n");
    if text.chars().count() > MAX_PREVIEW_CHARS {
        preview.push('…');
    }
    format!("body preview: {preview:?}")
}

#[cfg(test)]
mod tests;
