use std::{collections::BTreeMap, time::Duration};

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    agent::{AgentEvent, AssistantMessage, EventSender, Message, MessageRole, ToolCall},
    provider::{OpenAiApi, Provider, ThinkingLevel, ToolDefinition},
};

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    endpoint: String,
    api: OpenAiApi,
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
        })
    }

    async fn send_request(
        &self,
        model: &str,
        thinking_level: Option<ThinkingLevel>,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<reqwest::Response> {
        let request = self.client.post(&self.endpoint).bearer_auth(&self.api_key);
        let response = match self.api {
            OpenAiApi::ChatCompletions => {
                request.json(&ChatRequest::new(model, thinking_level, messages, tools))
            }
            OpenAiApi::Responses => request.json(&ResponsesRequest::new(
                model,
                thinking_level,
                messages,
                tools,
            )),
        }
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
}

impl Provider for OpenAiProvider {
    fn supports_thinking(&self, model: &str) -> bool {
        model_supports_thinking(model)
    }

    async fn complete(
        &self,
        model: &str,
        thinking_level: Option<ThinkingLevel>,
        messages: &[Message],
        tools: &[ToolDefinition],
        events: &EventSender,
    ) -> Result<AssistantMessage> {
        let response = self
            .send_request(model, thinking_level, messages, tools)
            .await?;
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let body = response
            .bytes()
            .await
            .context("failed to read provider response body")?;
        parse_response_body(&body, &content_type, self.api, events)
    }
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

impl<'a> ChatRequest<'a> {
    fn new(
        model: &'a str,
        thinking_level: Option<ThinkingLevel>,
        messages: &'a [Message],
        tools: &'a [ToolDefinition],
    ) -> Self {
        Self {
            model,
            messages: messages.iter().map(WireMessage::from).collect(),
            stream: true,
            reasoning_effort: thinking_level.and_then(ThinkingLevel::as_provider_value),
            tools: tools.iter().map(WireTool::from).collect(),
            tool_choice: (!tools.is_empty()).then_some("auto"),
        }
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoning>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ResponsesTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

impl<'a> ResponsesRequest<'a> {
    fn new(
        model: &'a str,
        thinking_level: Option<ThinkingLevel>,
        messages: &'a [Message],
        tools: &'a [ToolDefinition],
    ) -> Self {
        Self {
            model,
            input: messages
                .iter()
                .flat_map(Vec::<ResponsesInputItem<'_>>::from)
                .collect(),
            stream: true,
            store: false,
            reasoning: thinking_level
                .and_then(ThinkingLevel::as_provider_value)
                .map(|effort| ResponsesReasoning { effort }),
            tools: tools.iter().map(ResponsesTool::from).collect(),
            tool_choice: (!tools.is_empty()).then_some("auto"),
        }
    }
}

#[derive(Debug, Serialize)]
struct ResponsesReasoning {
    effort: &'static str,
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

fn model_supports_thinking(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    model.starts_with("gpt-5")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.contains("deepseek")
        || model.contains("claude")
        || model.contains("gemini")
        || model.contains("qwen")
        || model.contains("grok")
        || model.contains("magistral")
        || model.contains("reasoning")
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
    let looks_like_sse = body
        .split(|byte| *byte == b'\n')
        .any(|line| line.trim_ascii_start().starts_with(b"data:"));

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

fn parse_chat_stream_body(body: &[u8], events: &EventSender) -> Result<AssistantMessage> {
    let mut content = String::new();
    let mut thinking = String::new();
    let mut provider_state = ChatProviderState::default();
    let mut tool_calls = BTreeMap::<usize, ToolCallAccumulator>::new();
    let (parsed_events, mut event_receiver) = tokio::sync::mpsc::unbounded_channel();

    for line in body.split(|byte| *byte == b'\n') {
        consume_sse_line(
            line,
            &mut content,
            &mut thinking,
            &mut provider_state,
            &mut tool_calls,
            &parsed_events,
        )?;
    }

    let message = finish_message(content, thinking, provider_state.finish(), tool_calls);
    drop(parsed_events);
    let parsed_events = std::iter::from_fn(|| event_receiver.try_recv().ok()).collect();
    emit_separated_message(events, &message, parsed_events);
    Ok(message)
}

fn emit_separated_message(
    events: &EventSender,
    message: &AssistantMessage,
    parsed_events: Vec<AgentEvent>,
) {
    let mut thinking_emitted = false;
    let mut content_emitted = false;
    for event in parsed_events {
        match event {
            AgentEvent::ThinkingDelta { delta } if message.thinking.is_some() => {
                thinking_emitted = true;
                let _ = events.send(AgentEvent::ThinkingDelta { delta });
            }
            AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta,
            } if message.thinking.is_none() => {
                content_emitted = true;
                let _ = events.send(AgentEvent::MessageDelta {
                    role: MessageRole::Assistant,
                    delta,
                });
            }
            _ => {}
        }
    }

    if !thinking_emitted && let Some(thinking) = &message.thinking {
        let _ = events.send(AgentEvent::ThinkingDelta {
            delta: thinking.clone(),
        });
    }
    if !content_emitted && !message.content.is_empty() {
        let _ = events.send(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: message.content.clone(),
        });
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
            reasoning_details,
            thinking_blocks,
            provider_specific_fields,
            tool_calls: delta_tool_calls,
        } = choice.delta;
        let reasoning_delta = reasoning_content.as_deref().or(reasoning.as_deref());
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
        reasoning_details,
        thinking_blocks,
        tool_calls,
    } = message;
    let content = content.unwrap_or_default();
    let raw_reasoning = reasoning_content.or(reasoning);
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
    let mut content = String::new();
    let mut thinking = String::new();
    let mut output = Vec::<Value>::new();
    let mut completed_output = None;
    let mut response_error = None;

    for line in body.split(|byte| *byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(data) = line.strip_prefix(b"data:") else {
            continue;
        };
        let data = data.strip_prefix(b" ").unwrap_or(data);
        if data == b"[DONE]" || data.is_empty() {
            continue;
        }

        let event: Value = serde_json::from_slice(data)
            .context("provider returned an invalid Responses API streaming event")?;
        match event.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    content.push_str(delta);
                    let _ = events.send(AgentEvent::MessageDelta {
                        role: MessageRole::Assistant,
                        delta: delta.to_owned(),
                    });
                }
            }
            Some("response.reasoning_summary_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    thinking.push_str(delta);
                    let _ = events.send(AgentEvent::ThinkingDelta {
                        delta: delta.to_owned(),
                    });
                }
            }
            Some("response.output_item.done") => {
                if let Some(item) = event.get("item") {
                    output.push(item.clone());
                }
            }
            Some("response.completed") => {
                completed_output = event
                    .get("response")
                    .and_then(|response| response.get("output"))
                    .and_then(Value::as_array)
                    .cloned();
            }
            Some("response.failed") | Some("error") => {
                response_error = response_stream_error(&event);
            }
            _ => {}
        }
    }

    if let Some(error) = response_error {
        bail!("Responses API stream failed: {error}");
    }
    if let Some(completed_output) = completed_output {
        output = completed_output;
    }

    let (completion_events, mut completion_event_receiver) = tokio::sync::mpsc::unbounded_channel();
    let mut message = finish_responses_message(output, &completion_events, !content.is_empty())?;
    drop(completion_events);
    let completion_events =
        std::iter::from_fn(|| completion_event_receiver.try_recv().ok()).collect::<Vec<_>>();
    if !content.is_empty() {
        message.content = content;
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
mod tests {
    use std::collections::BTreeMap;

    use tokio::sync::mpsc;

    use crate::{
        agent::{AgentEvent, Message, MessageRole, ToolCall},
        provider::{OpenAiApi, ThinkingLevel, ToolDefinition},
    };

    use super::{
        ChatProviderState, ResponsesRequest, ToolCallAccumulator, consume_sse_line, finish_message,
        parse_response_body,
    };

    #[test]
    fn assembles_streamed_text_and_tool_calls() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let mut content = String::new();
        let mut thinking = String::new();
        let mut provider_state = ChatProviderState::default();
        let mut tool_calls = BTreeMap::<usize, ToolCallAccumulator>::new();

        consume_sse_line(
            br#"data: {"choices":[{"delta":{"content":"Zex "}}]}"#,
            &mut content,
            &mut thinking,
            &mut provider_state,
            &mut tool_calls,
            &events,
        )
        .unwrap();
        consume_sse_line(
            br#"data: {"choices":[{"delta":{"content":"streams","tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"pa"}}]}}]}"#,
            &mut content,
            &mut thinking,
            &mut provider_state,
            &mut tool_calls,
            &events,
        )
        .unwrap();
        consume_sse_line(
            br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"Cargo.toml\"}"}}]}}]}"#,
            &mut content,
            &mut thinking,
            &mut provider_state,
            &mut tool_calls,
            &events,
        )
        .unwrap();

        let message = finish_message(content, thinking, provider_state.finish(), tool_calls);
        assert_eq!(message.content, "Zex streams");
        assert_eq!(message.thinking, None);
        assert_eq!(message.tool_calls.len(), 1);
        assert_eq!(message.tool_calls[0].id, "call_1");
        assert_eq!(message.tool_calls[0].name, "read");
        assert_eq!(message.tool_calls[0].arguments, r#"{"path":"Cargo.toml"}"#);
        match receiver.try_recv().unwrap() {
            AgentEvent::MessageDelta { role, delta } => {
                assert_eq!(role, MessageRole::Assistant);
                assert_eq!(delta, "Zex ");
            }
            event => panic!("unexpected event: {event:?}"),
        }
        match receiver.try_recv().unwrap() {
            AgentEvent::MessageDelta { role, delta } => {
                assert_eq!(role, MessageRole::Assistant);
                assert_eq!(delta, "streams");
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[test]
    fn detects_sse_without_a_content_type() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let message = parse_response_body(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Zex\"}}]}\n\ndata: [DONE]\n\n",
            "",
            OpenAiApi::ChatCompletions,
            &events,
        )
        .unwrap();

        assert_eq!(message.content, "Zex");
        match receiver.try_recv().unwrap() {
            AgentEvent::MessageDelta { role, delta } => {
                assert_eq!(role, MessageRole::Assistant);
                assert_eq!(delta, "Zex");
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[test]
    fn detects_json_with_an_incorrect_content_type() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let message = parse_response_body(
            br#"{"choices":[{"message":{"content":"Zex JSON"}}]}"#,
            "text/plain",
            OpenAiApi::ChatCompletions,
            &events,
        )
        .unwrap();

        assert_eq!(message.content, "Zex JSON");
        match receiver.try_recv().unwrap() {
            AgentEvent::MessageDelta { role, delta } => {
                assert_eq!(role, MessageRole::Assistant);
                assert_eq!(delta, "Zex JSON");
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[test]
    fn prefers_explicit_reasoning_fields_over_think_tag_fallback() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let message = parse_response_body(
            br#"{"choices":[{"message":{
                "content":"<think>fallback</think>Final answer",
                "reasoning_content":"Provider reasoning"
            }}]}"#,
            "application/json",
            OpenAiApi::ChatCompletions,
            &events,
        )
        .unwrap();

        assert_eq!(message.thinking.as_deref(), Some("Provider reasoning"));
        assert_eq!(message.content, "Final answer");
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::ThinkingDelta {
                delta: "Provider reasoning".to_owned(),
            }
        );
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta: "Final answer".to_owned(),
            }
        );
    }

    #[test]
    fn separates_think_tags_when_provider_has_no_reasoning_field() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let message = parse_response_body(
            br#"{"choices":[{"message":{"content":"<think>\nInspect first.\n</think>\nFinal answer"}}]}"#,
            "application/json",
            OpenAiApi::ChatCompletions,
            &events,
        )
        .unwrap();

        assert_eq!(message.thinking.as_deref(), Some("Inspect first."));
        assert_eq!(message.content, "Final answer");
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::ThinkingDelta {
                delta: "Inspect first.".to_owned(),
            }
        );
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta: "Final answer".to_owned(),
            }
        );
    }

    #[test]
    fn separates_streamed_think_tags_before_emitting_final_answer() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let body = br#"
data: {"choices":[{"delta":{"content":"<thi"}}]}

data: {"choices":[{"delta":{"content":"nk>Inspect first."}}]}

data: {"choices":[{"delta":{"content":"</think>Final "}}]}

data: {"choices":[{"delta":{"content":"answer"}}]}

data: [DONE]

"#;
        let message = parse_response_body(
            body,
            "text/event-stream",
            OpenAiApi::ChatCompletions,
            &events,
        )
        .unwrap();

        assert_eq!(message.thinking.as_deref(), Some("Inspect first."));
        assert_eq!(message.content, "Final answer");
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::ThinkingDelta {
                delta: "Inspect first.".to_owned(),
            }
        );
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta: "Final answer".to_owned(),
            }
        );
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn parses_streamed_reasoning_and_thinking_blocks_without_mixing_answer() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let body = br#"
data: {"choices":[{"delta":{"reasoning_content":"Plan "}}]}

data: {"choices":[{"delta":{"thinking_blocks":[{"type":"thinking","thinking":"carefully."}]}}]}

data: {"choices":[{"delta":{"content":"Answer"}}]}

data: [DONE]

"#;

        let message = parse_response_body(
            body,
            "text/event-stream",
            OpenAiApi::ChatCompletions,
            &events,
        )
        .unwrap();

        assert_eq!(message.thinking.as_deref(), Some("Plan carefully."));
        assert_eq!(message.content, "Answer");
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::ThinkingDelta {
                delta: "Plan ".to_owned(),
            }
        );
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::ThinkingDelta {
                delta: "carefully.".to_owned(),
            }
        );
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta: "Answer".to_owned(),
            }
        );
        assert_eq!(
            message.provider_state.as_ref().unwrap()["thinking_blocks"][0]["thinking"],
            "carefully."
        );
        assert_eq!(
            message.provider_state.as_ref().unwrap()["reasoning_content"],
            "Plan "
        );
    }

    #[test]
    fn serializes_reasoning_state_for_chat_tool_continuation() {
        let messages = vec![Message::Assistant {
            content: String::new(),
            thinking: Some("Call the reader.".to_owned()),
            tool_calls: vec![ToolCall {
                id: "call_read".to_owned(),
                name: "read".to_owned(),
                arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
            }],
            provider_state: Some(serde_json::json!({
                "reasoning_content": "Call the reader.",
                "reasoning_details": [{"type": "reasoning.summary", "summary": "Call reader"}],
                "thinking_blocks": [{
                    "type": "thinking",
                    "thinking": "Call the reader.",
                    "signature": "signed"
                }]
            })),
        }];

        let request = serde_json::to_value(super::ChatRequest::new(
            "deepseek-reasoner",
            None,
            &messages,
            &[],
        ))
        .unwrap();
        let assistant = &request["messages"][0];

        assert_eq!(assistant["reasoning_content"], "Call the reader.");
        assert_eq!(assistant["reasoning_details"][0]["summary"], "Call reader");
        assert_eq!(assistant["thinking_blocks"][0]["signature"], "signed");
        assert_eq!(assistant["tool_calls"][0]["id"], "call_read");
    }

    #[test]
    fn reports_empty_and_unsupported_bodies_with_context() {
        let (events, _) = mpsc::unbounded_channel();

        let empty_error =
            parse_response_body(b"", "application/json", OpenAiApi::ChatCompletions, &events)
                .unwrap_err();
        assert!(empty_error.to_string().contains("empty response body"));

        let html_error = parse_response_body(
            b"<html>gateway login</html>",
            "text/html",
            OpenAiApi::ChatCompletions,
            &events,
        )
        .unwrap_err();
        let message = html_error.to_string();
        assert!(message.contains("Content-Type: text/html"));
        assert!(message.contains("gateway login"));
    }

    #[test]
    fn serializes_responses_input_and_flat_tools() {
        let messages = vec![
            Message::User {
                content: "read Cargo.toml".to_owned(),
            },
            Message::Assistant {
                content: String::new(),
                thinking: None,
                tool_calls: vec![ToolCall {
                    id: "call_read".to_owned(),
                    name: "read".to_owned(),
                    arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
                }],
                provider_state: None,
            },
            Message::Tool {
                tool_call_id: "call_read".to_owned(),
                content: "name = zex".to_owned(),
            },
        ];
        let tools = vec![ToolDefinition {
            name: "read".to_owned(),
            description: "Read a file".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        }];

        let request =
            serde_json::to_value(ResponsesRequest::new("test-model", None, &messages, &tools))
                .unwrap();
        assert_eq!(request["stream"], true);
        assert_eq!(request["store"], false);
        assert_eq!(request["tools"][0]["type"], "function");
        assert_eq!(request["tools"][0]["name"], "read");
        assert!(request["tools"][0].get("function").is_none());
        assert_eq!(request["input"][1]["type"], "function_call");
        assert_eq!(request["input"][1]["call_id"], "call_read");
        assert_eq!(request["input"][2]["type"], "function_call_output");
        assert_eq!(request["input"][2]["call_id"], "call_read");
    }

    #[test]
    fn serializes_supported_thinking_levels() {
        let messages = vec![Message::User {
            content: "think".to_owned(),
        }];
        let responses = serde_json::to_value(ResponsesRequest::new(
            "gpt-5",
            Some(ThinkingLevel::High),
            &messages,
            &[],
        ))
        .unwrap();
        assert_eq!(responses["reasoning"]["effort"], "high");

        let chat = serde_json::to_value(super::ChatRequest::new(
            "gpt-5",
            Some(ThinkingLevel::Low),
            &messages,
            &[],
        ))
        .unwrap();
        assert_eq!(chat["reasoning_effort"], "low");
        assert!(super::model_supports_thinking("deepseek-reasoner"));
        assert!(super::model_supports_thinking("anthropic/claude-sonnet-4"));
    }

    #[test]
    fn parses_responses_json_with_text_and_function_calls() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let message = parse_response_body(
            br#"{
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Checking."}]
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "read",
                        "arguments": "{\"path\":\"Cargo.toml\"}"
                    }
                ],
                "error": null
            }"#,
            "application/json",
            OpenAiApi::Responses,
            &events,
        )
        .unwrap();

        assert_eq!(message.content, "Checking.");
        assert_eq!(message.tool_calls[0].id, "call_1");
        assert_eq!(message.tool_calls[0].name, "read");
        assert!(message.provider_state.is_some());
        match receiver.try_recv().unwrap() {
            AgentEvent::MessageDelta { role, delta } => {
                assert_eq!(role, MessageRole::Assistant);
                assert_eq!(delta, "Checking.");
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[test]
    fn parses_responses_reasoning_summary_separately() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let message = parse_response_body(
            br#"{
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [{"type": "summary_text", "text": "Checked constraints."}]
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done."}]
                    }
                ],
                "error": null
            }"#,
            "application/json",
            OpenAiApi::Responses,
            &events,
        )
        .unwrap();

        assert_eq!(message.thinking.as_deref(), Some("Checked constraints."));
        assert_eq!(message.content, "Done.");
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::ThinkingDelta {
                delta: "Checked constraints.".to_owned(),
            }
        );
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta: "Done.".to_owned(),
            }
        );
    }

    #[test]
    fn parses_responses_sse_text_and_completed_output() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let body = br#"
event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Done"}

event: response.completed
data: {"type":"response.completed","response":{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done"}]}]}}

"#;
        let message =
            parse_response_body(body, "text/event-stream", OpenAiApi::Responses, &events).unwrap();

        assert_eq!(message.content, "Done");
        assert!(message.tool_calls.is_empty());
        match receiver.try_recv().unwrap() {
            AgentEvent::MessageDelta { role, delta } => {
                assert_eq!(role, MessageRole::Assistant);
                assert_eq!(delta, "Done");
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }
}
