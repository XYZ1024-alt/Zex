use std::{collections::BTreeMap, time::Duration};

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    agent::{AgentEvent, AssistantMessage, EventSender, Message, MessageRole, ToolCall},
    provider::{OpenAiApi, Provider, ToolDefinition},
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
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<reqwest::Response> {
        let request = self.client.post(&self.endpoint).bearer_auth(&self.api_key);
        let response = match self.api {
            OpenAiApi::ChatCompletions => request.json(&ChatRequest::new(model, messages, tools)),
            OpenAiApi::Responses => request.json(&ResponsesRequest::new(model, messages, tools)),
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
    async fn complete(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        events: &EventSender,
    ) -> Result<AssistantMessage> {
        let response = self.send_request(model, messages, tools).await?;
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

impl<'a> ChatRequest<'a> {
    fn new(model: &'a str, messages: &'a [Message], tools: &'a [ToolDefinition]) -> Self {
        Self {
            model,
            messages: messages.iter().map(WireMessage::from).collect(),
            stream: true,
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
                tool_calls,
                ..
            } => Self::Assistant {
                content: (!content.is_empty()).then_some(content),
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ResponsesTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

impl<'a> ResponsesRequest<'a> {
    fn new(model: &'a str, messages: &'a [Message], tools: &'a [ToolDefinition]) -> Self {
        Self {
            model,
            input: messages
                .iter()
                .flat_map(Vec::<ResponsesInputItem<'_>>::from)
                .collect(),
            stream: true,
            store: false,
            tools: tools.iter().map(ResponsesTool::from).collect(),
            tool_choice: (!tools.is_empty()).then_some("auto"),
        }
    }
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
    let mut tool_calls = BTreeMap::<usize, ToolCallAccumulator>::new();

    for line in body.split(|byte| *byte == b'\n') {
        consume_sse_line(line, &mut content, &mut tool_calls, events)?;
    }

    Ok(finish_message(content, tool_calls))
}

fn consume_sse_line(
    line: &[u8],
    content: &mut String,
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
        if let Some(delta) = choice.delta.content {
            content.push_str(&delta);
            let _ = events.send(AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta,
            });
        }

        for tool_call in choice.delta.tool_calls.unwrap_or_default() {
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
    tool_calls: BTreeMap<usize, ToolCallAccumulator>,
) -> AssistantMessage {
    AssistantMessage {
        content,
        tool_calls: tool_calls
            .into_values()
            .map(|call| ToolCall {
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            })
            .collect(),
        provider_state: None,
    }
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
    let content = message.content.unwrap_or_default();

    if !content.is_empty() {
        let _ = events.send(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: content.clone(),
        });
    }

    Ok(AssistantMessage {
        content,
        tool_calls: message
            .tool_calls
            .into_iter()
            .map(|call| ToolCall {
                id: call.id,
                name: call.function.name,
                arguments: call.function.arguments,
            })
            .collect(),
        provider_state: None,
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

    let mut message = finish_responses_message(output, events, !content.is_empty())?;
    if !content.is_empty() {
        message.content = content;
    }
    Ok(message)
}

fn finish_responses_message(
    output: Vec<Value>,
    events: &EventSender,
    text_already_emitted: bool,
) -> Result<AssistantMessage> {
    let mut content = String::new();
    let mut tool_calls = Vec::new();

    for item in &output {
        match item.get("type").and_then(Value::as_str) {
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

    if !text_already_emitted && !content.is_empty() {
        let _ = events.send(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: content.clone(),
        });
    }

    Ok(AssistantMessage {
        content,
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
        provider::{OpenAiApi, ToolDefinition},
    };

    use super::{
        ResponsesRequest, ToolCallAccumulator, consume_sse_line, finish_message,
        parse_response_body,
    };

    #[test]
    fn assembles_streamed_text_and_tool_calls() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let mut content = String::new();
        let mut tool_calls = BTreeMap::<usize, ToolCallAccumulator>::new();

        consume_sse_line(
            br#"data: {"choices":[{"delta":{"content":"Zex "}}]}"#,
            &mut content,
            &mut tool_calls,
            &events,
        )
        .unwrap();
        consume_sse_line(
            br#"data: {"choices":[{"delta":{"content":"streams","tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"pa"}}]}}]}"#,
            &mut content,
            &mut tool_calls,
            &events,
        )
        .unwrap();
        consume_sse_line(
            br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"Cargo.toml\"}"}}]}}]}"#,
            &mut content,
            &mut tool_calls,
            &events,
        )
        .unwrap();

        let message = finish_message(content, tool_calls);
        assert_eq!(message.content, "Zex streams");
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
            serde_json::to_value(ResponsesRequest::new("test-model", &messages, &tools)).unwrap();
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
