use std::collections::BTreeMap;

use tokio::sync::mpsc;

use crate::{
    agent::{AgentEvent, Message, MessageRole, ToolCall},
    provider::{OpenAiApi, ThinkingLevel, ToolDefinition},
};

#[test]
fn parses_and_normalizes_model_list_response() {
    assert_eq!(
        super::parse_models_response(
            br#"{"data":[{"id":"gpt-z"},{"id":" gpt-a "},{"id":"gpt-z"},{"id":""}]}"#,
        )
        .unwrap(),
        vec!["gpt-a", "gpt-z"]
    );
}

#[test]
fn rejects_invalid_model_list_response() {
    let error = super::parse_models_response(br#"{"data":[{"name":"missing-id"}]}"#).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("Provider model discovery returned invalid JSON")
    );
}

use super::{
    ChatProviderState, ResponsesRequest, ToolCallAccumulator, consume_sse_line, finish_message,
    parse_response_body, response_usage,
};
use crate::agent::CompletionUsage;

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
fn extracts_usage_from_chat_and_responses_payloads() {
    assert_eq!(
        response_usage(
            br#"{"usage":{"prompt_tokens":1200,"completion_tokens":37}}"#,
            "application/json",
            OpenAiApi::ChatCompletions,
        ),
        Some(CompletionUsage {
            input_tokens: Some(1200),
            output_tokens: Some(37),
        })
    );
    assert_eq!(
            response_usage(
                b"data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":5300,\"output_tokens\":91}}}\n\ndata: [DONE]\n\n",
                "text/event-stream",
                OpenAiApi::Responses,
            ),
            Some(CompletionUsage {
                input_tokens: Some(5300),
                output_tokens: Some(91),
            })
        );
    assert_eq!(
        response_usage(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":42}}\n\ndata: [DONE]\n\n",
            "text/event-stream",
            OpenAiApi::ChatCompletions,
        ),
        Some(CompletionUsage {
            input_tokens: Some(900),
            output_tokens: Some(42),
        })
    );
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

    let thinking = crate::provider::NormalizedThinking {
        requested: ThinkingLevel::Off,
        clamped: ThinkingLevel::Off,
        effective: ThinkingLevel::Off,
        provider_value: None,
    };
    let request = serde_json::to_value(super::ChatRequest::new(
        "deepseek-reasoner",
        &thinking,
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

    let thinking = crate::provider::NormalizedThinking {
        requested: ThinkingLevel::Off,
        clamped: ThinkingLevel::Off,
        effective: ThinkingLevel::Off,
        provider_value: None,
    };
    let request = serde_json::to_value(ResponsesRequest::new(
        "test-model",
        &thinking,
        &messages,
        &tools,
    ))
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
    let high = crate::provider::NormalizedThinking {
        requested: ThinkingLevel::Max,
        clamped: ThinkingLevel::High,
        effective: ThinkingLevel::High,
        provider_value: Some("high".to_owned()),
    };
    let responses =
        serde_json::to_value(ResponsesRequest::new("gpt-5", &high, &messages, &[])).unwrap();
    assert_eq!(responses["reasoning"]["effort"], "high");

    let low = crate::provider::NormalizedThinking {
        requested: ThinkingLevel::Low,
        clamped: ThinkingLevel::Low,
        effective: ThinkingLevel::Low,
        provider_value: Some("low".to_owned()),
    };
    let chat =
        serde_json::to_value(super::ChatRequest::new("gpt-5", &low, &messages, &[])).unwrap();
    assert_eq!(chat["reasoning_effort"], "low");
    assert_eq!(chat["stream_options"]["include_usage"], true);
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
fn parses_direct_thinking_before_think_tag_fallback() {
    let (events, mut receiver) = mpsc::unbounded_channel();
    let message = parse_response_body(
        br#"{
                "choices": [{
                    "message": {
                        "content": "<think>fallback</think>Final",
                        "thinking": "Explicit",
                        "tool_calls": []
                    }
                }]
            }"#,
        "application/json",
        OpenAiApi::ChatCompletions,
        &events,
    )
    .unwrap();

    assert_eq!(message.thinking.as_deref(), Some("Explicit"));
    assert_eq!(message.content, "Final");
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::ThinkingDelta {
            delta: "Explicit".to_owned(),
        }
    );
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "Final".to_owned(),
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

fn parse_one_shot(
    api: OpenAiApi,
    body: &[u8],
) -> (crate::agent::AssistantMessage, Vec<AgentEvent>) {
    let (events, mut receiver) = mpsc::unbounded_channel();
    let message = parse_response_body(body, "text/event-stream", api, &events).unwrap();
    let events = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
    (message, events)
}

/// Feed the body through the incremental stream machinery in fixed-size
/// chunks, mirroring how `read_response_body` drives it.
fn parse_chunked(
    api: OpenAiApi,
    body: &[u8],
    chunk_size: usize,
) -> (crate::agent::AssistantMessage, Vec<AgentEvent>) {
    let (events, mut receiver) = mpsc::unbounded_channel();
    let mut parser = super::SseParser::new(api);
    let mut carry = Vec::new();
    for chunk in body.chunks(chunk_size.max(1)) {
        carry.extend_from_slice(chunk);
        super::feed_complete_lines(&mut carry, &mut parser, &events).unwrap();
    }
    if !carry.is_empty() {
        parser.feed_line(&carry, &events).unwrap();
    }
    let message = parser.finish(&events).unwrap();
    let events = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
    (message, events)
}

fn assert_same_message(
    one_shot: &(crate::agent::AssistantMessage, Vec<AgentEvent>),
    chunked: &(crate::agent::AssistantMessage, Vec<AgentEvent>),
) {
    assert_eq!(one_shot.0.content, chunked.0.content);
    assert_eq!(one_shot.0.thinking, chunked.0.thinking);
    assert_eq!(one_shot.0.tool_calls, chunked.0.tool_calls);
    assert_eq!(one_shot.0.provider_state, chunked.0.provider_state);
    assert_eq!(one_shot.1, chunked.1);
}

#[test]
fn chunked_chat_stream_matches_one_shot_at_every_split_point() {
    let body: &[u8] = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"思考一下\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"pa\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\":\\\"x\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"你好，世界\"}}]}\n\n",
        "data: [DONE]\n\n",
    )
    .as_bytes();
    let expected = parse_one_shot(OpenAiApi::ChatCompletions, body);
    // Every two-chunk split, including splits inside multibyte UTF-8.
    for split in 0..=body.len() {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let mut parser = super::SseParser::new(OpenAiApi::ChatCompletions);
        let mut carry = body[..split].to_vec();
        super::feed_complete_lines(&mut carry, &mut parser, &events).unwrap();
        carry.extend_from_slice(&body[split..]);
        super::feed_complete_lines(&mut carry, &mut parser, &events).unwrap();
        if !carry.is_empty() {
            parser.feed_line(&carry, &events).unwrap();
        }
        let message = parser.finish(&events).unwrap();
        let collected: Vec<AgentEvent> = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
        assert_same_message(&expected, &(message, collected));
    }
    // Many small chunks exercise the carry buffer across partial lines.
    assert_same_message(
        &expected,
        &parse_chunked(OpenAiApi::ChatCompletions, body, 5),
    );
}

#[test]
fn chunked_responses_stream_matches_one_shot() {
    let body: &[u8] = concat!(
        "event: response.reasoning_summary_text.delta\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"想\"}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"你\"}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"好\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"想\"}]},{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"你好\"}]}],\"usage\":{\"output_tokens\":9}}}\n\n",
    )
    .as_bytes();
    let expected = parse_one_shot(OpenAiApi::Responses, body);
    assert_same_message(&expected, &parse_chunked(OpenAiApi::Responses, body, 1));
    assert_same_message(&expected, &parse_chunked(OpenAiApi::Responses, body, 7));
    assert_same_message(
        &expected,
        &parse_chunked(OpenAiApi::Responses, body, body.len()),
    );
}

#[test]
fn incremental_parsers_emit_deltas_before_finish() {
    let (events, mut receiver) = mpsc::unbounded_channel();

    let mut responses = super::ResponsesStreamParser::default();
    responses
        .feed_line(
            r#"data: {"type":"response.output_text.delta","delta":"你"}"#.as_bytes(),
            &events,
        )
        .unwrap();
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "你".to_owned(),
        }
    );

    let mut chat = super::ChatStreamParser::new();
    chat.feed_line(
        r#"data: {"choices":[{"delta":{"reasoning_content":"想"}}]}"#.as_bytes(),
        &events,
    )
    .unwrap();
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::ThinkingDelta {
            delta: "想".to_owned(),
        }
    );
    // Chat content stays buffered until finish so <think> splitting applies.
    chat.feed_line(
        r#"data: {"choices":[{"delta":{"content":"答"}}]}"#.as_bytes(),
        &events,
    )
    .unwrap();
    assert!(receiver.try_recv().is_err());
    let message = chat.finish(&events).unwrap();
    assert_eq!(message.content, "答");
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "答".to_owned(),
        }
    );
}
