//! Versioned contract between the Rust agent runtime, the inference backend and
//! the Kotlin UI.
//!
//! Everything that crosses a process or a persistence boundary is one of these
//! types, serialised as JSON. Two rules hold everywhere:
//!
//! * generation ending is not turn ending — a tool call keeps the turn alive;
//! * the backend that actually ran is always reported, never the one requested.

use super::json::Json;

pub const PROTOCOL_VERSION: u32 = 1;
pub const SESSION_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Pure Rust engine in this process.
    RustCpu,
    /// Pure Rust engine with the experimental OpenCL path.
    RustOpenCl,
    /// GenieX/llama.cpp reaching the Hexagon DSP through FastRPC.
    HexagonNpu,
    /// No backend is loaded.
    None,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::RustCpu => "rust-cpu",
            BackendKind::RustOpenCl => "rust-opencl",
            BackendKind::HexagonNpu => "hexagon-npu",
            BackendKind::None => "none",
        }
    }

    pub fn parse(text: &str) -> BackendKind {
        match text {
            "rust-cpu" => BackendKind::RustCpu,
            "rust-opencl" => BackendKind::RustOpenCl,
            "hexagon-npu" => BackendKind::HexagonNpu,
            _ => BackendKind::None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    TokenLimit,
    ContextFull,
    Cancelled,
    /// The backend returned or threw. `error` carries the classified detail.
    Error,
    StepLimit,
    TurnTimeout,
    ToolOutputLimit,
    RepeatedToolError,
    /// Nothing could be generated because no model is loaded.
    NoModel,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::Eos => "eos",
            StopReason::TokenLimit => "token_limit",
            StopReason::ContextFull => "context_full",
            StopReason::Cancelled => "cancelled",
            StopReason::Error => "error",
            StopReason::StepLimit => "step_limit",
            StopReason::TurnTimeout => "turn_timeout",
            StopReason::ToolOutputLimit => "tool_output_limit",
            StopReason::RepeatedToolError => "repeated_tool_error",
            StopReason::NoModel => "no_model",
        }
    }
}

/// Stable wire names. Renaming one is a protocol version bump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    UserMessageAccepted,
    TurnStarted,
    StepStarted,
    TurnEnded,
    AssistantText,
    AssistantReasoning,
    ToolCalled,
    ToolResult,
    Error,
    Cancelled,
    Usage,
    Checkpoint,
    ModelState,
    DeltaText,
    DeltaReasoning,
    ModelError,
}

impl EventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::UserMessageAccepted => "user_message.accepted",
            EventKind::TurnStarted => "turn.started",
            EventKind::StepStarted => "step.started",
            EventKind::TurnEnded => "turn.ended",
            EventKind::AssistantText => "assistant.text",
            EventKind::AssistantReasoning => "assistant.reasoning",
            EventKind::ToolCalled => "tool.called",
            EventKind::ToolResult => "tool.result",
            EventKind::Error => "error",
            EventKind::Cancelled => "cancelled",
            EventKind::Usage => "usage",
            EventKind::Checkpoint => "checkpoint",
            EventKind::ModelState => "model.state",
            EventKind::DeltaText => "delta.text",
            EventKind::DeltaReasoning => "delta.reasoning",
            EventKind::ModelError => "model.error",
        }
    }

    /// Token deltas are cheap to rebuild from the durable record, so they are
    /// never written to the session log.
    pub fn is_durable(self) -> bool {
        !matches!(self, EventKind::DeltaText | EventKind::DeltaReasoning)
    }
}

/// One record of the session log, and also the shape streamed to the UI.
/// `seq` is monotonic per session and assigned by the runtime, never by a caller.
#[derive(Clone, Debug)]
pub struct AgentEvent {
    pub kind: EventKind,
    pub session_id: String,
    pub turn_id: String,
    pub step: u32,
    pub seq: u64,
    pub at_ms: u64,
    pub data: Json,
}

impl AgentEvent {
    pub fn new(kind: EventKind, session_id: &str, turn_id: &str, data: Json) -> Self {
        Self {
            kind,
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            step: 0,
            seq: 0,
            at_ms: now_ms(),
            data,
        }
    }

    pub fn at_step(mut self, step: u32) -> Self {
        self.step = step;
        self
    }

    pub fn with_seq(mut self, seq: u64) -> Self {
        self.seq = seq;
        self
    }

    pub fn to_json(&self) -> Json {
        Json::obj()
            .with("v", Json::int(PROTOCOL_VERSION as i64))
            .with("seq", Json::int(self.seq as i64))
            .with("at_ms", Json::int(self.at_ms as i64))
            .with("kind", Json::str(self.kind.as_str()))
            .with("session_id", Json::str(&self.session_id))
            .with("turn_id", Json::str(&self.turn_id))
            .with("step", Json::int(self.step as i64))
            .with("data", self.data.clone())
    }

    pub fn from_json(value: &Json) -> Option<AgentEvent> {
        let kind = match value.get("kind")?.as_str()? {
            "user_message.accepted" => EventKind::UserMessageAccepted,
            "turn.started" => EventKind::TurnStarted,
            "step.started" => EventKind::StepStarted,
            "turn.ended" => EventKind::TurnEnded,
            "assistant.text" => EventKind::AssistantText,
            "assistant.reasoning" => EventKind::AssistantReasoning,
            "tool.called" => EventKind::ToolCalled,
            "tool.result" => EventKind::ToolResult,
            "error" => EventKind::Error,
            "cancelled" => EventKind::Cancelled,
            "usage" => EventKind::Usage,
            "checkpoint" => EventKind::Checkpoint,
            "model.state" => EventKind::ModelState,
            "delta.text" => EventKind::DeltaText,
            "delta.reasoning" => EventKind::DeltaReasoning,
            "model.error" => EventKind::ModelError,
            _ => return None,
        };
        Some(AgentEvent {
            kind,
            session_id: value.get("session_id")?.as_str()?.to_string(),
            turn_id: value.get("turn_id")?.as_str()?.to_string(),
            step: value.get("step").and_then(|v| v.as_i64()).unwrap_or(0) as u32,
            seq: value.get("seq").and_then(|v| v.as_i64()).unwrap_or(0) as u64,
            at_ms: value.get("at_ms").and_then(|v| v.as_i64()).unwrap_or(0) as u64,
            data: value.get("data").cloned().unwrap_or_else(Json::obj),
        })
    }
}

/// Everything a tool declares to the model. `parameters` is the JSON schema the
/// model is shown; the executor validates against it before touching the disk.
#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: &'static str,
    pub version: &'static str,
    pub description: &'static str,
    pub parameters: fn() -> Json,
    /// Capability the tool needs. Checked against the session's grants.
    pub capability: &'static str,
    pub max_output_bytes: usize,
    pub timeout_ms: u64,
}

impl ToolSpec {
    pub fn schema_json(&self) -> Json {
        Json::obj()
            .with("name", Json::str(self.name))
            .with("version", Json::str(self.version))
            .with("description", Json::str(self.description))
            .with("capability", Json::str(self.capability))
            .with("parameters", (self.parameters)())
    }
}

/// Hard bounds of one turn. Every one of these is surfaced as a stop reason
/// rather than turned into an invented final answer.
#[derive(Clone, Copy, Debug)]
pub struct TurnLimits {
    pub max_steps: u32,
    pub turn_timeout_ms: u64,
    pub max_tool_output_bytes: usize,
    pub max_context_tokens: usize,
    pub max_repeated_tool_errors: u32,
    pub max_queued_messages: usize,
}

impl Default for TurnLimits {
    fn default() -> Self {
        Self {
            max_steps: 12,
            turn_timeout_ms: 600_000,
            max_tool_output_bytes: 16 * 1024,
            max_context_tokens: 8192,
            max_repeated_tool_errors: 3,
            max_queued_messages: 8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub threads: usize,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.95,
            seed: 0xC0FFEE,
            threads: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub load_ms: f64,
}

impl TokenUsage {
    pub fn to_json(self) -> Json {
        Json::obj()
            .with("prompt_tokens", Json::int(self.prompt_tokens as i64))
            .with(
                "cached_prompt_tokens",
                Json::int(self.cached_prompt_tokens as i64),
            )
            .with("completion_tokens", Json::int(self.completion_tokens as i64))
            .with("prefill_ms", Json::Num(round2(self.prefill_ms)))
            .with("decode_ms", Json::Num(round2(self.decode_ms)))
            .with("load_ms", Json::Num(round2(self.load_ms)))
    }
}

fn round2(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    (value * 100.0).round() / 100.0
}

/// A tool call the model produced. `id` is minted by the runtime so a result can
/// always be correlated with the call that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Json,
}

impl ToolCall {
    pub fn to_json(&self) -> Json {
        Json::obj()
            .with("call_id", Json::str(&self.id))
            .with("name", Json::str(&self.name))
            .with("arguments", self.arguments.clone())
    }
}

/// The runtime's own verdict on a tool execution. The model never supplies this.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutcome {
    pub call_id: String,
    pub name: String,
    pub ok: bool,
    pub result: Json,
    pub error_class: Option<String>,
    pub error: Option<String>,
    pub duration_ms: u64,
    pub truncated: bool,
}

impl ToolOutcome {
    pub fn ok(call_id: &str, name: &str, result: Json, duration_ms: u64) -> Self {
        Self {
            call_id: call_id.to_string(),
            name: name.to_string(),
            ok: true,
            result,
            error_class: None,
            error: None,
            duration_ms,
            truncated: false,
        }
    }

    pub fn failed(call_id: &str, name: &str, class: &str, message: &str, duration_ms: u64) -> Self {
        Self {
            call_id: call_id.to_string(),
            name: name.to_string(),
            ok: false,
            result: Json::obj(),
            error_class: Some(class.to_string()),
            error: Some(message.to_string()),
            duration_ms,
            truncated: false,
        }
    }

    pub fn to_json(&self) -> Json {
        Json::obj()
            .with("call_id", Json::str(&self.call_id))
            .with("name", Json::str(&self.name))
            .with("ok", Json::Bool(self.ok))
            .with("result", self.result.clone())
            .with(
                "error_class",
                match &self.error_class {
                    Some(c) => Json::str(c),
                    None => Json::Null,
                },
            )
            .with(
                "error",
                match &self.error {
                    Some(e) => Json::str(e),
                    None => Json::Null,
                },
            )
            .with("duration_ms", Json::int(self.duration_ms as i64))
            .with("truncated", Json::Bool(self.truncated))
    }

    /// What the model sees as the `tool` message. Errors stay visible: hiding
    /// them teaches the model to retry blindly.
    pub fn to_model_message(&self) -> String {
        if self.ok {
            let mut text = self.result.to_string();
            if text.len() > MAX_MODEL_RESULT {
                let mut cut = MAX_MODEL_RESULT;
                while cut > 0 && !text.is_char_boundary(cut) {
                    cut -= 1;
                }
                text.truncate(cut);
                text.push_str("\n[output truncated]");
            }
            text
        } else {
            format!(
                "{{\"error\":\"{}\",\"class\":\"{}\"}}",
                self.error.as_deref().unwrap_or("tool failed"),
                self.error_class.as_deref().unwrap_or("tool_error")
            )
        }
    }
}

pub const MAX_MODEL_RESULT: usize = 12_000;

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_round_trip() {
        let event = AgentEvent::new(
            EventKind::ToolResult,
            "s1",
            "t1",
            Json::obj().with("ok", Json::Bool(true)),
        )
        .at_step(3);
        let restored = AgentEvent::from_json(&event.to_json()).unwrap();
        assert_eq!(restored.kind, event.kind);
        assert_eq!(restored.session_id, "s1");
        assert_eq!(restored.turn_id, "t1");
        assert_eq!(restored.step, 3);
        assert_eq!(restored.data.get("ok"), Some(&Json::Bool(true)));
    }

    #[test]
    fn deltas_are_not_durable() {
        assert!(!EventKind::DeltaText.is_durable());
        assert!(!EventKind::DeltaReasoning.is_durable());
        assert!(EventKind::AssistantText.is_durable());
        assert!(EventKind::ToolResult.is_durable());
    }

    #[test]
    fn tool_outcome_shows_errors_to_the_model() {
        let bad = ToolOutcome::failed("c1", "fs_read", "invalid_argument", "path is outside the workspace", 3);
        let shown = bad.to_model_message();
        assert!(shown.contains("outside the workspace"));
        let good = ToolOutcome::ok("c2", "fs_read", Json::obj().with("bytes", Json::int(3)), 1);
        assert_eq!(good.to_model_message(), "{\"bytes\":3}");
    }

    #[test]
    fn model_result_truncation_keeps_valid_utf8() {
        let outcome = ToolOutcome::ok("c", "t", Json::str("à".repeat(MAX_MODEL_RESULT)), 0);
        let shown = outcome.to_model_message();
        assert!(shown.ends_with("[output truncated]"));
        assert!(shown.is_char_boundary(shown.len()));
    }
}
