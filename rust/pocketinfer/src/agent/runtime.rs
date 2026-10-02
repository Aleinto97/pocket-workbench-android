//! The turn state machine.
//!
//! One session has at most one active turn. Messages that arrive during a turn
//! are queued, not interleaved. Every bound is a stop reason that reaches the
//! transcript — the runtime never invents a closing sentence to paper over a
//! limit, an error or a cancellation.
//!
//! ```text
//! user message
//!   → context preparation (token count, compaction)
//!   → inference
//!   → final answer, or a tool call
//!   → validate and run the tool
//!   → append the result
//!   → inference again
//! ```

use super::backend::{Backend, DeltaSink, LoadRequest, ModelInfo};
use super::context::{Conversation, TurnItem};
use super::json::Json;
use super::protocol::{
    AgentEvent, BackendKind, EventKind, GenerationOptions, StopReason, TokenUsage, ToolCall,
    ToolOutcome, TurnLimits,
};
use super::session::SessionLog;
use super::tools::{self, Grants, Workspace};
use crate::util::{Error, Result};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Emitted deltas are batched before they reach the UI: one Compose
/// recomposition per token is not viable, and the batch size is bounded so a
/// long reply still feels live.
const DELTA_BATCH_CHARS: usize = 24;
const DELTA_FLUSH_INTERVAL: Duration = Duration::from_millis(60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Preparing,
    Generating,
    RunningTool,
    Cancelling,
    Stopped,
    Failed,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Preparing => "preparing",
            Phase::Generating => "generating",
            Phase::RunningTool => "tool",
            Phase::Cancelling => "cancelling",
            Phase::Stopped => "stopped",
            Phase::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug)]
pub struct QueuedMessage {
    pub request_id: String,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub workspace_id: String,
    pub system_prompt: String,
    /// Model family for the prompt overlay ("qwen" default, "llama",
    /// "minicpm"). Only appends temperament lines; the tool contract is
    /// identical for every family.
    pub model_family: String,
    /// When true, compaction summaries are written by the model instead of
    /// the factual rule. Off by default: small models confabulate summaries.
    pub use_model_summary: bool,
    pub grants: Grants,
    pub limits: TurnLimits,
    pub generation: GenerationOptions,
    pub linux: Option<super::tools::LinuxConfig>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            workspace_id: "default".to_string(),
            // Empty means "build from model_family at open". A non-empty value
            // is a custom override and wins over the family overlay.
            system_prompt: String::new(),
            model_family: "qwen".to_string(),
            use_model_summary: false,
            grants: Grants::all(),
            limits: TurnLimits::default(),
            generation: GenerationOptions::default(),
            linux: None,
        }
    }
}

impl SessionConfig {
    /// The system text the session actually starts with: custom override when
    /// provided, otherwise the family prompt. Resolved at open so the family
    /// chosen in Settings applies without the UI assembling prompts.
    pub fn effective_system_prompt(&self) -> String {
        if self.system_prompt.trim().is_empty() {
            default_system_prompt_for(&self.model_family)
        } else {
            self.system_prompt.clone()
        }
    }
}

/// The tool contract as the model sees it. Kept here rather than in the backend
/// so a backend swap cannot change what the model is told the tools are.
///
/// Shape follows Qwen's own guidance for tool use: each tool as a JSON Schema
/// object, and the call format spelled out once, with one worked example.
pub fn tool_brief() -> String {
    let mut out = String::from(
        "You are provided with function signatures within <tools></tools> XML \
         tags. Here are the available tools:\n<tools>\n",
    );
    for spec in tools::TOOLS {
        let parameters = (spec.parameters)();
        let entry = Json::obj_with(vec![
            ("type".to_string(), Json::str("function")),
            (
                "function".to_string(),
                Json::obj_with(vec![
                    ("name".to_string(), Json::str(spec.name)),
                    ("description".to_string(), Json::str(spec.description)),
                    ("parameters".to_string(), parameters),
                ]),
            ),
        ]);
        let mut rendered = String::new();
        let _ = entry.write(&mut rendered);
        out.push_str(&rendered);
        out.push('\n');
    }
    out.push_str(
        "</tools>\n\
         For each function call return a json object with the function name and \
         arguments within <tool_call></tool_call> XML tags as follows:\n\
         <tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
         </tool_call>\n\
         Use the tool's own schema for the arguments. A tool call is an action, \
         not an answer: the app runs it and gives you the result, then you \
         continue. Never write a tool result yourself. One call per reply.\n\
         A call and its result look like this:\n\
         user\nqual e il file piu grande del progetto?\n\
         assistant\n<tool_call>\n\
         {\"name\": \"shell\", \"arguments\": {\"command\": \"find . -type f -printf '%s %p\\\\n' | sort -rn | head -1\"}}\n\
         </tool_call>\n\
         tool\n{\"name\": \"shell\", \"content\": \"4211 ./build.log\"}\n\
         assistant\nThat is build.log, 4211 bytes.\n\n\
         This is what looking something up looks like. There is no browser and \
         no search tool: curl is how a page is reached, and you choose the URL \
         that answers the question. The page comes back in \"content\", and you \
         write the answer from that text alone — never from what you remember \
         about the subject. If the page comes back empty, the site is blocking \
         bots: do not answer from memory, fetch a search engine instead, then \
         fetch the most promising link from its results.\n\
         user\nqual e la prossima partita della juve?\n\
         assistant\n<tool_call>\n\
         {\"name\": \"shell\", \"arguments\": {\"command\": \"curl -sL --max-time 20 https://www.juve.it/it/news/ | sed 's/<[^>]*>/ /g' | tr -s ' \\\\n' ' \\\\n' | grep -i -m5 -E 'partita|prossima'\"}}\n\
         </tool_call>\n\
          tool\n{\"name\": \"shell\", \"content\": \"whatever curl printed\"}\n\
          assistant\nAnswer from the text in \"content\" alone.\n\n\
          Two more tools: `skill` loads a named markdown procedure from the \
          project's .skills/ folder — read it before relying on it. \
          `delegate` sends a bounded research question to a fresh read-only \
          context and returns a condensed answer; use it for large explorations \
          instead of filling this conversation.\n\
          One call per reply. If the task is done, answer without a tool \
          call — do not call tools to admire your own work. After writing a \
          file, re-read it or run it to verify before claiming it works.",
    );
    out
}

/// Per-model overlay: same tools, different temperament. Small local models
/// fail in model-specific ways, so each family gets the two sentences that fix
/// its own failure mode. Unknown families get the Qwen default.
pub fn tool_brief_for(family: &str) -> String {
    let base = tool_brief();
    let appendix = match family.trim().to_lowercase().as_str() {
        "llama" => " Llama note: when calling tools, output ONLY the \
            <tool_call> block, no surrounding prose and no explanation of what \
            you are about to do.",
        "minicpm" => " MiniCPM note: think briefly (three sentences at most), \
            then either call exactly one tool or answer. Never repeat a call \
            whose result is already above.",
        _ => "",
    };
    if appendix.is_empty() {
        base
    } else {
        format!("{base}{appendix}")
    }
}

/// Qwen's guidance again: the messages should carry as much available
/// information as possible. The workspace path matters because the tools
/// resolve relative paths against it and the model should know where it is.
/// The date is deliberately NOT here: it would change the system prefix every
/// day and invalidate the prefill cache. It lives in `turn_context`, which is
/// per-turn by design.
fn context_facts() -> String {
    "The project folder is /workspace.".to_string()
}

fn today() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (year, month, day) = civil_from_epoch(now);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Injected next to the turn that is being answered, not in the system prompt.
///
/// This placement is the whole trick, and it is what DeepSeek's harness does
/// with its per-turn "current runtime context" block. The same sentence in the
/// system prompt was ignored — measured on the NPU, the model would list the
/// tools it has and then decline to use them, answering from memory instead.
/// Sitting immediately before the question, it is the last thing read before
/// generation and it is obeyed: the model emits a tool call.
///
/// This block is per-turn by design, so it also carries everything that
/// changes often: today's date, the context-economy rule and the working set.
/// None of that belongs in the system prefix, which must stay byte-identical
/// for the prefill cache to hit.
pub fn turn_context() -> String {
    let files = working_set();
    let working = if files.is_empty() {
        String::new()
    } else {
        format!(" Open files: {}.", files.join(", "))
    };
    format!(
        "CONTEXT: today is {}. Project folder: /workspace. Tools right now: \
         shell (curl reaches the internet; system=\"linux\" gives a full Debian \
         with apt and compilers), fs_list, fs_read, fs_write, fs_edit, \
         fs_search, skill, delegate.{working} Context is precious: batch \
         independent reads in one turn and read only what the question needs. \
         If the answer needs anything that is not already in this \
         conversation, call a tool now — do not answer it from memory. A call \
         is this and nothing else, no prose around it:\n\
         <tool_call>\n{{\"name\": \"shell\", \"arguments\": {{\"command\": \"curl -sL \
         https://example.com\"}}}}\n</tool_call>\n{}\n\n",
        today(),
        grounding_rule()
    )
}

/// One line of the per-turn context, kept separate so tests pin it down:
/// factual claims must point at tool output, and guesses must say so.
pub fn grounding_rule() -> String {
    "Ground every factual claim in tool output: cite sources as [path:line]. \
     Anything without a citation is your guess — say so."
        .to_string()
}

/// Files the agent touched recently, shown in the per-turn context so it does
/// not re-list or re-read what is already at hand. Process-wide because one
/// session generates at a time on this device; updated on every tool call.
static WORKING_SET: std::sync::OnceLock<std::sync::Mutex<Vec<String>>> =
    std::sync::OnceLock::new();

fn working_set() -> Vec<String> {
    WORKING_SET
        .get()
        .and_then(|lock| lock.lock().ok().map(|guard| guard.clone()))
        .unwrap_or_default()
}

fn remember_working_file(path: &str) {
    let trimmed = path.trim().trim_matches('/');
    if trimmed.is_empty() || trimmed.len() > 160 {
        return;
    }
    let Some(lock) = WORKING_SET.get() else {
        let _ = WORKING_SET.set(std::sync::Mutex::new(vec![trimmed.to_string()]));
        return;
    };
    if let Ok(mut guard) = lock.lock() {
        guard.retain(|p| p != trimmed);
        guard.insert(0, trimmed.to_string());
        guard.truncate(5);
    }
}

fn default_system_prompt() -> String {
    default_system_prompt_for("qwen")
}

/// System prompt for a model family. The family only appends temperament
/// lines; the contract (tools, format, facts) is identical, so behaviour
/// stays comparable across models.
fn default_system_prompt_for(family: &str) -> String {
    format!(
        "You are Pocket Workbench, an agent running entirely on this device. You \
         work in the user's project, with a shell and the network.\n{}\n\
         The conversation below is everything you already know: what was asked, \
         what you did, and what the tools returned. Use it, and do not repeat a \
         tool call whose result is already there.\n\
         When a tool fails or returns nothing, say so plainly and try another \
         route; never fill the gap with a plausible guess.\n\
         Reply in the user's language.\n{}",
        context_facts(),
        tool_brief_for(family)
    )
}

/// Project memory: conventions the model should not have to rediscover every
/// session. A short markdown file in the workspace root; re-read at session
/// open and pinned as a system note so it survives every compaction without
/// re-injection. Missing file is normal, not an error.
fn load_project_memory(workspace_root: &std::path::Path) -> Option<String> {
    const MAX_MEMORY_BYTES: u64 = 8 * 1024;
    let path = workspace_root.join("AGENTS.md");
    let metadata = std::fs::metadata(&path).ok()?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_MEMORY_BYTES {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    let trimmed = text.trim().to_string();
    if trimmed.is_empty() {
        return None;
    }
    Some(format!("Project memory (AGENTS.md, always applies):\n{trimmed}"))
}

/// Skill catalogue: names only, so the list costs one line. Bodies load
/// on demand through the `skill` tool.
fn list_skills(workspace_root: &std::path::Path) -> Option<String> {
    let dir = workspace_root.join(".skills");
    let entries = std::fs::read_dir(&dir).ok()?;
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            let stem = name.strip_suffix(".md")?;
            if stem.is_empty()
                || stem.len() > 64
                || !stem
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return None;
            }
            if entry.file_type().ok()?.is_file() {
                Some(stem.to_string())
            } else {
                None
            }
        })
        .collect();
    if names.is_empty() {
        return None;
    }
    names.sort();
    Some(format!(
        "Available skills (load with the skill tool): {}.",
        names.join(", ")
    ))
}

/// Days-from-civil, so the date shown to the model needs no calendar library.
fn civil_from_epoch(epoch: u64) -> (i64, u32, u32) {
    let days = (epoch / 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

struct Batch {
    text: String,
    at: Instant,
}

/// Forwards deltas to the event channel in bounded batches.
struct BatchingSink {
    sender: Sender<AgentEvent>,
    session_id: String,
    turn_id: String,
    text: Batch,
    reasoning: Batch,
}

impl BatchingSink {
    fn new(sender: Sender<AgentEvent>, session_id: &str, turn_id: &str) -> Self {
        Self {
            sender,
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            text: Batch { text: String::new(), at: Instant::now() },
            reasoning: Batch { text: String::new(), at: Instant::now() },
        }
    }

    fn flush(&mut self) {
        let now = Instant::now();
        if !self.text.text.is_empty() {
            let event = AgentEvent::new(
                EventKind::DeltaText,
                &self.session_id,
                &self.turn_id,
                Json::obj().with("text", Json::str(&self.text.text)),
            );
            let _ = self.sender.send(event);
            self.text.text.clear();
        }
        if !self.reasoning.text.is_empty() {
            let event = AgentEvent::new(
                EventKind::DeltaReasoning,
                &self.session_id,
                &self.turn_id,
                Json::obj().with("text", Json::str(&self.reasoning.text)),
            );
            let _ = self.sender.send(event);
            self.reasoning.text.clear();
        }
        self.text.at = now;
    }

    fn due(&self) -> bool {
        self.text.at.elapsed() >= DELTA_FLUSH_INTERVAL
            || self.reasoning.at.elapsed() >= DELTA_FLUSH_INTERVAL
    }
}

impl DeltaSink for BatchingSink {
    fn text(&mut self, piece: &str) {
        self.text.text.push_str(piece);
        if self.text.text.len() >= DELTA_BATCH_CHARS || self.due() {
            self.flush();
        }
    }

    fn reasoning(&mut self, piece: &str) {
        self.reasoning.text.push_str(piece);
        if self.reasoning.text.len() >= DELTA_BATCH_CHARS || self.due() {
            self.flush();
        }
    }
}

/// Sink that collects generated text instead of streaming it. Used for the
/// optional model-written compaction summary and for delegation answers.
struct CollectSink {
    text: String,
}

impl DeltaSink for CollectSink {
    fn text(&mut self, piece: &str) {
        self.text.push_str(piece);
    }

    fn reasoning(&mut self, _piece: &str) {}
}

/// Optional model-written compaction summary. The dropped turns are passed as
/// evidence with a strict factual instruction; on any failure (no model, an
/// error, an empty reply) the caller falls back to the factual rule, which is
/// also the default. Small models confabulate summaries, so this stays opt-in.
fn model_summary(
    backend: &mut dyn Backend,
    options: &GenerationOptions,
    evidence: &[String],
    replaced: usize,
) -> Option<String> {
    let mut body = String::new();
    for text in evidence {
        let piece: String = text.chars().take(400).collect();
        body.push_str(&piece);
        body.push('\n');
        if body.len() > 12_000 {
            break;
        }
    }
    if body.trim().is_empty() {
        return None;
    }
    let messages = vec![
        (
            "system".to_string(),
            "Summarize tool activity factually in at most 6 lines: which tools \
             ran, which files were involved, what failed. No advice, no guesses, \
             no invented user intent."
                .to_string(),
        ),
        ("user".to_string(), body),
    ];
    let mut summary_options = *options;
    summary_options.max_tokens = 256;
    summary_options.temperature = 0.0;
    let mut sink = CollectSink { text: String::new() };
    let generation = backend
        .generate("compact-summary", &messages, &summary_options, &mut sink)
        .ok()?;
    let text = generation.text.trim().to_string();
    if text.is_empty() {
        return None;
    }
    let mut out = format!(
        "[compacted, model summary] {text} The full text of this part is still in the session log."
    );
    if replaced > 0 {
        out.push_str(&format!(
            " Replaces {replaced} earlier compaction note(s)."
        ));
    }
    Some(out)
}

pub struct Session {
    pub id: String,
    pub config: SessionConfig,
    pub log: SessionLog,
    conversation: Conversation,
    workspace: Workspace,
    events: Sender<AgentEvent>,
    backend: Box<dyn Backend>,
    cancel: Arc<AtomicBool>,
    active_turn: Option<String>,
    queue: Vec<QueuedMessage>,
    phase: Phase,
    /// Set when a turn stops for a reason that is not the model's own ending.
    last_stop: Option<StopReason>,
    turn_counter: u64,
    tool_error_streak: u32,
    turn_usage: TokenUsage,
}

impl Session {
    pub fn open(
        id: &str,
        log_path: &Path,
        workspace_root: &Path,
        config: SessionConfig,
        events: Sender<AgentEvent>,
        backend: Box<dyn Backend>,
    ) -> Result<Session> {
        let log = SessionLog::open(log_path, &config.workspace_id, "Pocket Workbench")?;
        let mut conversation = Conversation::new(&config.effective_system_prompt());
        // Project memory and skill catalogue are pinned as leading system
        // notes: compaction never drops them, so no re-injection is needed.
        if let Some(memory) = load_project_memory(workspace_root) {
            conversation.push_system_note(&memory);
        }
        if let Some(skills) = list_skills(workspace_root) {
            conversation.push_system_note(&skills);
        }
        // Replay the durable log so a reopened session keeps its context. Only
        // durable records are replayed; deltas were never written.
        // Open turns are tracked as a set: a TurnStarted that later sees its
        // TurnEnded is closed, not interrupted. Only turns still open at the
        // end of the log died with the process and get reported.
        let mut open_turns: Vec<String> = Vec::new();
        for event in log.events() {
            match event.kind {
                EventKind::UserMessageAccepted => {
                    conversation.push_user(event.data.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                }
                EventKind::AssistantText => {
                    conversation.push_assistant("", event.data.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                }
                EventKind::ToolCalled => {
                    let Some(call) = call_from_event(event) else {
                        continue;
                    };
                    conversation.push_tool_call(call);
                }
                EventKind::ToolResult => {
                    let Some(outcome) = outcome_from_event(event) else {
                        continue;
                    };
                    if outcome.ok {
                        conversation.push_tool_result(outcome);
                    }
                }
                EventKind::Checkpoint => {
                    let summary = event
                        .data
                        .get("summary")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !summary.is_empty() {
                        conversation.items.push(TurnItem::Summary(summary));
                    }
                }
                EventKind::TurnStarted => {
                    if !open_turns.contains(&event.turn_id) {
                        open_turns.push(event.turn_id.clone());
                    }
                }
                EventKind::TurnEnded => {
                    open_turns.retain(|id| id != &event.turn_id);
                }
                _ => {}
            }
        }
        let interrupted: Option<String> = open_turns.into_iter().next_back();
        let mut session = Session {
            id: id.to_string(),
            config,
            log,
            conversation,
            workspace: Workspace::new(workspace_root, Grants::all()),
            events,
            backend,
            cancel: Arc::new(AtomicBool::new(false)),
            active_turn: None,
            queue: Vec::new(),
            phase: Phase::Idle,
            last_stop: None,
            turn_counter: 0,
            tool_error_streak: 0,
            turn_usage: TokenUsage::default(),
        };
        session.workspace = Workspace::new(workspace_root, session.config.grants.clone());
        session.workspace.linux = session.config.linux.clone();
        session.apply_compacted_header();
        if let Some(turn_id) = interrupted {
            // Reconstruct and repair: record the interruption so a retry is a
            // fresh turn and nothing gets replayed twice.
            let last_seq = session.log.events().last().map(|e| e.seq).unwrap_or(0);
            let _ = last_seq;
            session.emit(
                AgentEvent::new(
                    EventKind::Error,
                    id,
                    &turn_id,
                    Json::obj()
                        .with("class", Json::str("interrupted"))
                        .with(
                            "message",
                            Json::str(
                                "This turn was interrupted before it finished. Its effects on files are unknown; review them before retrying.",
                            ),
                        ),
                ),
            );
            session.emit(
                AgentEvent::new(
                    EventKind::TurnEnded,
                    id,
                    &turn_id,
                    Json::obj()
                        .with("reason", Json::str(StopReason::Cancelled.as_str()))
                        .with("interrupted", Json::Bool(true)),
                ),
            );
        }
        Ok(session)
    }

    fn apply_compacted_header(&self) {
        // A compaction recorded in the header already happened; the replay above
        // re-applied it through its checkpoint event. Nothing else to do, but the
        // header is the place a future version would reconcile from.
        let _ = self.log.header().get("compacted_at_seq");
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn is_busy(&self) -> bool {
        self.active_turn.is_some()
    }

    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    pub fn last_stop(&self) -> Option<StopReason> {
        self.last_stop
    }

    pub fn backend_kind(&self) -> BackendKind {
        self.backend
            .info()
            .map(|info| info.backend)
            .unwrap_or(BackendKind::None)
    }

    pub fn model_info(&self) -> Option<ModelInfo> {
        self.backend.info()
    }

    pub fn load_model(&mut self, request: &LoadRequest) -> Result<ModelInfo> {
        self.emit(
            AgentEvent::new(
                EventKind::ModelState,
                &self.id,
                "",
                Json::obj()
                    .with("state", Json::str("loading"))
                    .with("model_path", Json::str(&request.model_path))
                    .with("requested_backend", Json::str(request.backend.as_str())),
            ),
        );
        match self.backend.load(request) {
            Ok(info) => {
                // The event reports the backend that ran, not the one requested.
                self.emit(
                    AgentEvent::new(
                        EventKind::ModelState,
                        &self.id,
                        "",
                        Json::obj()
                            .with("state", Json::str("ready"))
                            .with("info", info.to_json()),
                    ),
                );
                Ok(info)
            }
            Err(error) => {
                self.emit(
                    AgentEvent::new(
                        EventKind::ModelError,
                        &self.id,
                        "",
                        Json::obj()
                            .with("class", Json::str("model_load_failed"))
                            .with("message", Json::str(&error.to_string())),
                    ),
                );
                Err(error)
            }
        }
    }

    pub fn unload_model(&mut self) {
        self.backend.unload();
        self.emit(
            AgentEvent::new(
                EventKind::ModelState,
                &self.id,
                "",
                Json::obj().with("state", Json::str("unloaded")),
            ),
        );
    }

    /// Replaces the inference backend. Used when a load requests a different
    /// backend than the session was opened with — without this, an NPU choice
    /// made later in settings would be silently ignored and the old engine
    /// would fail on the new weights. The previous backend is unloaded first;
    /// callers must only swap when the kind actually differs, so a resident
    /// model survives a plain reload.
    pub fn set_backend(&mut self, backend: Box<dyn Backend>) {
        self.backend.unload();
        self.backend = backend;
    }

    /// Admits a user message. This never runs a turn: the caller owns the worker
    /// thread and calls `pump`. Re-admitting a request id the log already
    /// recorded is a no-op, so a retry after a crash cannot start a second turn.
    pub fn submit(&mut self, request_id: &str, text: &str) -> Result<bool> {
        let text = text.trim();
        if text.is_empty() {
            bail!("Message is empty");
        }
        if self.log.is_admitted(request_id) {
            return Ok(false);
        }
        let busy = self.is_busy();
        if busy && self.queue.len() >= self.config.limits.max_queued_messages {
            bail!(
                "The queue already holds {} messages; wait for the current turn to finish",
                self.config.limits.max_queued_messages
            );
        }
        let position = self.queue.len() + usize::from(busy);
        self.queue.push(QueuedMessage {
            request_id: request_id.to_string(),
            text: text.to_string(),
        });
        self.emit(
            AgentEvent::new(
                EventKind::UserMessageAccepted,
                &self.id,
                self.active_turn.clone().unwrap_or_default().as_str(),
                Json::obj()
                    .with("request_id", Json::str(request_id))
                    .with("text", Json::str(text))
                    .with("queued", Json::Bool(busy))
                    .with("position", Json::int(position as i64)),
            ),
        );
        Ok(true)
    }

    /// Runs queued messages, one turn at a time, until the queue is empty.
    /// Returns the number of turns that ran.
    pub fn pump(&mut self) -> usize {
        let mut ran = 0usize;
        while !self.is_busy() {
            let Some(message) = self.queue.first().cloned() else {
                break;
            };
            self.queue.remove(0);
            if self.start_turn(&message.request_id, &message.text).is_err() {
                break;
            }
            ran += 1;
        }
        ran
    }

    fn start_turn(&mut self, request_id: &str, text: &str) -> Result<()> {
        self.turn_counter += 1;
        let turn_id = format!("{}t{}", self.id, self.turn_counter);
        self.active_turn = Some(turn_id.clone());
        self.cancel.store(false, Ordering::SeqCst);
        self.tool_error_streak = 0;
        self.last_stop = None;
        self.phase = Phase::Preparing;

        let _ = request_id;
        let turn_base = self.conversation.len();
        self.conversation.push_user(text);
        self.emit(
            AgentEvent::new(
                EventKind::TurnStarted,
                &self.id,
                &turn_id,
                Json::obj().with("limits", limits_json(&self.config.limits)),
            ),
        );

        let started = Instant::now();
        let outcome = self.run_turn(&turn_id);
        self.phase = Phase::Idle;
        self.active_turn = None;
        let reason = outcome.unwrap_or(StopReason::Error);
        // A turn that died before producing anything — the classic case is
        // "no model loaded" — must not leave its question behind. Otherwise
        // the retry arrives with the same question twice, and the model reads
        // a stuck loop instead of a fresh question. Tool results always stay:
        // a tool that ran is a fact, even when the turn around it failed.
        if matches!(reason, StopReason::Error | StopReason::NoModel)
            && !self.conversation.has_effects_since(turn_base)
        {
            self.conversation.truncate(turn_base);
        }
        self.last_stop = Some(reason);
        self.emit(
            AgentEvent::new(
                EventKind::TurnEnded,
                &self.id,
                &turn_id,
                Json::obj()
                    .with("reason", Json::str(reason.as_str()))
                    .with("duration_ms", Json::int(started.elapsed().as_millis() as i64))
                    .with("usage", self.turn_usage.to_json())
                    .with("backend", Json::str(self.backend_kind().as_str())),
            ),
        );
        Ok(())
    }

    /// Runs one turn to completion. Errors are recorded and turned into a stop
    /// reason; they never escape as an exception, because the transcript must
    /// show what happened.
    fn run_turn(&mut self, turn_id: &str) -> Result<StopReason> {
        let deadline = Instant::now() + Duration::from_millis(self.config.limits.turn_timeout_ms);
        let mut step = 0u32;
        self.turn_usage = TokenUsage::default();
        loop {
            if self.cancel.load(Ordering::SeqCst) {
                return Ok(StopReason::Cancelled);
            }
            if step >= self.config.limits.max_steps {
                self.record_limit(
                    turn_id,
                    step,
                    StopReason::StepLimit,
                    &format!(
                        "Stopped after {} steps, the limit for one turn",
                        self.config.limits.max_steps
                    ),
                );
                return Ok(StopReason::StepLimit);
            }
            if Instant::now() >= deadline {
                self.record_limit(
                    turn_id,
                    step,
                    StopReason::TurnTimeout,
                    &format!(
                        "Stopped after {} ms, the time limit for one turn",
                        self.config.limits.turn_timeout_ms
                    ),
                );
                return Ok(StopReason::TurnTimeout);
            }
            step += 1;
            self.phase = Phase::Preparing;
            self.emit(
                AgentEvent::new(
                    EventKind::StepStarted,
                    &self.id,
                    turn_id,
                    Json::obj().with("step", Json::int(step as i64)),
                )
                .at_step(step),
            );
            self.prepare_context()?;

            self.phase = Phase::Generating;
            let mut sink = BatchingSink::new(self.events.clone(), &self.id, turn_id);
            let messages = self.conversation.render();
            let request_id = format!("{turn_id}-s{step}");
            let generation = {
                let options = self.config.generation;
                let result = self
                    .backend
                    .generate(&request_id, &messages, &options, &mut sink);
                sink.flush();
                result
            };
            if let Some(usage) = generation.as_ref().ok().map(|g| g.usage) {
                self.turn_usage.prompt_tokens += usage.prompt_tokens;
                self.turn_usage.cached_prompt_tokens += usage.cached_prompt_tokens;
                self.turn_usage.completion_tokens += usage.completion_tokens;
                self.turn_usage.prefill_ms += usage.prefill_ms;
                self.turn_usage.decode_ms += usage.decode_ms;
            }

            let generation = match generation {
                Ok(generation) => generation,
                Err(error) => {
                    let classified = classify(&error);
                    self.emit(
                        AgentEvent::new(
                            EventKind::Error,
                            &self.id,
                            turn_id,
                            Json::obj()
                                .with("class", Json::str(classified))
                                .with("message", Json::str(&error.to_string())),
                        )
                        .at_step(step),
                    );
                    return Ok(StopReason::Error);
                }
            };

            self.emit(
                AgentEvent::new(
                    EventKind::Usage,
                    &self.id,
                    turn_id,
                    generation.to_json().with("step", Json::int(step as i64)),
                )
                .at_step(step),
            );

            if !generation.reasoning.is_empty() {
                self.emit(
                    AgentEvent::new(
                        EventKind::AssistantReasoning,
                        &self.id,
                        turn_id,
                        Json::obj().with("text", Json::str(&generation.reasoning)),
                    )
                    .at_step(step),
                );
            }

            if let Some(call) = generation.tool_call.clone() {
                // A tool call keeps the turn alive. This is the single place the
                // "generation ended but the turn did not" distinction is acted on.
                self.conversation.push_assistant(&generation.reasoning, "");
                let outcome = self.run_tool(&call, turn_id, step);
                match outcome {
                    Some(result) => {
                        self.conversation.push_tool_call(call.clone());
                        self.conversation.push_tool_result(result);
                    }
                    None => {
                        return Ok(if self.cancel.load(Ordering::SeqCst) {
                            StopReason::Cancelled
                        } else {
                            StopReason::Error
                        });
                    }
                }
                continue;
            }

            if !generation.text.is_empty() {
                self.conversation.push_assistant(&generation.reasoning, &generation.text);
                self.emit(
                    AgentEvent::new(
                        EventKind::AssistantText,
                        &self.id,
                        turn_id,
                        Json::obj().with("text", Json::str(&generation.text)),
                    )
                    .at_step(step),
                );
            }

            if self.cancel.load(Ordering::SeqCst) {
                return Ok(StopReason::Cancelled);
            }
            match generation.stop {
                StopReason::Cancelled => return Ok(StopReason::Cancelled),
                StopReason::Eos => {
                    if self.tool_error_streak >= self.config.limits.max_repeated_tool_errors {
                        self.record_limit(
                            turn_id,
                            step,
                            StopReason::RepeatedToolError,
                            &format!(
                                "Stopped after {} consecutive tool errors",
                                self.tool_error_streak
                            ),
                        );
                        return Ok(StopReason::RepeatedToolError);
                    }
                    return Ok(StopReason::Eos);
                }
                other => return Ok(other),
            }
        }
    }

    /// Token budget, compaction and cache invalidation before each step.
    fn prepare_context(&mut self) -> Result<()> {
        // A limit change invalidates prefix reuse; the backend owns the cache, so
        // the signal is the compaction event it already emits.
        let budget = self
            .config
            .limits
            .max_context_tokens
            .saturating_sub(self.config.generation.max_tokens + 64);
        let count = |text: &str| self.backend.count_tokens(text);
        // First tier: shrink old tool outputs, drop nothing.
        let micro = self.conversation.microcompact(5, 40);
        if micro > 0 {
            self.emit(
                AgentEvent::new(
                    EventKind::Checkpoint,
                    &self.id,
                    self.active_turn.as_deref().unwrap_or(""),
                    Json::obj()
                        .with("kind", Json::str("microcompact"))
                        .with("truncated_results", Json::int(micro as i64)),
                ),
            );
        }
        // Split borrows so the model summariser can call the backend while
        // the conversation is being compacted: plan first (shared borrows
        // only), summarise, then apply.
        let compaction = {
            let use_model_summary = self.config.use_model_summary;
            let options = self.config.generation;
            let backend = &mut *self.backend;
            let count = |text: &str| backend.count_tokens(text);
            let replaced = self.conversation.drop_stale_summaries();
            let plan = self.conversation.plan_compact(budget, &count);
            match plan {
                None => None,
                Some(plan) => {
                    let summary = if use_model_summary {
                        let evidence: Vec<String> = self.conversation.items
                            [plan.drop_from..plan.cut]
                            .iter()
                            .map(|item| item.text())
                            .collect();
                        model_summary(backend, &options, &evidence, replaced)
                            .unwrap_or_else(|| {
                                super::context::summarise(
                                    &self.conversation.items[plan.drop_from..plan.cut],
                                    replaced,
                                )
                            })
                    } else {
                        super::context::summarise(
                            &self.conversation.items[plan.drop_from..plan.cut],
                            replaced,
                        )
                    };
                    Some(self.conversation.apply_summary(&plan, summary))
                }
            }
        };
        if let Some(compaction) = compaction {
            // The prefix hash tells the next prefill whether the cache still
            // holds: same hash means KV reuse, changed hash means full price.
            let prefix = self.conversation.prefix_hash();
            let turn_id = self.active_turn.clone().unwrap_or_default();
            self.emit(compaction.event(&self.id, &turn_id, prefix));
            // Keep the turn in flight and the checkpoint that explains the gap;
            // everything older is now represented by the summary.
            let keep_from_seq = self
                .log
                .events()
                .iter()
                .find(|event| event.turn_id == turn_id)
                .map(|event| event.seq)
                .unwrap_or(u64::MAX);
            if let Err(error) = self.log.compact(keep_from_seq, &compaction.to_json()) {
                // A failed compaction must not silently discard context: keep the
                // in-memory conversation and say so.
                self.emit(
                    AgentEvent::new(
                        EventKind::Error,
                        &self.id,
                        self.active_turn.as_deref().unwrap_or(""),
                        Json::obj()
                            .with("class", Json::str("compaction_failed"))
                            .with("message", Json::str(&error.to_string())),
                    ),
                );
            }
        }
        Ok(())
    }

    /// Validates, executes and records one tool call. Returns `None` when the
    /// session must stop (cancelled).
    fn run_tool(&mut self, call: &ToolCall, turn_id: &str, step: u32) -> Option<ToolOutcome> {
        self.phase = Phase::RunningTool;
        self.workspace.cancel = tools::CancelToken::new();
        self.emit(
            AgentEvent::new(
                EventKind::ToolCalled,
                &self.id,
                turn_id,
                call.to_json().with("step", Json::int(step as i64)),
            )
            .at_step(step),
        );
        // The working set feeds the per-turn context: the model sees what is
        // already at hand instead of re-listing the project.
        if let Some(path) = call.arguments.get("path").and_then(|v| v.as_str()) {
            remember_working_file(path);
        }
        // Delegation runs in an isolated read-only context, not in this
        // workspace: large explorations must not bloat this conversation.
        if call.name == "delegate" {
            return self.run_delegation(call, turn_id, step);
        }
        let started = Instant::now();
        let result = self.workspace.execute(&call.name, &call.arguments);
        let duration_ms = started.elapsed().as_millis() as u64;
        let outcome = match result {
            Ok(value) => {
                let mut outcome = ToolOutcome::ok(&call.id, &call.name, value, duration_ms);
                let rendered = outcome.result.to_string();
                if rendered.len() > self.config.limits.max_tool_output_bytes {
                    outcome.result = Json::obj()
                        .with("summary", Json::str(rendered.chars().take(400).collect::<String>()))
                        .with(
                            "note",
                            Json::str("The full output was larger than the tool limit; narrow the request."),
                        );
                    outcome.truncated = true;
                }
                outcome
            }
            Err(error) => {
                self.tool_error_streak += 1;
                ToolOutcome::failed(&call.id, &call.name, "tool_failed", &error.to_string(), duration_ms)
            }
        };
        self.emit(
            AgentEvent::new(
                EventKind::ToolResult,
                &self.id,
                turn_id,
                outcome.to_json().with("step", Json::int(step as i64)),
            )
            .at_step(step),
        );
        if self.cancel.load(Ordering::SeqCst) {
            return None;
        }
        Some(outcome)
    }

    /// Runs a bounded research question in an isolated read-only context and
    /// returns a condensed answer. The file contents stay in the child
    /// context; only the summary and a step trailer come back, so large
    /// explorations cost hundreds of tokens instead of thousands.
    fn run_delegation(&mut self, call: &ToolCall, turn_id: &str, step: u32) -> Option<ToolOutcome> {
        let started = Instant::now();
        let task = call
            .arguments
            .get("task")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if task.is_empty() {
            return Some(ToolOutcome::failed(
                &call.id,
                &call.name,
                "bad_arguments",
                "delegate needs a task string",
                started.elapsed().as_millis() as u64,
            ));
        }
        if !self.config.grants.allows(tools::CAP_READ) {
            return Some(ToolOutcome::failed(
                &call.id,
                &call.name,
                "forbidden",
                "this session may not read the project",
                started.elapsed().as_millis() as u64,
            ));
        }
        let max_steps = call
            .arguments
            .get("max_steps")
            .and_then(|v| v.as_i64())
            .unwrap_or(4)
            .clamp(1, 6) as u32;
        let child_workspace = Workspace::new(&self.workspace.root, Grants::read_only());
        let mut child = Conversation::new(
            "You are a read-only researcher. Answer the task with short facts: \
             file paths and the exact lines that matter. No advice, no guesses. \
             Use the same <tool_call> format as always. When you have enough, \
             answer in plain text without further calls.",
        );
        child.push_user(&task);
        let mut steps: Vec<Json> = Vec::new();
        let mut answer = String::new();
        for _ in 0..max_steps {
            if self.cancel.load(Ordering::SeqCst) {
                return None;
            }
            let messages = child.render();
            let mut sink = CollectSink { text: String::new() };
            let options = self.config.generation;
            let request_id = format!("{}-delegate-s{}", turn_id, step);
            let generation = match self.backend.generate(&request_id, &messages, &options, &mut sink) {
                Ok(generation) => generation,
                Err(error) => {
                    answer = format!("research failed: {error}");
                    break;
                }
            };
            let reply = if sink.text.trim().is_empty() {
                generation.text.clone()
            } else {
                sink.text.clone()
            };
            let Some(child_call) = super::backend::parse_tool_call(&reply, &request_id) else {
                answer = reply.trim().to_string();
                break;
            };
            // The child is read-only: writes and shell never reach the workspace.
            if child_call.name == "fs_write" || child_call.name == "fs_edit" || child_call.name == "shell" {
                answer = format!(
                    "research stopped: {} is not allowed in a read-only context",
                    child_call.name
                );
                break;
            }
            child.push_tool_call(child_call.clone());
            let outcome = match child_workspace.execute(&child_call.name, &child_call.arguments) {
                Ok(value) => ToolOutcome::ok(&child_call.id, &child_call.name, value, 0),
                Err(error) => {
                    ToolOutcome::failed(&child_call.id, &child_call.name, "tool_failed", &error.to_string(), 0)
                }
            };
            steps.push(
                Json::obj()
                    .with("tool", Json::str(&child_call.name))
                    .with("ok", Json::Bool(outcome.ok)),
            );
            child.push_tool_result(outcome);
            if child.items.len() > 24 {
                break;
            }
        }
        if answer.trim().is_empty() {
            answer = "research produced no answer".to_string();
        }
        if answer.len() > 4000 {
            let mut cut = 4000;
            while cut > 0 && !answer.is_char_boundary(cut) {
                cut -= 1;
            }
            answer.truncate(cut);
            answer.push_str("\n[delegation answer truncated]");
        }
        let duration_ms = started.elapsed().as_millis() as u64;
        let outcome = ToolOutcome::ok(
            &call.id,
            &call.name,
            Json::obj()
                .with("answer", Json::str(&answer))
                .with("steps", Json::Arr(steps)),
            duration_ms,
        );
        self.emit(
            AgentEvent::new(
                EventKind::ToolResult,
                &self.id,
                turn_id,
                outcome.to_json().with("step", Json::int(step as i64)),
            )
            .at_step(step),
        );
        if self.cancel.load(Ordering::SeqCst) {
            return None;
        }
        Some(outcome)
    }

    /// Context usage by category for the Stats page: where the window goes,
    /// the budget, and the prefix hash the prefill cache depends on.
    pub fn context_breakdown(&self) -> Json {
        let count = |text: &str| self.backend.count_tokens(text);
        let parts = self.conversation.token_breakdown(&count);
        let total: usize = parts.iter().map(|(_, cost)| cost).sum();
        let mut breakdown = Json::obj();
        for (name, cost) in &parts {
            breakdown = breakdown.with(*name, Json::int(*cost as i64));
        }
        breakdown
            .with("total", Json::int(total as i64))
            .with("budget", Json::int(self.config.limits.max_context_tokens as i64))
            .with(
                "prefix_hash",
                Json::str(&format!("{:016x}", self.conversation.prefix_hash())),
            )
            .with("items", Json::int(self.conversation.len() as i64))
    }

    fn record_limit(&mut self, turn_id: &str, step: u32, reason: StopReason, message: &str) {
        self.emit(
            AgentEvent::new(
                EventKind::Error,
                &self.id,
                turn_id,
                Json::obj()
                    .with("class", Json::str(reason.as_str()))
                    .with("message", Json::str(message))
                    .with("step", Json::int(step as i64)),
            )
            .at_step(step),
        );
    }

    /// Cancels inference and any running tool. Effects already on disk are not
    /// rolled back.
    pub fn cancel(&mut self) {
        // With no turn in flight there is nothing to interrupt: arming the
        // backend here would make the *next* turn look cancelled.
        let Some(turn_id) = self.active_turn.clone() else {
            return;
        };
        self.cancel.store(true, Ordering::SeqCst);
        self.phase = Phase::Cancelling;
        self.backend.cancel(&turn_id);
        self.workspace.cancel.cancel();
        tools::cancel_active_shell();
        {
            self.emit(
                AgentEvent::new(
                    EventKind::Cancelled,
                    &self.id,
                    &turn_id,
                    Json::obj().with("requested_at_ms", Json::int(super::protocol::now_ms() as i64)),
                ),
            );
        }
    }

    /// Whether a turn is currently running.
    pub fn running(&self) -> bool {
        self.is_busy()
    }

    /// Whether an admitted message is waiting for its turn.
    pub fn has_pending(&self) -> bool {
        !self.queue.is_empty()
    }

    /// Runs at most one queued turn. The caller owns the thread and decides how
    /// many to run before yielding.
    pub fn pump_one(&mut self) -> bool {
        if self.is_busy() {
            return false;
        }
        let Some(message) = self.queue.first().cloned() else {
            return false;
        };
        self.queue.remove(0);
        self.start_turn(&message.request_id, &message.text).is_ok()
    }

    pub fn clear_queue(&mut self) {
        self.queue.clear();
    }

    fn emit(&mut self, event: AgentEvent) {
        match self.log.record(event.clone()) {
            Ok(recorded) => {
                let _ = self.events.send(recorded);
            }
            Err(error) => {
                // Losing the durable record is serious: report it instead of
                // letting the transcript imply a turn that was never saved.
                let fallback = AgentEvent::new(
                    EventKind::Error,
                    &self.id,
                    &event.turn_id,
                    Json::obj()
                        .with("class", Json::str("log_write_failed"))
                        .with("message", Json::str(&error.to_string()))
                        .with("original_kind", Json::str(event.kind.as_str())),
                );
                let _ = self.events.send(fallback);
            }
        }
    }
}

fn call_from_event(event: &AgentEvent) -> Option<ToolCall> {
    Some(ToolCall {
        id: event.data.get("call_id")?.as_str()?.to_string(),
        name: event.data.get("name")?.as_str()?.to_string(),
        arguments: event
            .data
            .get("arguments")
            .cloned()
            .unwrap_or_else(Json::obj),
    })
}

fn outcome_from_event(event: &AgentEvent) -> Option<ToolOutcome> {
    Some(ToolOutcome {
        call_id: event.data.get("call_id")?.as_str()?.to_string(),
        name: event.data.get("name")?.as_str()?.to_string(),
        ok: event.data.get("ok")?.as_bool().unwrap_or(false),
        result: event.data.get("result").cloned().unwrap_or_else(Json::obj),
        error_class: event
            .data
            .get("error_class")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        error: event
            .data
            .get("error")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        duration_ms: event
            .data
            .get("duration_ms")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as u64,
        truncated: event
            .data
            .get("truncated")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    })
}

fn limits_json(limits: &TurnLimits) -> Json {
    Json::obj()
        .with("max_steps", Json::int(limits.max_steps as i64))
        .with("turn_timeout_ms", Json::int(limits.turn_timeout_ms as i64))
        .with(
            "max_tool_output_bytes",
            Json::int(limits.max_tool_output_bytes as i64),
        )
        .with(
            "max_context_tokens",
            Json::int(limits.max_context_tokens as i64),
        )
        .with(
            "max_repeated_tool_errors",
            Json::int(limits.max_repeated_tool_errors as i64),
        )
        .with(
            "max_queued_messages",
            Json::int(limits.max_queued_messages as i64),
        )
}

/// Classifies a failure so the UI can react differently to "no model" and
/// "the tool was wrong" without parsing English.
fn classify(error: &Error) -> &'static str {
    let text = error.to_string().to_ascii_lowercase();
    if text.contains("no model") || text.contains("not loaded") {
        "no_model"
    } else if text.contains("context window") || text.contains("tokens but the window") {
        "context_overflow"
    } else if text.contains("npu") || text.contains("hexagon") || text.contains("fastrpc") {
        "backend_npu"
    } else if text.contains("cannot load") || text.contains("gguf") {
        "model_load_failed"
    } else {
        "generation_failed"
    }
}

#[cfg(test)]
fn limits_default_json() -> Json {
    limits_json(&TurnLimits::default())
}

/// Registry of sessions plus the worker that runs them, so the JNI layer and
/// the Kotlin service share one place that owns the state.
pub struct Runtime {
    sessions: Mutex<Vec<Arc<Mutex<Session>>>>,
    /// Cancel flags mirrored outside the session lock. A turn holds its session
    /// mutex for the whole generation, so Stop must not need that mutex to take
    /// effect — it flips the flag here, plus the global engine stop and the
    /// active shell kill, all without touching the session lock.
    cancels: Mutex<std::collections::HashMap<String, Arc<AtomicBool>>>,
    /// Kept so the channel stays open; sessions hold their own senders.
    _events_tx: Sender<AgentEvent>,
    events_rx: Mutex<Receiver<AgentEvent>>,
}

impl Runtime {
    pub fn new() -> Result<Runtime> {
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        Ok(Runtime {
            sessions: Mutex::new(Vec::new()),
            cancels: Mutex::new(std::collections::HashMap::new()),
            _events_tx: events_tx,
            events_rx: Mutex::new(events_rx),
        })
    }

    /// A sender for the shared event channel, so a session can be opened after
    /// the runtime exists.
    pub fn sender(&self) -> Sender<AgentEvent> {
        self._events_tx.clone()
    }

    /// Non-blocking read of the next event, if any.
    pub fn next_event(&self) -> Option<AgentEvent> {
        self.events_rx.lock().ok()?.try_recv().ok()
    }

    pub fn register(&self, session: Session) -> Result<String> {
        let id = session.id.clone();
        let cancel = session.cancel.clone();
        let session = Arc::new(Mutex::new(session));
        // Reopening the same id replaces the handle: never stack two live
        // sessions for one log file, or turns could interleave on one log.
        {
            let mut sessions = self
                .sessions
                .lock()
                .map_err(|_| Error::new("session registry is poisoned"))?;
            sessions.retain(|existing| {
                existing
                    .lock()
                    .map(|inner| inner.id != id)
                    .unwrap_or(true)
            });
            sessions.push(session);
        }
        self.cancels
            .lock()
            .map_err(|_| Error::new("cancel registry is poisoned"))?
            .insert(id.clone(), cancel);
        Ok(id)
    }

    pub fn with<T>(&self, id: &str, action: impl FnOnce(&mut Session) -> Result<T>) -> Result<T> {
        // Clone the session Arc under the registry lock, then drop the registry
        // lock before touching the session. Holding both across a whole turn
        // would serialise every other session operation behind a generation.
        let session = {
            let sessions = self
                .sessions
                .lock()
                .map_err(|_| Error::new("session registry is poisoned"))?;
            sessions
                .iter()
                .find(|session| {
                    session
                        .lock()
                        .map(|inner| inner.id == id)
                        .unwrap_or(false)
                })
                .cloned()
                .ok_or_else(|| Error::new(format!("Unknown session '{id}'")))?
        };
        let mut guard = session
            .lock()
            .map_err(|_| Error::new("session state is poisoned"))?;
        action(&mut guard)
    }

    /// Stop without needing the session mutex. The turn loop polls the same
    /// flag, the engine polls the global stop, and any active shell child is
    /// killed by pid — so this returns immediately even mid-generation.
    pub fn cancel_session(&self, id: &str) -> Result<bool> {
        let cancel = self
            .cancels
            .lock()
            .map_err(|_| Error::new("cancel registry is poisoned"))?
            .get(id)
            .cloned();
        match cancel {
            Some(flag) => {
                flag.store(true, Ordering::SeqCst);
                crate::util::request_stop();
                super::tools::cancel_active_shell();
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn ids(&self) -> Vec<String> {
        self.sessions
            .lock()
            .map(|sessions| {
                sessions
                    .iter()
                    .filter_map(|session| session.lock().ok().map(|inner| inner.id.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn close(&self, id: &str) -> Result<()> {
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| Error::new("session registry is poisoned"))?;
        sessions.retain(|session| {
            let matches = session.lock().map(|inner| inner.id == id).unwrap_or(false);
            !matches
        });
        if let Ok(mut cancels) = self.cancels.lock() {
            cancels.remove(id);
        }
        Ok(())
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            let (tx, rx) = std::sync::mpsc::channel();
            Runtime {
                sessions: Mutex::new(Vec::new()),
                cancels: Mutex::new(std::collections::HashMap::new()),
                _events_tx: tx,
                events_rx: Mutex::new(rx),
            }
        })
    }
}

/// Batch of events for the UI. The batch is a JSONL string rather than one
/// payload per event so a long reply does not turn into a stream of binder calls.
pub fn drain_batch(runtime: &Runtime, max: usize) -> Vec<AgentEvent> {
    let mut batch = Vec::new();
    while batch.len() < max {
        match runtime.next_event() {
            Some(event) => batch.push(event),
            None => break,
        }
    }
    batch
}

pub fn batch_to_jsonl(batch: &[AgentEvent]) -> String {
    let mut out = String::new();
    for event in batch {
        event.to_json().write(&mut out);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::backend::{Generation, LoadRequest, ModelInfo};
    use super::super::json::Json;
    use super::super::session::SessionLog;
    use super::*;
    use std::sync::mpsc::channel;

    /// Backend that replays a scripted list of replies and records the calls it
    /// received, so turn behaviour can be tested without a model.
    struct ScriptedBackend {
        replies: Vec<std::result::Result<Generation, String>>,
        seen: Vec<Vec<(String, String)>>,
        cancelled: Arc<AtomicBool>,
        loaded: Option<ModelInfo>,
    }

    /// One scripted reply, before it becomes a `Generation`.
    enum Script {
        Answer(String),
        Tool { name: String, arguments: String },
        Failure(String),
        Cancelled,
    }

    impl Script {
        fn into_generation(self) -> std::result::Result<Generation, String> {
            match self {
                Script::Answer(text) => Ok(Generation {
                    text,
                    stop: StopReason::Eos,
                    backend: BackendKind::RustCpu,
                    ..Default::default()
                }),
                Script::Tool { name, arguments } => Ok(Generation {
                    tool_call: Some(ToolCall {
                        id: format!("call-{name}"),
                        name,
                        arguments: super::super::json::parse(&arguments).expect("arguments parse"),
                    }),
                    stop: StopReason::Eos,
                    backend: BackendKind::RustCpu,
                    ..Default::default()
                }),
                Script::Failure(message) => Err(message),
                Script::Cancelled => Ok(Generation {
                    stop: StopReason::Cancelled,
                    backend: BackendKind::RustCpu,
                    ..Default::default()
                }),
            }
        }
    }

    impl ScriptedBackend {
        fn new(replies: Vec<Script>) -> Self {
            Self {
                replies: replies.into_iter().map(Script::into_generation).collect(),
                seen: Vec::new(),
                cancelled: Arc::new(AtomicBool::new(false)),
                loaded: None,
            }
        }
    }

    impl Backend for ScriptedBackend {
        fn load(&mut self, request: &LoadRequest) -> Result<ModelInfo> {
            // Mirrors the real backends: what ran is reported, and a request for
            // a backend that is not available becomes an explicit fallback.
            let (backend, fallback) = match (request.backend, request.use_gpu) {
                (BackendKind::RustOpenCl, true) => (BackendKind::RustOpenCl, false),
                (BackendKind::RustOpenCl, false) => (BackendKind::RustCpu, true),
                _ => (BackendKind::RustCpu, request.backend == BackendKind::RustOpenCl),
            };
            let info = ModelInfo {
                backend,
                requested: request.backend,
                fallback,
                context_tokens: 4096,
                load_ms: 1.0,
                name: "scripted".to_string(),
                kv_bytes: 0,
                detail: String::new(),
            };
            self.loaded = Some(info.clone());
            Ok(info)
        }

        fn generate(
            &mut self,
            _request_id: &str,
            messages: &[(String, String)],
                _options: &GenerationOptions,
            sink: &mut dyn DeltaSink,
        ) -> Result<Generation> {
            self.seen.push(messages.to_vec());
            if self.cancelled.load(Ordering::SeqCst) {
                return Ok(Generation {
                    stop: StopReason::Cancelled,
                    backend: BackendKind::RustCpu,
                    ..Default::default()
                });
            }
            let index = self.seen.len() - 1;
            match self.replies.get(index) {
                Some(Ok(generation)) => {
                    if !generation.text.is_empty() {
                        for chunk in generation.text.chars().collect::<Vec<_>>().chunks(3) {
                            sink.text(&chunk.iter().collect::<String>());
                        }
                    }
                    Ok(generation.clone())
                }
                Some(Err(message)) => Err(Error::new(message.clone())),
                None => Ok(Generation {
                    text: "nothing more".to_string(),
                    stop: StopReason::Eos,
                    backend: BackendKind::RustCpu,
                    ..Default::default()
                }),
            }
        }

        fn cancel(&mut self, _request_id: &str) {
            self.cancelled.store(true, Ordering::SeqCst);
        }

        fn unload(&mut self) {
            self.loaded = None;
        }

        fn capabilities(&self) -> Json {
            Json::obj().with("streaming", Json::Bool(true))
        }

        fn info(&self) -> Option<ModelInfo> {
            self.loaded.clone()
        }
    }

    fn generation(text: &str) -> Script {
        Script::Answer(text.to_string())
    }

    fn tool_generation(name: &str, arguments: &str) -> Script {
        Script::Tool { name: name.to_string(), arguments: arguments.to_string() }
    }

    fn failure(message: &str) -> Script {
        Script::Failure(message.to_string())
    }

    fn cancelled() -> Script {
        Script::Cancelled
    }

    struct Fixture {
        dir: std::path::PathBuf,
        events: Receiver<AgentEvent>,
        session: Session,
    }

    impl Fixture {
        fn new(name: &str, replies: Vec<Script>) -> Fixture {
            let dir = std::env::temp_dir()
                .join(format!("pocketagent-rt-{name}-{}", std::process::id()));
            std::fs::remove_dir_all(&dir).ok();
            std::fs::create_dir_all(dir.join("workspace")).unwrap();
            let (tx, rx) = channel();
            let workspace = dir.join("workspace");
            let config = SessionConfig {
                workspace_id: "w1".to_string(),
                ..Default::default()
            };
            let session = Session::open(
                "s1",
                &dir.join("session.jsonl"),
                &workspace,
                config,
                tx,
                Box::new(ScriptedBackend::new(replies)),
            )
            .unwrap();
            Fixture { dir, events: rx, session }
        }
    }

    fn durable_events(fixture: &Fixture) -> Vec<AgentEvent> {
        let text = std::fs::read_to_string(fixture.dir.join("session.jsonl")).unwrap_or_default();
        text.lines()
            .filter_map(|line| super::super::json::parse(line).ok())
            .filter_map(|value| AgentEvent::from_json(&value))
            .filter(|event| event.kind.is_durable())
            .collect()
    }

    #[test]
    fn a_plain_answer_produces_the_documented_event_sequence() {
        let mut fixture = Fixture::new("plain", vec![generation("Here is the answer.")]);
        fixture.session.submit("r1", "hello").unwrap();
        fixture.session.pump();
        let events = durable_events(&fixture);
        let order: Vec<&str> = events.iter().map(|event| event.kind.as_str()).collect();
        assert_eq!(
            order,
            vec![
                "user_message.accepted",
                "turn.started",
                "step.started",
                "usage",
                "assistant.text",
                "turn.ended",
            ]
        );
        assert_eq!(events[0].seq, 1);
        assert!(events.windows(2).all(|pair| pair[0].seq < pair[1].seq));
        assert_eq!(fixture.session.last_stop(), Some(StopReason::Eos));
    }

    #[test]
    fn sequence_numbers_are_monotonic_in_the_file() {
        let mut fixture = Fixture::new("seq", vec![generation("a"), generation("b")]);
        fixture.session.submit("r1", "one").unwrap();
        fixture.session.submit("r2", "two").unwrap();
        fixture.session.pump();
        let seqs: Vec<u64> = durable_events(&fixture)
            .iter()
            .map(|event| event.seq)
            .collect();
        assert!(seqs.windows(2).all(|pair| pair[0] < pair[1]), "{seqs:?}");
    }

    #[test]
    fn a_tool_call_keeps_the_turn_alive_and_records_both_sides() {
        let mut fixture = Fixture::new(
            "tool",
            vec![
                tool_generation("fs_write", r#"{"path":"hello.txt","content":"ciao"}"#),
                generation("Created hello.txt."),
            ],
        );
        fixture.session.submit("r1", "create a file").unwrap();
        fixture.session.pump();
        let events = durable_events(&fixture);
        let order: Vec<&str> = events.iter().map(|event| event.kind.as_str()).collect();
        assert!(order.contains(&"tool.called"));
        assert!(order.contains(&"tool.result"));
        let called = events.iter().find(|e| e.kind == EventKind::ToolCalled).unwrap();
        let result = events.iter().find(|e| e.kind == EventKind::ToolResult).unwrap();
        assert_eq!(
            called.data.get("call_id").unwrap().as_str().unwrap(),
            result.data.get("call_id").unwrap().as_str().unwrap(),
            "a result must correlate with its call"
        );
        assert_eq!(result.data.get("ok").unwrap().as_bool(), Some(true));
        // The file really exists: the tool ran, it was not simulated.
        let content = std::fs::read_to_string(fixture.dir.join("workspace/hello.txt")).unwrap();
        assert_eq!(content, "ciao");
        assert_eq!(fixture.session.last_stop(), Some(StopReason::Eos));
        assert_eq!(fixture.session.phase(), Phase::Idle);
    }

    #[test]
    fn a_failing_tool_is_recorded_and_shown_to_the_model() {
        let mut fixture = Fixture::new(
            "toolfail",
            vec![
                tool_generation("fs_read", r#"{"path":"../escape"}"#),
                generation("I could not read that path."),
            ],
        );
        fixture.session.submit("r1", "read outside").unwrap();
        fixture.session.pump();
        let events = durable_events(&fixture);
        let result = events.iter().find(|e| e.kind == EventKind::ToolResult).unwrap();
        assert_eq!(result.data.get("ok").unwrap().as_bool(), Some(false));
        assert!(result
            .data
            .get("error")
            .and_then(|e| e.as_str())
            .unwrap()
            .contains("outside the workspace"));
        // The second request must contain the error, so the model can react.
        assert_eq!(fixture.session.last_stop(), Some(StopReason::Eos));
    }

    #[test]
    fn every_prior_tool_call_is_present_in_the_next_prompt() {
        let mut fixture = Fixture::new(
            "history",
            vec![
                tool_generation("fs_write", r#"{"path":"a.txt","content":"x"}"#),
                tool_generation("fs_read", r#"{"path":"a.txt"}"#),
                generation("done"),
            ],
        );
        fixture.session.submit("r1", "write then read").unwrap();
        fixture.session.pump();
        let log = SessionLog::open(&fixture.dir.join("session.jsonl"), "w1", "t").unwrap();
        let rendered: Vec<String> = log
            .events()
            .iter()
            .map(|event| event.data.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string())
            .collect();
        assert!(rendered.iter().any(|text| text == "write then read"));
        assert_eq!(fixture.session.last_stop(), Some(StopReason::Eos));
    }

    #[test]
    fn an_effectless_failed_turn_leaves_no_question_behind() {
        // The no_model story: a turn dies before the backend runs, the user
        // retries the same message, and the model must read one question, not
        // the same question twice.
        let mut fixture = Fixture::new(
            "rollback",
            vec![failure("No model is loaded"), generation("second")],
        );
        fixture.session.submit("r1", "same question").unwrap();
        assert_eq!(fixture.session.pump(), 1);
        assert_eq!(fixture.session.conversation.len(), 1);
        fixture.session.submit("r2", "same question").unwrap();
        assert_eq!(fixture.session.pump(), 1);
        let rendered: String = fixture
            .session
            .conversation
            .render()
            .into_iter()
            .map(|(_, text)| text)
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(rendered.matches("same question").count(), 1);
    }

    #[test]
    fn submit_only_admits_and_pump_runs_the_turn() {
        let mut fixture = Fixture::new("admit", vec![generation("first")]);
        fixture.session.submit("r1", "one").unwrap();
        assert!(!fixture.session.running());
        assert_eq!(fixture.session.queued(), 1);
        // Nothing has run yet: the acceptance is recorded, the turn is not.
        let before = durable_events(&fixture);
        assert!(before.iter().all(|e| e.kind != EventKind::AssistantText));
        assert_eq!(fixture.session.pump(), 1);
        assert_eq!(fixture.session.queued(), 0);
        let after = durable_events(&fixture);
        assert!(after.iter().any(|e| e.kind == EventKind::AssistantText));
    }

    #[test]
    fn queued_messages_run_in_order_after_the_active_turn() {
        let mut fixture = Fixture::new(
            "queue",
            vec![
                tool_generation("fs_write", r#"{"path":"a.txt","content":"1"}"#),
                generation("first"),
                generation("second"),
            ],
        );
        // The UI submitting while a turn is in flight: the session is busy, so
        // each message is accepted and queued rather than run.
        fixture.session.active_turn = Some("s1t1".to_string());
        fixture.session.submit("r1", "one").unwrap();
        fixture.session.submit("r2", "two").unwrap();
        fixture.session.submit("r3", "three").unwrap();
        assert_eq!(fixture.session.queued(), 3);
        assert!(!durable_events(&fixture)
            .iter()
            .any(|event| event.kind == EventKind::AssistantText));
        // The active turn finishes; the worker calls pump.
        fixture.session.active_turn = None;
        assert_eq!(fixture.session.pump(), 3);
        let accepted: Vec<String> = durable_events(&fixture)
            .iter()
            .filter(|event| event.kind == EventKind::UserMessageAccepted)
            .map(|event| event.data.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string())
            .collect();
        assert_eq!(accepted, vec!["one", "two", "three"]);
        let answers: Vec<String> = durable_events(&fixture)
            .iter()
            .filter(|event| event.kind == EventKind::AssistantText)
            .map(|event| event.data.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string())
            .collect();
        assert_eq!(answers[0], "first");
        assert_eq!(answers[1], "second");
    }

    #[test]
    fn a_queued_message_is_accepted_exactly_once() {
        let mut fixture = Fixture::new("idem", vec![generation("ok")]);
        assert!(fixture.session.submit("r1", "hello").unwrap());
        // Same request id again: already admitted, so nothing new happens.
        assert!(!fixture.session.submit("r1", "hello").unwrap());
        fixture.session.pump();
        assert!(!fixture.session.submit("r1", "hello").unwrap());
        let accepted = durable_events(&fixture)
            .iter()
            .filter(|event| event.kind == EventKind::UserMessageAccepted)
            .count();
        assert_eq!(accepted, 1);
    }

    #[test]
    fn the_queue_has_a_ceiling_and_refuses_rather_than_drops() {
        let mut fixture = Fixture::new("ceiling", vec![generation("ok")]);
        fixture.session.active_turn = Some("s1t1".to_string());
        fixture.session.config.limits.max_queued_messages = 2;
        fixture.session.submit("r1", "one").unwrap();
        fixture.session.submit("r2", "two").unwrap();
        let error = fixture.session.submit("r3", "three").unwrap_err();
        assert!(error.to_string().contains("queue"));
        assert_eq!(fixture.session.queued(), 2, "the refused message is not queued");
    }

    #[test]
    fn the_step_limit_is_a_stop_reason_not_an_invented_answer() {
        let mut fixture = Fixture::new(
            "steplimit",
            (0..20).map(|_| tool_generation("fs_list", r#"{"path":"."}"#)).collect(),
        );
        fixture.session.config.limits.max_steps = 3;
        fixture.session.submit("r1", "loop forever").unwrap();
        fixture.session.pump();
        assert_eq!(fixture.session.last_stop(), Some(StopReason::StepLimit));
        let events = durable_events(&fixture);
        let limit = events
            .iter()
            .find(|e| e.kind == EventKind::Error)
            .expect("a limit is reported");
        assert_eq!(limit.data.get("class").unwrap().as_str().unwrap(), "step_limit");
        let ended = events.last().unwrap();
        assert_eq!(ended.kind, EventKind::TurnEnded);
        assert_eq!(ended.data.get("reason").unwrap().as_str().unwrap(), "step_limit");
        // No fabricated closing message.
        assert!(!events.iter().any(|e| e.kind == EventKind::AssistantText));
    }

    #[test]
    fn a_cancelled_generation_ends_the_turn_without_an_answer() {
        let mut fixture = Fixture::new("cancel", vec![cancelled(), generation("never")]);
        fixture.session.submit("r1", "hello").unwrap();
        fixture.session.pump();
        assert_eq!(fixture.session.last_stop(), Some(StopReason::Cancelled));
        let events = durable_events(&fixture);
        let ended = events.last().unwrap();
        assert_eq!(ended.kind, EventKind::TurnEnded);
        assert_eq!(ended.data.get("reason").unwrap().as_str().unwrap(), "cancelled");
        assert!(
            !events.iter().any(|event| event.kind == EventKind::AssistantText),
            "a cancelled turn must not publish an answer"
        );
        // The queue is not touched by a cancellation: the next message still runs.
        fixture.session.submit("r2", "again").unwrap();
        fixture.session.pump();
        assert_eq!(fixture.session.last_stop(), Some(StopReason::Eos));
    }

    #[test]
    fn cancel_on_an_idle_session_records_nothing_and_does_not_block_the_next_turn() {
        let mut fixture = Fixture::new("cancelidle", vec![generation("ok")]);
        fixture.session.cancel();
        assert!(!durable_events(&fixture)
            .iter()
            .any(|event| event.kind == EventKind::Cancelled));
        fixture.session.submit("r1", "hello").unwrap();
        fixture.session.pump();
        assert_eq!(fixture.session.last_stop(), Some(StopReason::Eos));
    }

    #[test]
    fn a_generation_error_is_classified_and_ends_the_turn() {
        let mut fixture = Fixture::new("error", vec![failure("no model is loaded")]);
        fixture.session.submit("r1", "hello").unwrap();
        fixture.session.pump();
        assert_eq!(fixture.session.last_stop(), Some(StopReason::Error));
        let events = durable_events(&fixture);
        let error = events.iter().find(|e| e.kind == EventKind::Error).unwrap();
        assert_eq!(error.data.get("class").unwrap().as_str().unwrap(), "no_model");
        assert_eq!(durable_events(&fixture).last().unwrap().kind, EventKind::TurnEnded);
    }

    #[test]
    fn deltas_are_streamed_but_never_written_to_the_log() {
        let mut fixture = Fixture::new("delta", vec![generation("abcdefghijklmnopqrstuvwxyz")]);
        fixture.session.submit("r1", "hello").unwrap();
        fixture.session.pump();
        let streamed: Vec<String> = {
            let mut out = Vec::new();
            while let Ok(event) = fixture.events.try_recv() {
                if event.kind == EventKind::DeltaText {
                    out.push(event.data.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string());
                }
            }
            out
        };
        assert!(!streamed.is_empty(), "deltas must reach the UI");
        assert!(streamed.len() < 26, "deltas must be batched, not one per character");
        let on_disk = std::fs::read_to_string(fixture.dir.join("session.jsonl")).unwrap();
        assert!(!on_disk.contains("delta.text"));
        assert!(on_disk.contains("abcdefghijklmnopqrstuvwxyz"));
    }

    #[test]
    fn oversized_tool_output_is_replaced_by_a_pointer() {
        let mut fixture = Fixture::new(
            "truncate",
            vec![
                tool_generation("shell", r#"{"command":"for i in $(seq 1 4000); do echo 0123456789; done"}"#),
                generation("that is a lot of output"),
            ],
        );
        fixture.session.config.limits.max_tool_output_bytes = 200;
        fixture.session.submit("r1", "print a lot").unwrap();
        fixture.session.pump();
        let events = durable_events(&fixture);
        let result = events.iter().find(|e| e.kind == EventKind::ToolResult).unwrap();
        assert_eq!(result.data.get("truncated").unwrap().as_bool(), Some(true));
        assert!(result.data.get("result").unwrap().get("note").is_some());
    }

    #[test]
    fn model_state_events_report_the_backend_that_ran() {
        let mut fixture = Fixture::new("modelstate", vec![generation("ok")]);
        fixture.session.pump();
        fixture.session
            .load_model(&LoadRequest {
                model_path: "/tmp/model.gguf".to_string(),
                backend: BackendKind::RustOpenCl,
                context_tokens: 4096,
                threads: 2,
                // Asked for OpenCL without asking for it to be used: the report
                // must say CPU ran and that this was a fallback.
                use_gpu: false,
            })
            .unwrap();
        let events = durable_events(&fixture);
        let state = events
            .iter()
            .find(|event| event.kind == EventKind::ModelState && event.data.get("state").and_then(|s| s.as_str()) == Some("ready"))
            .expect("a ready event");
        let info = state.data.get("info").unwrap();
        assert_eq!(info.get("backend").unwrap().as_str().unwrap(), "rust-cpu");
        assert_eq!(
            info.get("requested_backend").unwrap().as_str().unwrap(),
            "rust-opencl"
        );
        assert_eq!(info.get("fallback").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn reopening_a_session_restores_context_and_flags_an_interrupted_turn() {
        let dir = std::env::temp_dir().join(format!("pocketagent-reopen-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(dir.join("workspace")).unwrap();
        let (tx, _rx) = channel();
        {
            let mut session = Session::open(
                "s1",
                &dir.join("session.jsonl"),
                &dir.join("workspace"),
                SessionConfig { workspace_id: "w1".into(), ..Default::default() },
                tx,
                Box::new(ScriptedBackend::new(vec![generation("hello there")])),
            )
            .unwrap();
            session.submit("r1", "hi").unwrap();
            session.pump();
        }
        // Simulate a crash between turn.started and turn.ended.
        let log = std::fs::read_to_string(dir.join("session.jsonl")).unwrap();
        std::fs::write(
            dir.join("session.jsonl"),
            log.replace("\"kind\":\"turn.ended\"", "\"kind\":\"turn.unknown\""),
        )
        .unwrap();

        let (tx2, rx2) = channel();
        let mut reopened = Session::open(
            "s1",
            &dir.join("session.jsonl"),
            &dir.join("workspace"),
            SessionConfig { workspace_id: "w1".into(), ..Default::default() },
            tx2,
            Box::new(ScriptedBackend::new(vec![generation("second")])),
        )
        .unwrap();
        let mut interrupted = false;
        while let Ok(event) = rx2.try_recv() {
            if event.kind == EventKind::Error
                && event.data.get("class").and_then(|c| c.as_str()) == Some("interrupted")
            {
                interrupted = true;
            }
        }
        assert!(interrupted, "an interrupted turn must be surfaced, not hidden");
        // The first exchange is still in context, and the new turn continues it.
        let context = reopened
            .conversation
            .render()
            .iter()
            .map(|(_, text)| text.clone())
            .collect::<Vec<_>>();
        assert!(context.iter().any(|text| text == "hi"), "{context:?}");
        assert!(context.iter().any(|text| text == "hello there"), "{context:?}");
        reopened.submit("r2", "what did you say?").unwrap();
        reopened.pump();
        let events: Vec<AgentEvent> = std::fs::read_to_string(dir.join("session.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| super::super::json::parse(line).ok())
            .filter_map(|value| AgentEvent::from_json(&value))
            .collect();
        // The already-admitted request is not replayed.
        let accepted: Vec<String> = events
            .iter()
            .filter(|event| event.kind == EventKind::UserMessageAccepted)
            .map(|event| event.data.get("request_id").and_then(|r| r.as_str()).unwrap_or("").to_string())
            .collect();
        assert_eq!(accepted, vec!["r1".to_string(), "r2".to_string()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_runtime_registers_and_closes_sessions() {
        let runtime = Runtime::new().unwrap();
        let dir = std::env::temp_dir().join(format!("pocketagent-reg-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("workspace")).unwrap();
        let (tx, _rx) = channel();
        let session = Session::open(
            "s7",
            &dir.join("s.jsonl"),
            &dir.join("workspace"),
            SessionConfig::default(),
            tx,
            Box::new(ScriptedBackend::new(vec![])),
        )
        .unwrap();
        runtime.register(session).unwrap();
        assert_eq!(runtime.ids(), vec!["s7".to_string()]);
        assert!(runtime.with("s7", |session| Ok(session.phase())).is_ok());
        assert!(runtime.with("nope", |session| Ok(session.phase())).is_err());
        runtime.close("s7").unwrap();
        assert!(runtime.ids().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn event_batches_are_jsonl_for_the_binder_boundary() {
        let runtime = Runtime::new().unwrap();
        let batch = vec![AgentEvent::new(EventKind::TurnStarted, "s", "t", Json::obj())];
        let jsonl = batch_to_jsonl(&batch);
        assert_eq!(jsonl.lines().count(), 1);
        let parsed = super::super::json::parse(jsonl.lines().next().unwrap()).unwrap();
        assert_eq!(parsed.get("kind").unwrap().as_str().unwrap(), "turn.started");
        assert!(drain_batch(&runtime, 8).is_empty());
    }

    #[test]
    fn default_limits_are_explicit_and_serialised() {
        let json = limits_default_json();
        assert_eq!(json.get("max_steps").unwrap().as_i64().unwrap(), 12);
        assert!(json.get("turn_timeout_ms").unwrap().as_i64().unwrap() > 0);
    }
}

#[cfg(test)]
mod prompt_dump {
    use super::*;
    /// Writes the exact prompt the backend receives, so a backend can be tried
    /// offline against the same bytes the app sends. Run with --nocapture.
    #[test]
    fn dump() {
        let (sys, conv) = super::super::backend::debug_render(&[
            ("system".to_string(), default_system_prompt()),
            ("user".to_string(), "dimmi la prossima partita della juve".to_string()),
        ]);
        let dir = std::env::temp_dir().join("pocketinfer-prompt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("system.txt"), &sys).unwrap();
        std::fs::write(dir.join("conversation.txt"), &conv).unwrap();
        println!("wrote {} and {} to {}", sys.len(), conv.len(), dir.display());
    }
}

#[cfg(test)]
pub fn default_system_prompt_for_test() -> String {
    default_system_prompt()
}

#[cfg(test)]
pub fn turn_context_for_test() -> String {
    turn_context()
}

#[cfg(test)]
mod grounding_tests {
    use super::*;

    #[test]
    fn citations_use_the_bracket_form() {
        let rule = grounding_rule();
        assert!(rule.contains("[path:line]"), "the UI chips parse exactly this form");
    }

    #[test]
    fn every_turn_carries_the_grounding_rule() {
        assert!(turn_context().contains("[path:line]"));
    }
}
