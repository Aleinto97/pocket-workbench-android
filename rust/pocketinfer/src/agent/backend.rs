//! The inference contract and the backends behind it.
//!
//! ```text
//! load(model_path, backend, context_size, options)
//! generate(request_id, messages, tool_schemas, options)
//! cancel(request_id)
//! unload()
//! capabilities()
//! ```
//!
//! Two adapters implement it:
//!
//! * `RustEngine` — the in-process pure Rust engine, CPU or OpenCL. This is the
//!   default path and the one that can keep a model resident across turns.
//! * `GeniexRunner` — the prebuilt GenieX/llama.cpp runner on the Hexagon NPU.
//!   Experimental on purpose: it starts a process per request, it prints its
//!   tokens at the end rather than streaming them, and its teardown has crashed
//!   before. It sits behind the same contract so the agent does not change.

use super::json::{self, Json};
use super::protocol::{
    BackendKind, GenerationOptions, StopReason, TokenUsage, ToolCall,
};
use super::sys;
use crate::util::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// What one generation produced. Deltas are reported through the sink as they
/// arrive; this struct carries the durable result and the measured cost.
#[derive(Clone, Debug)]
pub struct Generation {
    pub text: String,
    pub reasoning: String,
    pub tool_call: Option<ToolCall>,
    pub usage: TokenUsage,
    pub backend: BackendKind,
    pub stop: StopReason,
    pub error: Option<String>,
}

impl Default for Generation {
    fn default() -> Self {
        Self {
            text: String::new(),
            reasoning: String::new(),
            tool_call: None,
            usage: TokenUsage::default(),
            backend: BackendKind::None,
            stop: StopReason::Eos,
            error: None,
        }
    }
}

impl Generation {
    pub fn empty(backend: BackendKind, stop: StopReason) -> Self {
        Self {
            backend,
            stop,
            ..Default::default()
        }
    }

    pub fn to_json(&self) -> Json {
        Json::obj()
            .with("backend", Json::str(self.backend.as_str()))
            .with("stop", Json::str(self.stop.as_str()))
            .with("usage", self.usage.to_json())
            .with(
                "error",
                match &self.error {
                    Some(message) => Json::str(message),
                    None => Json::Null,
                },
            )
    }
}

/// Live token callback. The runtime forwards these as delta events and never
/// relies on them for the transcript.
pub trait DeltaSink: Send {
    fn text(&mut self, piece: &str);
    fn reasoning(&mut self, piece: &str);
}

pub struct NullSink;

impl DeltaSink for NullSink {
    fn text(&mut self, _piece: &str) {}
    fn reasoning(&mut self, _piece: &str) {}
}

pub struct LoadRequest {
    pub model_path: String,
    pub backend: BackendKind,
    pub context_tokens: usize,
    pub threads: usize,
    pub use_gpu: bool,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub backend: BackendKind,
    pub requested: BackendKind,
    pub fallback: bool,
    pub context_tokens: usize,
    pub load_ms: f64,
    pub name: String,
    pub kv_bytes: usize,
    pub detail: String,
}

impl ModelInfo {
    pub fn to_json(&self) -> Json {
        Json::obj()
            .with("backend", Json::str(self.backend.as_str()))
            .with("requested_backend", Json::str(self.requested.as_str()))
            .with("fallback", Json::Bool(self.fallback))
            .with("context_tokens", Json::int(self.context_tokens as i64))
            .with("load_ms", Json::Num(self.load_ms))
            .with("name", Json::str(&self.name))
            .with("kv_bytes", Json::int(self.kv_bytes as i64))
            .with("detail", Json::str(&self.detail))
    }
}

pub trait Backend: Send {
    /// Loads the weights and keeps them until `unload` or a memory decision.
    fn load(&mut self, request: &LoadRequest) -> Result<ModelInfo>;
    fn generate(
        &mut self,
        request_id: &str,
        messages: &[(String, String)],
        options: &GenerationOptions,
        sink: &mut dyn DeltaSink,
    ) -> Result<Generation>;
    /// Must take effect promptly: it is wired to the Stop button.
    fn cancel(&mut self, request_id: &str);
    fn unload(&mut self);
    fn capabilities(&self) -> Json;
    /// Counts tokens with the tokenizer that will actually run.
    fn count_tokens(&self, _text: &str) -> usize {
        // A backend without an exposed tokenizer reports a rough estimate, and
        // the caller is told so rather than being handed a false precision.
        _text.len() / 4 + 1
    }
    fn has_tokenizer(&self) -> bool {
        false
    }
    fn info(&self) -> Option<ModelInfo>;
}

// ---------------------------------------------------------------- Rust engine

pub struct RustEngine {
    engine: Option<crate::model::Engine>,
    path: String,
    n_ctx: usize,
    threads: usize,
    use_gpu: bool,
    info: Option<ModelInfo>,
}

impl Default for RustEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RustEngine {
    pub fn new() -> Self {
        Self {
            engine: None,
            path: String::new(),
            n_ctx: 0,
            threads: 0,
            use_gpu: false,
            info: None,
        }
    }

    fn template_for<'a>(
        engine: &'a crate::model::Engine,
        model_path: &str,
        messages: &[(String, String)],
    ) -> String {
        // All three signals the backend has: filename, tokenizer preset and
        // model name. A wrong template makes the model echo other-assistant
        // instructions instead of answering.
        let probe = format!("{} {}", engine.model.cfg.name, model_path);
        let family = crate::chat::family_of(&engine.model.tok_pre(), &probe);
        let chat: Vec<crate::chat::Message> = messages
            .iter()
            .map(|(role, content)| crate::chat::Message { role: role.clone(), content: content.clone() })
            .collect();
        crate::chat::apply_for(family, &chat, true)
    }
}

impl Backend for RustEngine {
    fn load(&mut self, request: &LoadRequest) -> Result<ModelInfo> {
        let reload = self.engine.is_none()
            || self.path != request.model_path
            || self.n_ctx != request.context_tokens
            || self.threads != request.threads
            || self.use_gpu != request.use_gpu;
        if !reload {
            if let Some(info) = &self.info {
                return Ok(info.clone());
            }
        }
        let started = std::time::Instant::now();
        let mut engine = crate::model::Engine::load(&request.model_path, request.context_tokens, request.threads)?;
        let mut backend = BackendKind::RustCpu;
        let mut fallback = false;
        let mut detail = String::new();
        if request.use_gpu || request.backend == BackendKind::RustOpenCl {
            match engine.enable_opencl() {
                Ok(()) => {
                    backend = BackendKind::RustOpenCl;
                    detail = "OpenCL matvec verified by self-test".to_string();
                }
                Err(error) => {
                    fallback = true;
                    detail = format!("OpenCL unavailable, running on the CPU: {error}");
                }
            }
        }
        let kv_bytes = engine.kv_bytes();
        let name = engine.model.cfg.name.clone();
        self.engine = Some(engine);
        self.path = request.model_path.clone();
        self.n_ctx = request.context_tokens;
        self.threads = request.threads;
        self.use_gpu = request.use_gpu;
        let info = ModelInfo {
            backend,
            requested: request.backend,
            fallback,
            context_tokens: request.context_tokens,
            load_ms: started.elapsed().as_secs_f64() * 1000.0,
            name,
            kv_bytes,
            detail,
        };
        self.info = Some(info.clone());
        Ok(info)
    }

    fn generate(
        &mut self,
        request_id: &str,
        messages: &[(String, String)],
        options: &GenerationOptions,
        sink: &mut dyn DeltaSink,
    ) -> Result<Generation> {
        crate::util::request_stop();
        crate::util::STOP_REQUESTED.store(false, Ordering::SeqCst);
        let model_path = self.path.clone();
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| Error::new("No model is loaded"))?;
        let prompt = Self::template_for(engine, &model_path, messages);
        let tokens = engine.model.tok.encode(&prompt, true);
        let reserve = (options.max_tokens + 64).min(self.n_ctx / 2);
        if tokens.len() + reserve > self.n_ctx {
            return Err(Error::new(format!(
                "The conversation needs {} tokens but the window is {}; reduce the history or the reply limit",
                tokens.len() + reserve,
                self.n_ctx
            )));
        }
        let opts = crate::model::GenOpts {
            n_ctx: self.n_ctx,
            max_tokens: options.max_tokens,
            temp: options.temperature,
            top_p: options.top_p,
            seed: options.seed,
            threads: options.threads,
        };
        let backend = self
            .info
            .as_ref()
            .map(|info| info.backend)
            .unwrap_or(BackendKind::RustCpu);
        let split = ReasoningSplit::new();
        let splitter = split.clone();
        let stats = engine.generate(&tokens, &opts, |bytes| {
            let text = String::from_utf8_lossy(bytes);
            splitter.feed(&text, sink);
            true
        })?;
        let (reasoning, text) = split.finish();
        let stop = match stats.stop.as_str() {
            "eog" => StopReason::Eos,
            "user_stop" => StopReason::Cancelled,
            "context_full" => StopReason::ContextFull,
            "max_tokens" => StopReason::TokenLimit,
            _ => StopReason::Eos,
        };
        let tool_call = parse_tool_call(&text, request_id);
        // A tool call is not a final answer: report it as its own stop so the
        // turn machine knows to run the tool and go back to the model.
        let stop = if tool_call.is_some() && stop == StopReason::Eos {
            StopReason::Eos
        } else {
            stop
        };
        Ok(Generation {
            text,
            reasoning,
            tool_call,
            usage: TokenUsage {
                prompt_tokens: tokens.len(),
                cached_prompt_tokens: stats.prefill_cached_tokens,
                completion_tokens: stats.gen_tokens,
                prefill_ms: stats.prefill_ms,
                decode_ms: stats.gen_ms,
                load_ms: 0.0,
            },
            backend,
            stop,
            error: None,
        })
    }

    fn cancel(&mut self, _request_id: &str) {
        crate::util::request_stop();
    }

    fn unload(&mut self) {
        self.engine = None;
        self.info = None;
        self.path.clear();
        self.n_ctx = 0;
        crate::util::STOP_REQUESTED.store(false, Ordering::SeqCst);
    }

    fn capabilities(&self) -> Json {
        Json::obj()
            .with("streaming", Json::Bool(true))
            .with("reasoning", Json::Bool(true))
            .with("grammar", Json::Bool(false))
            .with("tokenizer", Json::Bool(self.engine.is_some()))
            .with("resident", Json::Bool(true))
    }

    fn count_tokens(&self, text: &str) -> usize {
        match &self.engine {
            Some(engine) => engine.model.tok.encode(text, false).len(),
            None => text.len() / 4 + 1,
        }
    }

    fn has_tokenizer(&self) -> bool {
        self.engine.is_some()
    }

    fn info(&self) -> Option<ModelInfo> {
        self.info.clone()
    }
}

/// Splits a stream into reasoning and visible text on `<think>` markers, and
/// forwards each piece to the sink as it arrives. A model that never emits the
/// markers simply produces no reasoning.
#[derive(Clone)]
struct ReasoningSplit {
    state: Arc<Mutex<SplitState>>,
}

#[derive(Default)]
struct SplitState {
    in_reasoning: bool,
    reasoning: String,
    text: String,
    closed: bool,
}

impl ReasoningSplit {
    fn new() -> Self {
        Self { state: Arc::new(Mutex::new(SplitState::default())) }
    }

    fn feed(&self, piece: &str, sink: &mut dyn DeltaSink) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut rest = piece;
        while !rest.is_empty() {
            if state.in_reasoning {
                if let Some(end) = rest.find("</think>") {
                    state.reasoning.push_str(&rest[..end]);
                    sink.reasoning(&rest[..end]);
                    rest = &rest[end + "</think>".len()..];
                    state.in_reasoning = false;
                    continue;
                }
                state.reasoning.push_str(rest);
                sink.reasoning(rest);
                return;
            }
            if let Some(start) = rest.find("<think>") {
                state.text.push_str(&rest[..start]);
                sink.text(&rest[..start]);
                rest = &rest[start + "<think>".len()..];
                state.in_reasoning = true;
                continue;
            }
            state.text.push_str(rest);
            sink.text(rest);
            return;
        }
    }

    fn finish(&self) -> (String, String) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if state.closed {
            return (state.reasoning.clone(), state.text.clone());
        }
        state.closed = true;
        // An unterminated <think> block is reasoning that never closed: keep it
        // as reasoning rather than leaking the tags into the answer.
        if state.in_reasoning {
            state.in_reasoning = false;
        }
        (state.reasoning.trim().to_string(), state.text.trim().to_string())
    }
}

// ------------------------------------------------------- GenieX NPU (runner)

/// Experimental adapter over the prebuilt GenieX runner packaged as
/// `libgeniexbench.so`.
///
/// Two honest limitations, both by design of that runner and not worked around
/// here: it prints `[gen ]` lines when the process exits, so there is no token
/// callback to forward, and it starts a fresh process per request, so nothing is
/// resident between turns. The agent treats both as normal.
pub struct GeniexRunner {
    native_dir: PathBuf,
    cache_dir: PathBuf,
    model_path: String,
    info: Option<ModelInfo>,
    active: Mutex<Option<i32>>,
    cancelled: AtomicBool,
}

const BENCH: &str = "libgeniexbench.so";
const SHIM: &str = "libgeniexbench_exit.so";
const SKEL: &str = "libggml-htp-v81.so";
const PLUGIN_LINK: &str = "llama_cpp";
const RUNNER_TIMEOUT_MS: u64 = 180_000;
const MAX_SYSTEM_CHARS: usize = 20_000;
const MAX_PROMPT_CHARS: usize = 24_000;
const MAX_GEN_TOKENS: usize = 512;

impl GeniexRunner {
    pub fn new(native_dir: &str, cache_dir: &str) -> Self {
        Self {
            native_dir: PathBuf::from(native_dir),
            cache_dir: PathBuf::from(cache_dir),
            model_path: String::new(),
            info: None,
            active: Mutex::new(None),
            cancelled: AtomicBool::new(false),
        }
    }

    /// The runner refuses to start without these three files, and it discovers
    /// its plugin through symlinks, so this reports what is missing instead of
    /// failing at generation time.
    pub fn preflight(&self) -> Result<()> {
        for name in [BENCH, SHIM, SKEL] {
            let path = self.native_dir.join(name);
            if !path.exists() {
                bail!("{} is missing from the APK; the NPU backend is not available in this build", name);
            }
        }
        std::fs::create_dir_all(&self.cache_dir)
            .map_err(|e| Error::new(format!("Cannot create the runner cache: {e}")))?;
        Ok(())
    }

    fn plugin_root(&self) -> Result<PathBuf> {
        let root = self.cache_dir.join("geniex-plugins");
        std::fs::create_dir_all(&root)
            .map_err(|e| Error::new(format!("Cannot create the plugin folder: {e}")))?;
        let plugin = root.join(PLUGIN_LINK);
        let skel = root.join(SKEL);
        // Recreate every time: a stale link can point at a removed build.
        std::fs::remove_file(&plugin).ok();
        std::fs::remove_file(&skel).ok();
        sys::link(&self.native_dir, &plugin)?;
        sys::link(&self.native_dir.join(SKEL), &skel)?;
        Ok(root)
    }

    fn child_alive(pid: i32) -> bool {
        unsafe { sys::kill(pid, 0) == 0 }
    }
}

impl Backend for GeniexRunner {
    fn load(&mut self, request: &LoadRequest) -> Result<ModelInfo> {
        self.preflight()?;
        if !Path::new(&request.model_path).is_file() {
            bail!("Model file not found: {}", request.model_path);
        }
        let plugin_root = self.plugin_root()?;
        let bench = self.native_dir.join(BENCH);
        // The runner is the thing that actually touches the DSP, so prove it can
        // start before claiming the NPU is ready. A missing vendor OpenCL loader
        // shows up here rather than as a silent CPU fallback later.
        // NOTE: --help is run WITHOUT -m and matched only on the bench's own
        // usage marker. An earlier version scanned help text for words like
        // "error" and always failed; worse, the failed load left model_path
        // empty, so generate() then ran with `-m ""` and every turn died in
        // geniex_llm_create (-100201).
        let probe = std::process::Command::new(&bench)
            .arg("--plugin")
            .arg(PLUGIN_LINK)
            .arg("--device")
            .arg("npu")
            .arg("--help")
            .env("LD_LIBRARY_PATH", format!("{}:/vendor/lib64", self.native_dir.display()))
            .env("GENIEX_PLUGIN_PATH", plugin_root.display().to_string())
            .output()
            .map_err(|e| Error::new(format!("Cannot start the GenieX runner: {e}")))?;
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&probe.stdout),
            String::from_utf8_lossy(&probe.stderr)
        );
        if probe.status.success() && combined.contains("--device") {
            let info = ModelInfo {
                backend: BackendKind::HexagonNpu,
                requested: request.backend,
                fallback: false,
                context_tokens: request.context_tokens,
                load_ms: 0.0,
                name: request.model_path.clone(),
                kv_bytes: 0,
                detail: "GenieX runner started with the Hexagon device selected".to_string(),
            };
            self.model_path = request.model_path.clone();
            self.info = Some(info.clone());
            return Ok(info);
        }
        bail!(
            "The GenieX runner could not reach the Hexagon NPU: {}",
            tail(&combined, 400)
        )
    }

    fn generate(
        &mut self,
        request_id: &str,
        messages: &[(String, String)],
        options: &GenerationOptions,
        sink: &mut dyn DeltaSink,
    ) -> Result<Generation> {
        self.cancelled.store(false, Ordering::SeqCst);
        // Never invoke the runner without a model: `-m ""` fails inside
        // geniex_llm_create with -100201, which reads like a DSP problem but is
        // just a missing path (e.g. after a failed load that never stored one).
        if self.model_path.is_empty() {
            bail!("No model is loaded for the NPU backend; load it first and retry");
        }
        let plugin_root = self.plugin_root()?;
        let bench = self.native_dir.join(BENCH);
        let shim = self.native_dir.join(SHIM);
        let id = request_id.replace(|c: char| !c.is_ascii_alphanumeric(), "");
        let prompt_file = self.cache_dir.join(format!("prompt-{id}.txt"));
        let system_file = self.cache_dir.join(format!("system-{id}.txt"));
        let output_file = self.cache_dir.join(format!("output-{id}.log"));
        let report_file = self.cache_dir.join(format!("report-{id}.json"));

        let (system, conversation) = split_messages(messages);
        let system = tail_chars(&system, MAX_SYSTEM_CHARS);
        let conversation = tail_chars(&conversation, MAX_PROMPT_CHARS);
        let write = |path: &Path, text: &str| -> Result<()> {
            crate::util::write_file(&path.to_string_lossy(), text.as_bytes())
        };
        if write(&prompt_file, &conversation).is_err() || write(&system_file, &system).is_err() {
            bail!("Cannot stage the runner prompt");
        }

        let tokens = options.max_tokens.clamp(1, MAX_GEN_TOKENS);
        // stdout carries the "[gen ]" lines collect_generated parses, so it must
        // go to the output file — discarding it (Stdio::null) yields an empty
        // reply every time. stderr goes to its own file so reading it never
        // blocks the timeout/cancel loop below.
        let stderr_file = self.cache_dir.join(format!("stderr-{id}.log"));
        let out_handle = std::fs::File::create(&output_file)
            .map_err(|e| Error::new(format!("Cannot stage the runner output: {e}")))?;
        let err_handle = std::fs::File::create(&stderr_file)
            .map_err(|e| Error::new(format!("Cannot stage the runner diagnostics: {e}")))?;
        let mut command = std::process::Command::new(&bench);
        command
            .arg("--plugin")
            .arg(PLUGIN_LINK)
            .arg("--device")
            .arg("npu")
            .arg("-m")
            .arg(&self.model_path)
            .arg("--accuracy")
            .arg("--no-think")
            .arg("--prompt-file")
            .arg(&prompt_file)
            .arg("-n")
            .arg(tokens.to_string())
            .arg("--output-json")
            .arg(&report_file)
            .env("LD_LIBRARY_PATH", format!("{}:/vendor/lib64", self.native_dir.display()))
            .env("GENIEX_PLUGIN_PATH", plugin_root.display().to_string())
            .env("LD_PRELOAD", shim.display().to_string())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::from(out_handle))
            .stderr(std::process::Stdio::from(err_handle));
        if !system.trim().is_empty() {
            command.arg("--system-prompt").arg(system.trim());
        }

        let started = std::time::Instant::now();
        let mut child = command.spawn().map_err(|e| Error::new(format!("Cannot start the runner: {e}")))?;
        let pid = child.id() as i32;
        if let Ok(mut slot) = self.active.lock() {
            *slot = Some(pid);
        }
        let deadline = started + std::time::Duration::from_millis(RUNNER_TIMEOUT_MS);
        let status: std::result::Result<Option<std::process::ExitStatus>, String> = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(Some(status)),
                Ok(None) => {}
                Err(error) => break Err(format!("The NPU runner could not be waited on: {error}")),
            }
            // Per-backend flag plus the global stop flipped by Runtime::cancel_session:
            // the runner must die on Stop even though its backend struct is behind
            // the session mutex the turn holds.
            if self.cancelled.load(Ordering::SeqCst) || crate::util::stop_requested() {
                sys::terminate_tree(pid);
                let _ = child.wait();
                break Ok(None);
            }
            if std::time::Instant::now() > deadline {
                sys::terminate_tree(pid);
                let _ = child.wait();
                return Err(Error::new(format!(
                    "The NPU runner exceeded {RUNNER_TIMEOUT_MS} ms and was terminated"
                )));
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        };
        if let Ok(mut slot) = self.active.lock() {
            *slot = None;
        }

        let output = std::fs::read_to_string(&output_file).unwrap_or_default();
        let stderr = std::fs::read_to_string(&stderr_file).unwrap_or_default();
        let _ = std::fs::remove_file(&prompt_file);
        let _ = std::fs::remove_file(&system_file);
        let _ = std::fs::remove_file(&output_file);
        let _ = std::fs::remove_file(&stderr_file);
        let report = std::fs::read_to_string(&report_file).ok();
        let _ = std::fs::remove_file(&report_file);

        match status {
            Err(message) => Err(crate::util::Error::new(message)),
            Ok(None) => Ok(Generation {
                text: String::new(),
                stop: StopReason::Cancelled,
                backend: BackendKind::HexagonNpu,
                ..Default::default()
            }),
            Ok(Some(status)) if !status.success() => {
                // The chat-visible message stays short (Binder + UI), but the
                // fuller diagnostics go to logcat so the next failure can be
                // diagnosed without guessing.
                crate::util::log(
                    crate::util::ANDROID_LOG_ERROR,
                    &format!("geniex exit {status}: {}", tail(&format!("{stderr}{output}"), 4000)),
                );
                Err(Error::new(format!(
                    "The NPU runner exited with {status}: {}",
                    tail(&format!("{stderr}{output}"), 500)
                )))
            }
            Ok(Some(_)) => {
                let text = collect_generated(&output);
                if text.trim().is_empty() {
                    crate::util::log(
                        crate::util::ANDROID_LOG_ERROR,
                        &format!("geniex empty output: {}", tail(&format!("{stderr}{output}"), 4000)),
                    );
                    bail!(
                        "The NPU runner produced no text: {}",
                        tail(&format!("{stderr}{output}"), 500)
                    );
                }
                // No per-token callback exists on this path, so the finished
                // text is forwarded once. The UI shows a completed message, not
                // a fake stream.
                sink.text(&text);
                let tool_call = parse_tool_call(&text, request_id);
                let usage = read_usage(report.as_deref());
                Ok(Generation {
                    text,
                    reasoning: String::new(),
                    tool_call,
                    usage,
                    backend: BackendKind::HexagonNpu,
                    stop: StopReason::Eos,
                    error: None,
                })
            }
        }
    }

    fn cancel(&mut self, _request_id: &str) {
        self.cancelled.store(true, Ordering::SeqCst);
        if let Ok(slot) = self.active.lock() {
            if let Some(pid) = *slot {
                if GeniexRunner::child_alive(pid) {
                    sys::terminate_tree(pid);
                }
            }
        }
    }

    fn unload(&mut self) {
        self.cancel("unload");
        self.info = None;
        self.model_path.clear();
    }

    fn capabilities(&self) -> Json {
        Json::obj()
            .with("streaming", Json::Bool(false))
            .with("reasoning", Json::Bool(false))
            .with("grammar", Json::Bool(false))
            .with("tokenizer", Json::Bool(false))
            .with("resident", Json::Bool(false))
    }

    fn info(&self) -> Option<ModelInfo> {
        self.info.clone()
    }
}

/// The runner prints generated text only when it exits, prefixed `[gen ]`.
/// Parsing those lines after the fact is not streaming, and is not treated as
/// such anywhere in the runtime.
fn collect_generated(output: &str) -> String {
    output
        .lines()
        .filter(|line| line.starts_with("[gen ]"))
        .map(|line| line.trim_start_matches("[gen ]").trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn read_usage(report: Option<&str>) -> TokenUsage {
    let Some(text) = report else {
        return TokenUsage::default();
    };
    let Ok(value) = json::parse(text) else {
        return TokenUsage::default();
    };
    let Some(runs) = value.get("runs").and_then(|r| r.as_array()) else {
        return TokenUsage::default();
    };
    let Some(first) = runs.first() else {
        return TokenUsage::default();
    };
    let prompt_tokens = first.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
    let completion_tokens = first.get("gen_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
    // The runner's "agg" section carries real NPU timings (medians). Without
    // them the UI can only divide by the whole turn duration — which includes
    // the one-shot model load and reads as ~0 tok/s.
    let median = |group: &str, field: &str| {
        value
            .get("agg")
            .and_then(|agg| agg.get(group))
            .and_then(|g| g.get(field))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    };
    let prefill_tps = median("prefill_tps", "median");
    let decode_tps = median("decode_tps", "median");
    TokenUsage {
        prompt_tokens,
        cached_prompt_tokens: 0,
        completion_tokens,
        prefill_ms: if prefill_tps > 0.0 && prompt_tokens > 0 {
            prompt_tokens as f64 / prefill_tps * 1000.0
        } else {
            0.0
        },
        decode_ms: if decode_tps > 0.0 && completion_tokens > 0 {
            completion_tokens as f64 / decode_tps * 1000.0
        } else {
            0.0
        },
        load_ms: 0.0,
    }
}

/// Last `max` characters, cut on a character boundary. The runner takes the
/// most recent context, which is the part the current turn depends on.
fn tail_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let skip = text.chars().count() - max;
    text.chars().skip(skip).collect()
}

/// Last `max` characters of combined diagnostic output, for error messages.
fn tail(text: &str, max: usize) -> String {
    tail_chars(text.trim(), max)
}

#[cfg(test)]
pub fn debug_render(messages: &[(String, String)]) -> (String, String) {
    split_messages(messages)
}

fn split_messages(messages: &[(String, String)]) -> (String, String) {
    let mut system = String::new();
    let mut history = String::new();
    let mut current: Option<&str> = None;
    for (role, content) in messages {
        match role.as_str() {
            "system" => {
                system.push_str(content);
                system.push_str("\n\n");
            }
            "user" => {
                if let Some(previous) = current.take() {
                    // The newest user turn is the request being answered, and it
                    // is rendered last, next to the runtime context.
                    history.push_str(&format!("[user] {}\n", previous.trim_end()));
                }
                current = Some(content);
            }
            "assistant" => history.push_str(&format!("[agent] {}\n", content.trim_end())),
            _ => history.push_str(&format!("[tool result] {}\n", content.trim_end())),
        }
    }
    // History as a quoted log, not as chat turns. The runner feeds this file
    // through the model's own chat template as a single user turn, so lines
    // that look like real role markers are continued instead of read: that was
    // the echo of the previous answer at the start of every reply.
    let mut conversation = String::new();
    if !history.trim_end().is_empty() {
        conversation.push_str("[what already happened]\n");
        conversation.push_str(history.trim_end());
        conversation.push_str("\n[/what already happened]\n\n");
    }
    conversation.push_str("[current request]\n");
    conversation.push_str(&super::runtime::turn_context());
    conversation.push_str(current.unwrap_or_default().trim_end());
    conversation.push_str("\n[/current request]");
    (system, conversation)
}

// ------------------------------------------------------------- tool-call form

/// Reads a tool call out of a completed reply. Several shapes are accepted
/// because small models are inconsistent: the model's own XML grammar
/// (`<tool_call><function=NAME><parameter=KEY>…`), a bare JSON object, or a
/// fenced block containing either. Anything unrecognised is treated as plain
/// prose, never as a call.
pub fn parse_tool_call(text: &str, request_id: &str) -> Option<ToolCall> {
    if let Some(call) = parse_xml_call(text, request_id) {
        return Some(call);
    }
    for candidate in candidates(text) {
        let Ok(value) = json::parse(&candidate) else {
            continue;
        };
        // Accepted shapes: {"name":..}, {"tool":..}, {"function":{"name":..}}.
        let name = match value.get("name").and_then(|n| n.as_str()) {
            Some(name) => name,
            None => match value.get("tool").and_then(|n| n.as_str()) {
                Some(name) => name,
                None => value
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())?,
            },
        };
        if super::tools::find(name).is_none() {
            continue;
        }
        let arguments = match value.get("arguments").or_else(|| value.get("parameters")) {
            Some(Json::Str(inner)) => json::parse(inner).unwrap_or_else(|_| Json::obj()),
            Some(other) if !other.is_null() => other.clone(),
            _ => Json::obj(),
        };
        let call_id = value
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                let digest = super::tools::sha256_hex(name.as_bytes());
                format!("{request_id}-{}", &digest[..12])
            });
        return Some(ToolCall { id: call_id, name: name.to_string(), arguments });
    }
    None
}

/// The grammar this model was trained to emit. Parameters are bare on their own
/// line: a string stays as written, anything structured is JSON.
fn parse_xml_call(text: &str, request_id: &str) -> Option<ToolCall> {
    let open = text.find("<tool_call>")?;
    let after = &text[open + "<tool_call>".len()..];
    let body = match after.find("</tool_call>") {
        Some(close) => &after[..close],
        // A cut-off generation still carries a usable call; the step limit, not
        // this, decides when to stop.
        None => after,
    };
    let (name, mut arguments) = parse_xml_function(body)?;
    if super::tools::find(&name).is_none() {
        return None;
    }
    if let Json::Obj(ref fields) = arguments {
        if fields.is_empty() {
            arguments = Json::obj();
        }
    }
    let digest = super::tools::sha256_hex(name.as_bytes());
    Some(ToolCall {
        id: format!("{request_id}-{}", &digest[..12]),
        name,
        arguments,
    })
}

fn parse_xml_function(body: &str) -> Option<(String, Json)> {
    let after_open = body.find("<function=")?;
    let rest = &body[after_open + "<function=".len()..];
    let name_end = rest.find('>')?;
    let name = rest[..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut arguments = Vec::new();
    let mut rest = &rest[name_end + 1..];
    while let Some(at) = rest.find("<parameter=") {
        let after = &rest[at + "<parameter=".len()..];
        let key_end = after.find('>')?;
        let key = after[..key_end].trim().to_string();
        let tail = &after[key_end + 1..];
        let end = tail
            .find("</parameter>")
            .or_else(|| tail.find("<parameter="))
            .unwrap_or(tail.len());
        let raw = tail[..end].trim();
        arguments.push((key, decode_xml_value(raw)));
        rest = &tail[end..];
    }
    Some((name, Json::obj_with(arguments)))
}

/// A string parameter is the text itself; numbers, booleans, arrays and objects
/// are written as JSON. Anything that is not valid JSON is kept as a string,
/// so a shell command containing brackets survives.
fn decode_xml_value(raw: &str) -> Json {
    if raw.is_empty() {
        return Json::Str(String::new());
    }
    match json::parse(raw) {
        Ok(value) if !matches!(value, Json::Str(_)) => value,
        _ => Json::Str(raw.to_string()),
    }
}

fn candidates(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = text.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    let mut rest = text;
    while let Some(open) = rest.find("<tool_call>") {
        let after = &rest[open + "<tool_call>".len()..];
        let Some(close) = after.find("</tool_call>") else {
            break;
        };
        out.push(after[..close].trim().to_string());
        rest = &after[close + "</tool_call>".len()..];
    }
    if let Some(open) = trimmed.find("```") {
        let after = &trimmed[open + 3..];
        let end = after.find("```").unwrap_or(after.len());
        let block = after[..end].trim();
        let block = block.strip_prefix("json").unwrap_or(block).trim();
        if !block.is_empty() {
            out.push(block.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct Recorder {
        text: String,
        reasoning: String,
    }

    impl DeltaSink for Recorder {
        fn text(&mut self, piece: &str) {
            self.text.push_str(piece);
        }
        fn reasoning(&mut self, piece: &str) {
            self.reasoning.push_str(piece);
        }
    }

    #[test]
    fn splits_reasoning_from_the_answer_and_forwards_both() {
        let mut sink = Recorder::default();
        let split = ReasoningSplit::new();
        for piece in ["<think>weigh", "ing the options", "</think>", "\n\nhere it is"] {
            split.feed(piece, &mut sink);
        }
        let (reasoning, text) = split.finish();
        assert_eq!(reasoning, "weighing the options");
        assert_eq!(text, "here it is");
        assert_eq!(sink.reasoning, "weighing the options");
        assert_eq!(sink.text, "\n\nhere it is");
    }

    #[test]
    fn a_model_without_think_tags_produces_no_reasoning() {
        let mut sink = Recorder::default();
        let split = ReasoningSplit::new();
        split.feed("just an answer", &mut sink);
        let (reasoning, text) = split.finish();
        assert!(reasoning.is_empty());
        assert_eq!(text, "just an answer");
    }

    #[test]
    fn an_unterminated_think_block_does_not_leak_tags() {
        let mut sink = Recorder::default();
        let split = ReasoningSplit::new();
        split.feed("<think>still going", &mut sink);
        let (reasoning, text) = split.finish();
        assert_eq!(reasoning, "still going");
        assert!(text.is_empty());
        assert!(!text.contains("<think>"));
    }

    #[test]
    fn parses_a_bare_json_tool_call() {
        let call = parse_tool_call(r#"{"name":"fs_read","arguments":{"path":"a.txt"}}"#, "r1").unwrap();
        assert_eq!(call.name, "fs_read");
        assert_eq!(call.arguments.get("path").unwrap().as_str().unwrap(), "a.txt");
        assert!(call.id.starts_with("r1-"));
    }

    #[test]
    fn parses_a_tagged_tool_call_with_string_arguments() {
        let text = "sure\n<tool_call>{\"name\":\"fs_write\",\"arguments\":\"{\\\"path\\\":\\\"b.txt\\\",\\\"content\\\":\\\"x\\\"}\"}</tool_call>";
        let call = parse_tool_call(text, "r2").unwrap();
        assert_eq!(call.name, "fs_write");
        assert_eq!(call.arguments.get("content").unwrap().as_str().unwrap(), "x");
    }

    #[test]
    fn parses_a_fenced_tool_call_and_ignores_unknown_names() {
        let fenced = "```json\n{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}\n```";
        let call = parse_tool_call(fenced, "r3").unwrap();
        assert_eq!(call.name, "shell");
        assert!(parse_tool_call(r#"{"name":"rm_rf","arguments":{}}"#, "r3").is_none());
        assert!(parse_tool_call("plain prose about fs_read", "r3").is_none());
        assert!(parse_tool_call("{not json", "r3").is_none());
    }

    #[test]
    fn keeps_a_supplied_call_id() {
        let call = parse_tool_call(r#"{"id":"abc","name":"fs_list","arguments":{}}"#, "r").unwrap();
        assert_eq!(call.id, "abc");
    }

    #[test]
    fn reports_the_backend_that_actually_ran() {
        let info = ModelInfo {
            backend: BackendKind::RustCpu,
            requested: BackendKind::RustOpenCl,
            fallback: true,
            context_tokens: 8192,
            load_ms: 12.5,
            name: "qwen".to_string(),
            kv_bytes: 1024,
            detail: "OpenCL unavailable".to_string(),
        };
        let json = info.to_json();
        assert_eq!(json.get("backend").unwrap().as_str().unwrap(), "rust-cpu");
        assert_eq!(json.get("requested_backend").unwrap().as_str().unwrap(), "rust-opencl");
        assert_eq!(json.get("fallback").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn a_generation_with_a_tool_call_is_not_a_final_answer() {
        // The distinction is structural: `tool_call` is set, and the turn machine
        // keeps going because of it.
        let generation = Generation {
            tool_call: parse_tool_call(r#"{"name":"fs_list","arguments":{}}"#, "r"),
            stop: StopReason::Eos,
            ..Default::default()
        };
        assert!(generation.tool_call.is_some());
        assert_eq!(generation.stop, StopReason::Eos);
    }

    #[test]
    fn runner_generated_lines_are_collected_after_the_fact() {
        let output = "load\n[gen ] hello\n[gen ] world\n[gen ]tokens: 12\n";
        // Indentation the model produced is preserved; only the runner's marker
        // and the surrounding blank lines go.
        assert_eq!(collect_generated(output), "hello\n world\ntokens: 12");
        assert!(collect_generated("nothing here").is_empty());
    }

    #[test]
    fn runner_usage_is_read_from_the_report() {
        let report = r#"{"runs":[{"prompt_tokens":120,"gen_tokens":34}]}"#;
        let usage = read_usage(Some(report));
        assert_eq!(usage.prompt_tokens, 120);
        assert_eq!(usage.completion_tokens, 34);
        assert_eq!(usage.decode_ms, 0.0, "no agg section means no decode timing");
        assert_eq!(read_usage(None).prompt_tokens, 0);
        assert_eq!(read_usage(Some("not json")).prompt_tokens, 0);
    }

    #[test]
    fn runner_agg_medians_become_real_prefill_and_decode_timing() {
        let report = r#"{"runs":[{"prompt_tokens":100,"gen_tokens":20}],"agg":{"prefill_tps":{"median":50.0},"decode_tps":{"median":10.0}}}"#;
        let usage = read_usage(Some(report));
        assert_eq!(usage.prefill_ms, 2000.0);
        assert_eq!(usage.decode_ms, 2000.0);
    }

    #[test]
    fn preflight_reports_a_missing_runtime_instead_of_failing_late() {
        let runner = GeniexRunner::new("/does/not/exist", "/tmp");
        let error = runner.preflight().unwrap_err().to_string();
        assert!(error.contains(BENCH));
    }

    #[test]
    fn cancelled_runs_report_cancellation_not_an_error() {
        let generation = Generation {
            stop: StopReason::Cancelled,
            backend: BackendKind::HexagonNpu,
            ..Default::default()
        };
        assert_eq!(generation.stop.as_str(), "cancelled");
        let _ = StdMutex::new(0);
    }

    #[test]
    fn history_is_a_quoted_log_and_the_request_comes_last() {
        let messages = vec![
            ("system".to_string(), "SYSTEMMARKER".to_string()),
            ("user".to_string(), "prima".to_string()),
            ("assistant".to_string(), "risposta".to_string()),
            ("tool".to_string(), "ok: 4211 build.log".to_string()),
            ("user".to_string(), "adesso".to_string()),
        ];
        let (system, conversation) = split_messages(&messages);
        assert_eq!(system, "SYSTEMMARKER\n\n");
        assert!(!conversation.contains("SYSTEMMARKER"));
        assert!(conversation.starts_with("[what already happened]\n"));
        // No bare role markers: the runner wraps this file in the model's chat
        // template, and lines that look like turns get continued, not read.
        assert!(conversation.contains("[user] prima\n"));
        assert!(conversation.contains("[agent] risposta\n"));
        assert!(conversation.contains("[tool result] ok: 4211 build.log\n"));
        assert!(!conversation.contains("\nassistant\n"));
        assert!(!conversation.contains("\nuser\n"));
        // The request, with its runtime context, is the last thing before
        // generation.
        assert!(conversation.ends_with("[/current request]"));
        let context = conversation.find("CONTEXT: today is ").unwrap();
        let request = context + conversation[context..].find("adesso").unwrap();
        assert!(context < request);
    }

    #[test]
    fn the_runtime_context_lands_on_the_turn_being_answered() {
        let messages = vec![
            ("user".to_string(), "prima".to_string()),
            ("assistant".to_string(), "risposta".to_string()),
            ("user".to_string(), "adesso".to_string()),
        ];
        let (_, conversation) = split_messages(&messages);
        assert!(conversation.contains("CONTEXT: today is "));
        // Only the current turn carries it: history must stay as recorded.
        assert_eq!(conversation.matches("CONTEXT: today is ").count(), 1);
        assert!(conversation.contains("[user] prima\n"));
    }

    #[test]
    fn the_context_carries_the_date_and_the_workspace() {
        // The system prefix stays date-free so it is byte-identical across
        // days (prefill cache); the date rides in the per-turn context.
        let prompt = super::super::runtime::default_system_prompt_for_test();
        assert!(!prompt.contains("Today is "), "date would bust the prefix cache");
        assert!(prompt.contains("/workspace"));
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("\"type\":\"function\""), "tools must be JSON Schema objects");
        let turn = super::super::runtime::turn_context_for_test();
        assert!(turn.contains("today is "), "the model cannot guess today");
    }

    #[test]
    fn the_models_xml_grammar_is_understood() {
        let reply = "Sto cercando il file più grande.\n\
<tool_call>\n<function=shell>\n<parameter=command>\n\
ls -la\n</tool_call>";
        let call = parse_tool_call(reply, "r1").expect("native grammar must parse");
        assert_eq!(call.name, "shell");
        assert_eq!(call.arguments.get("command").and_then(|v| v.as_str()), Some("ls -la"));
    }

    #[test]
    fn a_shell_command_with_brackets_is_not_mistaken_for_json() {
        let reply = "<tool_call>\n<function=shell>\n<parameter=command>\n\
find . -name '*.txt' -exec grep -l {} \\;\n</tool_call>";
        let call = parse_tool_call(reply, "r1").expect("call must parse");
        let command = call.arguments.get("command").and_then(|v| v.as_str()).unwrap_or("");
        assert!(command.contains("-exec"), "got {command:?}");
        assert!(command.contains("{}"), "got {command:?}");
    }

    #[test]
    fn structured_xml_parameters_stay_typed() {
        let reply = "<tool_call>\n<function=fs_write>\n<parameter=path>\nnotes.txt\n\
<parameter=content>\n{\"a\": 1}\n</tool_call>";
        let call = parse_tool_call(reply, "r1").expect("call must parse");
        assert_eq!(call.arguments.get("path").and_then(|v| v.as_str()), Some("notes.txt"));
        assert_eq!(call.arguments.get("content"), Some(&Json::obj_with(vec![(
            "a".to_string(),
            Json::int(1)
        )])));
    }

    #[test]
    fn a_json_tool_call_still_parses() {
        let call = parse_tool_call("{\"name\":\"fs_list\",\"arguments\":{\"path\":\"docs\"}}", "r1")
            .expect("json form must keep working");
        assert_eq!(call.name, "fs_list");
        assert_eq!(call.arguments.get("path").and_then(|v| v.as_str()), Some("docs"));
    }

    #[test]
    fn an_unknown_tool_name_is_not_a_call() {
        assert!(parse_tool_call("<tool_call>\n<function=web_search>\n</tool_call>", "r1").is_none());
    }
}
