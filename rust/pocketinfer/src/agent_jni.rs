//! JNI surface of the agent runtime.
//!
//! Everything crosses as a JSON string in one direction or the other. That keeps
//! the boundary small and versioned (`protocol.version`) rather than spreading a
//! dozen JNI signatures that would all have to change together.
//!
//! The Kotlin side owns the worker thread: it calls `submit` to admit a message
//! and `pump` to run queued turns, then `drainEvents` to collect a batch. This
//! layer never blocks on a turn.

use super::agent::backend::{Backend, GeniexRunner, LoadRequest, RustEngine};
use super::agent::json::{self, Json};
use super::agent::protocol::{BackendKind, EventKind, GenerationOptions, PROTOCOL_VERSION, TurnLimits};
use super::agent::runtime::{
    drain_batch, tool_brief, Runtime, Session, SessionConfig,
};
use super::agent::tools::{grant_list, Grants};
use super::jni::{JString, Jni};
use super::util::Result;
use std::path::PathBuf;
use std::sync::OnceLock;

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::new().unwrap_or_else(|_| Runtime::default()))
}

/// Directories the GenieX runner needs, captured from the open-session call.
/// They never change for the lifetime of the process, so first-writer-wins is
/// safe here.
static BACKEND_DIRS: OnceLock<(String, String)> = OnceLock::new();

fn make_backend(choice: BackendKind) -> Box<dyn Backend> {
    match BACKEND_DIRS.get().cloned() {
        Some((native, cache)) if choice == BackendKind::HexagonNpu => {
            Box::new(GeniexRunner::new(&native, &cache))
        }
        _ => Box::new(RustEngine::new()),
    }
}

macro_rules! answer {
    ($env:expr, $jni:expr, $value:expr) => {{
        let owned: String = $value;
        let js = $jni.jstring(&owned);
        if js.is_null() {
            return core::ptr::null_mut();
        }
        js
    }};
}

/// Runs `$body` with a JNI handle, turning a panic into a Java exception rather
/// than letting it cross the FFI boundary.
fn guarded<F>(env: *mut *const super::jni::JniTable, body: F) -> JString
where
    F: FnOnce(&Jni) -> JString,
{
    let jni = Jni::new(env);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(&jni)));
    match result {
        Ok(value) => value,
        Err(_) => {
            jni.throw_state("The native agent runtime panicked; the turn was aborted");
            core::ptr::null_mut()
        }
    }
}

fn ok_json(fields: Vec<(&str, Json)>) -> Json {
    let mut value = Json::obj().with("ok", Json::Bool(true));
    for (key, item) in fields {
        value.set(key, item);
    }
    value
}

fn fail_json(message: &str) -> Json {
    Json::obj()
        .with("ok", Json::Bool(false))
        .with("error", Json::str(message))
}

fn json_result(result: Result<Json>) -> String {
    match result {
        Ok(value) => value.to_string(),
        Err(error) => fail_json(&error.to_string()).to_string(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeProtocolVersion(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
) -> i32 {
    let _ = env;
    PROTOCOL_VERSION as i32
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeCapabilitiesJson(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
) -> JString {
    let jni = Jni::new(env);
    answer!(
        env,
        jni,
        ok_json(vec![
            ("protocol_version", Json::int(PROTOCOL_VERSION as i64)),
            ("tools", super::agent::tools::schemas_json()),
            ("tool_brief", Json::str(tool_brief())),
            (
                "backends",
                Json::Arr(vec![
                    Json::obj().with("id", Json::str(BackendKind::RustCpu.as_str()))
                        .with("label", Json::str("CPU (in-process Rust engine)"))
                        .with("resident", Json::Bool(true))
                        .with("streaming", Json::Bool(true)),
                    Json::obj().with("id", Json::str(BackendKind::RustOpenCl.as_str()))
                        .with("label", Json::str("OpenCL GPU with CPU fallback"))
                        .with("resident", Json::Bool(true))
                        .with("streaming", Json::Bool(true)),
                    Json::obj().with("id", Json::str(BackendKind::HexagonNpu.as_str()))
                        .with("label", Json::str("Hexagon NPU via the GenieX runner (experimental)"))
                        .with("resident", Json::Bool(false))
                        .with("streaming", Json::Bool(false)),
                ]),
            ),
            ("capabilities", Json::obj()
                .with("streaming", Json::Bool(true))
                .with("resident_model", Json::Bool(true))
                .with("cancellation", Json::Bool(true))
                .with("jsonl_session_log", Json::Bool(true))),
        ])
        .to_string()
    )
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeOpenSession(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    session_id: JString,
    log_path: JString,
    workspace_root: JString,
    config_json: JString,
    native_dir: JString,
    cache_dir: JString,
) -> JString {
    guarded(env, |jni| {
        let id = jni.get_string_utf(session_id);
        let log = PathBuf::from(jni.get_string_utf(log_path));
        let workspace = PathBuf::from(jni.get_string_utf(workspace_root));
        let config_text = jni.get_string_utf(config_json);
        let native = jni.get_string_utf(native_dir);
        let cache = jni.get_string_utf(cache_dir);
        let config = build_config(&config_text);
        let choice = config.1.unwrap_or(BackendKind::RustCpu);
        let _ = BACKEND_DIRS.set((native, cache));
        let backend = make_backend(choice);
        let workspace_id = config.0.workspace_id.clone();
        // Sessions publish on the runtime's shared channel, which is the one the
        // service drains.
        let session = Session::open(
            &id,
            &log,
            &workspace,
            config.0,
            runtime().sender(),
            backend,
        );
        let payload = match session {
            Ok(session) => match runtime().register(session) {
                Ok(_) => ok_json(vec![
                    ("session_id", Json::str(&id)),
                    ("workspace_id", Json::str(&workspace_id)),
                ]),
                Err(error) => fail_json(&error.to_string()),
            },
            Err(error) => fail_json(&error.to_string()),
        };
        answer!(env, jni, payload.to_string())
    })
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeCloseSession(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    session_id: JString,
) -> JString {
    guarded(env, |jni| {
        let id = jni.get_string_utf(session_id);
        answer!(env, jni, json_result(runtime().close(&id).map(|()| ok_json(vec![]))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeSessionIds(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
) -> JString {
    let jni = Jni::new(env);
    let mut ids = Json::Arr(Vec::new());
    for id in runtime().ids() {
        ids.push(Json::str(id));
    }
    answer!(env, jni, ok_json(vec![("ids", ids)]).to_string())
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeSubmit(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    session_id: JString,
    request_id: JString,
    text: JString,
) -> JString {
    guarded(env, |jni| {
        let id = jni.get_string_utf(session_id);
        let request = jni.get_string_utf(request_id);
        let message = jni.get_string_utf(text);
        answer!(env, jni, json_result(runtime().with(&id, |session| {
            let admitted = session.submit(&request, &message)?;
            Ok(ok_json(vec![
                ("admitted", Json::Bool(admitted)),
                ("queued", Json::Bool(session.is_busy())),
                ("pending", Json::int(session.queued() as i64)),
            ]))
        })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativePump(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    session_id: JString,
    max_turns: i32,
) -> JString {
    guarded(env, |jni| {
        let id = jni.get_string_utf(session_id);
        let limit = max_turns.max(1) as usize;
        answer!(env, jni, json_result(runtime().with(&id, |session| {
            let mut ran = 0usize;
            while ran < limit && session.pump_one() {
                ran += 1;
            }
            Ok(ok_json(vec![
                ("turns", Json::int(ran as i64)),
                ("pending", Json::int(session.queued() as i64)),
            ]))
        })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeCancel(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    session_id: JString,
) -> JString {
    guarded(env, |jni| {
        let id = jni.get_string_utf(session_id);
        // Non-blocking by design: cancel_session flips the turn flag, the
        // global engine stop and the active shell kill without taking the
        // session mutex, which the running turn holds until it observes the
        // flag. The turn loop then emits turn.ended with reason cancelled.
        answer!(env, jni, json_result(runtime().cancel_session(&id).map(|signalled| {
            ok_json(vec![("signalled", Json::Bool(signalled))])
        })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeLoadModel(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    session_id: JString,
    model_path: JString,
    backend: JString,
    context_tokens: i32,
    threads: i32,
    use_gpu: u8,
) -> JString {
    guarded(env, |jni| {
        let id = jni.get_string_utf(session_id);
        let model = jni.get_string_utf(model_path);
        let requested = BackendKind::parse(&jni.get_string_utf(backend));
        let request = LoadRequest {
            model_path: model,
            backend: requested,
            context_tokens: context_tokens.clamp(2048, 32768) as usize,
            threads: threads.clamp(1, 8) as usize,
            use_gpu: use_gpu != 0,
        };
        answer!(env, jni, json_result(runtime().with(&id, |session| {
            // The session keeps whatever backend it was opened with. If the
            // user picked another one afterwards (e.g. Hexagon NPU for the
            // Qwen3.8 hybrid weights the Rust engine cannot load), swap now —
            // otherwise the stale backend fails on the new model and the choice
            // looks silently ignored. Same kind: no swap, resident model stays.
            if requested != BackendKind::None && session.backend_kind() != requested {
                session.set_backend(make_backend(requested));
            }
            let info = session.load_model(&request)?;
            Ok(ok_json(vec![("info", info.to_json())]))
        })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeUnloadModel(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    session_id: JString,
) -> JString {
    guarded(env, |jni| {
        let id = jni.get_string_utf(session_id);
        answer!(env, jni, json_result(runtime().with(&id, |session| {
            session.unload_model();
            Ok(ok_json(vec![]))
        })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeSessionState(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    session_id: JString,
) -> JString {
    guarded(env, |jni| {
        let id = jni.get_string_utf(session_id);
        let payload = runtime()
            .with(&id, |session| {
                Ok(ok_json(vec![
                    ("session_id", Json::str(&session.id)),
                    ("phase", Json::str(session.phase().as_str())),
                    ("running", Json::Bool(session.running())),
                    ("pending", Json::int(session.queued() as i64)),
                    (
                        "last_stop",
                        match session.last_stop() {
                            Some(reason) => Json::str(reason.as_str()),
                            None => Json::Null,
                        },
                    ),
                    (
                        "model",
                        match session.model_info() {
                            Some(info) => info.to_json(),
                            None => Json::Null,
                        },
                    ),
                    ("context", session.context_breakdown()),
                ]))
            })
            .unwrap_or_else(|error| fail_json(&error.to_string()));
        answer!(env, jni, payload.to_string())
    })
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeDrainEvents(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    max_events: i32,
) -> JString {
    guarded(env, |jni| {
        let batch = drain_batch(runtime(), max_events.clamp(1, 512) as usize);
        let jsonl = super::agent::runtime::batch_to_jsonl(&batch);
        answer!(
            env,
            jni,
            ok_json(vec![("count", Json::int(batch.len() as i64)), ("jsonl", Json::str(jsonl))])
                .to_string()
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeSessionEvents(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    log_path: JString,
) -> JString {
    guarded(env, |jni| {
        let path = PathBuf::from(jni.get_string_utf(log_path));
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        // Every complete line is returned; a truncated tail is simply absent,
        // which is what the reader expects.
        let mut jsonl = String::new();
        let mut last_seq = 0u64;
        let mut dropped = 0usize;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(value) = json::parse(line) else {
                dropped += 1;
                continue;
            };
            if value.get("kind").and_then(|k| k.as_str()) == Some("session") {
                continue;
            }
            if let Some(seq) = value.get("seq").and_then(|s| s.as_i64()) {
                last_seq = last_seq.max(seq as u64);
            }
            value.write(&mut jsonl);
            jsonl.push('\n');
        }
        answer!(
            env,
            jni,
            ok_json(vec![
                ("count", Json::int(jsonl.lines().count() as i64)),
                ("last_seq", Json::int(last_seq as i64)),
                ("skipped_lines", Json::int(dropped as i64)),
                ("jsonl", Json::str(jsonl)),
            ])
            .to_string()
        )
    })
}

/// Reads the durable record and projects it for the UI, so the transcript is
/// derived from the log rather than from a second in-memory copy that could
/// disagree with it.
#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeTranscript(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
    log_path: JString,
) -> JString {
    guarded(env, |jni| {
        let path = PathBuf::from(jni.get_string_utf(log_path));
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let mut projection = TranscriptProjection::default();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(value) = json::parse(line) else {
                continue;
            };
            let Some(event) = super::agent::protocol::AgentEvent::from_json(&value) else {
                continue;
            };
            projection.push(&event);
        }
        answer!(env, jni, projection.to_json().to_string())
    })
}

/// Turns the durable event stream into the items the chat renders. Kept as its
/// own type so the projection rules can be tested without JNI.
#[derive(Default)]
struct TranscriptProjection {
    items: Vec<Json>,
    interrupted: Vec<String>,
    open_turns: Vec<String>,
}

impl TranscriptProjection {
    fn push(&mut self, event: &super::agent::protocol::AgentEvent) {
        use EventKind as K;
        let text = |key: &str| {
            event
                .data
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let base = Json::obj()
            .with("seq", Json::int(event.seq as i64))
            .with("at_ms", Json::int(event.at_ms as i64))
            .with("turn_id", Json::str(&event.turn_id));
        match event.kind {
            K::UserMessageAccepted => {
                let mut item = base
                    .clone()
                    .with("kind", Json::str("user"))
                    .with("text", Json::str(text("text")))
                    .with(
                        "queued",
                        Json::Bool(
                            event
                                .data
                                .get("queued")
                                .and_then(|q| q.as_bool())
                                .unwrap_or(false),
                        ),
                    );
                if let Some(position) = event.data.get("position").and_then(|p| p.as_i64()) {
                    item.set("position", Json::int(position));
                }
                self.items.push(item);
            }
            K::AssistantText => {
                self.items.push(
                    base.clone()
                        .with("kind", Json::str("assistant"))
                        .with("text", Json::str(text("text"))),
                );
            }
            K::AssistantReasoning => {
                self.items.push(
                    base.clone()
                        .with("kind", Json::str("reasoning"))
                        .with("text", Json::str(text("text"))),
                );
            }
            K::ToolCalled => {
                self.items.push(
                    base.clone()
                        .with("kind", Json::str("tool"))
                        .with("call_id", Json::str(text("call_id")))
                        .with("name", Json::str(text("name")))
                        .with("phase", Json::str("called"))
                        .with(
                            "arguments",
                            event
                                .data
                                .get("arguments")
                                .cloned()
                                .unwrap_or_else(Json::obj),
                        ),
                );
            }
            K::ToolResult => {
                let ok = event.data.get("ok").and_then(|o| o.as_bool()).unwrap_or(false);
                self.items.push(
                    base.clone()
                        .with("kind", Json::str("tool"))
                        .with("call_id", Json::str(text("call_id")))
                        .with("name", Json::str(text("name")))
                        .with("phase", Json::str("done"))
                        .with("ok", Json::Bool(ok))
                        .with("result", event.data.get("result").cloned().unwrap_or_else(Json::obj))
                        .with(
                            "error",
                            match event.data.get("error").and_then(|e| e.as_str()) {
                                Some(message) => Json::str(message),
                                None => Json::Null,
                            },
                        )
                        .with(
                            "duration_ms",
                            Json::int(
                                event
                                    .data
                                    .get("duration_ms")
                                    .and_then(|d| d.as_i64())
                                    .unwrap_or(0),
                            ),
                        )
                        .with(
                            "truncated",
                            Json::Bool(
                                event
                                    .data
                                    .get("truncated")
                                    .and_then(|t| t.as_bool())
                                    .unwrap_or(false),
                            ),
                        ),
                );
            }
            K::Error => {
                self.items.push(
                    base.clone()
                        .with("kind", Json::str("error"))
                        .with("class", Json::str(text("class")))
                        .with("message", Json::str(text("message"))),
                );
            }
            K::Cancelled => {
                self.items.push(base.clone().with("kind", Json::str("cancelled")));
            }
            K::Checkpoint => {
                // Microcompaction only shrinks old outputs; it explains nothing
                // to the user, so it stays out of the transcript.
                let sub = event.data.get("kind").and_then(|k| k.as_str()).unwrap_or("");
                if sub == "microcompact" {
                    return;
                }
                self.items.push(
                    base.clone()
                        .with("kind", Json::str("checkpoint"))
                        .with("summary", Json::str(text("summary")))
                        .with(
                            "dropped_items",
                            Json::int(
                                event
                                    .data
                                    .get("dropped_items")
                                    .and_then(|d| d.as_i64())
                                    .unwrap_or(0),
                            ),
                        ),
                );
            }
            K::TurnStarted => self.open_turns.push(event.turn_id.clone()),
            K::TurnEnded => {
                self.open_turns.retain(|id| id != &event.turn_id);
                if event
                    .data
                    .get("interrupted")
                    .and_then(|i| i.as_bool())
                    == Some(true)
                {
                    self.interrupted.push(event.turn_id.clone());
                }
            }
            K::ModelError => {
                self.items.push(
                    base.clone()
                        .with("kind", Json::str("error"))
                        .with("class", Json::str(text("class")))
                        .with("message", Json::str(text("message"))),
                );
            }
            _ => {}
        }
    }

    fn to_json(&self) -> Json {
        Json::obj()
            .with("ok", Json::Bool(true))
            .with("items", Json::Arr(self.items.clone()))
            .with(
                "interrupted_turns",
                Json::Arr(self.interrupted.iter().map(Json::str).collect()),
            )
            .with(
                "open_turns",
                Json::Arr(self.open_turns.iter().map(Json::str).collect()),
            )
    }
}

/// Parses the session config the Kotlin side sends. Unknown fields are ignored
/// so a newer UI can talk to an older runtime.
fn build_config(text: &str) -> (SessionConfig, Option<BackendKind>) {
    let mut config = SessionConfig::default();
    let Ok(value) = json::parse(text) else {
        return (config, None);
    };
    if let Some(prompt) = value.get("system_prompt").and_then(|p| p.as_str()) {
        if !prompt.trim().is_empty() {
            config.system_prompt = prompt.to_string();
        }
    }
    if let Some(family) = value.get("model_family").and_then(|f| f.as_str()) {
        let family = family.trim().to_lowercase();
        if ["qwen", "llama", "minicpm"].contains(&family.as_str()) {
            config.model_family = family;
        }
    }
    if let Some(use_model) = value.get("use_model_summary").and_then(|v| v.as_bool()) {
        config.use_model_summary = use_model;
    }
    if let Some(workspace) = value.get("workspace_id").and_then(|w| w.as_str()) {
        config.workspace_id = workspace.to_string();
    }
    if let Some(grants) = value.get("grants").and_then(|g| g.as_array()) {
        let read = grants.iter().any(|g| g.as_str() == Some(super::agent::tools::CAP_READ));
        let write = grants.iter().any(|g| g.as_str() == Some(super::agent::tools::CAP_WRITE));
        let shell = grants.iter().any(|g| g.as_str() == Some(crate::agent::tools::CAP_SHELL));
        config.grants = Grants { read, write, shell };
    }
    let backend = value
        .get("backend")
        .and_then(|b| b.as_str())
        .map(BackendKind::parse)
        .filter(|backend| *backend != BackendKind::None);
    // Debian module paths. Always accepted when present; the tool checks the
    // tree on every call, so installing afterwards needs no session reopen.
    if let Some(linux) = value.get("linux") {
        let text = |key: &str| {
            linux
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let config_linux = super::agent::tools::LinuxConfig {
            proot: text("proot"),
            rootfs: text("rootfs"),
            tmp: text("tmp"),
            tools: text("tools"),
        };
        if !config_linux.proot.is_empty() {
            config.linux = Some(config_linux);
        }
    }
    if let Some(limits) = value.get("limits") {
        config.limits = TurnLimits {
            max_steps: limits
                .get("max_steps")
                .and_then(|v| v.as_i64())
                .unwrap_or(config.limits.max_steps as i64)
                .clamp(1, 64) as u32,
            turn_timeout_ms: limits
                .get("turn_timeout_ms")
                .and_then(|v| v.as_i64())
                .unwrap_or(config.limits.turn_timeout_ms as i64)
                .clamp(5_000, 3_600_000) as u64,
            max_tool_output_bytes: limits
                .get("max_tool_output_bytes")
                .and_then(|v| v.as_i64())
                .unwrap_or(config.limits.max_tool_output_bytes as i64)
                .clamp(512, 1 << 20) as usize,
            max_context_tokens: limits
                .get("max_context_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(config.limits.max_context_tokens as i64)
                .clamp(1024, 131_072) as usize,
            max_repeated_tool_errors: limits
                .get("max_repeated_tool_errors")
                .and_then(|v| v.as_i64())
                .unwrap_or(config.limits.max_repeated_tool_errors as i64)
                .clamp(1, 16) as u32,
            max_queued_messages: limits
                .get("max_queued_messages")
                .and_then(|v| v.as_i64())
                .unwrap_or(config.limits.max_queued_messages as i64)
                .clamp(0, 64) as usize,
        };
    }
    if let Some(generation) = value.get("generation") {
        config.generation = GenerationOptions {
            max_tokens: generation
                .get("max_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(config.generation.max_tokens as i64)
                .clamp(16, 4096) as usize,
            temperature: generation
                .get("temperature")
                .and_then(|v| v.as_f64())
                .unwrap_or(config.generation.temperature as f64) as f32,
            top_p: generation
                .get("top_p")
                .and_then(|v| v.as_f64())
                .unwrap_or(config.generation.top_p as f64) as f32,
            seed: generation
                .get("seed")
                .and_then(|v| v.as_i64())
                .unwrap_or(config.generation.seed as i64) as u64,
            threads: generation
                .get("threads")
                .and_then(|v| v.as_i64())
                .unwrap_or(config.generation.threads as i64)
                .clamp(1, 8) as usize,
        };
    }
    (config, backend)
}

/// Exposes the grant list so the settings screen can show what is allowed.
#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_AgentRuntime_nativeGrantsJson(
    env: *mut *const super::jni::JniTable,
    _this: super::jni::JObject,
) -> JString {
    let jni = Jni::new(env);
    answer!(env, jni, grant_list(&Grants::all()).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_apply_to_an_empty_document() {
        let (config, backend) = build_config("{}");
        assert_eq!(backend, None);
        assert_eq!(config.limits.max_steps, TurnLimits::default().max_steps);
        assert!(config.grants.allows(crate::agent::tools::CAP_SHELL));
    }

    #[test]
    fn config_reads_limits_generation_and_grants() {
        let (config, backend) = build_config(
            r#"{"backend":"hexagon-npu","workspace_id":"w7",
                "grants":["workspace:read","workspace:write"],
                "limits":{"max_steps":3,"max_context_tokens":2048,"max_queued_messages":1},
                "generation":{"max_tokens":128,"temperature":0.2,"threads":2}}"#,
        );
        assert_eq!(backend, Some(BackendKind::HexagonNpu));
        assert_eq!(config.workspace_id, "w7");
        assert_eq!(config.limits.max_steps, 3);
        assert_eq!(config.limits.max_context_tokens, 2048);
        assert_eq!(config.limits.max_queued_messages, 1);
        assert_eq!(config.generation.max_tokens, 128);
        assert_eq!(config.generation.threads, 2);
        assert!(!config.grants.shell, "the shell capability was not granted");
        assert!(config.grants.write);
    }

    #[test]
    fn out_of_range_limits_are_clamped_not_trusted() {
        let (config, _) = build_config(
            r#"{"limits":{"max_steps":100000,"max_context_tokens":1,"turn_timeout_ms":1},
                "generation":{"max_tokens":999999}}"#,
        );
        assert_eq!(config.limits.max_steps, 64);
        assert_eq!(config.limits.max_context_tokens, 1024);
        assert_eq!(config.limits.turn_timeout_ms, 5_000);
        assert_eq!(config.generation.max_tokens, 4096);
    }

    #[test]
    fn a_malformed_config_falls_back_to_defaults_instead_of_failing() {
        let (config, backend) = build_config("{not json");
        assert_eq!(backend, None);
        assert_eq!(config.limits.max_steps, TurnLimits::default().max_steps);
        assert_eq!(config.limits.max_context_tokens, TurnLimits::default().max_context_tokens);
        assert!(!config.effective_system_prompt().is_empty());
    }

    #[test]
    fn ok_and_fail_payloads_are_shaped_consistently() {
        assert_eq!(ok_json(vec![("a", Json::int(1))]).get("ok").unwrap().as_bool(), Some(true));
        let failure = fail_json("boom");
        assert_eq!(failure.get("ok").unwrap().as_bool(), Some(false));
        assert_eq!(failure.get("error").unwrap().as_str().unwrap(), "boom");
    }

    #[test]
    fn protocol_version_is_one() {
        assert_eq!(PROTOCOL_VERSION, 1);
    }
}
