use crate::chat::{self, Message};
use crate::forensics;
use crate::model::{Engine, GenOpts};
use crate::util;
use core::ffi::c_void;
use std::sync::{Mutex, OnceLock};

type JObject = *mut c_void;
type JClass = *mut c_void;
type JMethodId = *mut c_void;
type JString = *mut c_void;
type JArray = *mut c_void;

#[repr(C)]
struct JniTable {
    entries: [usize; 233],
}

#[repr(C)]
union JValue {
    l: JObject,
    i: i32,
    j: i64,
}

const I_FIND_CLASS: usize = 6;
const I_THROW_NEW: usize = 14;
const I_EXCEPTION_CLEAR: usize = 17;
const I_DELETE_LOCAL_REF: usize = 23;
const I_GET_OBJECT_CLASS: usize = 31;
const I_GET_METHOD_ID: usize = 33;
const I_CALL_VOID_METHOD_A: usize = 63;
const I_NEW_STRING_UTF: usize = 167;
const I_GET_STRING_UTF_CHARS: usize = 169;
const I_RELEASE_STRING_UTF_CHARS: usize = 170;
const I_GET_ARRAY_LENGTH: usize = 171;
const I_GET_OBJECT_ARRAY_ELEMENT: usize = 173;
const I_EXCEPTION_CHECK: usize = 228;

struct Jni {
    env: *mut *const JniTable,
}

impl Jni {
    fn table(&self) -> &JniTable {
        unsafe { &**self.env }
    }

    fn fn_ptr<F: Copy>(&self, idx: usize) -> F {
        let p = self.table().entries[idx];
        unsafe { core::mem::transmute_copy::<usize, F>(&p) }
    }

    fn find_class(&self, name: &str) -> JClass {
        let mut c = Vec::with_capacity(name.len() + 1);
        c.extend_from_slice(name.as_bytes());
        c.push(0);
        let f: extern "C" fn(*mut *const JniTable, *const u8) -> JClass = self.fn_ptr(I_FIND_CLASS);
        f(self.env, c.as_ptr())
    }

    fn throw_new(&self, class: JClass, msg: &str) {
        let c = nul(msg);
        let f: extern "C" fn(*mut *const JniTable, JClass, *const u8) -> i32 =
            self.fn_ptr(I_THROW_NEW);
        f(self.env, class, c.as_ptr());
    }

    fn throw_state(&self, msg: &str) {
        let cls = self.find_class("java/lang/IllegalStateException");
        self.throw_new(cls, msg);
    }

    fn exception_check(&self) -> bool {
        let f: extern "C" fn(*mut *const JniTable) -> u8 = self.fn_ptr(I_EXCEPTION_CHECK);
        f(self.env) != 0
    }

    fn exception_clear(&self) {
        let f: extern "C" fn(*mut *const JniTable) = self.fn_ptr(I_EXCEPTION_CLEAR);
        f(self.env);
    }

    fn get_object_class(&self, obj: JObject) -> JClass {
        let f: extern "C" fn(*mut *const JniTable, JObject) -> JClass =
            self.fn_ptr(I_GET_OBJECT_CLASS);
        f(self.env, obj)
    }

    fn get_method_id(&self, class: JClass, name: &str, sig: &str) -> JMethodId {
        let n = nul(name);
        let s = nul(sig);
        let f: extern "C" fn(*mut *const JniTable, JClass, *const u8, *const u8) -> JMethodId =
            self.fn_ptr(I_GET_METHOD_ID);
        f(self.env, class, n.as_ptr(), s.as_ptr())
    }

    fn call_void_method_a(&self, obj: JObject, mid: JMethodId, args: &[JValue]) {
        let f: extern "C" fn(*mut *const JniTable, JObject, JMethodId, *const JValue) =
            self.fn_ptr(I_CALL_VOID_METHOD_A);
        f(self.env, obj, mid, args.as_ptr());
    }

    fn new_string_utf(&self, modified_utf8: &[u8]) -> JString {
        let c = nul_bytes(modified_utf8);
        let f: extern "C" fn(*mut *const JniTable, *const u8) -> JString =
            self.fn_ptr(I_NEW_STRING_UTF);
        f(self.env, c.as_ptr())
    }

    fn get_string_utf(&self, s: JString) -> String {
        if s.is_null() {
            return String::new();
        }
        let f: extern "C" fn(*mut *const JniTable, JString, *mut u8) -> *const u8 =
            self.fn_ptr(I_GET_STRING_UTF_CHARS);
        let p = f(self.env, s, core::ptr::null_mut());
        if p.is_null() {
            return String::new();
        }
        let mut n = 0usize;
        unsafe {
            while *p.add(n) != 0 {
                n += 1;
            }
        }
        let out = unsafe { core::slice::from_raw_parts(p, n) };
        let res = String::from_utf8_lossy(out).into_owned();
        let rel: extern "C" fn(*mut *const JniTable, JString, *const u8) =
            self.fn_ptr(I_RELEASE_STRING_UTF_CHARS);
        rel(self.env, s, p);
        res
    }

    fn array_len(&self, a: JArray) -> usize {
        let f: extern "C" fn(*mut *const JniTable, JArray) -> i32 = self.fn_ptr(I_GET_ARRAY_LENGTH);
        f(self.env, a).max(0) as usize
    }

    fn array_get(&self, a: JArray, i: usize) -> JObject {
        let f: extern "C" fn(*mut *const JniTable, JArray, i32) -> JObject =
            self.fn_ptr(I_GET_OBJECT_ARRAY_ELEMENT);
        f(self.env, a, i as i32)
    }

    fn delete_local(&self, o: JObject) {
        let f: extern "C" fn(*mut *const JniTable, JObject) = self.fn_ptr(I_DELETE_LOCAL_REF);
        f(self.env, o);
    }
}

fn nul(s: &str) -> Vec<u8> {
    nul_bytes(s.as_bytes())
}

fn nul_bytes(b: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(b.len() + 1);
    v.extend_from_slice(b);
    v.push(0);
    v
}

fn to_modified_utf8(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() + 8);
    let mut i = 0usize;
    let n = bytes.len();
    while i < n {
        let c = bytes[i];
        if c < 0x80 {
            if c == 0 {
                out.push(0xC0);
                out.push(0x80);
            } else {
                out.push(c);
            }
            i += 1;
        } else if c & 0xE0 == 0xC0 && i + 1 < n {
            out.push(c);
            out.push(bytes[i + 1]);
            i += 2;
        } else if c & 0xF0 == 0xE0 && i + 2 < n {
            out.extend_from_slice(&bytes[i..i + 3]);
            i += 3;
        } else if c & 0xF8 == 0xF0 && i + 3 < n {
            let cp = ((c as u32 & 0x07) << 18)
                | ((bytes[i + 1] as u32 & 0x3F) << 12)
                | ((bytes[i + 2] as u32 & 0x3F) << 6)
                | (bytes[i + 3] as u32 & 0x3F);
            i += 4;
            if (0x10000..=0x10FFFF).contains(&cp) {
                let v = cp - 0x10000;
                let pair = [0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)];
                for u in pair {
                    out.push((0xE0 | (u >> 12)) as u8);
                    out.push((0x80 | ((u >> 6) & 0x3F)) as u8);
                    out.push((0x80 | (u & 0x3F)) as u8);
                }
            }
        } else {
            out.extend_from_slice(&[0xEF, 0xBF, 0xBD]);
            i += 1;
        }
    }
    out
}

struct Cached {
    engine: Option<Engine>,
    path: String,
    n_ctx: usize,
    threads: usize,
    backend: &'static str,
}

static CACHE: OnceLock<Mutex<Cached>> = OnceLock::new();

fn cache() -> &'static Mutex<Cached> {
    CACHE.get_or_init(|| {
        Mutex::new(Cached {
            engine: None,
            path: String::new(),
            n_ctx: 0,
            threads: 0,
            backend: "CPU",
        })
    })
}

unsafe fn build_engine(
    model_path: &str,
    n_ctx: usize,
    threads: usize,
    use_gpu: bool,
) -> util::Result<(Engine, &'static str, bool)> {
    let mut engine = Engine::load(model_path, n_ctx, threads)?;
    let mut backend = "CPU";
    let mut fallback = false;
    if use_gpu {
        match engine.enable_opencl() {
            Ok(()) => {
                backend = "OpenCL GPU";
            }
            Err(e) => {
                util::log(util::ANDROID_LOG_WARN, &format!("OpenCL unavailable: {e}"));
                fallback = true;
            }
        }
        if let Err(e) = engine.enable_npu(model_path) {
            util::log(util::ANDROID_LOG_INFO, &format!("NPU: {e}"));
        }
    }
    Ok((engine, backend, fallback))
}

unsafe fn emit_error_stats(
    j: &Jni,
    callback: JObject,
    on_stats: JMethodId,
    backend: &str,
    threads: usize,
    ctx: usize,
    reasoning: &str,
    gpu_fallback: bool,
    error: &str,
) {
    if on_stats.is_null() {
        return;
    }
    let json = format!(
        "{{\"backend\":\"{}\",\"threads\":{},\"ctx\":{},\"reasoning\":\"{}\",\"load_ms\":0.0,\"model_cached\":1,\"gpu_fallback\":{},\"prefill_tokens\":0,\"prefill_ms\":0.0,\"gen_tokens\":0,\"gen_ms\":0.0,\"history_dropped\":0,\"stop\":\"error\",\"error\":\"{}\"}}",
        json_escape_str(backend),
        threads,
        ctx,
        json_escape_str(reasoning),
        gpu_fallback,
        json_escape_str(error)
    );
    let js = j.new_string_utf(&to_modified_utf8(json.as_bytes()));
    if !js.is_null() {
        j.call_void_method_a(callback, on_stats, &[JValue { l: js }]);
        j.delete_local(js);
    }
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_NativeEngine_stop(
    _env: *mut *const JniTable,
    _this: JObject,
) {
    util::request_stop();
}

static PANIC_HOOK: OnceLock<()> = OnceLock::new();

fn install_panic_hook() {
    PANIC_HOOK.get_or_init(|| {
        std::panic::set_hook(Box::new(|info| {
            util::log(util::ANDROID_LOG_ERROR, &format!("engine panic: {info}"));
        }));
    });
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_NativeEngine_generate(
    env: *mut *const JniTable,
    _this: JObject,
    path: JString,
    roles: JArray,
    contents: JArray,
    callback: JObject,
    log_dir: JString,
    threads_j: i32,
    context_j: i32,
    use_gpu_j: u8,
    direct_answer_j: u8,
) {
    install_panic_hook();
    let j = Jni { env };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        generate_impl(
            &j, path, roles, contents, callback, log_dir, threads_j, context_j, use_gpu_j,
            direct_answer_j,
        );
    }));
    if result.is_err() && !j.exception_check() {
        j.throw_state("Native inference engine panicked; generation aborted");
    }
}

unsafe fn generate_impl(
    j: &Jni,
    path: JString,
    roles: JArray,
    contents: JArray,
    callback: JObject,
    log_dir: JString,
    threads_j: i32,
    context_j: i32,
    use_gpu_j: u8,
    direct_answer_j: u8,
) {
    util::request_stop();
    util::STOP_REQUESTED.store(false, core::sync::atomic::Ordering::SeqCst);
    forensics::GEN_TOKENS.store(0, core::sync::atomic::Ordering::SeqCst);
    let model_path = j.get_string_utf(path);
    let logs = j.get_string_utf(log_dir);
    forensics::set_log_dir(&logs);
    forensics::set_model_hint(&model_path);
    let threads = threads_j.clamp(1, 6) as usize;
    let n_ctx = context_j.clamp(4096, 16384) as usize;
    forensics::THREADS.store(threads as i32, core::sync::atomic::Ordering::SeqCst);
    forensics::CTX.store(n_ctx as i32, core::sync::atomic::Ordering::SeqCst);

    let cb_class = j.get_object_class(callback);
    let on_token = j.get_method_id(cb_class, "onToken", "(Ljava/lang/String;)V");
    let on_stats = j.get_method_id(cb_class, "onStats", "(Ljava/lang/String;)V");
    if on_token.is_null() {
        j.throw_state("TokenCallback.onToken not found");
        return;
    }

    let n_roles = j.array_len(roles);
    let n_contents = j.array_len(contents);
    let mut messages = Vec::with_capacity(n_roles.min(n_contents));
    for i in 0..n_roles.min(n_contents) {
        let r = j.array_get(roles, i);
        let c = j.array_get(contents, i);
        let role = j.get_string_utf(r);
        let content = j.get_string_utf(c);
        j.delete_local(r);
        j.delete_local(c);
        messages.push(Message { role, content });
    }

    forensics::set_phase(1);
    let t_load = std::time::Instant::now();
    let mut guard = match cache().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let reload = guard.engine.is_none()
        || guard.path != model_path
        || guard.n_ctx != n_ctx
        || guard.threads != threads;
    let mut loaded_now = false;
    let mut gpu_fallback = false;
    if reload {
        forensics::set_phase(1);
        match build_engine(&model_path, n_ctx, threads, use_gpu_j != 0) {
            Ok((engine, backend, fallback)) => {
                guard.engine = Some(engine);
                guard.path = model_path.clone();
                guard.n_ctx = n_ctx;
                guard.threads = threads;
                guard.backend = backend;
                gpu_fallback = fallback;
                loaded_now = true;
            }
            Err(e) => {
                drop(guard);
                forensics::set_phase(10);
                emit_error_stats(&j, callback, on_stats, "CPU", threads, n_ctx, "automatic", false, &format!("Cannot load GGUF model: {e}"));
                if !j.exception_check() {
                    j.throw_state(&format!("Cannot load GGUF model: {e}"));
                }
                return;
            }
        }
    }
    let load_ms = if loaded_now {
        t_load.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    let backend = guard.backend;
    forensics::set_backend_hint(backend);
    let engine = guard.engine.as_mut().unwrap();

    let minicpm5 = chat::is_minicpm5(
        &engine.model.tok_pre(),
        &engine.model.cfg.name,
    ) || model_path.to_ascii_lowercase().contains("minicpm5");

    forensics::set_phase(3);
    let mut kept: Vec<usize> = (0..messages.len()).collect();
    let mut prompt;
    let mut direct_applied = false;
    let mut tokens;
    loop {
        let view: Vec<Message> = kept.iter().map(|i| messages[*i].clone()).collect();
        prompt = chat::apply(&view, minicpm5, true);
        direct_applied = direct_answer_j != 0
            && minicpm5
            && prompt.ends_with("<|im_start|>assistant\n");
        if direct_applied {
            prompt.push_str(chat::direct_suffix());
        }
        forensics::set_phase(4);
        tokens = engine.model.tok.encode(&prompt, true);
        if tokens.len() <= n_ctx - 1024 {
            break;
        }
        let oldest = if !kept.is_empty() && messages[kept[0]].role == "system" {
            2
        } else {
            1
        };
        if kept.len() <= oldest + 1 {
            drop(guard);
            forensics::set_phase(10);
            j.throw_state(
                "Latest request exceeds the context window. Select a larger window or shorten the message.",
            );
            return;
        }
        kept.remove(oldest);
    }
    let dropped = messages.len() - kept.len();

    forensics::PROMPT_TOKENS.store(tokens.len() as u64, core::sync::atomic::Ordering::SeqCst);
    forensics::set_phase(5);
    let opts = GenOpts {
        n_ctx,
        max_tokens: 2048.min(n_ctx.saturating_sub(tokens.len()).saturating_sub(8)),
        temp: 0.7,
        top_p: 0.95,
        seed: 0xC0FFEE,
        threads,
    };
    let jref = j;
    let mut callback_failed = false;
    let gen_result = engine.generate(&tokens, &opts, |bytes| {
        if callback_failed {
            return false;
        }
        let mu8 = to_modified_utf8(bytes);
        forensics::LAST_OP.store(5, core::sync::atomic::Ordering::SeqCst);
        let js = jref.new_string_utf(&mu8);
        if js.is_null() {
            return false;
        }
        forensics::LAST_OP.store(6, core::sync::atomic::Ordering::SeqCst);
        jref.call_void_method_a(callback, on_token, &[JValue { l: js }]);
        jref.delete_local(js);
        forensics::LAST_OP.store(7, core::sync::atomic::Ordering::SeqCst);
        if jref.exception_check() {
            callback_failed = true;
            return false;
        }
        true
    });

    let stats = match gen_result {
        Ok(s) => s,
        Err(e) => {
            drop(guard);
            forensics::set_phase(10);
            emit_error_stats(
                &j,
                callback,
                on_stats,
                backend,
                threads,
                n_ctx,
                if direct_applied { "direct" } else { "automatic" },
                gpu_fallback,
                &format!("Inference failed: {e}"),
            );
            if !j.exception_check() {
                j.throw_state(&format!("Inference failed: {e}"));
            }
            return;
        }
    };
    forensics::GEN_TOKENS.store(stats.gen_tokens as u64, core::sync::atomic::Ordering::SeqCst);
    forensics::set_phase(7);
    if !on_stats.is_null() && !j.exception_check() {
        let json = format!(
            "{{\"backend\":\"{backend}\",\"threads\":{threads},\"ctx\":{n_ctx},\"reasoning\":\"{}\",\
             \"load_ms\":{load_ms:.1},\"model_cached\":{},\"gpu_fallback\":{gpu_fallback},\
             \"prefill_tokens\":{},\"prefill_ms\":{:.1},\"gen_tokens\":{},\"gen_ms\":{:.1},\
             \"history_dropped\":{dropped},\"stop\":\"{}\"}}",
            if direct_applied { "direct" } else { "automatic" },
            if loaded_now { 0 } else { 1 },
            stats.prefill_tokens,
            stats.prefill_ms,
            stats.gen_tokens,
            stats.gen_ms,
            stats.stop
        );
        forensics::set_phase(8);
        let js = j.new_string_utf(&to_modified_utf8(json.as_bytes()));
        if !js.is_null() {
            j.call_void_method_a(callback, on_stats, &[JValue { l: js }]);
            j.delete_local(js);
        }
    }
    if j.exception_check() {
        j.exception_clear();
    }
    forensics::set_phase(9);
    drop(guard);
    forensics::set_phase(10);
}

fn json_escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_NativeEngine_engineInfo(
    env: *mut *const JniTable,
    _this: JObject,
) -> JString {
    let j = Jni { env };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine_info_json(&j)));
    match result {
        Ok(json) => j.new_string_utf(&to_modified_utf8(json.as_bytes())),
        Err(_) => j.new_string_utf(&to_modified_utf8(b"{\"loaded\":false,\"error\":\"internal panic\"}")),
    }
}

fn engine_info_json(_j: &Jni) -> String {
    let mut caps = String::new();
    caps.push_str(&format!("\"engine\":\"pocketinfer-{}\"", env!("CARGO_PKG_VERSION")));
    caps.push_str(&format!(",\"abi\":\"aarch64\""));
    caps.push_str(",\"neon\":true");
    caps.push_str(&format!(",\"dotprod\":{}", crate::quant_int::int8_available()));
    match crate::backend::opencl::probe() {
        Some(lib) => caps.push_str(&format!(",\"opencl\":\"{}\"", json_escape_str(&lib))),
        None => caps.push_str(",\"opencl\":null"),
    }
    let npu = crate::backend::npu::inspect("");
    caps.push_str(&format!(",\"qnn\":\"{}\"", json_escape_str(&npu.description)));
    let mut model = "null".to_string();
    if let Ok(guard) = cache().lock() {
        if let Some(engine) = guard.engine.as_ref() {
            let cfg = &engine.model.cfg;
            let mut types = String::new();
            for (i, (ttype, bytes, count)) in engine.model.weight_stats().iter().enumerate() {
                if i > 0 {
                    types.push(',');
                }
                types.push_str(&format!(
                    "{{\"type\":{},\"name\":\"{}\",\"bytes\":{},\"tensors\":{}}}",
                    ttype,
                    crate::quant::type_name(*ttype),
                    bytes,
                    count
                ));
            }
            let fname = guard.path.rsplit('/').next().unwrap_or(&guard.path).to_string();
            let fsize = std::fs::metadata(&guard.path).map(|m| m.len()).unwrap_or(0);
            model = format!(
                "{{\"file\":\"{}\",\"file_bytes\":{},\"arch\":\"{}\",\"layers\":{},\"hidden\":{},\"ff\":{},\"heads\":{},\"kv_heads\":{},\"head_dim\":{},\"vocab\":{},\"ctx\":{},\"threads\":{},\"kv_bytes\":{},\"backend\":\"{}\",\"quant\":[{}]}}",
                json_escape_str(&fname),
                fsize,
                json_escape_str(&cfg.arch),
                cfg.n_layer,
                cfg.n_embd,
                cfg.n_ff,
                cfg.n_head,
                cfg.n_head_kv,
                cfg.head_dim,
                engine.model.vocab_size(),
                engine.n_ctx,
                engine.pool.n(),
                engine.kv_bytes(),
                json_escape_str(&guard.backend),
                types
            );
        }
    }
    format!("{{\"loaded\":true,{},\"model\":{}}}", caps, model)
}

#[no_mangle]
pub unsafe extern "C" fn Java_com_pocketworkbench_app_NativeEngine_diagnose(
    env: *mut *const JniTable,
    _this: JObject,
    path: JString,
    threads_j: i32,
    context_j: i32,
) -> JString {
    let j = Jni { env };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let model_path = j.get_string_utf(path);
        let threads = threads_j.clamp(1, 6) as usize;
        let n_ctx = context_j.clamp(4096, 16384) as usize;
        let json = diagnose_json(&model_path, threads, n_ctx);
        j.new_string_utf(&to_modified_utf8(json.as_bytes()))
    }));
    match result {
        Ok(js) => js,
        Err(_) => j.new_string_utf(&to_modified_utf8(b"{\"ok\":false,\"steps\":[{\"name\":\"panic\",\"ok\":false,\"ms\":0,\"error\":\"internal panic\"}]}")),
    }
}

struct Diag {
    steps: Vec<(String, bool, f64, String)>,
    ok: bool,
}

impl Diag {
    fn new() -> Self {
        Self { steps: Vec::new(), ok: true }
    }
    fn run<T>(&mut self, name: &str, f: impl FnOnce() -> util::Result<T>) -> Option<T> {
        let t = std::time::Instant::now();
        match f() {
            Ok(v) => {
                self.steps.push((name.to_string(), true, t.elapsed().as_secs_f64() * 1000.0, String::new()));
                Some(v)
            }
            Err(e) => {
                self.steps.push((name.to_string(), false, t.elapsed().as_secs_f64() * 1000.0, e.msg));
                self.ok = false;
                None
            }
        }
    }
    fn steps_json(&self) -> String {
        let mut out = String::from("[");
        for (i, (name, ok, ms, err)) in self.steps.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"name\":\"{}\",\"ok\":{},\"ms\":{:.1},\"error\":\"{}\"}}",
                json_escape_str(name),
                ok,
                ms,
                json_escape_str(err)
            ));
        }
        out.push(']');
        out
    }
}

fn diagnose_json(model_path: &str, threads: usize, n_ctx: usize) -> String {
    for (i, name) in crate::forensics::PHASES.iter().enumerate() {
        if *name == "load_model" {
            crate::forensics::PHASE.store(i as i32, core::sync::atomic::Ordering::Relaxed);
        }
    }
    let mut d = Diag::new();
    d.run("file", || {
        if model_path.is_empty() {
            bail!("no model selected");
        }
        let md = std::fs::metadata(model_path).map_err(|e| crate::err!("{e}"))?;
        if md.len() < 1024 {
            bail!("file too small ({} bytes)", md.len());
        }
        use std::io::Read;
        let mut f = std::fs::File::open(model_path).map_err(|e| crate::err!("{e}"))?;
        let mut hdr = [0u8; 8];
        f.read_exact(&mut hdr).map_err(|e| crate::err!("{e}"))?;
        let v = crate::gguf::parse_version(&hdr)?;
        Ok((md.len(), v))
    });
    if d.ok {
        let g = d.run("gguf_metadata", || crate::gguf::Gguf::load(model_path));
        if let Some(g) = g {
            d.run("config", || crate::model::Config::from_gguf(&g).map(|c| c));
            let tok = d.run("tokenizer", || {
                let tok = crate::tokenizer::Tokenizer::from_gguf(&g)?;
                let ids = tok.encode("Ciao mondo 123", true);
                if ids.is_empty() {
                    bail!("tokenizer produced no tokens");
                }
                Ok(tok)
            });
            let _ = tok;
            if d.ok {
                let engine = d.run("engine_load", || crate::model::Engine::load(model_path, n_ctx, threads));
                if let Some(mut engine) = engine {
                    let ids = engine.model.tok.encode("Ciao", true);
                    let b = ids.len().max(1);
                    d.run("prefill", || {
                        let positions: Vec<usize> = (0..b).collect();
                        engine.forward(&ids, &positions, true)
                    });
                    d.run("sample", || {
                        let lg = engine.logits();
                        if lg.is_empty() {
                            bail!("no logits produced");
                        }
                        Ok(())
                    });
                }
            }
        }
    }
    let caps = format!(
        "\"engine\":\"pocketinfer-{}\",\"dotprod\":{},\"opencl\":{},\"threads\":{},\"ctx\":{}",
        env!("CARGO_PKG_VERSION"),
        crate::quant_int::int8_available(),
        crate::backend::opencl::probe().map(|s| format!("\"{}\"", json_escape_str(&s))).unwrap_or_else(|| "null".into()),
        threads,
        n_ctx
    );
    format!("{{\"ok\":{},\"caps\":{{{}}},\"steps\":{}}}", d.ok, caps, d.steps_json())
}
