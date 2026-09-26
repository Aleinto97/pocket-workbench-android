use core::sync::atomic::{AtomicI32, AtomicU64, Ordering};

pub const PHASES: [&str; 11] = [
    "idle",
    "load_model",
    "alloc_context",
    "chat_template",
    "tokenize",
    "prefill",
    "generating",
    "stats_build",
    "stats_callback",
    "cleanup",
    "done",
];

pub const OPS: [&str; 10] = [
    "none",
    "prefill_decode",
    "sample",
    "accept",
    "to_piece",
    "jni_str",
    "jni_call",
    "jni_exc",
    "gen_decode",
    "state_save",
];

pub static PHASE: AtomicI32 = AtomicI32::new(0);
pub static LAST_OP: AtomicI32 = AtomicI32::new(0);
pub static GEN_TOKENS: AtomicU64 = AtomicU64::new(0);
pub static PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
pub static THREADS: AtomicI32 = AtomicI32::new(0);
pub static CTX: AtomicI32 = AtomicI32::new(0);

static mut CRASH_FILE: [u8; 512] = [0; 512];
static mut STATE_FILE: [u8; 512] = [0; 512];
static mut MODEL_HINT: [u8; 128] = [0; 128];
static mut BACKEND_HINT: [u8; 24] = [0; 24];
static HANDLERS: AtomicI32 = AtomicI32::new(0);

extern "C" {
    fn open(path: *const u8, flags: i32, mode: u32) -> i32;
    fn close(fd: i32) -> i32;
    fn write(fd: i32, buf: *const u8, n: usize) -> isize;
    fn getpid() -> i32;
    fn time(t: *mut i64) -> i64;
}

fn set_cstr<const N: usize>(dst: &mut [u8; N], s: &str) {
    let b = s.as_bytes();
    let n = b.len().min(dst.len() - 1);
    dst[..n].copy_from_slice(&b[..n]);
    dst[n] = 0;
}

pub fn set_log_dir(dir: &str) {
    unsafe {
        if !dir.is_empty() {
            set_cstr(&mut CRASH_FILE, &format!("{dir}/native_crash.txt"));
            set_cstr(&mut STATE_FILE, &format!("{dir}/native_state.txt"));
            if HANDLERS.swap(1, Ordering::SeqCst) == 0 {
                install_handlers();
            }
        }
    }
}

pub fn set_model_hint(path: &str) {
    let base = path.rsplit('/').next().unwrap_or(path);
    unsafe {
        set_cstr(&mut MODEL_HINT, base);
    }
}

pub fn set_backend_hint(s: &str) {
    unsafe {
        let b = s.as_bytes();
        let n = b.len().min(23);
        BACKEND_HINT[..n].copy_from_slice(&b[..n]);
        BACKEND_HINT[n] = 0;
    }
}

pub fn set_phase(p: usize) {
    PHASE.store(p as i32, Ordering::SeqCst);
    save_state();
}

pub fn save_state() {
    unsafe {
        if STATE_FILE[0] == 0 {
            return;
        }
        let fd = open(STATE_FILE.as_ptr(), 1 | 64 | 512, 0o644);
        if fd < 0 {
            return;
        }
        let p = PHASE.load(Ordering::Relaxed).max(0) as usize;
        let phase = PHASES.get(p).copied().unwrap_or("idle");
        let mut s = String::new();
        s.push_str(&format!("pid={}\n", getpid()));
        s.push_str(&format!("phase={phase}\n"));
        s.push_str(&format!(
            "detail=tokens={} model={}\n",
            GEN_TOKENS.load(Ordering::Relaxed),
            cstr_to_str(&MODEL_HINT)
        ));
        s.push_str(&format!("ts={}\n", time(core::ptr::null_mut()) * 1000));
        s.push_str(&format!(
            "backend={}\nthreads={}\n",
            cstr_to_str(&BACKEND_HINT),
            THREADS.load(Ordering::Relaxed)
        ));
        let _ = write(fd, s.as_ptr(), s.len());
        close(fd);
    }
}

fn cstr_to_str(buf: &[u8]) -> String {
    let n = buf.iter().position(|c| *c == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

#[cfg(target_os = "android")]
fn install_handlers() {
    #[repr(C)]
    struct SigAction {
        handler: usize,
        mask: [u64; 16],
        flags: i32,
        restorer: usize,
    }
    extern "C" {
        fn sigaction(sig: i32, act: *const SigAction, old: *mut SigAction) -> i32;
    }
    const SA_SIGINFO: i32 = 4;
    extern "C" fn handler(sig: i32, info: *mut core::ffi::c_void, uctx: *mut core::ffi::c_void) {
        let _ = (info, uctx);
        unsafe {
            if CRASH_FILE[0] != 0 {
                let fd = open(CRASH_FILE.as_ptr(), 1 | 64 | 512, 0o644);
                if fd >= 0 {
                    let p = PHASE.load(Ordering::Relaxed).max(0) as usize;
                    let op = LAST_OP.load(Ordering::Relaxed).max(0) as usize;
                    let mut s = String::new();
                    s.push_str(&format!("signal={sig}\n"));
                    s.push_str(&format!(
                        "phase={}\n",
                        PHASES.get(p).copied().unwrap_or("idle")
                    ));
                    s.push_str(&format!("last_op={}\n", OPS.get(op).copied().unwrap_or("none")));
                    s.push_str(&format!(
                        "gen_tokens={}\nprompt_tokens={}\n",
                        GEN_TOKENS.load(Ordering::Relaxed),
                        PROMPT_TOKENS.load(Ordering::Relaxed)
                    ));
                    s.push_str(&format!(
                        "backend={}\nthreads={}\nctx={}\n",
                        cstr_to_str(&BACKEND_HINT),
                        THREADS.load(Ordering::Relaxed),
                        CTX.load(Ordering::Relaxed)
                    ));
                    s.push_str(&format!("model={}\n", cstr_to_str(&MODEL_HINT)));
                    let _ = write(fd, s.as_ptr(), s.len());
                    close(fd);
                }
            }
        }
        extern "C" {
            fn signal(sig: i32, handler: usize) -> usize;
            fn raise(sig: i32) -> i32;
        }
        const SIG_DFL: usize = 0;
        unsafe {
            signal(sig, SIG_DFL);
            raise(sig);
        }
    }
    unsafe {
        let mut sa = SigAction { handler: handler as usize, mask: [0; 16], flags: SA_SIGINFO, restorer: 0 };
        for sig in [11, 6, 7, 4, 8] {
            sigaction(sig, &sa, core::ptr::null_mut());
        }
    }
}

#[cfg(not(target_os = "android"))]
fn install_handlers() {}
