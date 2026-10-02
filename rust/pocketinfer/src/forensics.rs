use core::ffi::c_void;
use core::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Mutex, Once, OnceLock};

pub const PHASES: [&str; 11] = [
    "idle", "load_model", "alloc_context", "chat_template", "tokenize",
    "prefill", "generating", "stats_build", "stats_callback", "cleanup", "done",
];

pub const OPS: [&str; 10] = [
    "none", "prefill_decode", "sample", "accept", "to_piece", "jni_str",
    "jni_call", "jni_exc", "gen_decode", "state_save",
];

pub static PHASE: AtomicI32 = AtomicI32::new(0);
pub static LAST_OP: AtomicI32 = AtomicI32::new(0);
pub static GEN_TOKENS: AtomicU64 = AtomicU64::new(0);
pub static PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
pub static THREADS: AtomicI32 = AtomicI32::new(0);
pub static CTX: AtomicI32 = AtomicI32::new(0);

// Paths are frozen before the handler is installed; its only mutable state is
// atomics. Hint strings are read solely by save_state(), never by a handler.
static CRASH_FILE: OnceLock<[u8; 512]> = OnceLock::new();
static STATE_FILE: OnceLock<[u8; 512]> = OnceLock::new();
static MODEL_HINT: Mutex<String> = Mutex::new(String::new());
static BACKEND_HINT: Mutex<String> = Mutex::new(String::new());
static BACKEND_CODE: AtomicI32 = AtomicI32::new(0);
static HANDLERS: Once = Once::new();

extern "C" {
    fn open(path: *const u8, flags: i32, ...) -> i32;
    fn close(fd: i32) -> i32;
    fn write(fd: i32, buf: *const c_void, n: usize) -> isize;
    fn getpid() -> i32;
    fn time(t: *mut i64) -> i64;
}

const O_WRONLY_CREATE_TRUNC: i32 = 1 | 64 | 512;

fn path_cstr(dir: &str, name: &str) -> Option<[u8; 512]> {
    let path = format!("{dir}/{name}");
    if path.len() >= 512 || path.contains('\0') {
        return None;
    }
    let mut bytes = [0; 512];
    bytes[..path.len()].copy_from_slice(path.as_bytes());
    Some(bytes)
}

pub fn set_log_dir(dir: &str) {
    if dir.is_empty() {
        return;
    }
    // On the first call, initialize BOTH paths before registering the signal
    // handler. Subsequent generations reuse the same app-private directory.
    let Some(crash) = path_cstr(dir, "native_crash.txt") else { return };
    let Some(state) = path_cstr(dir, "native_state.txt") else { return };
    HANDLERS.call_once(|| {
        let _ = CRASH_FILE.set(crash);
        let _ = STATE_FILE.set(state);
        install_handlers();
    });
}

pub fn set_model_hint(path: &str) {
    let base = path.rsplit('/').next().unwrap_or(path);
    *MODEL_HINT.lock().unwrap_or_else(|e| e.into_inner()) = base.to_string();
}

pub fn set_backend_hint(s: &str) {
    BACKEND_CODE.store(match s {
        "opencl" => 1,
        "vulkan" => 2,
        _ => 0,
    }, Ordering::Release);
    *BACKEND_HINT.lock().unwrap_or_else(|e| e.into_inner()) = s.to_string();
}

pub fn set_phase(p: usize) {
    PHASE.store(p as i32, Ordering::SeqCst);
    save_state();
}

pub fn save_state() {
    let Some(path) = STATE_FILE.get() else { return };
    let fd = unsafe { open(path.as_ptr(), O_WRONLY_CREATE_TRUNC, 0o644) };
    if fd < 0 {
        return;
    }
    let p = PHASE.load(Ordering::Relaxed).max(0) as usize;
    let phase = PHASES.get(p).copied().unwrap_or("idle");
    let model = MODEL_HINT.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let backend = BACKEND_HINT.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let state = format!(
        "pid={}\nphase={phase}\ndetail=tokens={} model={model}\nts={}\nbackend={backend}\nthreads={}\n",
        unsafe { getpid() }, GEN_TOKENS.load(Ordering::Relaxed),
        unsafe { time(core::ptr::null_mut()) } * 1000, THREADS.load(Ordering::Relaxed),
    );
    unsafe {
        let _ = write(fd, state.as_ptr() as *const c_void, state.len());
        close(fd);
    }
}

// The Android crash handler must not allocate, format strings, acquire locks
// or call Rust logging: all of those can deadlock if a signal interrupts the
// allocator itself. The last state snapshot includes model/backend strings.
#[cfg(target_os = "android")]
unsafe fn write_signal_bytes(fd: i32, bytes: &[u8]) {
    let mut offset = 0;
    while offset < bytes.len() {
        let count = write(fd, bytes.as_ptr().add(offset) as *const c_void, bytes.len() - offset);
        if count <= 0 {
            break;
        }
        offset += count as usize;
    }
}

#[cfg(target_os = "android")]
unsafe fn write_signal_num(fd: i32, mut value: u64) {
    let mut digits = [0u8; 20];
    let mut offset = digits.len();
    loop {
        offset -= 1;
        digits[offset] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    write_signal_bytes(fd, &digits[offset..]);
    write_signal_bytes(fd, b"\n");
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
        fn signal(sig: i32, handler: usize) -> usize;
        fn raise(sig: i32) -> i32;
    }
    extern "C" fn handler(sig: i32) {
        unsafe {
            signal(sig, 0); // Restore the default disposition before logging.
            if let Some(path) = CRASH_FILE.get() {
                let fd = open(path.as_ptr(), O_WRONLY_CREATE_TRUNC, 0o644);
                if fd >= 0 {
                    write_signal_bytes(fd, b"signal=");
                    write_signal_num(fd, sig as u64);
                    write_signal_bytes(fd, b"phase=");
                    let phase = PHASE.load(Ordering::Relaxed).max(0) as usize;
                    write_signal_bytes(fd, PHASES.get(phase).copied().unwrap_or("idle").as_bytes());
                    write_signal_bytes(fd, b"\nlast_op=");
                    let op = LAST_OP.load(Ordering::Relaxed).max(0) as usize;
                    write_signal_bytes(fd, OPS.get(op).copied().unwrap_or("none").as_bytes());
                    write_signal_bytes(fd, b"\nbackend=");
                    let backend = match BACKEND_CODE.load(Ordering::Acquire) {
                        1 => b"opencl" as &[u8],
                        2 => b"vulkan" as &[u8],
                        _ => b"cpu" as &[u8],
                    };
                    write_signal_bytes(fd, backend);
                    write_signal_bytes(fd, b"\n");
                    write_signal_bytes(fd, b"gen_tokens=");
                    write_signal_num(fd, GEN_TOKENS.load(Ordering::Relaxed));
                    write_signal_bytes(fd, b"prompt_tokens=");
                    write_signal_num(fd, PROMPT_TOKENS.load(Ordering::Relaxed));
                    write_signal_bytes(fd, b"threads=");
                    write_signal_num(fd, THREADS.load(Ordering::Relaxed).max(0) as u64);
                    write_signal_bytes(fd, b"ctx=");
                    write_signal_num(fd, CTX.load(Ordering::Relaxed).max(0) as u64);
                    close(fd);
                }
            }
            raise(sig);
        }
    }
    let sa = SigAction {
        handler: handler as *const () as usize,
        mask: [0; 16], flags: 0, restorer: 0,
    };
    unsafe {
        for sig in [11, 6, 7, 4, 8] {
            sigaction(sig, &sa, core::ptr::null_mut());
        }
    }
}

#[cfg(not(target_os = "android"))]
fn install_handlers() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_snapshot_keeps_backend_and_model_for_diagnostics() {
        let dir = std::env::temp_dir().join(format!("pocketinfer-forensics-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        set_log_dir(dir.to_str().unwrap());
        set_model_hint("/tmp/MiniCPM5-Q4_K_M.gguf");
        set_backend_hint("opencl");
        set_phase(5);
        let snapshot = std::fs::read_to_string(dir.join("native_state.txt")).unwrap();
        assert!(snapshot.contains("phase=prefill\n"));
        assert!(snapshot.contains("model=MiniCPM5-Q4_K_M.gguf"));
        assert!(snapshot.contains("backend=opencl\n"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
