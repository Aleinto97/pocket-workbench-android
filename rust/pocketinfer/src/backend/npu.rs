//! QNN runtime diagnostics only. A GGUF is not an executable QNN graph.
//! Do not advertise NPU execution until a versioned QNN ABI, model export,
//! tensor I/O and KV state have been implemented and tested on Android.

use crate::util::Result;
use core::ffi::c_void;
use std::path::Path;

extern "C" {
    fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}

const RTLD_NOW: i32 = 2;

pub struct NpuStatus {
    pub context_binary: Option<String>,
    pub runtime_lib: Option<String>,
    pub description: String,
    pub fastrpc: bool,
    /// True only once token generation actually executes on Hexagon.
    pub ready: bool,
}

fn loadable(name: &str) -> bool {
    let mut c_name = Vec::with_capacity(name.len() + 1);
    c_name.extend_from_slice(name.as_bytes());
    c_name.push(0);
    let handle = unsafe { dlopen(c_name.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        return false;
    }
    unsafe { dlclose(handle) };
    true
}

fn context_binary(model_path: &str) -> Option<String> {
    let stem = model_path.strip_suffix(".gguf")?;
    [
        format!("{model_path}.qnn.bin"),
        format!("{stem}.qnn.bin"),
        format!("{stem}.qnn"),
    ]
    .into_iter()
    .find(|p| Path::new(p).is_file())
}

pub fn inspect(model_path: &str) -> NpuStatus {
    let context_binary = context_binary(model_path);
    let fastrpc = ["libcdsprpc.so", "/vendor/lib64/libcdsprpc.so"]
        .into_iter()
        .any(loadable);
    let candidates = ["libQnnHtp.so", "/vendor/lib64/libQnnHtp.so"];
    let runtime_lib = candidates.iter().find(|name| loadable(name)).map(|s| s.to_string());
    let blocked = runtime_lib.is_none()
        && candidates.iter().any(|name| Path::new(name).is_file());
    let mut description = match (&context_binary, &runtime_lib, blocked) {
        (Some(bin), Some(lib), _) => format!(
            "Context {bin} and QNN runtime {lib} detected; QNN execution is not implemented, using CPU/GPU"
        ),
        (Some(bin), None, _) => format!(
            "Context {bin} detected, but QNN execution is not implemented; using CPU/GPU"
        ),
        (None, Some(lib), _) => format!(
            "QNN runtime {lib} detected, but no context is present and execution is not implemented; using CPU/GPU"
        ),
        (None, None, true) =>
            "Vendor QNN library present but inaccessible to the app; QNN execution is not implemented".into(),
        _ => "QNN execution is not implemented; using CPU/GPU".into(),
    };
    if !fastrpc && Path::new("/vendor/lib64/libcdsprpc.so").is_file() {
        description.push_str("; FastRPC is present under /vendor but inaccessible to this app");
    }
    NpuStatus { context_binary, runtime_lib, description, fastrpc, ready: false }
}

pub fn enable(model_path: &str) -> Result<()> {
    Err(crate::err!("{}", inspect(model_path).description))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_never_claims_execution_from_available_libraries() {
        let status = inspect("model.gguf");
        assert!(!status.ready);
        assert!(status.description.contains("not implemented"));
    }

    #[test]
    fn context_file_never_enables_unimplemented_backend() {
        let path = std::env::temp_dir().join(format!("pocketinfer-qnn-{}.gguf", std::process::id()));
        let binary = format!("{}.qnn.bin", path.display());
        std::fs::write(&binary, b"not a valid context").unwrap();
        let status = inspect(path.to_str().unwrap());
        assert_eq!(status.context_binary.as_deref(), Some(binary.as_str()));
        assert!(!status.ready);
        assert!(enable(path.to_str().unwrap()).is_err());
        std::fs::remove_file(binary).unwrap();
    }
}
