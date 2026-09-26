use crate::util::{self, Result};
use core::ffi::c_void;

extern "C" {
    fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;

pub struct NpuStatus {
    pub context_binary: Option<String>,
    pub runtime_lib: Option<String>,
    pub description: String,
}

pub fn inspect(model_path: &str) -> NpuStatus {
    let mut context_binary = None;
    let candidates = [
        format!("{model_path}.qnn.bin"),
        format!("{model_path}.qnn"),
        model_path.replace(".gguf", ".qnn.bin"),
        model_path.replace(".gguf", ".qnn"),
    ];
    for c in &candidates {
        if std::path::Path::new(c).exists() {
            context_binary = Some(c.clone());
            break;
        }
    }
    let lib_candidates = [
        "libQnnHtp.so",
        "libQnnHtpV73Stub.so",
        "libQnnHtpV75Stub.so",
        "libQnnHtpV79Stub.so",
        "/vendor/lib64/libQnnHtp.so",
        "/system/lib64/libQnnHtp.so",
        "/vendor/lib64/libQnnHtpV79Stub.so",
    ];
    let mut runtime_lib = None;
    for c in lib_candidates {
        let mut b = Vec::with_capacity(c.len() + 1);
        b.extend_from_slice(c.as_bytes());
        b.push(0);
        let h = unsafe { dlopen(b.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            runtime_lib = Some(c.to_string());
            break;
        }
    }
    let description = match (&context_binary, &runtime_lib) {
        (Some(bin), Some(lib)) => format!(
            "QNN/Hexagon runtime ready ({lib}) with context binary {bin}; on-device execution is scheduled for the next engine revision"
        ),
        (Some(bin), None) => format!(
            "QNN context binary {bin} found but no libQnnHtp runtime is bundled; keep GPU/CPU"
        ),
        (None, Some(lib)) => format!(
            "{lib} present but no .qnn context binary next to the GGUF; export one with Qualcomm AI Hub to enable the NPU path"
        ),
        (None, None) => "no Qualcomm QNN runtime or context binary on this device".to_string(),
    };
    NpuStatus { context_binary, runtime_lib, description }
}

pub fn enable(model_path: &str) -> Result<()> {
    let st = inspect(model_path);
    util::log(util::ANDROID_LOG_INFO, &format!("NPU: {}", st.description));
    match (st.context_binary, st.runtime_lib) {
        (Some(_), Some(_)) => Err(crate::err!(
            "NPU context binary detected; NPU execution is not enabled in this build (GPU/CPU used)"
        )),
        _ => Err(crate::err!("{}", st.description)),
    }
}
