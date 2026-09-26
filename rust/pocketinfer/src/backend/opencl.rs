use crate::quant::{
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
};
use crate::util::{self, Error, Result};
use core::ffi::c_void;
use std::collections::HashMap;

extern "C" {
    fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}

const RTLD_NOW: i32 = 2;

type ClInt = i32;
type ClUint = u32;
type ClPlatformId = *mut c_void;
type ClDeviceId = *mut c_void;
type ClContext = *mut c_void;
type ClQueue = *mut c_void;
type ClMem = *mut c_void;
type ClProgram = *mut c_void;
type ClKernel = *mut c_void;

const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_MEM_READ_ONLY: u64 = 1 << 2;
const CL_MEM_READ_WRITE: u64 = 1 << 0;
const CL_MEM_COPY_HOST_PTR: u64 = 1 << 5;
const CL_TRUE: u32 = 1;

struct Api {
    get_platform_ids: extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt,
    get_device_ids: extern "C" fn(
        ClPlatformId,
        u64,
        ClUint,
        *mut ClDeviceId,
        *mut ClUint,
    ) -> ClInt,
    create_context: extern "C" fn(
        *const i64,
        ClUint,
        *const ClDeviceId,
        extern "C" fn(*const u8, *const c_void, usize, *mut c_void),
        *mut c_void,
        *mut ClInt,
    ) -> ClContext,
    create_command_queue: extern "C" fn(ClContext, ClDeviceId, u64, *mut ClInt) -> ClQueue,
    create_buffer:
        extern "C" fn(ClContext, u64, usize, *mut c_void, *mut ClInt) -> ClMem,
    create_program_with_source: extern "C" fn(ClContext, ClUint, *const *const u8, *const usize, *mut ClInt) -> ClProgram,
    build_program:
        extern "C" fn(ClProgram, ClUint, *const ClDeviceId, *const u8, *mut c_void, *mut c_void) -> ClInt,
    get_program_build_info: extern "C" fn(ClProgram, ClDeviceId, u32, usize, *mut c_void, *mut usize) -> ClInt,
    create_kernel: extern "C" fn(ClProgram, *const u8, *mut ClInt) -> ClKernel,
    set_kernel_arg: extern "C" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt,
    enqueue_nd_range_kernel: extern "C" fn(
        ClQueue,
        ClKernel,
        ClUint,
        *const usize,
        *const usize,
        *const usize,
        ClUint,
        *const c_void,
        *mut c_void,
    ) -> ClInt,
    enqueue_read_buffer: extern "C" fn(ClQueue, ClMem, u32, usize, usize, *mut c_void, ClUint, *const c_void, *mut c_void) -> ClInt,
    enqueue_write_buffer: extern "C" fn(ClQueue, ClMem, u32, usize, usize, *const c_void, ClUint, *const c_void, *mut c_void) -> ClInt,
    finish: extern "C" fn(ClQueue) -> ClInt,
    release_mem_object: extern "C" fn(ClMem) -> ClInt,
    release_kernel: extern "C" fn(ClKernel) -> ClInt,
    release_program: extern "C" fn(ClProgram) -> ClInt,
    release_command_queue: extern "C" fn(ClQueue) -> ClInt,
    release_context: extern "C" fn(ClContext) -> ClInt,
}

pub struct OpenClBackend {
    api: Api,
    lib: *mut c_void,
    ctx: ClContext,
    queue: ClQueue,
    program: ClProgram,
    device: ClDeviceId,
    kernels: HashMap<u32, ClKernel>,
    weight_buffers: HashMap<(usize, usize), ClMem>,
    x_buf: ClMem,
    out_buf: ClMem,
    x_cap: usize,
    out_cap: usize,
    device_name: String,
}

unsafe impl Send for OpenClBackend {}

const SOURCE: &str = r#"
static inline void gsm(int j, __global const uchar* q, uint* d, uint* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}
__kernel void matvec_q4k(__global const uchar* w, __global const float* x, __global float* out, const int nrows, const int k) {
    int r = get_global_id(0);
    if (r >= nrows) return;
    int nb = k / 256;
    __global const uchar* row = w + (size_t)r * (size_t)nb * 144;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        __global const uchar* blk = row + b * 144;
        float d = vload_half(0, (__global const half*)blk);
        float dmin = vload_half(1, (__global const half*)blk);
        __global const uchar* sc = blk + 4;
        __global const uchar* qs = blk + 16;
        __global const float* xb = x + b * 256;
        int is = 0;
        for (int g = 0; g < 8; g += 2) {
            uint s1, m1, s2, m2;
            gsm(is, sc, &s1, &m1);
            gsm(is + 1, sc, &s2, &m2);
            is += 2;
            __global const uchar* q = qs + (g >> 1) * 32;
            __global const float* x1 = xb + g * 32;
            float a1 = 0.f, a2 = 0.f, sx1 = 0.f, sx2 = 0.f;
            for (int l = 0; l < 32; ++l) {
                a1 += (float)(q[l] & 0xF) * x1[l];
                sx1 += x1[l];
                a2 += (float)(q[l] >> 4) * x1[32 + l];
                sx2 += x1[32 + l];
            }
            acc += d * (float)s1 * a1 - dmin * (float)m1 * sx1;
            acc += d * (float)s2 * a2 - dmin * (float)m2 * sx2;
        }
    }
    out[r] = acc;
}
__kernel void matvec_q6k(__global const uchar* w, __global const float* x, __global float* out, const int nrows, const int k) {
    int r = get_global_id(0);
    if (r >= nrows) return;
    int nb = k / 256;
    __global const uchar* row = w + (size_t)r * (size_t)nb * 210;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        __global const uchar* blk = row + b * 210;
        __global const uchar* ql = blk;
        __global const uchar* qh = blk + 128;
        __global const char* sc = (__global const char*)(blk + 192);
        float d = vload_half(0, (__global const half*)(blk + 208));
        __global const float* xb = x + b * 256;
        for (int half = 0; half < 2; ++half) {
            __global const uchar* qlb = ql + half * 64;
            __global const uchar* qhb = qh + half * 32;
            __global const char* scb = sc + half * 8;
            __global const float* x2 = xb + half * 128;
            for (int l = 0; l < 32; ++l) {
                int is = l >> 4;
                int q1 = (int)((qlb[l] & 0xF) | (((qhb[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((qlb[l + 32] & 0xF) | (((qhb[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((qlb[l] >> 4) | (((qhb[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((qlb[l + 32] >> 4) | (((qhb[l] >> 6) & 3) << 4)) - 32;
                acc += d * (float)scb[is] * (float)q1 * x2[l];
                acc += d * (float)scb[is + 2] * (float)q2 * x2[32 + l];
                acc += d * (float)scb[is + 4] * (float)q3 * x2[64 + l];
                acc += d * (float)scb[is + 6] * (float)q4 * x2[96 + l];
            }
        }
    }
    out[r] = acc;
}
__kernel void matvec_q40(__global const uchar* w, __global const float* x, __global float* out, const int nrows, const int k) {
    int r = get_global_id(0);
    if (r >= nrows) return;
    int nb = k / 32;
    __global const uchar* row = w + (size_t)r * (size_t)nb * 18;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        __global const uchar* blk = row + b * 18;
        float d = vload_half(0, (__global const half*)blk);
        __global const uchar* q = blk + 2;
        __global const float* xb = x + b * 32;
        float s = 0.f;
        float sx = 0.f;
        for (int l = 0; l < 16; ++l) {
            s += (float)(q[l] & 0xF) * xb[l] + (float)(q[l] >> 4) * xb[16 + l];
            sx += xb[l] + xb[16 + l];
        }
        acc += d * (s - 8.f * sx);
    }
    out[r] = acc;
}
__kernel void matvec_q80(__global const uchar* w, __global const float* x, __global float* out, const int nrows, const int k) {
    int r = get_global_id(0);
    if (r >= nrows) return;
    int nb = k / 32;
    __global const uchar* row = w + (size_t)r * (size_t)nb * 34;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        __global const uchar* blk = row + b * 34;
        float d = vload_half(0, (__global const half*)blk);
        __global const char* q = (__global const char*)(blk + 2);
        __global const float* xb = x + b * 32;
        float s = 0.f;
        for (int l = 0; l < 32; ++l) s += (float)q[l] * xb[l];
        acc += d * s;
    }
    out[r] = acc;
}
__kernel void matvec_f32(__global const float* w, __global const float* x, __global float* out, const int nrows, const int k) {
    int r = get_global_id(0);
    if (r >= nrows) return;
    __global const float* row = w + (size_t)r * (size_t)k;
    float acc = 0.f;
    for (int l = 0; l < k; ++l) acc += row[l] * x[l];
    out[r] = acc;
}
__kernel void matvec_f16(__global const uchar* w, __global const float* x, __global float* out, const int nrows, const int k) {
    int r = get_global_id(0);
    if (r >= nrows) return;
    __global const half* row = (__global const half*)(w + (size_t)r * (size_t)k * 2);
    float acc = 0.f;
    for (int l = 0; l < k; ++l) acc += vload_half(l, row) * x[l];
    out[r] = acc;
}
"#;

fn sym<T: Copy>(handle: *mut c_void, name: &str) -> Result<T> {
    let mut c = Vec::new();
    c.extend_from_slice(name.as_bytes());
    c.push(0);
    let p = unsafe { dlsym(handle, c.as_ptr()) };
    if p.is_null() {
        bail!("symbol {name} missing");
    }
    Ok(unsafe { core::mem::transmute_copy::<*mut c_void, T>(&p) })
}

pub fn probe() -> Option<String> {
    let candidates = [
        "libOpenCL.so",
        "libOpenCL_adreno.so",
        "/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
        "/vendor/lib/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
    ];
    let mut blocked: Option<String> = None;
    for c in candidates {
        let mut b = Vec::with_capacity(c.len() + 1);
        b.extend_from_slice(c.as_bytes());
        b.push(0);
        let h = unsafe { dlopen(b.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            return Some(c.to_string());
        }
        if blocked.is_none() && std::path::Path::new(c).exists() {
            blocked = Some(format!("{c} (present, blocked by linker namespace)"));
        }
    }
    blocked
}

impl OpenClBackend {
    pub fn new() -> Result<Self> {
        let candidates = [
            "libOpenCL.so",
            "libOpenCL_adreno.so",
            "/vendor/lib64/libOpenCL.so",
            "/system/vendor/lib64/libOpenCL.so",
            "/vendor/lib/libOpenCL.so",
            "/system/lib64/libOpenCL.so",
        ];
        let mut handle = core::ptr::null_mut();
        for c in candidates {
            let mut b = Vec::new();
            b.extend_from_slice(c.as_bytes());
            b.push(0);
            handle = unsafe { dlopen(b.as_ptr(), RTLD_NOW) };
            if !handle.is_null() {
                util::log(util::ANDROID_LOG_INFO, &format!("OpenCL loaded from {c}"));
                break;
            }
        }
        if handle.is_null() {
            bail!("libOpenCL not found");
        }
        let api = Api {
            get_platform_ids: sym(handle, "clGetPlatformIDs")?,
            get_device_ids: sym(handle, "clGetDeviceIDs")?,
            create_context: sym(handle, "clCreateContext")?,
            create_command_queue: sym(handle, "clCreateCommandQueue")?,
            create_buffer: sym(handle, "clCreateBuffer")?,
            create_program_with_source: sym(handle, "clCreateProgramWithSource")?,
            build_program: sym(handle, "clBuildProgram")?,
            get_program_build_info: sym(handle, "clGetProgramBuildInfo")?,
            create_kernel: sym(handle, "clCreateKernel")?,
            set_kernel_arg: sym(handle, "clSetKernelArg")?,
            enqueue_nd_range_kernel: sym(handle, "clEnqueueNDRangeKernel")?,
            enqueue_read_buffer: sym(handle, "clEnqueueReadBuffer")?,
            enqueue_write_buffer: sym(handle, "clEnqueueWriteBuffer")?,
            finish: sym(handle, "clFinish")?,
            release_mem_object: sym(handle, "clReleaseMemObject")?,
            release_kernel: sym(handle, "clReleaseKernel")?,
            release_program: sym(handle, "clReleaseProgram")?,
            release_command_queue: sym(handle, "clReleaseCommandQueue")?,
            release_context: sym(handle, "clReleaseContext")?,
        };
        let mut platform = core::ptr::null_mut();
        let mut n = 0u32;
        let rc = (api.get_platform_ids)(1u32, &mut platform, &mut n);
        if rc != 0 || platform.is_null() {
            bail!("no OpenCL platform ({rc})");
        }
        let mut device = core::ptr::null_mut();
        let rc = (api.get_device_ids)(platform, CL_DEVICE_TYPE_GPU, 1, &mut device, &mut n);
        if rc != 0 || device.is_null() {
            bail!("no OpenCL GPU device ({rc})");
        }
        extern "C" fn ctx_error(_a: *const u8, _b: *const c_void, _c: usize, _d: *mut c_void) {}
        let mut err = 0;
        let ctx = (api.create_context)(core::ptr::null(), 1, &device, ctx_error, core::ptr::null_mut(), &mut err);
        if ctx.is_null() || err != 0 {
            bail!("clCreateContext failed ({err})");
        }
        let queue = (api.create_command_queue)(ctx, device, 0, &mut err);
        if queue.is_null() || err != 0 {
            bail!("clCreateCommandQueue failed ({err})");
        }
        let src_c = SOURCE.as_bytes();
        let len = src_c.len();
        let program = (api.create_program_with_source)(ctx, 1, &src_c.as_ptr(), &len, &mut err);
        if program.is_null() || err != 0 {
            bail!("clCreateProgramWithSource failed ({err})");
        }
        let rc = (api.build_program)(program, 1, &device, core::ptr::null(), core::ptr::null_mut(), core::ptr::null_mut());
        if rc != 0 {
            let mut log = vec![0u8; 8192];
            let mut got = 0usize;
            const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
            (api.get_program_build_info)(program, device, CL_PROGRAM_BUILD_LOG, log.len(), log.as_mut_ptr() as *mut c_void, &mut got);
            let msg = String::from_utf8_lossy(&log[..got.min(log.len())]).into_owned();
            bail!("clBuildProgram failed ({rc}): {msg}");
        }
        let mut kernels = HashMap::new();
        for (ttype, name) in [
            (GGML_TYPE_Q4_K, "matvec_q4k"),
            (GGML_TYPE_Q6_K, "matvec_q6k"),
            (GGML_TYPE_Q4_0, "matvec_q40"),
            (GGML_TYPE_Q8_0, "matvec_q80"),
            (GGML_TYPE_F32, "matvec_f32"),
            (GGML_TYPE_F16, "matvec_f16"),
        ] {
            let cname = nul(name);
            let mut err = 0;
            let k = (api.create_kernel)(program, cname.as_ptr(), &mut err);
            if k.is_null() || err != 0 {
                util::log(util::ANDROID_LOG_WARN, &format!("kernel {name} failed ({err})"));
                continue;
            }
            kernels.insert(ttype, k);
        }
        let x_buf = (api.create_buffer)(ctx, CL_MEM_READ_WRITE, 1 << 20, core::ptr::null_mut(), &mut err);
        let out_buf = (api.create_buffer)(ctx, CL_MEM_READ_WRITE, 1 << 20, core::ptr::null_mut(), &mut err);
        if x_buf.is_null() || out_buf.is_null() {
            bail!("clCreateBuffer for IO failed");
        }
        let device_name = String::from("Adreno OpenCL");
        Ok(Self {
            api,
            lib: handle,
            ctx,
            queue,
            program,
            device,
            kernels,
            weight_buffers: HashMap::new(),
            x_buf,
            out_buf,
            x_cap: 1 << 20,
            out_cap: 1 << 20,
            device_name,
        })
    }

    pub fn name(&self) -> &str {
        &self.device_name
    }

    fn clear_weight_cache(&mut self) {
        for (_, b) in self.weight_buffers.drain() {
            (self.api.release_mem_object)(b);
        }
    }

    pub fn selftest(&mut self) -> Result<()> {
        let k = 256usize;
        let mut x = vec![0f32; k];
        for j in 0..k {
            x[j] = ((j % 7) as f32) * 0.1 - 0.3;
        }
        let mut w4 = vec![0u8; 144];
        w4[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        w4[4] = 1;
        for i in 0..128 {
            w4[16 + i] = 0x21;
        }
        let mut out4 = vec![0f32; 1];
        self.matvec(GGML_TYPE_Q4_K, &w4, k, &x, &mut out4)?;
        let want4 = crate::quant::dot_row(GGML_TYPE_Q4_K, &w4, k, &x);
        if (out4[0] - want4).abs() > want4.abs() * 0.02 + 0.01 {
            self.clear_weight_cache();
            bail!("q4_K self-test failed: gpu={} cpu={}", out4[0], want4);
        }
        let mut w6 = vec![0u8; 210];
        for i in 0..128 {
            w6[i] = 0x95;
        }
        for i in 0..64 {
            w6[128 + i] = 0x1B;
        }
        for i in 0..16 {
            w6[192 + i] = 2;
        }
        w6[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
        let mut out6 = vec![0f32; 1];
        self.matvec(GGML_TYPE_Q6_K, &w6, k, &x, &mut out6)?;
        let want6 = crate::quant::dot_row(GGML_TYPE_Q6_K, &w6, k, &x);
        if (out6[0] - want6).abs() > want6.abs() * 0.02 + 0.01 {
            self.clear_weight_cache();
            bail!("q6_K self-test failed: gpu={} cpu={}", out6[0], want6);
        }
        self.clear_weight_cache();
        Ok(())
    }

    pub fn supports(&self, ttype: u32) -> bool {
        self.kernels.contains_key(&ttype)
    }

    pub fn matvec(&mut self, ttype: u32, w: &[u8], k: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        let kern = *self.kernels.get(&ttype).ok_or_else(|| Error::new("type not supported"))?;
        let nrows = out.len();
        let key = (w.as_ptr() as usize, w.len());
        let wbuf = match self.weight_buffers.get(&key) {
            Some(b) => *b,
            None => {
                let mut err = 0;
                // COPY_HOST_PTR, not USE_HOST_PTR: GGUF tensor slices are only
                // 32-byte aligned inside an mmap, and Adreno's driver can crash
                // on non page-aligned host pointers with USE_HOST_PTR.
                let b = (self.api.create_buffer)(
                    self.ctx,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    w.len(),
                    w.as_ptr() as *mut c_void,
                    &mut err,
                );
                if b.is_null() || err != 0 {
                    bail!("clCreateBuffer weights failed ({err})");
                }
                self.weight_buffers.insert(key, b);
                b
            }
        };
        let xbytes = k * 4;
        let obytes = nrows * 4;
        if xbytes > self.x_cap {
            let mut err = 0;
            let new_buf = (self.api.create_buffer)(self.ctx, CL_MEM_READ_WRITE, xbytes, core::ptr::null_mut(), &mut err);
            if new_buf.is_null() || err != 0 {
                bail!("clCreateBuffer input failed ({err})");
            }
            (self.api.release_mem_object)(self.x_buf);
            self.x_buf = new_buf;
            self.x_cap = xbytes;
        }
        if obytes > self.out_cap {
            let mut err = 0;
            let new_buf = (self.api.create_buffer)(self.ctx, CL_MEM_READ_WRITE, obytes, core::ptr::null_mut(), &mut err);
            if new_buf.is_null() || err != 0 {
                bail!("clCreateBuffer output failed ({err})");
            }
            (self.api.release_mem_object)(self.out_buf);
            self.out_buf = new_buf;
            self.out_cap = obytes;
        }
        let mut err = 0;
        err |= (self.api.enqueue_write_buffer)(self.queue, self.x_buf, CL_TRUE, 0, xbytes, x.as_ptr() as *const c_void, 0, core::ptr::null(), core::ptr::null_mut());
        let nrows_i = nrows as i32;
        let k_i = k as i32;
        err |= (self.api.set_kernel_arg)(kern, 0, core::mem::size_of::<ClMem>(), &wbuf as *const ClMem as *const c_void);
        err |= (self.api.set_kernel_arg)(kern, 1, core::mem::size_of::<ClMem>(), &self.x_buf as *const ClMem as *const c_void);
        err |= (self.api.set_kernel_arg)(kern, 2, core::mem::size_of::<ClMem>(), &self.out_buf as *const ClMem as *const c_void);
        err |= (self.api.set_kernel_arg)(kern, 3, core::mem::size_of::<i32>(), &nrows_i as *const i32 as *const c_void);
        err |= (self.api.set_kernel_arg)(kern, 4, core::mem::size_of::<i32>(), &k_i as *const i32 as *const c_void);
        let global = [nrows];
        err |= (self.api.enqueue_nd_range_kernel)(self.queue, kern, 1, global.as_ptr(), core::ptr::null(), core::ptr::null(), 0, core::ptr::null(), core::ptr::null_mut());
        err |= (self.api.enqueue_read_buffer)(self.queue, self.out_buf, CL_TRUE, 0, obytes, out.as_mut_ptr() as *mut c_void, 0, core::ptr::null(), core::ptr::null_mut());
        err |= (self.api.finish)(self.queue);
        if err != 0 {
            bail!("OpenCL matvec failed ({err})");
        }
        Ok(())
    }
}

impl Drop for OpenClBackend {
    fn drop(&mut self) {
        self.clear_weight_cache();
        (self.api.release_mem_object)(self.x_buf);
        (self.api.release_mem_object)(self.out_buf);
        for (_, kernel) in self.kernels.drain() {
            (self.api.release_kernel)(kernel);
        }
        (self.api.release_program)(self.program);
        (self.api.release_command_queue)(self.queue);
        (self.api.release_context)(self.ctx);
        unsafe { dlclose(self.lib); }
    }
}

fn nul(s: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len() + 1);
    v.extend_from_slice(s.as_bytes());
    v.push(0);
    v
}
