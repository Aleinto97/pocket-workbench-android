use core::sync::atomic::{AtomicBool, Ordering};
use std::ffi::c_void;
use std::fmt;

pub struct Error {
    pub msg: String,
}

impl Error {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::new(format!("io: {e}"))
    }
}

pub type Result<T> = core::result::Result<T, Error>;

#[macro_export]
macro_rules! err {
    ($($arg:tt)*) => { $crate::util::Error::new(format!($($arg)*)) };
}

#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::err!($($arg)*)) };
}

pub const ANDROID_LOG_INFO: i32 = 4;
pub const ANDROID_LOG_WARN: i32 = 5;
pub const ANDROID_LOG_ERROR: i32 = 6;

#[cfg(target_os = "android")]
extern "C" {
    fn __android_log_write(prio: i32, tag: *const u8, text: *const u8) -> i32;
}

pub fn log(prio: i32, msg: &str) {
    #[cfg(target_os = "android")]
    {
        let tag = b"pocketinfer\0";
        let mut buf = Vec::with_capacity(msg.len() + 1);
        buf.extend_from_slice(msg.as_bytes());
        buf.push(0);
        unsafe {
            __android_log_write(prio, tag.as_ptr(), buf.as_ptr());
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        let label = match prio {
            ANDROID_LOG_ERROR => "E",
            ANDROID_LOG_WARN => "W",
            _ => "I",
        };
        eprintln!("pocketinfer {label}: {msg}");
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut shifts: i32 = 0;
            while m & 0x400 == 0 {
                m <<= 1;
                shifts += 1;
            }
            m &= 0x3FF;
            let exp32 = (113 - shifts) as u32;
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0xFF {
        let payload = if mant != 0 { 0x200 | (mant >> 13) as u16 } else { 0 };
        return sign | 0x7C00 | payload;
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = 14 - e;
        let mut half = (m >> shift) as u16;
        let rem = m & ((1 << shift) - 1);
        let half_bit = 1u32 << (shift - 1);
        if rem > half_bit || (rem == half_bit && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }
    let mut half = ((e as u32) << 10 | (mant >> 13)) as u16;
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1;
        if half & 0x7C00 == 0x7C00 {
            return sign | 0x7C00;
        }
    }
    sign | half
}

extern "C" {
    fn open(path: *const u8, flags: i32, mode: u32) -> i32;
    fn close(fd: i32) -> i32;
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn read(fd: i32, buf: *mut c_void, count: usize) -> isize;
    fn write(fd: i32, buf: *const c_void, count: usize) -> isize;
}

const O_RDONLY: i32 = 0;
const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;

pub struct MappedFile {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
    pub fn open(path: &str) -> Result<Self> {
        let mut cpath = Vec::with_capacity(path.len() + 1);
        cpath.extend_from_slice(path.as_bytes());
        cpath.push(0);
        let fd = unsafe { open(cpath.as_ptr(), O_RDONLY, 0) };
        if fd < 0 {
            bail!("cannot open {}", path);
        }
        let len = {
            let mut st: [u64; 32] = [0; 32];
            extern "C" {
                fn fstat(fd: i32, st: *mut c_void) -> i32;
            }
            if unsafe { fstat(fd, st.as_mut_ptr() as *mut c_void) } != 0 {
                unsafe { close(fd) };
                bail!("cannot stat {}", path);
            }
            st[6] as usize
        };
        if len == 0 {
            unsafe { close(fd) };
            bail!("empty file {}", path);
        }
        let ptr = unsafe { mmap(core::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };
        unsafe { close(fd) };
        if ptr == MAP_FAILED {
            bail!("mmap failed for {}", path);
        }
        Ok(Self { ptr: ptr as *mut u8, len })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr as *mut c_void, self.len);
        }
    }
}

pub fn read_file(path: &str) -> Result<Vec<u8>> {
    let mut cpath = Vec::with_capacity(path.len() + 1);
    cpath.extend_from_slice(path.as_bytes());
    cpath.push(0);
    let fd = unsafe { open(cpath.as_ptr(), O_RDONLY, 0) };
    if fd < 0 {
        bail!("cannot open {}", path);
    }
    let mut out = Vec::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = unsafe { read(fd, buf.as_mut_ptr() as *mut c_void, buf.len()) };
        if n < 0 {
            unsafe { close(fd) };
            bail!("read error on {}", path);
        }
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n as usize]);
    }
    unsafe { close(fd) };
    Ok(out)
}

pub fn write_file(path: &str, data: &[u8]) -> Result<()> {
    let mut cpath = Vec::with_capacity(path.len() + 1);
    cpath.extend_from_slice(path.as_bytes());
    cpath.push(0);
    const O_WRONLY: i32 = 1;
    const O_CREAT: i32 = 64;
    const O_TRUNC: i32 = 512;
    let fd = unsafe { open(cpath.as_ptr(), O_WRONLY | O_CREAT | O_TRUNC, 0o644) };
    if fd < 0 {
        bail!("cannot create {}", path);
    }
    let mut off = 0;
    while off < data.len() {
        let n = unsafe {
            write(
                fd,
                data[off..].as_ptr() as *const c_void,
                data.len() - off,
            )
        };
        if n <= 0 {
            unsafe { close(fd) };
            bail!("write error on {}", path);
        }
        off += n as usize;
    }
    unsafe { close(fd) };
    Ok(())
}

pub static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

pub fn request_stop() {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

pub fn stop_requested() -> bool {
    STOP_REQUESTED.load(Ordering::Relaxed)
}
