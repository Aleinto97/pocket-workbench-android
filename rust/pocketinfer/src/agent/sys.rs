//! The few libc calls the agent needs, declared directly so the crate keeps its
//! "no external dependencies" property.

use crate::util::Result;

extern "C" {
    pub fn kill(pid: i32, sig: i32) -> i32;
    pub fn symlink(target: *const u8, link: *const u8) -> i32;
    pub fn setpgid(pid: i32, pgid: i32) -> i32;
}

pub const SIGTERM: i32 = 15;
pub const SIGKILL: i32 = 9;

/// Terminates a process group and then the process itself. Used for Stop: a
/// shell or a runner that spawned children must not leave them behind.
pub fn terminate_tree(pid: i32) {
    if pid <= 0 {
        return;
    }
    unsafe {
        kill(-pid, SIGTERM);
        kill(pid, SIGKILL);
    }
}

/// NUL-terminated bytes of a path, for the C calls above.
pub fn c_path(path: &std::path::Path) -> Vec<u8> {
    let mut out = path.as_os_str().as_encoded_bytes().to_vec();
    out.push(0);
    out
}

pub fn link(target: &std::path::Path, link: &std::path::Path) -> Result<()> {
    let target_bytes = c_path(target);
    let link_bytes = c_path(link);
    if unsafe { symlink(target_bytes.as_ptr(), link_bytes.as_ptr()) } != 0 {
        bail!(
            "Cannot link {} to {}",
            link.display(),
            target.display()
        );
    }
    Ok(())
}
