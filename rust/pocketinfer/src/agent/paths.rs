//! Path handling for the workspace tools.
//!
//! `cwd` alone is not a sandbox: the app UID can still reach anything the app
//! can read. What these helpers guarantee is narrower and honest — every path a
//! tool touches is resolved to its canonical location and must stay inside the
//! workspace root, with symlinks resolved rather than trusted. The shell tool
//! is explicitly *not* covered by this and says so.

use crate::util::{Error, Result};
use std::path::{Component, Path, PathBuf};

/// Resolves `input` against `root` and refuses anything that lands outside.
///
/// Absolute inputs are accepted only if they already are inside the root; this
/// keeps `..` traversal, symlinked parents and `/data/data/...` guessing from
/// silently working.
pub fn resolve_inside(root: &Path, input: &str) -> Result<PathBuf> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(root.to_path_buf());
    }
    if trimmed.contains('\0') {
        return Err(Error::new("Path contains a NUL byte"));
    }
    let relative = Path::new(trimmed);
    let joined = if relative.is_absolute() {
        relative.to_path_buf()
    } else {
        root.join(relative)
    };
    // Resolve `.` and `..` lexically *before* touching the filesystem. Doing it
    // after would leave `root/sub/../../outside` as a string that still starts
    // with the root, which is exactly the case the containment check must catch.
    let normalized = normalize(&joined);
    let canonical = canonicalize_lenient(&normalized)?;
    if !is_inside(root, &canonical) {
        return Err(Error::new(
            "Path resolves outside the workspace; access is limited to the project folder",
        ));
    }
    Ok(canonical)
}

/// Purely lexical `.`/`..` folding. A `..` that would climb above an absolute
/// path's root is dropped, matching how a kernel would resolve it.
pub fn normalize(path: &Path) -> PathBuf {
    let mut parts: Vec<std::ffi::OsString> = Vec::new();
    let mut popped_below_root = false;
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.pop().is_none() {
                    popped_below_root = true;
                }
            }
            other => parts.push(other.as_os_str().to_os_string()),
        }
    }
    if popped_below_root && !path.is_absolute() {
        // Keep a relative result relative rather than inventing a root.
        return PathBuf::from(".");
    }
    let mut out = PathBuf::new();
    for part in parts {
        out.push(part);
    }
    if out.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        out
    }
}

/// Canonicalises as much of the path as exists, then appends the missing tail.
/// Needed so a tool can create a file inside folders that do not exist yet.
pub fn canonicalize_lenient(path: &Path) -> Result<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        match existing.canonicalize() {
            Ok(canonical) => {
                let mut out = canonical;
                for part in tail.iter().rev() {
                    out.push(part);
                }
                return Ok(out);
            }
            Err(_) => match (existing.file_name(), existing.parent()) {
                (Some(name), Some(parent)) => {
                    tail.push(name.to_os_string());
                    existing = parent.to_path_buf();
                }
                _ => return Ok(path.to_path_buf()),
            },
        }
    }
}

pub fn is_inside(root: &Path, candidate: &Path) -> bool {
    candidate == root || candidate.starts_with(root)
}

/// Rejects path components that have no business appearing in a tool argument.
pub fn reject_traversal(input: &str) -> Result<()> {
    let path = Path::new(input);
    for component in path.components() {
        match component {
            Component::ParentDir => {
                return Err(Error::new("Relative '..' segments are not accepted in a path"))
            }
            Component::Prefix(_) | Component::RootDir => {
                return Err(Error::new(
                    "Absolute paths are not accepted; give a path inside the project folder",
                ))
            }
            _ => {}
        }
    }
    Ok(())
}

/// Relative, forward-slash form for the model and the UI. Returns "." for the
/// root so a path is never rendered as an empty string.
pub fn display_relative(root: &Path, target: &Path) -> String {
    match target.strip_prefix(root) {
        Ok(rest) if rest.as_os_str().is_empty() => ".".to_string(),
        Ok(rest) => rest
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/"),
        Err(_) => target.to_string_lossy().into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pocketagent-paths-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.canonicalize().unwrap()
    }

    #[test]
    fn keeps_paths_inside_the_root() {
        let root = scratch("inside");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/file.txt"), "x").unwrap();
        assert!(resolve_inside(&root, "sub/file.txt").is_ok());
        assert!(resolve_inside(&root, ".").is_ok());
        assert!(resolve_inside(&root, "").is_ok());
        assert_eq!(resolve_inside(&root, "").unwrap(), root);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rejects_escapes() {
        let root = scratch("escape");
        assert!(resolve_inside(&root, "../outside.txt").is_err());
        assert!(resolve_inside(&root, "/etc/passwd").is_err());
        assert!(resolve_inside(&root, "sub/../../outside").is_err());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolves_symlinks_before_checking() {
        let root = scratch("symlink");
        let outside = scratch("symlink-target");
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, root.join("escape")).unwrap();
        #[cfg(unix)]
        assert!(resolve_inside(&root, "escape/secret.txt").is_err());
        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&outside).ok();
    }

    #[test]
    fn creates_missing_parents() {
        let root = scratch("missing");
        let resolved = resolve_inside(&root, "a/b/c/new.txt").unwrap();
        assert_eq!(resolved, root.join("a/b/c/new.txt"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn display_is_relative_and_slashed() {
        let root = Path::new("/data/user/0/app/files/workspaces/w1");
        assert_eq!(
            display_relative(root, Path::new("/data/user/0/app/files/workspaces/w1/a/b.kt")),
            "a/b.kt"
        );
        assert_eq!(display_relative(root, root), ".");
    }

    #[test]
    fn traversal_syntax_is_refused_early() {
        assert!(reject_traversal("a/../b").is_err());
        assert!(reject_traversal("/etc").is_err());
        assert!(reject_traversal("a/b").is_ok());
    }
}
