//! Tool registry and the workspace executors.
//!
//! Each tool declares a stable name, a version, a model-facing description, a
//! parameter schema and the capability it needs. Validation happens here, once,
//! before any executor runs: a call with a missing field or a wrong type never
//! reaches the filesystem.
//!
//! Capability note, stated plainly: these tools confine *themselves* to the
//! workspace by resolving canonical paths. That is not a kernel sandbox. The
//! app's UID can still reach anything the UID can reach, and the shell tool is
//! deliberately outside the path checks — it runs Android's `sh`, not a Linux
//! userland.

use super::json::Json;
use super::paths;
use super::protocol::ToolSpec;
use super::schema::{self, boolean, integer, object, string, string_enum};
use super::sys;
use crate::util::{Error, Result};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const CAP_READ: &str = "workspace:read";
pub const CAP_WRITE: &str = "workspace:write";
pub const CAP_SHELL: &str = "shell:android";

pub const MAX_READ_BYTES: usize = 256 * 1024;
pub const MAX_WRITE_BYTES: usize = 512 * 1024;
pub const MAX_LIST_ENTRIES: usize = 500;
pub const MAX_SEARCH_HITS: usize = 200;
pub const SHELL_TIMEOUT_MS: u64 = 120_000;
pub const MAX_SHELL_OUTPUT: usize = 32 * 1024;

static ACTIVE_SHELL: AtomicI32 = AtomicI32::new(-1);

pub fn cancel_active_shell() {
    let pid = ACTIVE_SHELL.load(Ordering::SeqCst);
    if pid > 0 {
        sys::terminate_tree(pid);
    }
}

fn fs_list_schema() -> Json {
    object(
        vec![
            ("path", string(512)),
            ("recursive", boolean()),
        ],
        vec![],
    )
}

fn fs_read_schema() -> Json {
    object(
        vec![
            ("path", string(512)),
            ("offset", integer(0, 1 << 24)),
            ("limit", integer(1, 200_000)),
        ],
        vec!["path"],
    )
}

fn fs_write_schema() -> Json {
    object(
        vec![
            ("path", string(512)),
            ("content", string(MAX_WRITE_BYTES)),
            ("overwrite", boolean()),
        ],
        vec!["path", "content"],
    )
}

fn fs_edit_schema() -> Json {
    object(
        vec![
            ("path", string(512)),
            ("old_text", string(64 * 1024)),
            ("new_text", string(64 * 1024)),
            ("expect_sha256", string(64)),
            ("replace_all", boolean()),
        ],
        vec!["path", "old_text", "new_text"],
    )
}

fn fs_search_schema() -> Json {
    object(
        vec![
            ("query", string(200)),
            ("path", string(512)),
            ("glob", string(64)),
        ],
        vec!["query"],
    )
}

fn skill_schema() -> Json {
    object(vec![("name", string(64))], vec!["name"])
}

fn delegate_schema() -> Json {
    object(
        vec![
            ("task", string(2000)),
            ("max_steps", integer(1, 6)),
        ],
        vec!["task"],
    )
}

fn shell_schema() -> Json {
    object(
        vec![
            ("command", string(4000)),
            ("timeout_ms", integer(1000, 600_000)),
            ("system", string_enum(["android", "linux"])),
        ],
        vec!["command"],
    )
}

pub const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        name: "fs_list",
        version: "1",
        description: "List files and folders in the project. path defaults to the project root.",
        parameters: fs_list_schema,
        capability: CAP_READ,
        max_output_bytes: 24 * 1024,
        timeout_ms: 5_000,
    },
    ToolSpec {
        name: "fs_read",
        version: "1",
        description: "Read a UTF-8 project file, optionally a window starting at a byte offset.",
        parameters: fs_read_schema,
        capability: CAP_READ,
        max_output_bytes: 32 * 1024,
        timeout_ms: 5_000,
    },
    ToolSpec {
        name: "fs_write",
        version: "1",
        description: "Create a new project file. Fails if it exists unless overwrite is true.",
        parameters: fs_write_schema,
        capability: CAP_WRITE,
        max_output_bytes: 4 * 1024,
        timeout_ms: 10_000,
    },
    ToolSpec {
        name: "fs_edit",
        version: "1",
        description: "Replace an exact snippet in a project file. Pass expect_sha256 to refuse editing content that changed since it was read.",
        parameters: fs_edit_schema,
        capability: CAP_WRITE,
        max_output_bytes: 8 * 1024,
        timeout_ms: 10_000,
    },
    ToolSpec {
        name: "fs_search",
        version: "1",
        description: "Search project files for a literal string and return matching lines.",
        parameters: fs_search_schema,
        capability: CAP_READ,
        max_output_bytes: 24 * 1024,
        timeout_ms: 20_000,
    },
    ToolSpec {
        name: "shell",
        version: "2",
        description: "Run a command in the project folder. system=\"android\" (default): Android's sh with toybox tools plus curl — no package manager, no compilers. system=\"linux\": full Debian under PRoot (apt, gcc, python3, git…; install what you need, it persists) with the project at /workspace — requires the Linux module, installed from Settings. Prefer linux for anything beyond small file/shell tasks. When curl gets nothing from a page (bot filters), on linux apt-install a real reader instead: lynx -dump URL, or chromium --headless --disable-gpu --dump-dom URL.",
        parameters: shell_schema,
        capability: CAP_SHELL,
        max_output_bytes: MAX_SHELL_OUTPUT,
        timeout_ms: SHELL_TIMEOUT_MS,
    },
    ToolSpec {
        name: "skill",
        version: "1",
        description: "Load a named markdown procedure from the project's .skills/ folder. Skills are used for multi-step routines the project defines; read the skill before relying on it.",
        parameters: skill_schema,
        capability: CAP_READ,
        max_output_bytes: 8 * 1024,
        timeout_ms: 5_000,
    },
    ToolSpec {
        name: "delegate",
        version: "1",
        description: "Send a bounded research question to a fresh read-only context (own conversation, no writes, no shell) and get back a condensed answer. Use for large explorations instead of filling this conversation; the answer arrives truncated to 4 KiB with a step trailer.",
        parameters: delegate_schema,
        capability: CAP_READ,
        max_output_bytes: 8 * 1024,
        timeout_ms: 120_000,
    },
];

pub fn find(name: &str) -> Option<&'static ToolSpec> {
    TOOLS.iter().find(|spec| spec.name == name)
}

pub fn schemas_json() -> Json {
    let mut list = Json::Arr(Vec::new());
    for spec in TOOLS {
        list.push(spec.schema_json());
    }
    list
}

/// Capabilities the session grants. A tool whose capability is absent is
/// refused before execution rather than silently unavailable.
#[derive(Clone, Debug, PartialEq)]
pub struct Grants {
    pub read: bool,
    pub write: bool,
    pub shell: bool,
}

impl Grants {
    pub fn all() -> Self {
        Self { read: true, write: true, shell: true }
    }

    pub fn read_only() -> Self {
        Self { read: true, write: false, shell: false }
    }

    pub fn allows(&self, capability: &str) -> bool {
        match capability {
            CAP_READ => self.read,
            CAP_WRITE => self.write,
            CAP_SHELL => self.shell,
            _ => false,
        }
    }
}

pub fn grant_list(grants: &Grants) -> Json {
    let mut list = Json::Arr(Vec::new());
    if grants.read {
        list.push(Json::str(CAP_READ));
    }
    if grants.write {
        list.push(Json::str(CAP_WRITE));
    }
    if grants.shell {
        list.push(Json::str(CAP_SHELL));
    }
    list
}

/// Cancels an in-flight shell command, used by the runtime's Stop path.
#[derive(Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self { flag: Arc::new(AtomicBool::new(false)) }
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        cancel_active_shell();
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// Paths of the optional Debian module. The Kotlin side owns the layout
/// (download, verify, extract); Rust only executes through it. `tools` holds
/// the `curl` symlink for Android-shell mode and is independent of `rootfs`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LinuxConfig {
    pub proot: String,
    pub rootfs: String,
    pub tmp: String,
    pub tools: String,
}

impl LinuxConfig {
    /// The module counts as installed when the Debian tree looks real.
    /// Checked on every call so a stale config fails loudly instead of
    /// running against half-extracted state.
    pub fn ready_rootfs(&self) -> Option<PathBuf> {
        if self.proot.is_empty() || self.rootfs.is_empty() {
            return None;
        }
        let root = PathBuf::from(&self.rootfs);
        let bash = root.join("bin/bash");
        if !Path::new(&self.proot).is_file() || !bash.is_file() {
            return None;
        }
        Some(root)
    }
}

/// Validated, path-resolved arguments. Executors take this, never raw `Json`, so
/// no executor can skip validation.
pub struct Workspace {
    pub root: PathBuf,
    pub grants: Grants,
    pub cancel: CancelToken,
    pub linux: Option<LinuxConfig>,
}

impl Workspace {
    pub fn new(root: &Path, grants: Grants) -> Self {
        Self { root: root.to_path_buf(), grants, cancel: CancelToken::new(), linux: None }
    }

    fn spec(&self, name: &str) -> Result<&'static ToolSpec> {
        let spec = find(name).ok_or_else(|| Error::new(format!("Unknown tool '{name}'")))?;
        if !self.grants.allows(spec.capability) {
            return Err(Error::new(format!(
                "Tool '{name}' needs the '{}' capability, which this session does not have",
                spec.capability
            )));
        }
        Ok(spec)
    }

    fn prepare(&self, name: &str, arguments: &Json) -> Result<(ToolCall, PathBuf)> {
        let spec = self.spec(name)?;
        schema::validate(&(spec.parameters)(), arguments).map_err(|e| {
            Error::new(format!("Invalid arguments for '{name}': {e}"))
        })?;
        let call = ToolCall::new(name, arguments.clone());
        let target = paths::resolve_inside(&self.root, call.text("path", ".").as_str())
            .map_err(|e| Error::new(format!("Invalid arguments for '{name}': {e}")))?;
        Ok((call, target))
    }

    /// Validates and runs one tool. Errors are returned, never swallowed, so the
    /// runtime can record a classified outcome and let the model see it.
    pub fn execute(&self, name: &str, arguments: &Json) -> Result<Json> {
        if self.cancel.is_cancelled() {
            return Err(Error::new("Cancelled before the tool started"));
        }
        match name {
            "fs_list" => self.list(arguments),
            "fs_read" => self.read(arguments),
            "fs_write" => self.write(arguments),
            "fs_edit" => self.edit(arguments),
            "fs_search" => self.search(arguments),
            "shell" => self.shell(arguments),
            "skill" => self.skill(arguments),
            // "delegate" never reaches the workspace: the session runs it in
            // an isolated read-only context (Session::run_delegation).
            "delegate" => Err(Error::new(
                "delegate runs in the session, not the workspace",
            )),
            other => Err(Error::new(format!("Unknown tool '{other}'"))),
        }
    }

    fn list(&self, arguments: &Json) -> Result<Json> {
        let (call, dir) = self.prepare("fs_list", arguments)?;
        let recursive = call.bool("recursive", false);
        if !dir.exists() {
            return Err(Error::new(format!(
                "'{}' does not exist",
                paths::display_relative(&self.root, &dir)
            )));
        }
        if !dir.is_dir() {
            return Err(Error::new(format!(
                "'{}' is a file, not a folder",
                paths::display_relative(&self.root, &dir)
            )));
        }
        let mut entries = Vec::new();
        collect(&dir, &self.root, recursive, 0, &mut entries)?;
        entries.truncate(MAX_LIST_ENTRIES);
        let mut items = Json::Arr(Vec::new());
        for (path, is_dir, size) in &entries {
            items.push(
                Json::obj()
                    .with("path", Json::str(paths::display_relative(&self.root, path)))
                    .with("type", Json::str(if *is_dir { "folder" } else { "file" }))
                    .with("bytes", Json::int(*size as i64)),
            );
        }
        Ok(Json::obj()
            .with("root", Json::str(paths::display_relative(&self.root, &dir)))
            .with("count", Json::int(entries.len() as i64))
            .with("items", items))
    }

    fn read(&self, arguments: &Json) -> Result<Json> {
        let (call, file) = self.prepare("fs_read", arguments)?;
        if !file.is_file() {
            return Err(Error::new(format!(
                "'{}' is not a file",
                paths::display_relative(&self.root, &file)
            )));
        }
        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        if size > MAX_READ_BYTES as u64 {
            return Err(Error::new(format!(
                "File is {size} bytes; read it in windows with offset and limit"
            )));
        }
        let offset = call.number("offset", 0).max(0) as usize;
        let limit = call.number("limit", 16_000).max(1) as usize;
        let bytes = crate::util::read_file(&file.to_string_lossy())?;
        let start = offset.min(bytes.len());
        let end = (start + limit).min(bytes.len());
        let window = &bytes[start..end];
        if window.contains(&0u8) {
            return Err(Error::new("Binary file; read it with an external tool instead"));
        }
        Ok(Json::obj()
            .with("path", Json::str(paths::display_relative(&self.root, &file)))
            .with("bytes", Json::int(size as i64))
            .with("offset", Json::int(start as i64))
            .with("sha256", Json::str(sha256_hex(&bytes)))
            .with("truncated", Json::Bool(end < bytes.len()))
            .with("content", Json::str(String::from_utf8_lossy(window))))
    }

    /// Loads a named skill from `.skills/<name>.md`. Names are restricted to a
    /// safe alphabet and resolved inside the workspace, so `..` cannot escape.
    fn skill(&self, arguments: &Json) -> Result<Json> {
        let spec = self.spec("skill")?;
        schema::validate(&(spec.parameters)(), arguments).map_err(|e| {
            Error::new(format!("Invalid arguments for 'skill': {e}"))
        })?;
        let call = ToolCall::new("skill", arguments.clone());
        let name = call.text("name", "");
        if name.is_empty()
            || name.len() > 64
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(Error::new(
                "Invalid skill name: use letters, digits, '_' or '-', max 64 chars",
            ));
        }
        let file = paths::resolve_inside(&self.root, &format!(".skills/{name}.md"))
            .map_err(|e| Error::new(format!("Invalid arguments for 'skill': {e}")))?;
        if !file.is_file() {
            return Err(Error::new(format!(
                "No skill named '{name}' in .skills/"
            )));
        }
        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        if size > 8 * 1024 {
            return Err(Error::new(format!(
                "Skill '{name}' is {size} bytes; keep skills under 8 KiB"
            )));
        }
        let bytes = crate::util::read_file(&file.to_string_lossy())?;
        Ok(Json::obj()
            .with("name", Json::str(&name))
            .with("content", Json::str(String::from_utf8_lossy(&bytes))))
    }

    fn write(&self, arguments: &Json) -> Result<Json> {
        let (call, file) = self.prepare("fs_write", arguments)?;
        let content = call.text("content", "");
        if content.len() > MAX_WRITE_BYTES {
            return Err(Error::new(format!(
                "Content is {} bytes; the limit is {MAX_WRITE_BYTES}",
                content.len()
            )));
        }
        let overwrite = call.bool("overwrite", false);
        if file.exists() && !overwrite {
            return Err(Error::new(format!(
                "'{}' already exists; read it and use fs_edit, or set overwrite",
                paths::display_relative(&self.root, &file)
            )));
        }
        let bytes = content.as_bytes();
        atomic_write(&file, bytes)?;
        Ok(Json::obj()
            .with("path", Json::str(paths::display_relative(&self.root, &file)))
            .with("bytes", Json::int(bytes.len() as i64))
            .with("created", Json::Bool(true))
            .with("sha256", Json::str(sha256_hex(bytes))))
    }

    fn edit(&self, arguments: &Json) -> Result<Json> {
        let (call, file) = self.prepare("fs_edit", arguments)?;
        if !file.is_file() {
            return Err(Error::new(format!(
                "'{}' is not a file",
                paths::display_relative(&self.root, &file)
            )));
        }
        let old_text = call.text("old_text", "");
        let new_text = call.text("new_text", "");
        if old_text.is_empty() {
            return Err(Error::new("old_text must not be empty"));
        }
        let current = crate::util::read_file(&file.to_string_lossy())?;
        let digest = sha256_hex(&current);
        if let Some(expected) = call.optional_text("expect_sha256") {
            if expected != digest {
                return Err(Error::new(format!(
                    "File changed since it was read (sha256 {}); read it again and retry",
                    &digest[..16]
                )));
            }
        }
        let text = String::from_utf8_lossy(&current).into_owned();
        let hits = text.matches(old_text.as_str()).count();
        if hits == 0 {
            return Err(Error::new("old_text was not found in the file, byte for byte"));
        }
        if hits > 1 && !call.bool("replace_all", false) {
            return Err(Error::new(format!(
                "old_text appears {hits} times; include more context or set replace_all"
            )));
        }
        let updated = if call.bool("replace_all", false) {
            text.replace(old_text.as_str(), new_text.as_str())
        } else {
            text.replacen(old_text.as_str(), new_text.as_str(), 1)
        };
        atomic_write(&file, updated.as_bytes())?;
        Ok(Json::obj()
            .with("path", Json::str(paths::display_relative(&self.root, &file)))
            .with("replacements", Json::int(if call.bool("replace_all", false) { hits as i64 } else { 1 }))
            .with("bytes", Json::int(updated.len() as i64))
            .with("sha256", Json::str(sha256_hex(updated.as_bytes()))))
    }

    fn search(&self, arguments: &Json) -> Result<Json> {
        let (call, base) = self.prepare("fs_search", arguments)?;
        let query = call.text("query", "");
        if query.is_empty() {
            return Err(Error::new("query must not be empty"));
        }
        let glob = call.optional_text("glob");
        if !base.exists() {
            return Err(Error::new("Search path does not exist"));
        }
        let files = if base.is_file() {
            vec![base.clone()]
        } else {
            let mut entries = Vec::new();
            collect(&base, &self.root, true, 0, &mut entries)?;
            entries
                .into_iter()
                .filter(|(_, is_dir, _)| !*is_dir)
                .map(|(path, _, _)| path)
                .collect()
        };
        let mut hits = Vec::new();
        let mut scanned = 0usize;
        for file in files {
            if self.cancel.is_cancelled() {
                break;
            }
            if let Some(pattern) = &glob {
                if !matches_glob(&file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), pattern) {
                    continue;
                }
            }
            let Ok(bytes) = crate::util::read_file(&file.to_string_lossy()) else {
                continue;
            };
            if bytes.len() > MAX_READ_BYTES || bytes.contains(&0u8) {
                continue;
            }
            scanned += 1;
            let text = String::from_utf8_lossy(&bytes).into_owned();
            for (number, line) in text.lines().enumerate() {
                if line.contains(query.as_str()) {
                    if hits.len() >= MAX_SEARCH_HITS {
                        break;
                    }
                    hits.push(
                        Json::obj()
                            .with("path", Json::str(paths::display_relative(&self.root, &file)))
                            .with("line", Json::int(number as i64 + 1))
                            .with("text", Json::str(line.trim().chars().take(300).collect::<String>())),
                    );
                }
            }
        }
        Ok(Json::obj()
            .with("query", Json::str(query))
            .with("files_scanned", Json::int(scanned as i64))
            .with("match_count", Json::int(hits.len() as i64))
            .with("matches", Json::Arr(hits)))
    }

    fn shell(&self, arguments: &Json) -> Result<Json> {
        let (call, _) = self.prepare("shell", arguments)?;
        let command_arg = call.text("command", "");
        if command_arg.trim().is_empty() {
            return Err(Error::new("command must not be empty"));
        }
        let timeout_ms = call.number("timeout_ms", SHELL_TIMEOUT_MS as i64).max(1000) as u64;
        let started = Instant::now();
        let system = call.optional_text("system").unwrap_or_else(|| "android".to_string());
        let (mut command, cwd_label) = match system.as_str() {
            "linux" => self.linux_command(&command_arg)?,
            "android" => {
                let mut command = Command::new("/system/bin/sh");
                command
                    .arg("-c")
                    .arg(&command_arg)
                    .current_dir(&self.root)
                    .env_remove("LD_PRELOAD")
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());
                // The system's own tools first, our bundled ones last. A static
                // curl is the fallback for devices that ship no curl, not the
                // preferred one: it is built for a normal Linux syscall policy,
                // and an app process is killed with SIGSYS ("Bad system call")
                // by the platform seccomp filter when it reaches for one.
                let system_paths = "/system/bin:/system/xbin:/vendor/bin";
                match &self.linux {
                    Some(linux) if !linux.tools.is_empty() => {
                        command.env("PATH", format!("{system_paths}:{}", linux.tools));
                    }
                    _ => {
                        command.env("PATH", system_paths);
                    }
                }
                (command, paths::display_relative(&self.root, &self.root))
            }
            other => {
                return Err(Error::new(format!(
                    "Invalid arguments for 'shell': unknown system '{other}' (android or linux)"
                )))
            }
        };
        // Its own process group, so cancelling kills the children the shell
        // started too, not just the shell itself.
        unsafe {
            command.pre_exec(|| {
                if sys::setpgid(0, 0) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command
            .spawn()
            .map_err(|e| Error::new(format!("Cannot start the {system} shell: {e}")))?;
        let pid = child.id() as i32;
        ACTIVE_SHELL.store(pid, Ordering::SeqCst);
        let handle = std::thread::spawn(move || child.wait_with_output());
        let deadline = Instant::now() + std::time::Duration::from_millis(timeout_ms);
        let timed_out = loop {
            if handle.is_finished() {
                break false;
            }
            if Instant::now() >= deadline || self.cancel.is_cancelled() {
                sys::terminate_tree(pid);
                break true;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        };
        let outcome = match handle.join() {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(e)) => {
                ACTIVE_SHELL.store(-1, Ordering::SeqCst);
                return Err(Error::new(format!("Shell failed: {e}")));
            }
            Err(_) => {
                ACTIVE_SHELL.store(-1, Ordering::SeqCst);
                return Err(Error::new("Shell thread panicked"));
            }
        };
        ACTIVE_SHELL.store(-1, Ordering::SeqCst);
        if timed_out {
            return Err(Error::new(if self.cancel.is_cancelled() {
                "Command cancelled".to_string()
            } else {
                format!("Command exceeded {timeout_ms} ms and was terminated")
            }));
        }
        let mut text = String::from_utf8_lossy(&outcome.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&outcome.stderr).into_owned();
        if !stderr.is_empty() {
            text.push('\n');
            text.push_str(&stderr);
        }
        let truncated = text.len() > MAX_SHELL_OUTPUT;
        if truncated {
            let mut cut = MAX_SHELL_OUTPUT;
            while cut > 0 && !text.is_char_boundary(cut) {
                cut -= 1;
            }
            text.truncate(cut);
        }
        Ok(Json::obj()
            .with("command", Json::str(command_arg))
            .with("system", Json::str(system))
            .with("cwd", Json::str(cwd_label))
            .with("exit_code", Json::int(outcome.status.code().unwrap_or(-1)))
            .with("timed_out", Json::Bool(false))
            .with("duration_ms", Json::int(started.elapsed().as_millis() as i64))
            .with("truncated", Json::Bool(truncated))
            .with("output", Json::str(&text))
            // An empty output is a fact the model must read, not a gap to
            // fill: grep exits 1 on no match, and bot filters serve empty
            // pages. Spelled out, the next step is another URL, not a guess.
            .with(
                "note",
                Json::str(if text.is_empty() {
                    "the command produced no output"
                } else {
                    ""
                }),
            ))
    }

    /// Builds the Debian-under-PRoot invocation. The project is visible at
    /// /workspace; /dev, /proc and /sys come from the host. --kill-on-exit
    /// plus our process-group kill means Stop leaves nothing behind.
    /// LD_PRELOAD and LD_LIBRARY_PATH are stripped: the GenieX shim and the
    /// Android linker paths must never leak into Debian processes.
    fn linux_command(&self, command_arg: &str) -> Result<(Command, String)> {
        let config = self.linux.clone().ok_or_else(|| {
            Error::new("The Debian module is not installed; install Linux from Settings first")
        })?;
        let root = config.ready_rootfs().ok_or_else(|| {
            Error::new(
                "The Debian module is not installed; install Linux from Settings first",
            )
        })?;
        for dir in [root.join(".l2s"), PathBuf::from(&config.tmp)] {
            std::fs::create_dir_all(&dir)
                .map_err(|e| Error::new(format!("Cannot prepare the Linux runtime: {e}")))?;
        }
        let mut command = Command::new(&config.proot);
        command
            .arg("--kill-on-exit")
            .arg("--link2symlink")
            .arg("-r")
            .arg(root.to_string_lossy().as_ref())
            .arg("-b")
            .arg("/dev")
            .arg("-b")
            .arg("/proc")
            .arg("-b")
            .arg("/sys")
            .arg("-b")
            .arg(format!("{}:/workspace", self.root.display()))
            .arg("-w")
            .arg("/workspace")
            .arg("/usr/bin/env")
            .arg("-i")
            .arg(format!("PATH={LINUX_PATH}"))
            .arg("HOME=/root")
            .arg("TERM=xterm-256color")
            .arg(format!("PROOT_TMP_DIR={}", config.tmp))
            .arg(format!("PROOT_L2S_DIR={}/.l2s", root.to_string_lossy()))
            .arg("/bin/bash")
            .arg("-c")
            .arg(command_arg)
            .env_remove("LD_PRELOAD")
            .env_remove("LD_LIBRARY_PATH")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        Ok((command, "/workspace".to_string()))
    }
}

const LINUX_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

/// Validated string arguments with typed defaults, so executors never index a
/// raw `Json` and never panic on a missing field.
pub struct ToolCall {
    name: String,
    arguments: Json,
}

impl ToolCall {
    pub fn new(name: &str, arguments: Json) -> Self {
        Self { name: name.to_string(), arguments }
    }

    pub fn raw(&self) -> &Json {
        &self.arguments
    }

    fn value(&self, key: &str) -> Option<&Json> {
        self.arguments.get(key).filter(|v| !v.is_null())
    }

    pub fn text(&self, key: &str, fallback: &str) -> String {
        self.value(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| fallback.to_string())
    }

    pub fn optional_text(&self, key: &str) -> Option<String> {
        self.value(key)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }

    pub fn number(&self, key: &str, fallback: i64) -> i64 {
        self.value(key).and_then(|v| v.as_f64()).map(|n| n as i64).unwrap_or(fallback)
    }

    pub fn bool(&self, key: &str, fallback: bool) -> bool {
        self.value(key).and_then(|v| v.as_bool()).unwrap_or(fallback)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

fn collect(
    dir: &Path,
    root: &Path,
    recursive: bool,
    depth: usize,
    out: &mut Vec<(PathBuf, bool, u64)>,
) -> Result<()> {
    if out.len() >= MAX_LIST_ENTRIES || depth > 12 {
        return Ok(());
    }
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| Error::new(format!("Cannot read folder: {e}")))?
        .filter_map(|entry| entry.ok())
        .collect();
    names.sort_by_key(|entry| {
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        (!is_dir, entry.file_name())
    });
    for entry in names {
        if out.len() >= MAX_LIST_ENTRIES {
            break;
        }
        let path = entry.path();
        let Ok(canonical) = path.canonicalize() else {
            continue;
        };
        // A symlink pointing out of the workspace is not listed: the tools would
        // refuse to read it anyway, and hiding it avoids advertising a path that
        // cannot be used.
        if !paths::is_inside(root, &canonical) {
            continue;
        }
        let is_dir = canonical.is_dir();
        let size = if is_dir { 0 } else { canonical.metadata().map(|m| m.len()).unwrap_or(0) };
        out.push((canonical.clone(), is_dir, size));
        if is_dir && recursive {
            collect(&canonical, root, true, depth + 1, out)?;
        }
    }
    Ok(())
}

fn matches_glob(name: &str, pattern: &str) -> bool {
    let pattern = pattern.trim();
    if pattern == "*" || pattern.is_empty() {
        return true;
    }
    match pattern.strip_prefix('*').and_then(|r| r.strip_suffix('*')) {
        Some(middle) => name.contains(middle),
        None => name.contains(pattern),
    }
}

/// Write through a temporary file in the same folder, then rename, so a crash
/// cannot leave a half-written project file behind.
fn atomic_write(target: &Path, bytes: &[u8]) -> Result<()> {
    let parent = target.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)
        .map_err(|e| Error::new(format!("Cannot create folder: {e}")))?;
    let temp = parent.join(format!(
        ".pocketagent-{}.tmp",
        target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "file".into())
    ));
    crate::util::write_file(&temp.to_string_lossy(), bytes)?;
    match std::fs::rename(&temp, target) {
        Ok(()) => Ok(()),
        Err(error) => {
            std::fs::remove_file(&temp).ok();
            Err(Error::new(format!("Cannot write file: {error}")))
        }
    }
}

/// FNV-1a based digest. This is an integrity check against a stale edit, not a
/// security boundary, so it stays dependency-free rather than pretending to be
/// a cryptographic hash.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = [0x6a09e667u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a];
    for chunk in bytes.chunks(4) {
        let mut block = [0u8; 4];
        block[..chunk.len()].copy_from_slice(chunk);
        let word = u32::from_le_bytes(block);
        for round in 0..4 {
            digest[round] = digest[round]
                .rotate_left(5)
                .wrapping_add(word ^ digest[(round + 1) % 4])
                .wrapping_mul(0x01000193);
        }
    }
    digest
        .iter()
        .map(|word| format!("{word:08x}"))
        .collect::<Vec<_>>()
        .concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace(name: &str) -> Workspace {
        let dir = std::env::temp_dir().join(format!("pocketagent-tools-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        Workspace::new(&dir.canonicalize().unwrap(), Grants::all())
    }

    fn args(text: &str) -> Json {
        super::super::json::parse(text).unwrap()
    }

    #[test]
    fn writes_then_reads_back_a_file() {
        let ws = workspace("roundtrip");
        ws.execute(
            "fs_write",
            &args(r#"{"path":"notes/hello.txt","content":"ciao\nmondo\n"}"#),
        )
        .unwrap();
        let read = ws.execute("fs_read", &args(r#"{"path":"notes/hello.txt"}"#)).unwrap();
        assert_eq!(read.get("content").unwrap().as_str().unwrap(), "ciao\nmondo\n");
        assert_eq!(read.get("bytes").unwrap().as_i64().unwrap(), 11);
        assert!(read.get("sha256").unwrap().as_str().unwrap().len() == 32);
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn write_refuses_to_clobber_without_permission() {
        let ws = workspace("clobber");
        ws.execute("fs_write", &args(r#"{"path":"a.txt","content":"1"}"#)).unwrap();
        let error = ws
            .execute("fs_write", &args(r#"{"path":"a.txt","content":"2"}"#))
            .unwrap_err();
        assert!(error.to_string().contains("already exists"));
        ws.execute("fs_write", &args(r#"{"path":"a.txt","content":"2","overwrite":true}"#))
            .unwrap();
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn edit_requires_the_exact_snippet_and_checks_the_digest() {
        let ws = workspace("edit");
        ws.execute("fs_write", &args(r#"{"path":"a.txt","content":"alpha\nbeta\ngamma\n"}"#)).unwrap();
        let read = ws.execute("fs_read", &args(r#"{"path":"a.txt"}"#)).unwrap();
        let digest = read.get("sha256").unwrap().as_str().unwrap().to_string();

        let error = ws
            .execute("fs_edit", &args(r#"{"path":"a.txt","old_text":"nope","new_text":"x"}"#))
            .unwrap_err();
        assert!(error.to_string().contains("not found"));

        let ambiguous = format!(
            r#"{{"path":"a.txt","old_text":"a","new_text":"z"}}"#
        );
        assert!(ws.execute("fs_edit", &args(&ambiguous)).is_err());

        let good = format!(
            r#"{{"path":"a.txt","old_text":"beta","new_text":"BETA","expect_sha256":"{digest}"}}"#
        );
        ws.execute("fs_edit", &args(&good)).unwrap();
        let after = ws.execute("fs_read", &args(r#"{"path":"a.txt"}"#)).unwrap();
        assert_eq!(after.get("content").unwrap().as_str().unwrap(), "alpha\nBETA\ngamma\n");

        let stale = format!(
            r#"{{"path":"a.txt","old_text":"alpha","new_text":"x","expect_sha256":"{digest}"}}"#
        );
        let error = ws.execute("fs_edit", &args(&stale)).unwrap_err();
        assert!(error.to_string().contains("changed since it was read"));
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn rejects_paths_outside_the_workspace_and_bad_types() {
        let ws = workspace("reject");
        assert!(ws.execute("fs_read", &args(r#"{"path":"../escape.txt"}"#)).is_err());
        assert!(ws.execute("fs_read", &args(r#"{"path":"/data/data/com.pocketworkbench.app/files/models"}"#)).is_err());
        assert!(ws.execute("fs_read", &args(r#"{}"#)).is_err());
        assert!(ws.execute("fs_read", &args(r#"{"path":12}"#)).is_err());
        assert!(ws.execute("fs_read", &args(r#"{"path":"a","limit":"lots"}"#)).is_err());
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn read_only_grants_refuse_writes_and_shell() {
        let dir = std::env::temp_dir().join(format!("pocketagent-tools-ro-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ws = Workspace::new(&dir.canonicalize().unwrap(), Grants::read_only());
        let error = ws.execute("fs_write", &args(r#"{"path":"a.txt","content":"x"}"#)).unwrap_err();
        assert!(error.to_string().contains("capability"));
        let error = ws.execute("shell", &args(r#"{"command":"echo hi"}"#)).unwrap_err();
        assert!(error.to_string().contains("capability"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lists_and_searches_the_tree() {
        let ws = workspace("list");
        ws.execute("fs_write", &args(r#"{"path":"src/main.txt","content":"target here\n"}"#)).unwrap();
        ws.execute("fs_write", &args(r#"{"path":"src/other.txt","content":"nothing\n"}"#)).unwrap();
        let listed = ws.execute("fs_list", &args(r#"{"path":".","recursive":true}"#)).unwrap();
        assert_eq!(listed.get("count").unwrap().as_i64().unwrap(), 3);

        let found = ws.execute("fs_search", &args(r#"{"query":"target"}"#)).unwrap();
        assert_eq!(found.get("match_count").unwrap().as_i64().unwrap(), 1);
        let match0 = &found.get("matches").unwrap().as_array().unwrap()[0];
        assert_eq!(match0.get("path").unwrap().as_str().unwrap(), "src/main.txt");

        let empty = ws.execute("fs_search", &args(r#"{"query":"absent"}"#)).unwrap();
        assert_eq!(empty.get("match_count").unwrap().as_i64().unwrap(), 0);
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn shell_runs_and_reports_the_exit_code() {
        let ws = workspace("shell");
        let out = ws.execute("shell", &args(r#"{"command":"echo hello && exit 3"}"#)).unwrap();
        assert_eq!(out.get("exit_code").unwrap().as_i64().unwrap(), 3);
        assert!(out.get("output").unwrap().as_str().unwrap().contains("hello"));
        assert_eq!(out.get("cwd").unwrap().as_str().unwrap(), ".");
        assert_eq!(out.get("system").unwrap().as_str().unwrap(), "android");
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn shell_rejects_an_unknown_system() {
        let ws = workspace("shell-system");
        // The schema validator fires first with the allowed values listed.
        let error = ws
            .execute("shell", &args(r#"{"command":"echo hi","system":"windows"}"#))
            .unwrap_err()
            .to_string();
        assert!(error.contains("android, linux"), "{error}");
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn shell_linux_without_the_module_fails_loudly() {
        let ws = workspace("shell-nolinux");
        assert!(ws.linux.is_none());
        let error = ws
            .execute("shell", &args(r#"{"command":"apt update","system":"linux"}"#))
            .unwrap_err();
        assert!(error.to_string().contains("Debian module"));
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn shell_linux_builds_the_proot_invocation() {
        let ws = workspace("shell-proot");
        let fake = ws.root.join("fake-linux");
        let proot = fake.join("proot");
        let rootfs_bin = fake.join("rootfs/bin");
        std::fs::create_dir_all(&rootfs_bin).unwrap();
        std::fs::write(&proot, b"fake").unwrap();
        std::fs::write(rootfs_bin.join("bash"), b"fake").unwrap();
        let mut ws = ws;
        ws.linux = Some(LinuxConfig {
            proot: proot.to_string_lossy().into_owned(),
            rootfs: fake.join("rootfs").to_string_lossy().into_owned(),
            tmp: fake.join("tmp").to_string_lossy().into_owned(),
            tools: String::new(),
        });
        let (command, cwd) = ws.linux_command("echo hi").unwrap();
        assert_eq!(cwd, "/workspace");
        let debug = format!("{command:?}");
        assert!(debug.contains("--kill-on-exit"), "{debug}");
        assert!(debug.contains("--link2symlink"), "{debug}");
        assert!(debug.contains("/workspace"), "{debug}");
        assert!(debug.contains("PROOT_TMP_DIR="), "{debug}");
        assert!(fake.join("tmp").is_dir());
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn shell_times_out_and_kills_the_tree() {
        let ws = workspace("shelltimeout");
        let error = ws
            .execute("shell", &args(r#"{"command":"sleep 30","timeout_ms":1000}"#))
            .unwrap_err();
        assert!(error.to_string().contains("exceeded"));
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn digest_is_stable_and_content_dependent() {
        assert_eq!(sha256_hex(b"pocket"), sha256_hex(b"pocket"));
        assert_ne!(sha256_hex(b"pocket"), sha256_hex(b"pocketa"));
    }
}
