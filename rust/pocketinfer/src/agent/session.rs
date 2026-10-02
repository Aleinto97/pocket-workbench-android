//! Append-only JSONL session log.
//!
//! The log is the authority for the transcript. The UI projects from it, it is
//! what survives a crash, and it is what compaction checkpoints against. Rules
//! that matter:
//!
//! * sequence numbers are monotonic and assigned here, never by a caller;
//! * re-admitting a request id that is already recorded is a no-op, so a retry
//!   after a crash cannot duplicate a turn;
//! * a truncated final line (killed mid-write) is dropped on load, never guessed.

use super::json::{parse, Json};
use super::protocol::{AgentEvent, EventKind, PROTOCOL_VERSION, SESSION_SCHEMA_VERSION};
use crate::util::{read_file, write_file, Result};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct SessionLog {
    path: PathBuf,
    header: Json,
    events: Vec<AgentEvent>,
    admitted: Vec<String>,
    next_seq: u64,
}

impl SessionLog {
    /// Opens the log, creating it with a versioned header when missing. A file
    /// whose last line is truncated keeps every complete record before it.
    pub fn open(path: &Path, workspace_id: &str, title: &str) -> Result<SessionLog> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                crate::util::Error::new(format!("cannot create {}: {e}", parent.display()))
            })?;
        }
        let mut log = SessionLog {
            path: path.to_path_buf(),
            header: Json::obj(),
            events: Vec::new(),
            admitted: Vec::new(),
            next_seq: 1,
        };
        if log.path.is_file() {
            log.load()?;
        } else {
            log.header = Json::obj()
                .with("v", Json::int(SESSION_SCHEMA_VERSION as i64))
                .with("protocol", Json::int(PROTOCOL_VERSION as i64))
                .with("kind", Json::str("session"))
                .with("workspace_id", Json::str(workspace_id))
                .with("title", Json::str(title));
            let header = log.header.clone();
            log.append(&header)?;
        }
        Ok(log)
    }

    fn load(&mut self) -> Result<()> {
        let bytes = read_file(&self.path.to_string_lossy())?;
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.split('\n').collect();
        let complete = match text.strip_suffix('\n') {
            Some(_) => lines.len().saturating_sub(1),
            // No trailing newline: the process died mid-append. Everything up to
            // the last newline is intact.
            None => lines.iter().rposition(|line| !line.is_empty()).map_or(0, |i| i),
        };
        let mut truncated_bytes = 0usize;
        for (index, line) in lines.iter().enumerate() {
            if index >= complete {
                truncated_bytes += line.len() + 1;
                continue;
            }
            if line.trim().is_empty() {
                continue;
            }
            let value = match parse(line) {
                Ok(value) => value,
                Err(error) => {
                    // A corrupt line in the middle is a real problem: refuse to
                    // guess, but keep the file so nothing is lost.
                    return Err(crate::util::Error::new(format!(
                        "session log {} line {} is corrupt: {error}",
                        self.path.display(),
                        index + 1
                    )));
                }
            };
            match value.get("kind").and_then(|k| k.as_str()) {
                Some("session") => self.header = value,
                _ => {
                    if let Some(event) = AgentEvent::from_json(&value) {
                        self.next_seq = self.next_seq.max(event.seq + 1);
                        if let Some(id) = admitted_id(&event) {
                            self.admitted.push(id);
                        }
                        self.events.push(event);
                    }
                }
            }
        }
        if truncated_bytes > 0 {
            // Rewrite without the partial tail so the next append starts clean.
            self.rewrite()?;
        }
        if self.header.is_null() || self.header.as_object().is_none() {
            self.header = Json::obj()
                .with("v", Json::int(SESSION_SCHEMA_VERSION as i64))
                .with("protocol", Json::int(PROTOCOL_VERSION as i64))
                .with("kind", Json::str("session"));
        }
        Ok(())
    }

    fn rewrite(&self) -> Result<()> {
        let mut out = String::new();
        self.header.write(&mut out);
        out.push('\n');
        for event in &self.events {
            event.to_json().write(&mut out);
            out.push('\n');
        }
        let temp = self.path.with_extension("log.tmp");
        write_file(&temp.to_string_lossy(), out.as_bytes())?;
        std::fs::rename(&temp, &self.path).map_err(|e| {
            crate::util::Error::new(format!("cannot repair {}: {e}", self.path.display()))
        })?;
        Ok(())
    }

    fn append(&mut self, value: &Json) -> Result<()> {
        let mut line = String::new();
        value.write(&mut line);
        line.push('\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| {
                crate::util::Error::new(format!("cannot open {}: {e}", self.path.display()))
            })?;
        file.write_all(line.as_bytes()).map_err(|e| {
            crate::util::Error::new(format!("cannot append to {}: {e}", self.path.display()))
        })?;
        file.sync_data().ok();
        Ok(())
    }

    pub fn events(&self) -> &[AgentEvent] {
        &self.events
    }

    pub fn header(&self) -> &Json {
        &self.header
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn is_admitted(&self, request_id: &str) -> bool {
        self.admitted.iter().any(|id| id == request_id)
    }

    /// Records an event. Durable kinds are flushed to disk immediately; token
    /// deltas are kept in memory only, because the durable text event that ends
    /// the step reconstructs them.
    pub fn record(&mut self, event: AgentEvent) -> Result<AgentEvent> {
        let mut event = event;
        event.seq = self.next_seq;
        self.next_seq += 1;
        if event.kind.is_durable() {
            self.append(&event.to_json())?;
        }
        if let Some(id) = admitted_id(&event) {
            self.admitted.push(id);
        }
        self.events.push(event.clone());
        Ok(event)
    }

    /// Bounds the in-memory working set after a compaction WITHOUT rewriting the
    /// file. The log stays append-only: every durable record ever written stays
    /// on disk, so request-id deduplication and crash recovery keep working
    /// after compaction, and the checkpoint event (already appended by the
    /// caller) explains the gap. Only the in-memory vector is trimmed; the file
    /// is never rewritten here.
    pub fn compact(&mut self, keep_from_seq: u64, summary: &Json) -> Result<()> {
        let kept: Vec<AgentEvent> = self
            .events
            .iter()
            .filter(|e| e.seq >= keep_from_seq && e.kind.is_durable())
            .cloned()
            .collect();
        // Header bookkeeping stays in memory only — rewriting the first line
        // would require rewriting the whole file, which is exactly what
        // append-only forbids.
        self.header = self
            .header
            .clone()
            .with("compacted_at_seq", Json::int(keep_from_seq as i64))
            .with("compaction", summary.clone());
        self.events = kept;
        Ok(())
    }
}

/// `admitted_id` is a method-like helper kept out of the impl because it needs
/// `&AgentEvent` while `record` borrows self mutably.
fn admitted_id(event: &AgentEvent) -> Option<String> {
    if event.kind != EventKind::UserMessageAccepted {
        return None;
    }
    event.data.get("request_id")?.as_str().map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::protocol::now_ms;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pocketagent-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn user_event(turn: &str, request: &str, text: &str) -> AgentEvent {
        AgentEvent::new(
            EventKind::UserMessageAccepted,
            "s1",
            turn,
            Json::obj()
                .with("request_id", Json::str(request))
                .with("text", Json::str(text)),
        )
        .with_seq(now_ms())
    }

    #[test]
    fn assigns_monotonic_sequence_numbers() {
        let dir = temp_dir("seq");
        let mut log = SessionLog::open(&dir.join("s.jsonl"), "w1", "T").unwrap();
        let first = log.record(user_event("t1", "r1", "hello")).unwrap();
        let second = log.record(AgentEvent::new(
            EventKind::TurnStarted,
            "s1",
            "t1",
            Json::obj(),
        ))
        .unwrap();
        assert!(second.seq > first.seq);
        drop(log);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn admission_is_idempotent_across_reopen() {
        let dir = temp_dir("admit");
        let path = dir.join("s.jsonl");
        {
            let mut log = SessionLog::open(&path, "w1", "T").unwrap();
            log.record(user_event("t1", "r1", "hello")).unwrap();
        }
        let log = SessionLog::open(&path, "w1", "T").unwrap();
        assert!(log.is_admitted("r1"));
        assert!(!log.is_admitted("r2"));
        drop(log);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn drops_a_truncated_tail_but_keeps_complete_records() {
        let dir = temp_dir("trunc");
        let path = dir.join("s.jsonl");
        {
            let mut log = SessionLog::open(&path, "w1", "T").unwrap();
            log.record(user_event("t1", "r1", "hello")).unwrap();
            log.record(AgentEvent::new(EventKind::TurnEnded, "s1", "t1", Json::obj()))
                .unwrap();
        }
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"{\"v\":1,\"kind\":\"assistant.tex").unwrap();
        drop(file);

        let log = SessionLog::open(&path, "w1", "T").unwrap();
        assert_eq!(log.events().len(), 2);
        // The repaired file must be appendable again without a corrupt line.
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.ends_with('\n'));
        for line in text.lines() {
            assert!(parse(line).is_ok(), "line not parseable: {line}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deltas_stay_out_of_the_log() {
        let dir = temp_dir("delta");
        let path = dir.join("s.jsonl");
        let mut log = SessionLog::open(&path, "w1", "T").unwrap();
        log.record(AgentEvent::new(
            EventKind::DeltaText,
            "s1",
            "t1",
            Json::obj().with("text", Json::str("he")),
        ))
        .unwrap();
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert!(!on_disk.contains("delta.text"));
        drop(log);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compaction_keeps_a_record_of_what_was_dropped() {
        let dir = temp_dir("compact");
        let path = dir.join("s.jsonl");
        let mut log = SessionLog::open(&path, "w1", "T").unwrap();
        let first = log.record(user_event("t1", "r1", "hello")).unwrap();
        let last = log
            .record(AgentEvent::new(
                EventKind::AssistantText,
                "s1",
                "t1",
                Json::obj().with("text", Json::str("world")),
            ))
            .unwrap();
        log.compact(last.seq, &Json::obj().with("dropped_events", Json::int(1)))
            .unwrap();
        // In-memory working set is trimmed, header bookkeeping is in-memory.
        assert_eq!(log.events().len(), 1);
        assert_eq!(log.header().get("compacted_at_seq"), Some(&Json::int(last.seq as i64)));
        drop(log);

        // The file stays append-only: reopening sees every durable record, so
        // request-id deduplication and audit survive compaction.
        let reopened = SessionLog::open(&path, "w1", "T").unwrap();
        assert_eq!(reopened.events().len(), 2);
        assert!(reopened.is_admitted("r1"));
        assert!(first.seq < last.seq);
        std::fs::remove_dir_all(&dir).ok();
    }
}
