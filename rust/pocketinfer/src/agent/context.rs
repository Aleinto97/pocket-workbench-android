//! Context assembly, token budgeting and compaction.
//!
//! Three rules shape this module:
//!
//! * token counts come from the tokenizer that will actually run, not an
//!   estimate, so the budget means something;
//! * a tool call and its result always travel together — a half pair is a
//!   protocol violation the model cannot recover from;
//! * compaction is structural and factual. It summarises by naming what was
//!   dropped, and it never invents prose about what the user wanted.

use super::json::Json;
use super::protocol::{AgentEvent, ToolCall, ToolOutcome};

/// One item of the model's view of the conversation.
#[derive(Clone, Debug, PartialEq)]
pub enum TurnItem {
    System(String),
    User(String),
    Assistant { reasoning: String, text: String },
    /// The assistant asked for a tool. Always followed by a `ToolResult`.
    ToolCall(ToolCall),
    /// Produced by the runtime, never by the model.
    ToolResult(ToolOutcome),
    /// Factual record of a compaction that already happened.
    Summary(String),
}

impl TurnItem {
    /// Role as the chat template sees it.
    pub fn role(&self) -> &'static str {
        match self {
            TurnItem::System(_) => "system",
            TurnItem::User(_) => "user",
            TurnItem::Assistant { .. } => "assistant",
            TurnItem::ToolCall(_) => "assistant",
            TurnItem::ToolResult(_) => "tool",
            TurnItem::Summary(_) => "system",
        }
    }

    /// Text handed to the model. Historical reasoning is deliberately left out:
    /// re-sending a past chain of thought as if it were context wastes the
    /// window and invites the model to imitate it. The reasoning stays in the
    /// log, where the UI shows it on demand.
    pub fn text(&self) -> String {
        match self {
            TurnItem::System(text)
            | TurnItem::User(text)
            | TurnItem::Summary(text) => text.clone(),
            TurnItem::Assistant { text, .. } => text.clone(),
            TurnItem::ToolCall(call) => {
                format!(
                    "Calling {}.\n{}",
                    call.name,
                    call.arguments.to_string()
                )
            }
            TurnItem::ToolResult(outcome) => outcome.to_model_message(),
        }
    }

}

/// A compacted prefix, reported as a checkpoint event.
#[derive(Clone, Debug, PartialEq)]
pub struct Compaction {
    pub kept_from: usize,
    pub dropped_items: usize,
    pub summary: String,
    pub freed_tokens: usize,
}

impl Compaction {
    pub fn to_json(&self) -> Json {        Json::obj()
            .with("kept_from", Json::int(self.kept_from as i64))
            .with("dropped_items", Json::int(self.dropped_items as i64))
            .with("summary", Json::str(&self.summary))
            .with("freed_tokens", Json::int(self.freed_tokens as i64))
    }

    pub fn event(&self, session_id: &str, turn_id: &str, prefix_hash: u64) -> AgentEvent {
        AgentEvent::new(
            super::protocol::EventKind::Checkpoint,
            session_id,
            turn_id,
            self.to_json()
                .with("kind", Json::str("compaction"))
                .with("prefix_hash", Json::str(&format!("{prefix_hash:016x}"))),
        )
    }
}

/// A cut that would bring the conversation within `budget`, computed without
/// mutating anything so callers can gather evidence (model summary) first.
#[derive(Clone, Copy, Debug)]
pub struct CompactionPlan {
    pub drop_from: usize,
    pub cut: usize,
    pub freed_tokens: usize,
}

#[derive(Default)]
pub struct Conversation {
    pub items: Vec<TurnItem>,
}

impl Conversation {
    pub fn new(system: &str) -> Self {
        let mut items = Vec::new();
        if !system.trim().is_empty() {
            items.push(TurnItem::System(system.trim().to_string()));
        }
        Self { items }
    }

    pub fn system(&self) -> Option<&str> {
        self.items.iter().find_map(|item| match item {
            TurnItem::System(text) => Some(text.as_str()),
            _ => None,
        })
    }

    pub fn push_user(&mut self, text: &str) {
        self.items.push(TurnItem::User(text.to_string()));
    }

    /// Appends a second (or third) system note: project memory, skill lists.
    /// Leading system items are never dropped by compaction, so memory loaded
    /// here survives every pass without re-injection.
    pub fn push_system_note(&mut self, text: &str) {
        if !text.trim().is_empty() {
            self.items.push(TurnItem::System(text.trim().to_string()));
        }
    }

    /// FNV-1a 64 over the leading system items: the stable prefix the prefill
    /// cache depends on. The runtime logs it after every compaction; a changed
    /// hash means the next prefill pays full price instead of reusing KV.
    pub fn prefix_hash(&self) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for item in &self.items {
            match item {
                TurnItem::System(text) => {
                    for byte in text.as_bytes() {
                        hash ^= *byte as u64;
                        hash = hash.wrapping_mul(0x100000001b3);
                    }
                    // Item boundary so ["ab","c"] != ["a","bc"].
                    hash ^= 0xff;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
                _ => break,
            }
        }
        hash
    }

    /// Tokens per category with the tokenizer that will actually run, so the
    /// Stats page can show where the window goes instead of one opaque total.
    pub fn token_breakdown(&self, count: &dyn Fn(&str) -> usize) -> [(&'static str, usize); 5] {
        let mut system = 0usize;
        let mut user = 0usize;
        let mut assistant = 0usize;
        let mut tools = 0usize;
        let mut summary = 0usize;
        for item in &self.items {
            let cost = item_cost(item, count);
            match item {
                TurnItem::System(_) => system += cost,
                TurnItem::User(_) => user += cost,
                TurnItem::Assistant { .. } => assistant += cost,
                TurnItem::ToolCall(_) | TurnItem::ToolResult(_) => tools += cost,
                TurnItem::Summary(_) => summary += cost,
            }
        }
        [
            ("system", system),
            ("user", user),
            ("assistant", assistant),
            ("tools", tools),
            ("summary", summary),
        ]
    }

    /// Microcompaction (first tier, before `compact`): truncate old tool
    /// results to `max_lines` lines, keeping the calls and the newest
    /// `keep_recent` results verbatim. Tool outputs dominate context, so this
    /// reclaims the most tokens while dropping no turn at all. Returns the
    /// number of results truncated.
    ///
    /// Truncation targets the semantic content (`output`/`content` fields),
    /// not the serialised JSON, and keeps the object shape so replay and the
    /// model message render unchanged apart from the shorter text.
    pub fn microcompact(&mut self, keep_recent: usize, max_lines: usize) -> usize {
        let result_idx: Vec<usize> = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                matches!(item, TurnItem::ToolResult(_)).then_some(index)
            })
            .collect();
        if result_idx.len() <= keep_recent {
            return 0;
        }
        let cutoff = result_idx.len() - keep_recent;
        let mut truncated = 0usize;
        for (position, &index) in result_idx.iter().enumerate() {
            if position >= cutoff {
                break;
            }
            if let TurnItem::ToolResult(outcome) = &mut self.items[index] {
                if outcome.truncated {
                    continue;
                }
                if let Some(replacement) = micro_truncate(&outcome.result, max_lines) {
                    outcome.result = replacement;
                    outcome.truncated = true;
                    truncated += 1;
                }
            }
        }
        truncated
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Drops everything from `len` on, so a turn that died before doing
    /// anything leaves no trace in what the model will read next.
    pub fn truncate(&mut self, len: usize) {
        self.items.truncate(len.min(self.items.len()));
    }

    /// True when anything added from `from` on is a real effect the next turn
    /// must know about: a tool ran and returned, so its result stays even if
    /// the turn that ran it died afterwards.
    pub fn has_effects_since(&self, from: usize) -> bool {
        self.items.iter().skip(from).any(|item| {
            matches!(
                item,
                TurnItem::ToolResult(_) | TurnItem::Assistant { .. } | TurnItem::Summary(_)
            )
        })
    }

    pub fn push_assistant(&mut self, reasoning: &str, text: &str) {
        self.items.push(TurnItem::Assistant {
            reasoning: reasoning.to_string(),
            text: text.to_string(),
        });
    }

    pub fn push_tool_call(&mut self, call: ToolCall) {
        self.items.push(TurnItem::ToolCall(call));
    }

    pub fn push_tool_result(&mut self, outcome: ToolOutcome) {
        self.items.push(TurnItem::ToolResult(outcome));
    }

    /// True when a `ToolResult` has no preceding `ToolCall` to pair with, which
    /// happens after a crash between the two durable records.
    pub fn has_orphan_results(&self) -> bool {
        let mut pending = 0i32;
        for item in &self.items {
            match item {
                TurnItem::ToolCall(_) => pending += 1,
                TurnItem::ToolResult(outcome) => {
                    if !self.items.iter().any(|other| match other {
                        TurnItem::ToolCall(call) => call.id == outcome.call_id,
                        _ => false,
                    }) {
                        return true;
                    }
                    pending -= 1;
                }
                _ => {}
            }
        }
        pending != 0
    }

    /// Token count with the tokenizer that will actually run.
    pub fn token_count(&self, count: &dyn Fn(&str) -> usize) -> usize {
        self.items
            .iter()
            .map(|item| {
                let rendered = item.text();
                count(&rendered) + count(item.role())
            })
            .sum()
    }

    /// Drops the oldest part of the conversation until it fits `budget`,
    /// respecting pair boundaries and never touching the system instructions or
    /// the newest user message. Returns `None` when nothing can be dropped.
    ///
    /// Previous compaction notes are replaced, never stacked: each pass is
    /// built from current content only, so summarised drift cannot accumulate
    /// across passes. The session log keeps the full record.
    pub fn compact(&mut self, budget: usize, count: &dyn Fn(&str) -> usize) -> Option<Compaction> {
        self.compact_with(budget, count, &|dropped, replaced| summarise(dropped, replaced))
    }

    /// Same as `compact`, but the summary text comes from `summarize`. The
    /// model-summary path uses this; on any failure the caller falls back to
    /// the factual `compact`.
    pub fn compact_with(
        &mut self,
        budget: usize,
        count: &dyn Fn(&str) -> usize,
        summarize: &dyn Fn(&[TurnItem], usize) -> String,
    ) -> Option<Compaction> {
        let replaced = self.drop_stale_summaries();
        let plan = self.plan_compact(budget, count)?;
        let summary = summarize(&self.items[plan.drop_from..plan.cut], replaced);
        Some(self.apply_summary(&plan, summary))
    }

    /// Removes stale compaction notes and reports how many there were. Never
    /// compact a compaction: each pass is built from current content only, so
    /// summarised drift cannot accumulate. The session log keeps the record.
    pub fn drop_stale_summaries(&mut self) -> usize {
        let replaced: usize = self
            .items
            .iter()
            .filter(|item| matches!(item, TurnItem::Summary(_)))
            .count();
        if replaced > 0 {
            self.items.retain(|item| !matches!(item, TurnItem::Summary(_)));
        }
        replaced
    }

    /// A cut that would bring the conversation within `budget`, without
    /// mutating anything. Pure so the model-summary path can gather evidence
    /// before calling the backend.
    pub fn plan_compact(
        &self,
        budget: usize,
        count: &dyn Fn(&str) -> usize,
    ) -> Option<CompactionPlan> {
        let costs: Vec<usize> = self.items.iter().map(|item| item_cost(item, count)).collect();
        let total: usize = costs.iter().sum();
        if total <= budget {
            return None;
        }
        let drop_from = self.first_droppable();
        let drop_until = self.last_protected();
        if drop_from >= drop_until {
            return None;
        }
        // Cut boundaries are precomputed so the budget check is a subtraction
        // instead of a full recount per iteration.
        let mut boundaries: Vec<(usize, usize)> = vec![(0, 0)];
        let mut kept: usize = 0;
        for index in 0..self.items.len() {
            kept += costs[index];
            if index + 1 >= self.items.len() || is_cut_boundary(&self.items, index) {
                boundaries.push((index + 1, kept));
            }
        }
        // Smallest sufficient cut: removing more history than the budget forces
        // would throw away context the model could still have used.
        let mut cut = None;
        for (boundary, cost) in boundaries.iter() {
            if *boundary > drop_from && *boundary <= drop_until && total - cost <= budget {
                cut = Some(*boundary);
                break;
            }
        }
        let cut = cut?;
        let freed_tokens: usize = costs[drop_from..cut].iter().sum();
        Some(CompactionPlan {
            drop_from,
            cut,
            freed_tokens,
        })
    }

    /// Drains the planned range and inserts the summary in its place.
    pub fn apply_summary(&mut self, plan: &CompactionPlan, summary: String) -> Compaction {
        let dropped_len = plan.cut - plan.drop_from;
        self.items.drain(plan.drop_from..plan.cut);
        // The summary goes in as its own item so the transcript can explain
        // the gap. The next pass replaces it (see `drop_stale_summaries`).
        self.items
            .insert(plan.drop_from, TurnItem::Summary(summary.clone()));
        Compaction {
            kept_from: plan.cut,
            dropped_items: dropped_len,
            summary,
            freed_tokens: plan.freed_tokens,
        }
    }

    /// Earliest index that may be dropped: never the system instructions and
    /// never a previous compaction summary, which is the record of what the
    /// earlier summary lost.
    fn first_droppable(&self) -> usize {
        let mut index = 0usize;
        while index < self.items.len()
            && matches!(self.items[index], TurnItem::System(_) | TurnItem::Summary(_))
        {
            index += 1;
        }
        index
    }

    /// Latest index that may be dropped. Dropping the newest user message, or
    /// anything after it, would leave the turn being answered out of context.
    fn last_protected(&self) -> usize {
        let last_user = self
            .items
            .iter()
            .rposition(|item| matches!(item, TurnItem::User(_)));
        match last_user {
            Some(position) => position,
            // No user message to protect: everything but the tail may go.
            None => self.items.len(),
        }
    }

    /// The messages handed to the backend, in template order.
    pub fn render(&self) -> Vec<(String, String)> {
        self.items
            .iter()
            .map(|item| (item.role().to_string(), item.text()))
            .collect()
    }
}

/// A cut may land after any item except a `ToolCall` that has not yet been
/// followed by its result. Cutting after a result is always safe: the call is
/// earlier in the list, so both stay on the same side of the cut.
fn is_cut_boundary(items: &[TurnItem], index: usize) -> bool {
    !matches!(items[index], TurnItem::ToolCall(_))
}

fn item_cost(item: &TurnItem, count: &dyn Fn(&str) -> usize) -> usize {
    count(&item.text()) + count(item.role())
}

/// Truncates a tool result to `max_lines` of semantic content, preserving the
/// object shape. Returns `None` when the result already fits.
fn micro_truncate(result: &Json, max_lines: usize) -> Option<Json> {
    if let Json::Obj(fields) = result {
        for key in ["output", "content"] {
            if let Some(text) = fields
                .iter()
                .find_map(|(k, v)| (k == key).then(|| v.as_str()).flatten())
            {
                let lines: Vec<&str> = text.lines().collect();
                if lines.len() <= max_lines {
                    return None;
                }
                let kept = lines[..max_lines].join("\n");
                let dropped = lines.len() - max_lines;
                let mut out = Json::obj();
                for (k, v) in fields {
                    if k == key {
                        out = out.with(k, Json::str(&kept));
                    } else {
                        out = out.with(k, v.clone());
                    }
                }
                out = out.with(
                    "note",
                    Json::str(&format!(
                        "{dropped} more lines in the session log; re-read the file or narrow the command."
                    )),
                );
                return Some(out);
            }
        }
    }
    // No text field (or not an object): fall back to whole-JSON lines.
    let text = result.to_string();
    let line_count = text.lines().count();
    if line_count <= max_lines {
        return None;
    }
    let kept: String = text.lines().take(max_lines).collect::<Vec<_>>().join("\n");
    let dropped = line_count - max_lines;
    Some(
        Json::obj()
            .with("truncated_output", Json::str(&kept))
            .with(
                "note",
                Json::str(&format!(
                    "{dropped} more lines in the session log; re-read the file or narrow the command."
                )),
            ),
    )
}

/// Factual, non-inventive summary of what compaction dropped. It names counts and
/// the paths and tools involved, because a wrong guess about the user's intent
/// would be worse than admitting the detail is gone.
pub(crate) fn summarise(dropped: &[TurnItem], replaced: usize) -> String {
    let mut turns = 0usize;
    let mut calls: Vec<String> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    let mut failures = 0usize;
    for item in dropped {
        match item {
            TurnItem::User(_) => turns += 1,
            TurnItem::ToolCall(call) => calls.push(call.name.clone()),
            TurnItem::ToolResult(outcome) => {
                if !outcome.ok {
                    failures += 1;
                }
                collect_paths(&outcome.result, &mut files);
            }
            TurnItem::Assistant { text, .. } => collect_paths_str(text, &mut files),
            _ => {}
        }
    }
    calls.sort();
    calls.dedup();
    files.sort();
    files.dedup();
    let mut lines = vec![format!(
        "[compacted] {turns} earlier user message(s), {} tool call(s){}.",
        calls.len(),
        if failures > 0 {
            format!(", {failures} failed")
        } else {
            String::new()
        }
    )];
    if !calls.is_empty() {
        lines.push(format!("Tools used: {}.", calls.join(", ")));
    }
    if !files.is_empty() {
        let shown: Vec<String> = files.iter().take(12).cloned().collect();
        let more = files.len().saturating_sub(shown.len());
        lines.push(format!(
            "Files involved: {}{}.",
            shown.join(", "),
            if more > 0 { format!(" (+{more} more)") } else { String::new() }
        ));
    }
    lines.push("The full text of this part is still in the session log.".to_string());
    if replaced > 0 {
        lines.push(format!(
            "Replaces {replaced} earlier compaction note(s); their detail stays in the session log."
        ));
    }
    lines.join(" ")
}

fn collect_paths(value: &Json, out: &mut Vec<String>) {
    if let Some(path) = value.get("path").and_then(|p| p.as_str()) {
        if !path.is_empty() {
            out.push(path.to_string());
        }
    }
    if let Some(items) = value.get("items").and_then(|i| i.as_array()) {
        for item in items {
            collect_paths(item, out);
        }
    }
    if let Some(matches) = value.get("matches").and_then(|m| m.as_array()) {
        for item in matches {
            collect_paths(item, out);
        }
    }
}

fn collect_paths_str(text: &str, out: &mut Vec<String>) {
    // Only single-token paths: no parsing, no guessing.
    for word in text.split_whitespace() {
        let trimmed = word.trim_matches(|c: char| matches!(c, '`' | '"' | '\'' | ',' | ')' | '('));
        if (trimmed.contains('/') || trimmed.contains('.'))
            && trimmed.len() <= 120
            && trimmed
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "._-/".contains(c))
        {
            out.push(trimmed.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::json::parse;
    use crate::agent::protocol::ToolCall;

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: parse(r#"{"path":"a.txt"}"#).unwrap(),
        }
    }

    fn ok(id: &str) -> ToolOutcome {
        ToolOutcome::ok(id, "fs_read", parse(r#"{"path":"src/main.rs"}"#).unwrap(), 1)
    }

    fn count(text: &str) -> usize {
        text.split_whitespace().count()
    }

    #[test]
    fn no_compaction_when_it_already_fits() {
        let mut conversation = Conversation::new("You are a local agent.");
        conversation.push_user("hello");
        assert!(conversation.compact(1000, &count).is_none());
        assert_eq!(conversation.items.len(), 2);
    }

    #[test]
    fn never_drops_the_system_instructions_or_the_last_request() {
        let mut conversation = Conversation::new("SYSTEM RULES");
        for index in 0..6 {
            conversation.push_user(&format!("request number {index} with some words"));
            conversation.push_assistant("", &format!("answer number {index} with more words"));
        }
        let compaction = conversation.compact(24, &count).unwrap();
        assert!(compaction.dropped_items > 0);
        assert_eq!(conversation.system(), Some("SYSTEM RULES"));
        let last_user = conversation
            .items
            .iter()
            .filter_map(|item| match item {
                TurnItem::User(text) => Some(text.clone()),
                _ => None,
            })
            .next_back()
            .unwrap();
        assert_eq!(last_user, "request number 5 with some words");
    }

    #[test]
    fn keeps_tool_calls_paired_with_their_results() {
        let mut conversation = Conversation::new("SYSTEM");
        conversation.push_user("first");
        conversation.push_assistant("", "sure");
        for index in 0..8 {
            let id = format!("call{index}");
            conversation.push_tool_call(call(&id, "fs_read"));
            conversation.push_tool_result(ok(&id));
            conversation.push_assistant("", "done with that");
        }
        conversation.push_user("now summarise what you read");
        // A budget that leaves roughly half the history: pairs must survive, and
        // survive whole.
        assert!(conversation.compact(45, &count).is_some());
        let pairs = paired_ids(&conversation);
        assert!(!pairs.is_empty(), "compaction should not have eaten every pair");
        assert_eq!(pairs.len(), conversation.items.iter().filter(|item| matches!(item, TurnItem::ToolCall(_))).count());
        assert!(!conversation.has_orphan_results());
    }

    #[test]
    fn a_tight_budget_drops_everything_but_the_request_in_flight() {
        let mut conversation = Conversation::new("SYSTEM");
        conversation.push_user("first");
        for index in 0..8 {
            let id = format!("call{index}");
            conversation.push_tool_call(call(&id, "fs_read"));
            conversation.push_tool_result(ok(&id));
        }
        conversation.push_user("now summarise what you read");
        assert!(conversation.compact(12, &count).is_some());
        let survivors = paired_ids(&conversation);
        assert!(survivors.len() <= 1, "a tight budget should leave almost nothing");
        assert!(!conversation.has_orphan_results());
        assert_eq!(
            conversation
                .items
                .iter()
                .filter(|item| matches!(item, TurnItem::User(_)))
                .count(),
            1
        );
    }

    fn paired_ids(conversation: &Conversation) -> Vec<String> {
        conversation
            .items
            .iter()
            .filter_map(|item| match item {
                TurnItem::ToolResult(outcome) => Some(outcome.call_id.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn summary_is_factual_and_mentions_the_log() {
        let mut conversation = Conversation::new("SYSTEM");
        conversation.push_user("please read src/main.rs");
        conversation.push_tool_call(call("c1", "fs_read"));
        conversation.push_tool_result(ok("c1"));
        conversation.push_assistant("", "read it");
        conversation.push_user("now edit it");
        conversation.push_assistant("", "edited");
        conversation.push_user("thanks");
        conversation.compact(6, &count).unwrap();
        let summary = conversation
            .items
            .iter()
            .find_map(|item| match item {
                TurnItem::Summary(text) => Some(text.clone()),
                _ => None,
            })
            .expect("a summary item");
        assert!(summary.contains("fs_read"));
        assert!(summary.contains("src/main.rs"));
        assert!(summary.contains("session log"));
    }

    #[test]
    fn historical_reasoning_is_not_replayed_as_an_instruction() {
        let mut conversation = Conversation::new("SYSTEM");
        conversation.push_assistant("deep private reasoning", "the answer");
        let rendered = conversation.render();
        assert_eq!(rendered[1].1, "the answer");
        assert!(!rendered[1].1.contains("private"));
    }

    #[test]
    fn detects_an_orphan_result_left_by_a_crash() {
        let mut conversation = Conversation::new("read it");
        conversation.push_user("read it");
        conversation.push_tool_result(ok("ghost"));
        assert!(conversation.has_orphan_results());
    }

    #[test]
    fn prefix_hash_is_stable_and_changes_with_system() {
        let mut first = Conversation::new("SYSTEM RULES");
        first.push_user("hello");
        let mut second = Conversation::new("SYSTEM RULES");
        second.push_user("something else entirely");
        assert_eq!(first.prefix_hash(), second.prefix_hash());
        // Project memory loaded at open sits before any user message, so it
        // is part of the stable prefix.
        let mut third = Conversation::new("SYSTEM RULES");
        third.push_system_note("extra memory");
        third.push_user("hello");
        assert_ne!(first.prefix_hash(), third.prefix_hash());
    }

    #[test]
    fn microcompact_truncates_old_results_and_keeps_pairs() {
        let mut conversation = Conversation::new("SYSTEM");
        conversation.push_user("go");
        for index in 0..4 {
            let id = format!("call{index}");
            conversation.push_tool_call(call(&id, "shell"));
            let mut outcome = ok(&id);
            outcome.result =
                parse(r#"{"output":"l0\nl1\nl2\nl3\nl4\nl5\nl6\nl7"}"#).unwrap();
            conversation.push_tool_result(outcome);
        }
        let truncated = conversation.microcompact(1, 3);
        assert_eq!(truncated, 3);
        assert!(!conversation.has_orphan_results());
        // The newest result stays verbatim.
        let last = conversation
            .items
            .iter()
            .filter_map(|item| match item {
                TurnItem::ToolResult(outcome) => Some(outcome),
                _ => None,
            })
            .next_back()
            .unwrap();
        assert!(!last.truncated);
        // An old one keeps its shape with shorter content.
        let first = conversation
            .items
            .iter()
            .filter_map(|item| match item {
                TurnItem::ToolResult(outcome) => Some(outcome),
                _ => None,
            })
            .next()
            .unwrap();
        assert!(first.truncated);
        assert_eq!(
            first
                .result
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap()
                .lines()
                .count(),
            3
        );
    }

    #[test]
    fn second_compaction_replaces_the_first_note() {
        let mut conversation = Conversation::new("SYSTEM");
        for index in 0..10 {
            conversation.push_user(&format!("request number {index} with some words"));
            conversation.push_assistant("", &format!("answer number {index} with more words"));
        }
        assert!(conversation.compact(40, &count).is_some());
        assert_eq!(
            conversation
                .items
                .iter()
                .filter(|item| matches!(item, TurnItem::Summary(_)))
                .count(),
            1
        );
        // More turns arrive, then compact again: still exactly one note.
        for index in 10..16 {
            conversation.push_user(&format!("request number {index} with some words"));
            conversation.push_assistant("", &format!("answer number {index} with more words"));
        }
        let second = conversation.compact(40, &count).unwrap();
        assert!(second.summary.contains("Replaces 1 earlier compaction note"));
        assert_eq!(
            conversation
                .items
                .iter()
                .filter(|item| matches!(item, TurnItem::Summary(_)))
                .count(),
            1
        );
    }
}
