#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

fn lstrip_nl(s: &str) -> &str {
    s.trim_start_matches('\n')
}

fn rstrip_nl(s: &str) -> &str {
    s.trim_end_matches('\n')
}

pub fn is_minicpm5(pre: &str, name: &str) -> bool {
    pre == "minicpm5" || name.to_ascii_lowercase().contains("minicpm5")
}

pub fn apply_minicpm5(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut out = String::from("<s>");
    if let Some(first) = messages.first() {
        if first.role == "system" {
            out.push_str("system\n");
            out.push_str(&first.content);
            out.push('\n');
        }
    }
    let mut i = 0usize;
    while i < messages.len() {
        let m = &messages[i];
        if m.role == "user" || (m.role == "system" && i > 0) {
            out.push_str(&m.role);
            out.push('\n');
            out.push_str(&m.content);
            out.push('\n');
        } else if m.role == "assistant" {
            let mut content = m.content.clone();
            let mut reasoning = String::new();
            if content.contains("</think>") {
                let head = content.split("</think>").next().unwrap_or("");
                reasoning = head.split(" thinking").last().unwrap_or("").to_string();
                reasoning = rstrip_nl(&reasoning).to_string();
                content = lstrip_nl(content.split("</think>").last().unwrap_or("")).to_string();
            }
            if !reasoning.is_empty() {
                out.push_str("assistant\n<think>\n");
                out.push_str(reasoning.trim_matches('\n'));
                out.push_str("\n\n\n");
                out.push_str(lstrip_nl(&content));
            } else if !content.contains(" thinking") && !content.contains("</think>") {
                out.push_str("assistant\n thinking\n\n\n\n");
                out.push_str(lstrip_nl(&content));
            } else {
                out.push_str("assistant\n");
                out.push_str(&content);
            }
            out.push('\n');
        } else if m.role == "tool" {
            out.push_str("user");
            while i < messages.len() && messages[i].role == "tool" {
                out.push_str("\n<tool_response>\n");
                out.push_str(&messages[i].content);
                out.push_str("\n</tool_response>");
                i += 1;
            }
            out.push('\n');
            continue;
        }
        i += 1;
    }
    if add_generation_prompt {
        out.push_str("assistant\n");
    }
    out
}

pub fn apply_generic(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(&m.role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

pub fn apply(messages: &[Message], minicpm5: bool, add_generation_prompt: bool) -> String {
    if minicpm5 {
        apply_minicpm5(messages, add_generation_prompt)
    } else {
        apply_generic(messages, add_generation_prompt)
    }
}
