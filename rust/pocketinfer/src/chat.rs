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

/// Model family for chat-template routing. The pre-tokenizer name and the
/// model filename are all the backend knows; both are checked.
pub fn family_of(pre: &str, name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    if pre == "minicpm5" || n.contains("minicpm") {
        "minicpm"
    } else if n.contains("tinyllama") {
        // Its own `<|user|>` format, not llama-2-chat despite the architecture.
        "tinyllama"
    } else if n.contains("llama-2") || n.contains("llama2") || n.contains("llama-1") {
        "llama2"
    } else if pre == "llama3" || n.contains("llama") {
        "llama3"
    } else {
        // ChatML (`<|im_start|>`): suits Qwen and every other instruct model
        // without a dedicated template.
        "generic"
    }
}

/// Routes to the template of the model's family. Using the wrong template
/// (e.g. minicpm5 tags on a llama-2 model) makes the model echo unrelated
/// instructions instead of answering.
pub fn apply_for(family: &str, messages: &[Message], add_generation_prompt: bool) -> String {
    match family {
        "minicpm" => apply_minicpm5(messages, add_generation_prompt),
        "tinyllama" => apply_tinyllama(messages, add_generation_prompt),
        "llama2" => apply_llama2(messages, add_generation_prompt),
        "llama3" => apply_llama3(messages, add_generation_prompt),
        _ => apply_generic(messages, add_generation_prompt),
    }
}

/// TinyLlama-Chat's native format (`<|user|>`). Despite the llama
/// architecture it was not trained on llama-2-chat tags: with `[INST]` it
/// echoes quiz text instead of answering.
pub fn apply_tinyllama(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for m in messages {
        match m.role.as_str() {
            "system" => {
                out.push_str("<|system|>\n");
                out.push_str(m.content.trim());
                out.push_str("</s>\n");
            }
            "user" => {
                out.push_str("<|user|>\n");
                out.push_str(m.content.trim());
                out.push_str("</s>\n");
            }
            "assistant" => {
                out.push_str("<|assistant|>\n");
                out.push_str(strip_think(&m.content).trim());
                out.push_str("</s>\n");
            }
            _ => {
                out.push_str("<|user|>\n[tool result]\n");
                out.push_str(m.content.trim());
                out.push_str("</s>\n");
            }
        }
    }
    if add_generation_prompt {
        out.push_str("<|assistant|>\n");
    }
    out
}

/// Llama-2-chat (`[INST]`) template. Tool results fold into user turns: the
/// format has no tool role, and inventing new markers teaches the model
/// tokens it never saw in training.
pub fn apply_llama2(messages: &[Message], _add_generation_prompt: bool) -> String {
    let systems: Vec<&str> = messages
        .iter()
        .filter(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .collect();
    let mut out = String::from("<s>");
    let mut sys_emitted = false;
    for m in messages.iter().filter(|m| m.role != "system") {
        match m.role.as_str() {
            "user" => {
                out.push_str("[INST] ");
                if !sys_emitted {
                    sys_emitted = true;
                    if !systems.is_empty() {
                        out.push_str("<<SYS>>\n");
                        out.push_str(&systems.join("\n"));
                        out.push_str("\n<</SYS>>\n\n");
                    }
                }
                out.push_str(m.content.trim());
                out.push_str(" [/INST]");
            }
            "assistant" => {
                out.push(' ');
                out.push_str(strip_think(&m.content).trim());
                out.push_str(" </s>");
            }
            _ => {
                // Tool results and anything else become user turns. The last
                // user-side turn leaves the prompt open for generation, so no
                // extra generation prompt is needed.
                out.push_str("<s>[INST] [tool result]\n");
                if !sys_emitted {
                    sys_emitted = true;
                    if !systems.is_empty() {
                        out.push_str("<<SYS>>\n");
                        out.push_str(&systems.join("\n"));
                        out.push_str("\n<</SYS>>\n\n");
                    }
                }
                out.push_str(m.content.trim());
                out.push_str(" [/INST]");
            }
        }
    }
    out
}

/// Llama-3 (`<|start_header_id|>`) template.
pub fn apply_llama3(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for m in messages {
        let header = match m.role.as_str() {
            "system" => "system",
            "assistant" => "assistant",
            _ => "user",
        };
        out.push_str("<|start_header_id|>");
        out.push_str(header);
        out.push_str("<|end_header_id|>\n\n");
        if m.role == "tool" {
            out.push_str("[tool result]\n");
        }
        let body = match m.role.as_str() {
            "assistant" => strip_think(&m.content),
            _ => m.content.clone(),
        };
        out.push_str(body.trim());
        out.push_str("<|eot_id|>");
    }
    if add_generation_prompt {
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    out
}

/// Removes `<think>...</think>` sections from replayed assistant text: the
/// reasoning stays in the session log, and other families must not imitate
/// minicpm's thinking tags.
fn strip_think(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            Some(end) => rest = &rest[start + end + "</think>".len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

pub fn apply_minicpm5(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut out = String::from("<s>");
    if let Some(first) = messages.first() {
        if first.role == "system" {
            out.push_str("<|im_start|>system\n");
            out.push_str(&first.content);
            out.push_str("<|im_end|>\n");
        }
    }
    let mut i = 0usize;
    while i < messages.len() {
        let m = &messages[i];
        if m.role == "user" || (m.role == "system" && i > 0) {
            out.push_str("<|im_start|>");
            out.push_str(&m.role);
            out.push('\n');
            out.push_str(&m.content);
            out.push_str("<|im_end|>\n");
        } else if m.role == "assistant" {
            let mut content = m.content.clone();
            let mut reasoning = String::new();
            if content.contains("</think>") {
                let head = content.split("</think>").next().unwrap_or("");
                reasoning = head.split("<think>").last().unwrap_or("").to_string();
                reasoning = rstrip_nl(&reasoning).to_string();
                content = lstrip_nl(content.split("</think>").last().unwrap_or("")).to_string();
            }
            out.push_str("<|im_start|>assistant\n");
            if !reasoning.is_empty() {
                out.push_str("<think>\n");
                out.push_str(reasoning.trim_matches('\n'));
                out.push_str("\n</think>\n\n");
                out.push_str(lstrip_nl(&content));
            } else if !content.contains("<think>") && !content.contains("</think>") {
                out.push_str("<think>\n\n</think>\n\n");
                out.push_str(lstrip_nl(&content));
            } else {
                out.push_str(&content);
            }
            out.push_str("<|im_end|>\n");
        } else if m.role == "tool" {
            out.push_str("<|im_start|>user");
            while i < messages.len() && messages[i].role == "tool" {
                out.push_str("\n");
                out.push_str("<tool_response>\n");
                out.push_str(&messages[i].content);
                out.push_str("\n");
                out.push_str("</tool_response>");
                i += 1;
            }
            out.push_str("<|im_end|>\n");
            continue;
        }
        i += 1;
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
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

pub fn thinking_on_suffix() -> &'static str {
    "<think>\n"
}

pub fn direct_suffix() -> &'static str {
    "<think>\n\n</think>\n\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Message {
        Message { role: "user".into(), content: text.into() }
    }

    #[test]
    fn family_routing_picks_the_right_template() {
        assert_eq!(family_of("minicpm5", "MiniCPM5-2B"), "minicpm");
        assert_eq!(family_of("gpt-2", "tinyllama-1.1b-chat.Q4_K_M"), "tinyllama");
        assert_eq!(family_of("llama3", "Llama-3.2-3B-Instruct"), "llama3");
        assert_eq!(family_of("qwen2", "Qwen3-4B"), "generic");
    }

    #[test]
    fn llama2_wraps_system_and_alternates() {
        let prompt = apply_llama2(
            &[
                Message { role: "system".into(), content: "Be brief.".into() },
                user("Hi"),
            ],
            true,
        );
        assert!(prompt.starts_with("<s>[INST] <<SYS>>"));
        assert!(prompt.contains("Be brief."));
        assert!(prompt.ends_with("Hi [/INST]"), "open for generation: {prompt}");
    }

    #[test]
    fn llama2_folds_tools_into_user_turns() {
        let prompt = apply_llama2(
            &[
                user("List files"),
                Message { role: "tool".into(), content: "{\"files\":[]}".into() },
            ],
            true,
        );
        assert!(prompt.contains("[tool result]"), "{prompt}");
        assert!(!prompt.contains("<|im_start|>"), "no ChatML leakage: {prompt}");
    }

    #[test]
    fn llama3_uses_headers_and_generation_prompt() {
        let prompt = apply_llama3(
            &[
                Message { role: "system".into(), content: "S.".into() },
                user("Q"),
            ],
            true,
        );
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn think_tags_are_stripped_for_other_families() {
        let prompt = apply_llama3(
            &[Message {
                role: "assistant".into(),
                content: "<think>\nsecret\n</think>\n\nanswer".into(),
            }],
            false,
        );
        assert!(!prompt.contains("secret"), "{prompt}");
        assert!(prompt.contains("answer"), "{prompt}");
    }
}

#[cfg(test)]
mod tinyllama_tests {
    use super::*;

    #[test]
    fn native_format_ends_open_for_generation() {
        let prompt = apply_tinyllama(
            &[Message { role: "user".into(), content: "Hi".into() }],
            true,
        );
        assert!(prompt.contains("<|user|>\nHi</s>"), "{prompt}");
        assert!(prompt.ends_with("<|assistant|>\n"), "{prompt}");
        assert!(!prompt.contains("[INST]"), "{prompt}");
    }
}
