use crate::gguf::Gguf;
use crate::regexlite;
use crate::util::Result;
use std::collections::HashMap;

const PRE_MINICPM5: [&str; 2] = [
    "\\p{N}{1,3}",
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

const PRE_QWEN2: [&str; 1] = [
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

const PRE_GPT2: [&str; 1] = [
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
];

const PRE_LLAMA3: [&str; 1] = [
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

pub struct Tokenizer {
    pub tokens: Vec<Vec<u8>>,
    pub token_types: Vec<i32>,
    token_to_id: HashMap<Vec<u8>, u32>,
    merges: HashMap<(Vec<u8>, Vec<u8>), u32>,
    byte_decoder: HashMap<char, u8>,
    byte_encoder: [char; 256],
    byte_to_id: [u32; 256],
    specials: Vec<(Vec<u8>, u32)>,
    patterns: Vec<&'static str>,
    pub pre: String,
    pub ignore_merges: bool,
    pub bos_id: Option<u32>,
    pub eos_ids: Vec<u32>,
    pub n_vocab: usize,
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let tokens = g.array_strings("tokenizer.ggml.tokens")?;
        let token_types = g.array_i32("tokenizer.ggml.token_type").unwrap_or_else(|_| vec![1; tokens.len()]);
        let merges_raw = g.array_strings("tokenizer.ggml.merges").unwrap_or_default();
        let mut merges = HashMap::with_capacity(merges_raw.len());
        for (i, m) in merges_raw.iter().enumerate() {
            if let Some(pos) = m.iter().skip(1).position(|c| *c == b' ') {
                let pos = pos + 1;
                merges.insert((m[..pos].to_vec(), m[pos + 1..].to_vec()), i as u32);
            }
        }
        let (byte_encoder, byte_decoder) = build_byte_map();
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut byte_to_id = [u32::MAX; 256];
        for (i, t) in tokens.iter().enumerate() {
            if !token_to_id.contains_key(t) {
                token_to_id.insert(t.clone(), i as u32);
            }
            if let Ok(s) = core::str::from_utf8(t) {
                let chars: Vec<char> = s.chars().collect();
                if chars.len() == 1 {
                    if let Some(b) = byte_decoder.get(&chars[0]) {
                        if byte_to_id[*b as usize] == u32::MAX {
                            byte_to_id[*b as usize] = i as u32;
                        }
                    }
                }
            }
        }
        let mut specials = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            let tt = token_types[i];
            if tt == 2 || tt == 3 || tt == 4 {
                specials.push((t.clone(), i as u32));
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let pre = g.get_str_string("tokenizer.ggml.pre").unwrap_or_default();
        let patterns = patterns_for(&pre);
        let ignore_merges = pre == "minicpm5";
        let bos_id = g.get_u32("tokenizer.ggml.bos_token_id");
        let mut eos_ids = Vec::new();
        if let Some(e) = g.get_u32("tokenizer.ggml.eos_token_id") {
            eos_ids.push(e);
        }
        for (t, i) in specials.iter() {
            if t.as_slice() == b"<|im_end|>" || t.as_slice() == b"<|eot|>" || t.as_slice() == b"<|endoftext|>" {
                eos_ids.push(*i);
            }
        }
        eos_ids.sort_unstable();
        eos_ids.dedup();
        Ok(Self {
            byte_encoder,
            n_vocab: tokens.len(),
            tokens,
            token_types,
            token_to_id,
            merges,
            byte_decoder,
            byte_to_id,
            specials,
            patterns,
            pre: pre.clone(),
            ignore_merges,
            bos_id,
            eos_ids,
        })
    }

    fn encode_word(&self, word: &str, out: &mut Vec<u32>) {
        let encoded: Vec<u8> = {
            let mut v = Vec::with_capacity(word.len() * 2);
            for b in word.as_bytes() {
                let mut buf = [0u8; 4];
                v.extend_from_slice(self.byte_encoder[*b as usize].encode_utf8(&mut buf).as_bytes());
            }
            v
        };
        if self.ignore_merges {
            if let Some(id) = self.token_to_id.get(&encoded) {
                out.push(*id);
                return;
            }
        }
        let mut symbols: Vec<Vec<u8>> = Vec::new();
        let s = unsafe { core::str::from_utf8_unchecked(&encoded) };
        for ch in s.chars() {
            let mut buf = [0u8; 4];
            symbols.push(ch.encode_utf8(&mut buf).as_bytes().to_vec());
        }
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let key = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(r) = self.merges.get(&key) {
                    if best.map(|(br, _)| *r < br).unwrap_or(true) {
                        best = Some((*r, i));
                    }
                }
            }
            match best {
                Some((_, i)) => {
                    let right = symbols.remove(i + 1);
                    symbols[i].extend_from_slice(&right);
                }
                None => break,
            }
        }
        for sym in &symbols {
            if let Some(id) = self.token_to_id.get(sym) {
                out.push(*id);
                continue;
            }
            if let Ok(s) = core::str::from_utf8(sym) {
                for ch in s.chars() {
                    let mut buf = [0u8; 4];
                    let cb = ch.encode_utf8(&mut buf).as_bytes().to_vec();
                    if let Some(id) = self.token_to_id.get(&cb) {
                        out.push(*id);
                    } else if let Some(b) = self.byte_decoder.get(&ch) {
                        let id = self.byte_to_id[*b as usize];
                        if id != u32::MAX {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let mut words: Vec<String> = vec![text.to_string()];
        for pat in &self.patterns {
            let mut next = Vec::new();
            for w in &words {
                for part in regexlite::split(pat, w) {
                    if !part.is_empty() {
                        next.push(part);
                    }
                }
            }
            words = next;
        }
        for w in &words {
            self.encode_word(w, out);
        }
    }

    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if !parse_special || self.specials.is_empty() {
            self.encode_plain(text, &mut out);
            return out;
        }
        let bytes = text.as_bytes();
        let mut pos = 0usize;
        let mut plain_start = 0usize;
        while pos < bytes.len() {
            let mut matched: Option<(usize, u32)> = None;
            for (s, id) in &self.specials {
                if !s.is_empty() && pos + s.len() <= bytes.len() && &bytes[pos..pos + s.len()] == s.as_slice() {
                    matched = Some((s.len(), *id));
                    break;
                }
            }
            if let Some((len, id)) = matched {
                if plain_start < pos {
                    self.encode_plain(&text[plain_start..pos], &mut out);
                }
                out.push(id);
                pos += len;
                plain_start = pos;
            } else {
                let ch = text[pos..].chars().next().unwrap();
                pos += ch.len_utf8();
            }
        }
        if plain_start < bytes.len() {
            self.encode_plain(&text[plain_start..], &mut out);
        }
        out
    }

    pub fn decode_token(&self, id: u32) -> Vec<u8> {
        let idx = id as usize;
        if idx >= self.tokens.len() {
            return Vec::new();
        }
        let tt = self.token_types[idx];
        if tt == 2 || tt == 3 || tt == 4 {
            return self.tokens[idx].clone();
        }
        let tok = &self.tokens[idx];
        let mut out = Vec::with_capacity(tok.len());
        if let Ok(s) = core::str::from_utf8(tok) {
            for ch in s.chars() {
                match self.byte_decoder.get(&ch) {
                    Some(b) => out.push(*b),
                    None => {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        } else {
            out.extend_from_slice(tok);
        }
        out
    }

    pub fn is_eog(&self, id: u32) -> bool {
        self.eos_ids.contains(&id)
    }

    pub fn is_special(&self, id: u32) -> bool {
        let tt = self.token_types[id as usize];
        tt == 2 || tt == 3 || tt == 4
    }
}

fn patterns_for(pre: &str) -> Vec<&'static str> {
    match pre {
        "minicpm5" => PRE_MINICPM5.to_vec(),
        "qwen2" | "qwen2.5" | "qwen3" => PRE_QWEN2.to_vec(),
        "llama3" | "llama-bpe" | "llama-v3" => PRE_LLAMA3.to_vec(),
        "gpt-2" | "phi-2" => PRE_GPT2.to_vec(),
        _ => PRE_GPT2.to_vec(),
    }
}

fn build_byte_map() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::new();
    for b in b'!'..=b'~' {
        bs.push(b as u32);
    }
    for b in 0xA1u32..=0xAC {
        bs.push(b);
    }
    for b in 0xAEu32..=0xFF {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut enc = ['\0'; 256];
    let mut dec = HashMap::new();
    for i in 0..bs.len() {
        let ch = char::from_u32(cs[i]).unwrap_or('\u{FFFD}');
        enc[bs[i] as usize] = ch;
        dec.insert(ch, bs[i] as u8);
    }
    (enc, dec)
}
