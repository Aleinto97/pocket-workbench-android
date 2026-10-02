//! Minimal JSON reader/writer.
//!
//! The crate deliberately has no external dependencies, and the agent protocol
//! needs to parse tool-call arguments and emit durable records. This module is
//! the only JSON implementation in the tree; it is strict about malformed input
//! because a half-parsed tool argument must never reach a filesystem tool.

use crate::util::{Error, Result};
use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn obj() -> Json {
        Json::Obj(Vec::new())
    }

    pub fn obj_with(fields: Vec<(String, Json)>) -> Json {
        Json::Obj(fields)
    }

    pub fn str(value: impl Into<String>) -> Json {
        Json::Str(value.into())
    }

    pub fn int(value: impl Into<i64>) -> Json {
        Json::Num(value.into() as f64)
    }

    pub fn set(&mut self, key: &str, value: Json) -> &mut Json {
        if let Json::Obj(fields) = self {
            if let Some(slot) = fields.iter_mut().find(|(k, _)| k == key) {
                slot.1 = value;
            } else {
                fields.push((key.to_string(), value));
            }
        }
        self
    }

    pub fn with(mut self, key: &str, value: Json) -> Json {
        self.set(key, value);
        self
    }

    pub fn push(&mut self, value: Json) -> &mut Json {
        if let Json::Arr(items) = self {
            items.push(value);
        }
        self
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|n| n as i64)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(fields) => Some(fields),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "boolean",
            Json::Num(_) => "number",
            Json::Str(_) => "string",
            Json::Arr(_) => "array",
            Json::Obj(_) => "object",
        }
    }

    pub fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Num(n) => {
                if n.is_finite() && n.fract() == 0.0 && n.abs() < 9.0e15 {
                    let _ = write!(out, "{}", *n as i64);
                } else if n.is_finite() {
                    let _ = write!(out, "{}", n);
                } else {
                    out.push_str("null");
                }
            }
            Json::Str(s) => write_string(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Obj(fields) => {
                out.push('{');
                for (i, (key, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

impl std::fmt::Display for Json {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();
        self.write(&mut out);
        f.write_str(&out)
    }
}

fn write_string(value: &str, out: &mut String) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn parse(input: &str) -> Result<Json> {
    let bytes = input.as_bytes();
    let mut p = Parser { bytes, at: 0 };
    p.skip_ws();
    let value = p.value(0)?;
    p.skip_ws();
    if p.at != bytes.len() {
        return Err(Error::new(format!("trailing data at byte {}", p.at)));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

const MAX_DEPTH: usize = 32;

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.at += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        if self.peek() == Some(c) {
            self.at += 1;
            Ok(())
        } else {
            Err(Error::new(format!(
                "expected '{}' at byte {}",
                c as char, self.at
            )))
        }
    }

    fn literal(&mut self, word: &str) -> Result<()> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(())
        } else {
            Err(Error::new(format!("unknown literal at byte {}", self.at)))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json> {
        if depth > MAX_DEPTH {
            return Err(Error::new("JSON nested too deeply"));
        }
        match self.peek() {
            None => Err(Error::new("unexpected end of JSON")),
            Some(b'n') => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            Some(b't') => {
                self.literal("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.skip_ws();
                if self.peek() == Some(b']') {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    self.skip_ws();
                    items.push(self.value(depth + 1)?);
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(Error::new(format!("malformed array at byte {}", self.at))),
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut fields: Vec<(String, Json)> = Vec::new();
                self.skip_ws();
                if self.peek() == Some(b'}') {
                    self.at += 1;
                    return Ok(Json::Obj(fields));
                }
                loop {
                    self.skip_ws();
                    let key = self.string()?;
                    self.skip_ws();
                    self.expect(b':')?;
                    self.skip_ws();
                    let value = self.value(depth + 1)?;
                    if let Some(slot) = fields.iter_mut().find(|(k, _)| *k == key) {
                        slot.1 = value;
                    } else {
                        fields.push((key, value));
                    }
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(Json::Obj(fields));
                        }
                        _ => return Err(Error::new(format!("malformed object at byte {}", self.at))),
                    }
                }
            }
            Some(_) => self.number(),
        }
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.at;
        if self.peek() == Some(b'-') || self.peek() == Some(b'+') {
            self.at += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == b'.' || c == b'e' || c == b'E' || c == b'-' || c == b'+' {
                self.at += 1;
            } else {
                break;
            }
        }
        if start == self.at {
            return Err(Error::new(format!("unexpected byte at {}", self.at)));
        }
        let text = core::str::from_utf8(&self.bytes[start..self.at])
            .map_err(|_| Error::new("number is not valid UTF-8"))?;
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| Error::new(format!("invalid number '{text}'")))
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let c = match self.peek() {
                Some(c) => c,
                None => return Err(Error::new("unterminated string")),
            };
            self.at += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = self
                        .peek()
                        .ok_or_else(|| Error::new("unterminated escape"))?;
                    self.at += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let high = self.hex4()?;
                            let ch = if (0xD800..0xDC00).contains(&high) {
                                if self.peek() == Some(b'\\') {
                                    self.at += 1;
                                    self.expect(b'u')?;
                                    let low = self.hex4()?;
                                    if !(0xDC00..0xE000).contains(&low) {
                                        return Err(Error::new("invalid low surrogate"));
                                    }
                                    let combined =
                                        0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                                    char::from_u32(combined)
                                } else {
                                    None
                                }
                            } else {
                                char::from_u32(high)
                            };
                            out.push(ch.ok_or_else(|| Error::new("invalid \\u escape"))?);
                        }
                        other => {
                            return Err(Error::new(format!(
                                "unknown escape '\\{}'",
                                other as char
                            )))
                        }
                    }
                }
                c if c < 0x20 => return Err(Error::new("control character in string")),
                c if c < 0x80 => out.push(c as char),
                _ => {
                    // Multi-byte UTF-8: copy the continuation bytes verbatim.
                    let start = self.at - 1;
                    let extra = if c >= 0xF0 {
                        3
                    } else if c >= 0xE0 {
                        2
                    } else {
                        1
                    };
                    let end = start + 1 + extra;
                    if end > self.bytes.len() {
                        return Err(Error::new("truncated UTF-8 sequence"));
                    }
                    let text = core::str::from_utf8(&self.bytes[start..end])
                        .map_err(|_| Error::new("invalid UTF-8 in string"))?;
                    out.push_str(text);
                    self.at = end;
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        if self.at + 4 > self.bytes.len() {
            return Err(Error::new("truncated \\u escape"));
        }
        let text = core::str::from_utf8(&self.bytes[self.at..self.at + 4])
            .map_err(|_| Error::new("invalid \\u escape"))?;
        let value =
            u32::from_str_radix(text, 16).map_err(|_| Error::new("invalid \\u escape"))?;
        self.at += 4;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_nested_values() {
        let value = parse(r#"{"a":[1,2.5,-3],"b":{"c":"x\ny"},"d":null,"e":true}"#).unwrap();
        assert_eq!(value.get("a").unwrap().as_array().unwrap().len(), 3);
        assert_eq!(value.get("b").unwrap().get("c").unwrap().as_str().unwrap(), "x\ny");
        assert!(value.get("d").unwrap().is_null());
        assert_eq!(value.get("e").unwrap().as_bool(), Some(true));
        let text = value.to_string();
        assert_eq!(parse(&text).unwrap(), value);
    }

    #[test]
    fn escapes_and_unicode() {
        let value = parse(r#""a\"b\\cé😀""#).unwrap();
        assert_eq!(value.as_str().unwrap(), "a\"b\\cé😀");
        assert_eq!(parse(&value.to_string()).unwrap(), value);
    }

    #[test]
    fn rejects_malformed_input() {
        assert!(parse("{").is_err());
        assert!(parse("{\"a\":}").is_err());
        assert!(parse("[1,2]extra").is_err());
        assert!(parse("{\"a\":1} {\"b\":2}").is_err());
        assert!(parse("").is_err());
        assert!(parse("\"unterminated").is_err());
    }

    #[test]
    fn writes_integers_without_exponent() {
        assert_eq!(Json::int(42).to_string(), "42");
        assert_eq!(Json::Num(1.5).to_string(), "1.5");
        assert_eq!(Json::Num(f64::NAN).to_string(), "null");
    }
}
