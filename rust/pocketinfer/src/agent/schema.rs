//! Validation of tool arguments against the schema the model was shown.
//!
//! The schema subset is deliberately small: object/array/string/number/
//! integer/boolean, `required`, `enum`, `properties`, `items`, numeric and
//! length bounds. Anything the model sends that does not satisfy it is rejected
//! before an executor runs, so a malformed call can never reach the filesystem.

use super::json::Json;
use crate::util::{Error, Result};

pub const INVALID_ARGUMENT: &str = "invalid_argument";

pub fn validate(schema: &Json, value: &Json) -> Result<()> {
    check(schema, value, "arguments")
}

fn check(schema: &Json, value: &Json, label: &str) -> Result<()> {
    let expected = schema.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let ok = match expected {
        "object" | "" => value.as_object().is_some() || value.is_null(),
        "array" => value.as_array().is_some(),
        "string" => value.as_str().is_some(),
        "integer" => value.as_f64().map(|n| n.fract() == 0.0).unwrap_or(false),
        "number" => value.as_f64().is_some(),
        "boolean" => value.as_bool().is_some(),
        _ => true,
    };
    if !ok {
        return Err(Error::new(format!(
            "{label} must be {}, received {}",
            if expected.is_empty() { "an object" } else { expected },
            value.type_name()
        )));
    }
    if let Some(options) = schema.get("enum").and_then(|e| e.as_array()) {
        if !options.iter().any(|option| option == value) {
            let names: Vec<&str> = options.iter().filter_map(|o| o.as_str()).collect();
            return Err(Error::new(format!(
                "{label} must be one of: {}",
                names.join(", ")
            )));
        }
    }
    if let Some(text) = value.as_str() {
        let length = text.chars().count();
        if let Some(min) = schema.get("minLength").and_then(|v| v.as_i64()) {
            if (length as i64) < min {
                return Err(Error::new(format!("{label} is shorter than {min} characters")));
            }
        }
        if let Some(max) = schema.get("maxLength").and_then(|v| v.as_i64()) {
            if (length as i64) > max {
                return Err(Error::new(format!("{label} is longer than {max} characters")));
            }
        }
    }
    if let Some(number) = value.as_f64() {
        if let Some(min) = schema.get("minimum").and_then(|v| v.as_f64()) {
            if number < min {
                return Err(Error::new(format!("{label} is below {min}")));
            }
        }
        if let Some(max) = schema.get("maximum").and_then(|v| v.as_f64()) {
            if number > max {
                return Err(Error::new(format!("{label} is above {max}")));
            }
        }
    }
    if let Some(items) = value.as_array() {
        if let Some(max) = schema.get("maxItems").and_then(|v| v.as_i64()) {
            if (items.len() as i64) > max {
                return Err(Error::new(format!("{label} has more than {max} items")));
            }
        }
        if let Some(item_schema) = schema.get("items") {
            for item in items {
                check(item_schema, item, &format!("{label} item"))?;
            }
        }
    }
    if let Some(fields) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
            for name in required.iter().filter_map(|n| n.as_str()) {
                let present = fields.iter().any(|(k, v)| k == name && !v.is_null());
                if !present {
                    return Err(Error::new(format!("{label} is missing the field '{name}'")));
                }
            }
        }
        if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
            for (key, item) in fields {
                let Some((_, field_schema)) = properties.iter().find(|(k, _)| k == key) else {
                    continue;
                };
                if !field_schema.is_null() {
                    check(field_schema, item, &format!("{label} field '{key}'"))?;
                }
            }
        }
        if let Some(max) = schema.get("maxProperties").and_then(|v| v.as_i64()) {
            if (fields.len() as i64) > max {
                return Err(Error::new(format!("{label} has more than {max} fields")));
            }
        }
    }
    Ok(())
}

/// Convenience for schemas built in Rust: `("path", "string")` tuples.
pub fn object(properties: Vec<(&str, Json)>, required: Vec<&str>) -> Json {
    let mut schema = Json::obj()
        .with("type", Json::str("object"))
        .with("additionalProperties", Json::Bool(false));
    let mut props = Json::obj();
    for (name, value) in properties {
        props.set(name, value);
    }
    schema.set("properties", props);
    if !required.is_empty() {
        let mut list = Json::Arr(Vec::new());
        for name in required {
            list.push(Json::str(name));
        }
        schema.set("required", list);
    }
    schema
}

pub fn string(max_length: usize) -> Json {
    Json::obj()
        .with("type", Json::str("string"))
        .with("maxLength", Json::int(max_length as i64))
}

pub fn string_enum<const N: usize>(options: [&str; N]) -> Json {
    let mut list = Json::Arr(Vec::new());
    for option in options {
        list.push(Json::str(option));
    }
    Json::obj().with("type", Json::str("string")).with("enum", list)
}

pub fn integer(minimum: i64, maximum: i64) -> Json {
    Json::obj()
        .with("type", Json::str("integer"))
        .with("minimum", Json::int(minimum))
        .with("maximum", Json::int(maximum))
}

pub fn boolean() -> Json {
    Json::obj().with("type", Json::str("boolean"))
}

#[cfg(test)]
mod tests {
    use super::super::json::parse;
    use super::*;

    fn schema() -> Json {
        object(
            vec![
                ("path", string(64)),
                ("limit", integer(1, 100)),
                ("mode", string_enum(["read", "write"])),
                ("flag", boolean()),
            ],
            vec!["path"],
        )
    }

    #[test]
    fn accepts_a_valid_object() {
        let value = parse(r#"{"path":"a.txt","limit":5,"mode":"read","flag":true}"#).unwrap();
        assert!(validate(&schema(), &value).is_ok());
    }

    #[test]
    fn rejects_missing_required_fields() {
        let value = parse(r#"{"limit":5}"#).unwrap();
        let error = validate(&schema(), &value).unwrap_err();
        assert!(error.to_string().contains("path"));
    }

    #[test]
    fn rejects_wrong_types_and_ranges() {
        assert!(validate(&schema(), &parse(r#"{"path":5}"#).unwrap()).is_err());
        assert!(validate(&schema(), &parse(r#"{"path":"a","limit":0}"#).unwrap()).is_err());
        assert!(validate(&schema(), &parse(r#"{"path":"a","limit":1000}"#).unwrap()).is_err());
        assert!(validate(&schema(), &parse(r#"{"path":"a","mode":"delete"}"#).unwrap()).is_err());
    }

    #[test]
    fn rejects_over_long_strings() {
        let long = "x".repeat(100);
        let value = parse(&format!(r#"{{"path":"{long}"}}"#)).unwrap();
        assert!(validate(&schema(), &value).is_err());
    }

    #[test]
    fn ignores_unknown_fields_but_validates_known_ones() {
        assert!(validate(&schema(), &parse(r#"{"path":"a","extra":1}"#).unwrap()).is_ok());
        assert!(validate(&schema(), &parse(r#"{"path":"a","limit":"x"}"#).unwrap()).is_err());
    }

    #[test]
    fn validates_nested_arrays() {
        let nested = object(
            vec![("items", Json::obj().with("type", Json::str("array"))
                .with("items", string(8))
                .with("maxItems", Json::int(2)))],
            vec!["items"],
        );
        assert!(validate(&nested, &parse(r#"{"items":["a","b"]}"#).unwrap()).is_ok());
        assert!(validate(&nested, &parse(r#"{"items":["a","b","c"]}"#).unwrap()).is_err());
        assert!(validate(&nested, &parse(r#"{"items":["aaaaaaaaaa"]}"#).unwrap()).is_err());
    }
}
