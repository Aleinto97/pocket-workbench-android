use crate::util::{Error, MappedFile, Result};
use std::collections::HashMap;

pub const GGUF_MAGIC: u32 = 0x4655_4747;

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub ne: Vec<u64>,
    pub ttype: u32,
    pub offset: u64,
}

#[derive(Clone, Copy)]
pub enum KvRef {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str { off: usize, len: usize },
    Arr { etype: u32, count: u64, off: usize },
    U64(u64),
    I64(i64),
    F64(f64),
    None,
}

pub struct Gguf {
    pub file: MappedFile,
    pub version: u32,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: usize,
    kv: HashMap<String, KvRef>,
    order: Vec<String>,
}

struct Reader<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, o: 0 }
    }
    fn need(&self, n: usize) -> Result<()> {
        if n > self.b.len().saturating_sub(self.o) {
            bail!("gguf truncated at offset {}", self.o);
        }
        Ok(())
    }
    fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.b[self.o];
        self.o += 1;
        Ok(v)
    }
    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }
    fn u16(&mut self) -> Result<u16> {
        self.need(2)?;
        let v = u16::from_le_bytes([self.b[self.o], self.b[self.o + 1]]);
        self.o += 2;
        Ok(v)
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn u32(&mut self) -> Result<u32> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.b[self.o..self.o + 4].try_into().unwrap());
        self.o += 4;
        Ok(v)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn u64(&mut self) -> Result<u64> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.b[self.o..self.o + 8].try_into().unwrap());
        self.o += 8;
        Ok(v)
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn string(&mut self) -> Result<(usize, usize)> {
        let n = self.u64()? as usize;
        self.need(n)?;
        let off = self.o;
        self.o += n;
        Ok((off, n))
    }
}

impl Gguf {
    pub fn load(path: &str) -> Result<Self> {
        let file = MappedFile::open(path)?;
        Self::parse(file)
    }

    pub fn parse(file: MappedFile) -> Result<Self> {
        let bytes = file.as_slice();
        let mut r = Reader::new(bytes);
        let magic = r.u32()?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (magic {magic:#x})");
        }
        let version = r.u32()?;
        if version < 2 || version > 3 {
            bail!("unsupported GGUF version {version}");
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;
        if n_tensors > 1_000_000 || n_kv > 1_000_000 {
            bail!("implausible gguf header");
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut kv = HashMap::new();
        let mut order = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let (koff, klen) = r.string()?;
            let key = String::from_utf8_lossy(&bytes[koff..koff + klen]).into_owned();
            let ttype = r.u32()?;
            let v = match ttype {
                0 => KvRef::U8(r.u8()?),
                1 => KvRef::I8(r.i8()?),
                2 => KvRef::U16(r.u16()?),
                3 => KvRef::I16(r.i16()?),
                4 => KvRef::U32(r.u32()?),
                5 => KvRef::I32(r.i32()?),
                6 => KvRef::F32(r.f32()?),
                7 => KvRef::Bool(r.u8()? != 0),
                8 => {
                    let (off, len) = r.string()?;
                    KvRef::Str { off, len }
                }
                9 => {
                    let etype = r.u32()?;
                    let count = r.u64()?;
                    let off = r.o;
                    skip_array(&mut r, etype, count)?;
                    KvRef::Arr { etype, count, off }
                }
                10 => KvRef::U64(r.u64()?),
                11 => KvRef::I64(r.i64()?),
                12 => KvRef::F64(r.f64()?),
                other => bail!("unknown gguf metadata type {other} for key {key}"),
            };
            order.push(key.clone());
            kv.insert(key, v);
        }
        for _ in 0..n_tensors {
            let (noff, nlen) = r.string()?;
            let name = String::from_utf8_lossy(&bytes[noff..noff + nlen]).into_owned();
            let nd = r.u32()? as usize;
            if nd > 4 {
                bail!("tensor {name} has {nd} dims");
            }
            let mut ne = Vec::with_capacity(nd);
            for _ in 0..nd {
                ne.push(r.u64()?);
            }
            let ttype = r.u32()?;
            let offset = r.u64()?;
            tensors.push(TensorInfo { name, ne, ttype, offset });
        }
        let alignment = match kv.get("general.alignment") {
            Some(KvRef::U32(v)) => *v as usize,
            Some(KvRef::U64(v)) => *v as usize,
            Some(KvRef::I32(v)) => *v as usize,
            _ => 32,
        };
        let alignment = alignment.max(1);
        let data_offset = r.o.checked_add(alignment - 1)
            .ok_or_else(|| crate::err!("invalid GGUF alignment"))? / alignment * alignment;
        if data_offset > bytes.len() {
            bail!("GGUF tensor data begins past the end of the file");
        }
        Ok(Self { file, version, tensors, data_offset, kv, order })
    }

    pub fn has(&self, key: &str) -> bool {
        self.kv.contains_key(key)
    }

    pub fn keys(&self) -> &[String] {
        &self.order
    }

    pub fn kv(&self, key: &str) -> Option<KvRef> {
        self.kv.get(key).copied()
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.kv.get(key)? {
            KvRef::U8(v) => Some(*v as u64),
            KvRef::U16(v) => Some(*v as u64),
            KvRef::U32(v) => Some(*v as u64),
            KvRef::U64(v) => Some(*v),
            KvRef::I8(v) => Some(*v as u64),
            KvRef::I16(v) => Some(*v as u64),
            KvRef::I32(v) => Some(*v as u64),
            KvRef::I64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get_u64(key).map(|v| v as u32)
    }

    pub fn get_i32(&self, key: &str) -> Option<i32> {
        self.get_u64(key).map(|v| v as i32)
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.kv.get(key)? {
            KvRef::F32(v) => Some(*v),
            KvRef::F64(v) => Some(*v as f32),
            _ => self.get_u64(key).map(|v| v as f32),
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.kv.get(key)? {
            KvRef::Bool(v) => Some(*v),
            KvRef::U8(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&[u8]> {
        match self.kv.get(key)? {
            KvRef::Str { off, len } => Some(&self.file.as_slice()[*off..*off + *len]),
            _ => None,
        }
    }

    pub fn get_str_string(&self, key: &str) -> Option<String> {
        self.get_str(key).map(|s| String::from_utf8_lossy(s).into_owned())
    }

    pub fn array_len(&self, key: &str) -> Option<u64> {
        match self.kv.get(key)? {
            KvRef::Arr { count, .. } => Some(*count),
            _ => None,
        }
    }

    pub fn array_type(&self, key: &str) -> Option<u32> {
        match self.kv.get(key)? {
            KvRef::Arr { etype, .. } => Some(*etype),
            _ => None,
        }
    }

    pub fn array_strings(&self, key: &str) -> Result<Vec<Vec<u8>>> {
        let bytes = self.file.as_slice();
        match self.kv.get(key) {
            Some(KvRef::Arr { etype: 8, count, off }) => {
                let mut r = Reader::new(bytes);
                r.o = *off;
                let mut out = Vec::with_capacity(*count as usize);
                for _ in 0..*count {
                    let (o, l) = r.string()?;
                    out.push(bytes[o..o + l].to_vec());
                }
                Ok(out)
            }
            _ => bail!("metadata {key} is not a string array"),
        }
    }

    pub fn array_i32(&self, key: &str) -> Result<Vec<i32>> {
        let bytes = self.file.as_slice();
        match self.kv.get(key) {
            Some(KvRef::Arr { etype: 5, count, off }) => {
                let mut r = Reader::new(bytes);
                r.o = *off;
                let mut out = Vec::with_capacity(*count as usize);
                for _ in 0..*count {
                    out.push(r.i32()?);
                }
                Ok(out)
            }
            _ => bail!("metadata {key} is not an i32 array"),
        }
    }

    pub fn array_f32(&self, key: &str) -> Result<Vec<f32>> {
        let bytes = self.file.as_slice();
        match self.kv.get(key) {
            Some(KvRef::Arr { etype: 6, count, off }) => {
                let mut r = Reader::new(bytes);
                r.o = *off;
                let mut out = Vec::with_capacity(*count as usize);
                for _ in 0..*count {
                    out.push(r.f32()?);
                }
                Ok(out)
            }
            _ => bail!("metadata {key} is not an f32 array"),
        }
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_data<'a>(&'a self, info: &TensorInfo) -> Result<&'a [u8]> {
        let bytes = self.file.as_slice();
        let start = usize::try_from(info.offset).ok()
            .and_then(|offset| self.data_offset.checked_add(offset))
            .ok_or_else(|| crate::err!("tensor {} offset overflow", info.name))?;
        if start > bytes.len() {
            bail!("tensor {} outside file", info.name);
        }
        let n = crate::quant::tensor_nbytes(info.ttype, &info.ne)?;
        let end = start.checked_add(n).ok_or_else(|| crate::err!("tensor {} size overflow", info.name))?;
        if end > bytes.len() {
            bail!("tensor {} truncated (need {n} bytes)", info.name);
        }
        Ok(&bytes[start..end])
    }
}

fn skip_array(r: &mut Reader, etype: u32, count: u64) -> Result<()> {
    for _ in 0..count {
        match etype {
            0 | 1 | 7 => {
                r.u8()?;
            }
            2 | 3 => {
                r.u16()?;
            }
            4 | 5 | 6 => {
                r.u32()?;
            }
            10 | 11 | 12 => {
                r.u64()?;
            }
            8 => {
                r.string()?;
            }
            9 => {
                let et = r.u32()?;
                let c = r.u64()?;
                skip_array(r, et, c)?;
            }
            other => bail!("unknown array element type {other}"),
        }
    }
    Ok(())
}

pub fn parse_version(bytes: &[u8]) -> Result<u32> {
    if bytes.len() < 8 {
        bail!("file too small");
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != GGUF_MAGIC {
        bail!("bad magic");
    }
    Ok(u32::from_le_bytes(bytes[4..8].try_into().unwrap()))
}

impl From<Error> for std::io::Error {
    fn from(e: Error) -> Self {
        std::io::Error::new(std::io::ErrorKind::Other, e.msg)
    }
}
