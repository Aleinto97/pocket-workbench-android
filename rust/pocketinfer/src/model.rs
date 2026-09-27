use crate::gguf::Gguf;
use crate::quant;
use crate::sampler::{sample, Rng};
use crate::tokenizer::Tokenizer;
use crate::util::{self, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

const MAX_BATCH: usize = 64;

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub n_ctx_train: usize,
    pub n_vocab: usize,
    pub name: String,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.get_str_string("general.architecture").unwrap_or_default();
        let p = if arch.is_empty() { "llama".to_string() } else { arch.clone() };
        let get = |key: &str| {
            g.get_u64(&format!("{p}.{key}"))
                .ok_or_else(|| crate::err!("missing metadata {p}.{key}"))
        };
        let n_layer = get("block_count")? as usize;
        let n_embd = get("embedding_length")? as usize;
        let n_ff = get("feed_forward_length")? as usize;
        let n_head = get("attention.head_count")? as usize;
        if n_layer == 0 || n_embd == 0 || n_ff == 0 || n_head == 0 {
            bail!("invalid model configuration: zero layer, hidden, FFN or head count");
        }
        let n_head_kv = g
            .get_u64(&format!("{p}.attention.head_count_kv"))
            .unwrap_or(n_head as u64) as usize;
        let head_dim = g
            .get_u64(&format!("{p}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head);
        if head_dim == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 {
            bail!("invalid GQA configuration: {n_head}/{n_head_kv} with head_dim={head_dim}");
        }
        if let Some(value_dim) = g.get_u64(&format!("{p}.attention.value_length")) {
            if value_dim != head_dim as u64 {
                bail!("value head dimension {value_dim} differs from key dimension {head_dim}");
            }
        }
        let n_rot = g
            .get_u64(&format!("{p}.rope.dimension_count"))
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let rope_theta = g.get_f32(&format!("{p}.rope.freq_base")).unwrap_or(10000.0);
        let rms_eps = g
            .get_f32(&format!("{p}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let n_ctx_train = g
            .get_u64(&format!("{p}.context_length"))
            .unwrap_or(4096) as usize;
        let n_vocab = g
            .get_u64(&format!("{p}.vocab_size"))
            .unwrap_or(0) as usize;
        let name = g.get_str_string("general.name").unwrap_or_default();
        if n_rot == 0 || n_rot > head_dim || n_rot % 2 != 0
            || !rope_theta.is_finite() || rope_theta <= 0.0
            || !rms_eps.is_finite() || rms_eps <= 0.0
        {
            bail!("invalid RoPE / RMSNorm configuration");
        }
        Ok(Self {
            arch: p,
            n_layer,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            n_rot,
            rope_theta,
            rms_eps,
            n_ctx_train,
            n_vocab,
            name,
        })
    }
}

#[derive(Clone)]
pub struct TensorRef {
    pub name: String,
    pub ttype: u32,
    pub ne0: usize,
    pub ne1: usize,
    pub row_bytes: usize,
    pub off: usize,
}

impl TensorRef {
    fn data<'a>(&self, base: &'a [u8]) -> &'a [u8] {
        &base[self.off..self.off + self.row_bytes * self.ne1]
    }
}

struct MatJob {
    n_rows: usize,
    chunk: usize,
    ttype: u32,
    w: *const u8,
    row_bytes: usize,
    k: usize,
    xt: *const f32,
    b: usize,
    out: *mut f32,
    act: *const crate::quant_int::Q8Act,
}


pub struct JobShared {
    desc: UnsafeCell<MatJob>,
    next: AtomicUsize,
    shutdown: AtomicBool,
}

unsafe impl Send for JobShared {}
unsafe impl Sync for JobShared {}

pub struct Pool {
    gen: Arc<AtomicUsize>,
    done: Arc<AtomicUsize>,
    shared: Arc<JobShared>,
    workers: Vec<JoinHandle<()>>,
    n: usize,
}

unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

impl Pool {
    pub fn new(n: usize) -> Pool {
        let n = n.max(1);
        let gen = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicUsize::new(0));
        let shared = Arc::new(JobShared {
            desc: UnsafeCell::new(MatJob::empty()),
            next: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        });
        let mut workers = Vec::new();
        // The calling thread also does row work, so n denotes total threads.
        for _ in 0..n - 1 {
            let gen = gen.clone();
            let done = done.clone();
            let shared = shared.clone();
            let h = thread::Builder::new()
                .stack_size(1 << 20)
                .spawn(move || {
                    let mut local = 0usize;
                    loop {
                        if shared.shutdown.load(Ordering::Acquire) {
                            return;
                        }
                        let mut spins = 0u32;
                        while gen.load(Ordering::Acquire) == local {
                            spins += 1;
                            if spins > 20_000 {
                                thread::park();
                                spins = 0;
                                if shared.shutdown.load(Ordering::Acquire) {
                                    return;
                                }
                            } else {
                                core::hint::spin_loop();
                            }
                        }
                        local = gen.load(Ordering::Acquire);
                        if shared.shutdown.load(Ordering::Acquire) {
                            return;
                        }
                        let job = unsafe { &*shared.desc.get() };
                        loop {
                            let start = shared.next.fetch_add(job.chunk, Ordering::Relaxed);
                            if start >= job.n_rows {
                                break;
                            }
                            let end = (start + job.chunk).min(job.n_rows);
                            for r in start..end {
                                let row = unsafe { job.w.add(r * job.row_bytes) };
                                let row_slice =
                                    unsafe { core::slice::from_raw_parts(row, job.row_bytes) };
                                let out = unsafe { job.out.add(r * job.b) };
                                let out_slice =
                                    unsafe { core::slice::from_raw_parts_mut(out, job.b) };
                                if !job.act.is_null() {
                                    crate::quant_int::dot_row_q8_lanes(
                                        job.ttype,
                                        row_slice,
                                        job.k,
                                        unsafe { &*job.act },
                                        out_slice,
                                        job.b,
                                    );
                                } else {
                                    quant::row_dot_multi(
                                        job.ttype,
                                        row_slice,
                                        job.k,
                                        unsafe {
                                            core::slice::from_raw_parts(job.xt, job.k * job.b)
                                        },
                                        job.b,
                                        out_slice,
                                    );
                                }
                            }
                        }
                        done.fetch_add(1, Ordering::Release);
                    }
                })
                .expect("worker thread");
            workers.push(h);
        }
        Pool { gen, done, shared, workers, n }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        ttype: u32,
        w: *const u8,
        row_bytes: usize,
        k: usize,
        xt: &[f32],
        b: usize,
        out: &mut [f32],
        n_rows: usize,
        act: *const crate::quant_int::Q8Act,
    ) {
        out[..n_rows * b].fill(0.0);
        if self.n == 1 {
            for r in 0..n_rows {
                let row = unsafe { w.add(r * row_bytes) };
                if r + 1 < n_rows {
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        crate::simd::prefetch_read(row, row_bytes);
                    }
                }
                let row = unsafe { core::slice::from_raw_parts(row, row_bytes) };
                if !act.is_null() {
                    crate::quant_int::dot_row_q8_lanes(ttype, row, k, unsafe { &*act }, &mut out[r * b..r * b + b], b);
                } else {
                    quant::row_dot_multi(ttype, row, k, xt, b, &mut out[r * b..r * b + b]);
                }
            }
            return;
        }
        let chunk = ((n_rows + self.n * 4 - 1) / (self.n * 4)).max(1);
        unsafe {
            let d = &mut *self.shared.desc.get();
            d.n_rows = n_rows;
            d.chunk = chunk;
            d.ttype = ttype;
            d.w = w;
            d.row_bytes = row_bytes;
            d.k = k;
            d.xt = xt.as_ptr();
            d.b = b;
            d.out = out.as_mut_ptr();
            d.act = act;
        }
        self.shared.next.store(0, Ordering::Relaxed);
        self.done.store(0, Ordering::Relaxed);
        self.gen.fetch_add(1, Ordering::Release);
        for h in &self.workers {
            h.thread().unpark();
        }
        // The calling thread also steals row chunks; workers are the ones that
        // bump `done` when the queue is exhausted.
        loop {
            let start = self.shared.next.fetch_add(chunk, Ordering::Relaxed);
            if start >= n_rows {
                break;
            }
            let end = (start + chunk).min(n_rows);
            for r in start..end {
                let row = unsafe { w.add(r * row_bytes) };
                if r + 1 < n_rows {
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        crate::simd::prefetch_read(row, row_bytes);
                    }
                }
                let row = unsafe { core::slice::from_raw_parts(row, row_bytes) };
                let dst = &mut out[r * b..r * b + b];
                if !act.is_null() {
                    crate::quant_int::dot_row_q8_lanes(ttype, row, k, unsafe { &*act }, dst, b);
                } else {
                    quant::row_dot_multi(ttype, row, k, xt, b, dst);
                }
            }
        }
        while self.done.load(Ordering::Acquire) < self.workers.len() {
            core::hint::spin_loop();
        }
    }
}

impl MatJob {
    fn empty() -> Self {
        Self {
            n_rows: 0,
            chunk: 1,
            ttype: 0,
            w: core::ptr::null(),
            row_bytes: 0,
            k: 0,
            xt: core::ptr::null(),
            b: 0,
            out: core::ptr::null_mut(),
            act: core::ptr::null(),
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.gen.fetch_add(1, Ordering::Release);
        for h in self.workers.drain(..) {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}

pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub n_ctx: usize,
    pub n_kv: usize,
    pub dh: usize,
    pub n_layer: usize,
}

impl KvCache {
    fn new(n_layer: usize, n_kv: usize, n_ctx: usize, dh: usize) -> Result<Self> {
        let n = n_layer.checked_mul(n_kv).and_then(|n| n.checked_mul(n_ctx))
            .and_then(|n| n.checked_mul(dh))
            .ok_or_else(|| crate::err!("KV cache dimensions overflow"))?;
        let mut k = Vec::new();
        let mut v = Vec::new();
        k.try_reserve_exact(n).map_err(|e| crate::err!("cannot allocate KV keys: {e}"))?;
        v.try_reserve_exact(n).map_err(|e| crate::err!("cannot allocate KV values: {e}"))?;
        k.resize(n, 0.0);
        v.resize(n, 0.0);
        Ok(Self { k, v, n_ctx, n_kv, dh, n_layer })
    }
    #[inline]
    fn idx(&self, l: usize, h: usize, pos: usize, d: usize) -> usize {
        ((l * self.n_kv + h) * self.n_ctx + pos) * self.dh + d
    }
    /// Base pointer of the key row for (layer, kv-head, position).
    #[inline]
    fn k_row(&self, l: usize, h: usize, pos: usize) -> *const f32 {
        &self.k[((l * self.n_kv + h) * self.n_ctx + pos) * self.dh] as *const f32
    }
    #[inline]
    fn v_row(&self, l: usize, h: usize, pos: usize) -> *const f32 {
        &self.v[((l * self.n_kv + h) * self.n_ctx + pos) * self.dh] as *const f32
    }
}

pub struct Scratch {
    x: Vec<f32>,
    x2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    row: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    rope_inv: Vec<f32>,
    rope_upto: usize,
}

impl Scratch {
    fn new(cfg: &Config, n_ctx: usize, n_batch: usize) -> Self {
        let ne = cfg.n_embd;
        let qdim = cfg.n_head * cfg.head_dim;
        let kdim = cfg.n_head_kv * cfg.head_dim;
        let half = cfg.n_rot / 2;
        let mut rope_inv = vec![0.0f32; half.max(1)];
        for i in 0..half {
            rope_inv[i] = 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / cfg.n_rot as f32);
        }
        // The vocabulary projection writes `output.ne1` rows; size the buffer
        // from the metadata vocab but never below what the embeddings imply.
        let n_log = cfg.n_vocab.max(1);
        Self {
            x: vec![0.0; ne * n_batch],
            x2: vec![0.0; ne * n_batch],
            q: vec![0.0; qdim * n_batch],
            k: vec![0.0; kdim * n_batch],
            v: vec![0.0; kdim * n_batch],
            attn: vec![0.0; qdim * n_batch],
            proj: vec![0.0; ne * n_batch],
            gate: vec![0.0; cfg.n_ff * n_batch],
            up: vec![0.0; cfg.n_ff * n_batch],
            row: vec![0.0; ne.max(cfg.n_ff)],
            scores: vec![0.0; n_ctx],
            logits: vec![0.0; n_log],
            rope_cos: vec![0.0; n_ctx * half],
            rope_sin: vec![0.0; n_ctx * half],
            rope_inv,
            rope_upto: 0,
        }
    }
    fn ensure_logits(&mut self, need: usize) {
        if self.logits.len() < need {
            self.logits.resize(need, 0.0);
        }
    }
    fn ensure_rope(&mut self, cfg: &Config, upto: usize) {
        let half = (cfg.n_rot / 2).max(1);
        let n = upto.min(self.rope_cos.len() / half);
        if n <= self.rope_upto {
            return;
        }
        let inv = &self.rope_inv;
        for pos in self.rope_upto..n {
            for i in 0..half {
                let a = pos as f32 * inv[i];
                self.rope_cos[pos * half + i] = a.cos();
                self.rope_sin[pos * half + i] = a.sin();
            }
        }
        self.rope_upto = n;
    }
}

pub struct Model {
    pub gguf: Gguf,
    pub cfg: Config,
    pub tok: Tokenizer,
    tensors: Vec<TensorRef>,
    embd: Option<TensorRef>,
    output: Option<TensorRef>,
    output_norm: Option<TensorRef>,
    layers: Vec<LayerWeights>,
}

struct LayerWeights {
    attn_q: TensorRef,
    attn_k: TensorRef,
    attn_v: TensorRef,
    attn_o: TensorRef,
    attn_norm: TensorRef,
    ffn_gate: TensorRef,
    ffn_up: TensorRef,
    ffn_down: TensorRef,
    ffn_norm: TensorRef,
}

impl Model {
    pub fn load(path: &str) -> Result<Model> {
        let gguf = Gguf::load(path)?;
        let cfg = Config::from_gguf(&gguf)?;
        let tok = Tokenizer::from_gguf(&gguf)?;
        let base = gguf.data_offset;
        let bytes = gguf.file.as_slice();
        let mut tensors = Vec::new();
        let find = |name: &str| -> Result<TensorRef> {
            let info = gguf
                .tensor(name)
                .ok_or_else(|| crate::err!("missing tensor {name}"))?;
            if !quant::is_supported(info.ttype) {
                bail!("tensor {name}: unsupported type {}", quant::type_name(info.ttype));
            }
            if info.ne.is_empty() || info.ne.len() > 2 || info.ne.iter().any(|n| *n == 0) {
                bail!("tensor {name}: expected one or two nonzero dimensions");
            }
            let ne0 = usize::try_from(info.ne[0]).map_err(|_| crate::err!("tensor {name}: dim0 overflow"))?;
            let ne1 = if info.ne.len() == 2 {
                usize::try_from(info.ne[1]).map_err(|_| crate::err!("tensor {name}: dim1 overflow"))?
            } else { 1 };
            let row_bytes = quant::tensor_nbytes(info.ttype, &[info.ne[0]])?;
            let off = base.checked_add(usize::try_from(info.offset).map_err(|_| crate::err!("tensor {name}: offset overflow"))?)
                .ok_or_else(|| crate::err!("tensor {name}: offset overflow"))?;
            let total = row_bytes.checked_mul(ne1).ok_or_else(|| crate::err!("tensor {name}: size overflow"))?;
            if off.checked_add(total).is_none_or(|end| end > bytes.len()) {
                bail!("tensor {name}: out of range");
            }
            Ok(TensorRef {
                name: name.to_string(),
                ttype: info.ttype,
                ne0,
                ne1,
                row_bytes,
                off,
            })
        };
        let embd = find("token_embd.weight")?;
        let output = if gguf.tensor("output.weight").is_some() {
            Some(find("output.weight")?)
        } else {
            Some(embd.clone())
        };
        let output_norm = Some(find("output_norm.weight")?);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let l = LayerWeights {
                attn_q: find(&format!("blk.{i}.attn_q.weight"))?,
                attn_k: find(&format!("blk.{i}.attn_k.weight"))?,
                attn_v: find(&format!("blk.{i}.attn_v.weight"))?,
                attn_o: find(&format!("blk.{i}.attn_output.weight"))?,
                attn_norm: find(&format!("blk.{i}.attn_norm.weight"))?,
                ffn_gate: find(&format!("blk.{i}.ffn_gate.weight"))?,
                ffn_up: find(&format!("blk.{i}.ffn_up.weight"))?,
                ffn_down: find(&format!("blk.{i}.ffn_down.weight"))?,
                ffn_norm: find(&format!("blk.{i}.ffn_norm.weight"))?,
            };
            layers.push(l);
        }
        tensors.extend(layers.iter().flat_map(|l| {
            vec![
                l.attn_q.clone(),
                l.attn_k.clone(),
                l.attn_v.clone(),
                l.attn_o.clone(),
                l.attn_norm.clone(),
                l.ffn_gate.clone(),
                l.ffn_up.clone(),
                l.ffn_down.clone(),
                l.ffn_norm.clone(),
            ]
        }));
        tensors.push(embd.clone());
        if let Some(o) = &output {
            tensors.push(o.clone());
        }
        if let Some(o) = &output_norm {
            tensors.push(o.clone());
        }
        let mut model = Model { gguf, cfg, tok, tensors, embd: Some(embd), output, output_norm, layers };
        model.finish()?;
        Ok(model)
    }

    fn finish(&mut self) -> Result<()> {
        let cfg = &self.cfg;
        let e = self.embd.as_ref().unwrap();
        let vocab = self.tok.n_vocab;
        if vocab == 0 || e.ne0 != cfg.n_embd || e.ne1 != vocab
            || (cfg.n_vocab != 0 && cfg.n_vocab != vocab)
        {
            bail!("embedding / metadata / tokenizer vocab dimensions disagree");
        }
        let output = self.output.as_ref().unwrap();
        check_shape(output, cfg.n_embd, vocab)?;
        check_shape(self.output_norm.as_ref().unwrap(), cfg.n_embd, 1)?;
        let qdim = cfg.n_head.checked_mul(cfg.head_dim).ok_or_else(|| crate::err!("query dimension overflow"))?;
        let kdim = cfg.n_head_kv.checked_mul(cfg.head_dim).ok_or_else(|| crate::err!("key dimension overflow"))?;
        for layer in &self.layers {
            for (tensor, columns, rows) in [
                (&layer.attn_norm, cfg.n_embd, 1),
                (&layer.ffn_norm, cfg.n_embd, 1),
                (&layer.attn_q, cfg.n_embd, qdim),
                (&layer.attn_k, cfg.n_embd, kdim),
                (&layer.attn_v, cfg.n_embd, kdim),
                (&layer.attn_o, qdim, cfg.n_embd),
                (&layer.ffn_gate, cfg.n_embd, cfg.n_ff),
                (&layer.ffn_up, cfg.n_embd, cfg.n_ff),
                (&layer.ffn_down, cfg.n_ff, cfg.n_embd),
            ] {
                check_shape(tensor, columns, rows)?;
            }
        }
        Ok(())
    }

    pub fn tok_pre(&self) -> String {
        self.tok.pre.clone()
    }

    pub fn vocab_size(&self) -> usize {
        self.embd.as_ref().map(|e| e.ne1).unwrap_or(0)
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    pub fn weight_stats(&self) -> Vec<(u32, u64, usize)> {
        use std::collections::BTreeMap;
        let mut map: BTreeMap<u32, (u64, usize)> = BTreeMap::new();
        for t in &self.tensors {
            let bytes = (t.row_bytes * t.ne1) as u64;
            let e = map.entry(t.ttype).or_insert((0, 0));
            e.0 += bytes;
            e.1 += 1;
        }
        map.into_iter().map(|(t, (b, c))| (t, b, c)).collect()
    }

    pub fn tensor_list(&self) -> Vec<(String, u32, usize, usize)> {
        self.tensors
            .iter()
            .map(|t| (t.name.clone(), t.ttype, t.ne0, t.ne1))
            .collect()
    }

    pub fn debug_tensor(&self, name: &str) -> Option<(usize, usize, usize, u32, [f32; 3])> {
        let t = self.tensors.iter().find(|t| t.name == name)?;
        let base = self.gguf.file.as_slice();
        let d = &base[t.off..t.off + t.row_bytes * t.ne1];
        let mut first = [0f32; 3];
        if t.ttype == quant::GGML_TYPE_F32 {
            let mut row = vec![0f32; t.ne0];
            quant::dequant_row(t.ttype, d, t.ne0, &mut row);
            first[0] = row[0];
            first[1] = row[1];
            first[2] = row[2];
        }
        Some((t.off, t.row_bytes, t.ne0, t.ttype, first))
    }
}

fn check_shape(t: &TensorRef, cols: usize, rows: usize) -> Result<()> {
    if t.ne0 != cols || t.ne1 != rows {
        bail!("tensor {} has shape {}x{}, expected {}x{}", t.name, t.ne0, t.ne1, cols, rows);
    }
    Ok(())
}

pub struct GenOpts {
    pub n_ctx: usize,
    pub max_tokens: usize,
    pub temp: f32,
    pub top_p: f32,
    pub seed: u64,
    pub threads: usize,
}

#[derive(Default, Clone)]
pub struct GenStats {
    pub prefill_tokens: usize,
    pub prefill_ms: f64,
    pub gen_tokens: usize,
    pub gen_ms: f64,
    pub stop: String,
    pub elapsed_load_ms: f64,
    pub gen_ids: Vec<u32>,
}

pub struct Engine {
    pub model: Model,
    pub pool: Pool,
    pub kv: KvCache,
    pub scratch: Scratch,
    pub n_ctx: usize,
    pub gpu: Option<crate::backend::opencl::OpenClBackend>,
    pub q8: crate::quant_int::Q8Act,
    pub int8_enabled: bool,
    pub dbg_on: bool,
    pub dbg: Vec<(&'static str, usize, Vec<f32>)>,
}

impl Engine {
    pub fn load(path: &str, n_ctx: usize, threads: usize) -> Result<Engine> {
        let model = Model::load(path)?;
        let kv = KvCache::new(
            model.cfg.n_layer,
            model.cfg.n_head_kv,
            n_ctx,
            model.cfg.head_dim,
        )?;
        let scratch = Scratch::new(&model.cfg, n_ctx, MAX_BATCH);
        let pool = Pool::new(threads);
        Ok(Engine {
            model,
            pool,
            kv,
            scratch,
            n_ctx,
            gpu: None,
            q8: crate::quant_int::Q8Act::new(),
            int8_enabled: true,
            dbg_on: false,
            dbg: Vec::new(),
        })
    }

    fn matmul(&mut self, w: &TensorRef, x: &[f32], b: usize, out: &mut [f32]) {
        let base = self.model.gguf.file.as_ptr();
        let ptr = unsafe { base.add(w.off) };
        if b == 1 {
            if let Some(gpu) = self.gpu.as_mut() {
                if gpu.supports(w.ttype) {
                    let wslice = unsafe {
                        core::slice::from_raw_parts(ptr, w.row_bytes * w.ne1)
                    };
                    match gpu.matvec(w.ttype, wslice, w.ne0, &x[..w.ne0], &mut out[..w.ne1]) {
                        Ok(()) => return,
                        Err(e) => {
                            util::log(
                                util::ANDROID_LOG_WARN,
                                &format!("OpenCL matvec failed ({}): falling back to CPU", e),
                            );
                            self.gpu = None;
                        }
                    }
                }
            }
        }
        if self.int8_enabled && crate::quant_int::supported(w.ttype) {
            self.q8.prepare(x, w.ne0, b);
            let act = &self.q8 as *const crate::quant_int::Q8Act;
            self.pool
                .matmul(w.ttype, ptr, w.row_bytes, w.ne0, x, b, out, w.ne1, act);
            return;
        }
        self.pool
            .matmul(w.ttype, ptr, w.row_bytes, w.ne0, x, b, out, w.ne1, core::ptr::null());
    }

    pub fn kv_bytes(&self) -> usize {
        self.kv.k.len() * 4 + self.kv.v.len() * 4
    }

    pub fn enable_opencl(&mut self) -> Result<()> {
        let mut gpu = crate::backend::opencl::OpenClBackend::new()?;
        gpu.selftest()?;
        util::log(
            util::ANDROID_LOG_INFO,
            &format!("OpenCL GPU enabled and verified: {}", gpu.name()),
        );
        self.gpu = Some(gpu);
        Ok(())
    }

    pub fn enable_npu(&mut self, model_path: &str) -> Result<()> {
        crate::backend::npu::enable(model_path)
    }

    fn rmsnorm(&mut self, x: &[f32], w: &TensorRef, b: usize, out: &mut [f32]) {
        let ne = self.model.cfg.n_embd;
        let eps = self.model.cfg.rms_eps;
        let base = self.model.gguf.file.as_slice();
        let wdata = w.data(base);
        let mut wbuf = Vec::new();
        let wptr: *const f32 = if w.ttype == quant::GGML_TYPE_F32
            && wdata.as_ptr().align_offset(core::mem::align_of::<f32>()) == 0 {
            wdata.as_ptr() as *const f32
        } else {
            wbuf.resize(ne, 0.0);
            quant::dequant_row(w.ttype, wdata, ne, &mut wbuf);
            wbuf.as_ptr()
        };
        if b == 1 {
            let ss: f32;
            #[cfg(target_arch = "aarch64")]
            { ss = unsafe { crate::simd::f32_sum_sq(x.as_ptr(), ne) }; }
            #[cfg(not(target_arch = "aarch64"))]
            {
                ss = x[..ne].iter().map(|v| v * v).sum();
            }
            let s = 1.0 / (ss / ne as f32 + eps).sqrt();
            #[cfg(target_arch = "aarch64")]
            unsafe {
                crate::simd::f32_rms_apply(x.as_ptr(), wptr, out.as_mut_ptr(), ne, s);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..ne {
                    out[i] = x[i] * s * unsafe { *wptr.add(i) };
                }
            }
            return;
        }
        for bi in 0..b {
            let mut ss = 0.0f32;
            for i in 0..ne {
                let v = x[i * b + bi];
                ss += v * v;
            }
            let s = 1.0 / (ss / ne as f32 + eps).sqrt();
            for i in 0..ne {
                out[i * b + bi] = x[i * b + bi] * s * unsafe { *wptr.add(i) };
            }
        }
    }

    pub fn forward(&mut self, ids: &[u32], positions: &[usize], want_logits: bool) -> Result<()> {
        self.forward_inner(ids, positions, want_logits, false)
    }

    fn forward_inner(&mut self, ids: &[u32], positions: &[usize], want_logits: bool, cancellable: bool) -> Result<()> {
        let b = ids.len();
        if b == 0 || b > MAX_BATCH {
            bail!("invalid batch {b}");
        }
        if positions.len() != b {
            bail!("batch positions do not match tokens");
        }
        let cfg = self.model.cfg.clone();
        let ne = cfg.n_embd;
        let dh = cfg.head_dim;
        let qdim = cfg.n_head * dh;
        let kdim = cfg.n_head_kv * dh;
        let ctx = self.n_ctx;
        for (i, pos) in positions.iter().enumerate() {
            if *pos >= ctx {
                bail!("context full");
            }
            if ids[i] as usize >= self.model.vocab_size() {
                bail!("token {} out of vocab", ids[i]);
            }
        }
        self.scratch.ensure_rope(&cfg, positions.iter().copied().max().unwrap_or(0) + 1);
        let base = self.model.gguf.file.as_ptr();
        {
            let embd = self.model.embd.clone().unwrap();
            let tok_bytes = embd.row_bytes;
            let x = &mut self.scratch.x;
            for (bi, t) in ids.iter().enumerate() {
                let row = unsafe { base.add(embd.off + (*t as usize) * tok_bytes) };
                let row = unsafe { core::slice::from_raw_parts(row, tok_bytes) };
                quant::dequant_row(embd.ttype, row, ne, &mut self.scratch.row[..ne]);
                for i in 0..ne {
                    x[i * b + bi] = self.scratch.row[i];
                }
            }
        }
        if self.dbg_on {
            self.dbg.push(("embd", b, self.scratch.x[..ne * b].to_vec()));
        }
        let mut x = core::mem::take(&mut self.scratch.x);
        let mut x2 = core::mem::take(&mut self.scratch.x2);
        let mut q = core::mem::take(&mut self.scratch.q);
        let mut k = core::mem::take(&mut self.scratch.k);
        let mut v = core::mem::take(&mut self.scratch.v);
        let mut attn = core::mem::take(&mut self.scratch.attn);
        let mut proj = core::mem::take(&mut self.scratch.proj);
        let mut gate = core::mem::take(&mut self.scratch.gate);
        let mut up = core::mem::take(&mut self.scratch.up);
        let result = {
            let layers = std::mem::take(&mut self.model.layers);
            let result = self.forward_layers(
                ids, positions, want_logits, cancellable, &cfg, b, ne, dh, qdim, kdim, &layers, x.as_mut_slice(),
                x2.as_mut_slice(), q.as_mut_slice(), k.as_mut_slice(), v.as_mut_slice(),
                attn.as_mut_slice(), proj.as_mut_slice(), gate.as_mut_slice(), up.as_mut_slice(),
            );
            self.model.layers = layers;
            result
        };
        self.scratch.x = x;
        self.scratch.x2 = x2;
        self.scratch.q = q;
        self.scratch.k = k;
        self.scratch.v = v;
        self.scratch.attn = attn;
        self.scratch.proj = proj;
        self.scratch.gate = gate;
        self.scratch.up = up;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layers(
        &mut self,
        _ids: &[u32],
        positions: &[usize],
        want_logits: bool,
        cancellable: bool,
        cfg: &Config,
        b: usize,
        ne: usize,
        dh: usize,
        qdim: usize,
        kdim: usize,
        layers: &[LayerWeights],
        x: &mut [f32],
        x2: &mut [f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        attn: &mut [f32],
        proj: &mut [f32],
        gate: &mut [f32],
        up: &mut [f32],
    ) -> Result<()> {
        let n_head = cfg.n_head;
        let n_kv = cfg.n_head_kv;
        let group = n_head / n_kv;
        let scale = 1.0 / (dh as f32).sqrt();
        let ctx = self.n_ctx;
        // Reuse one correctly sized buffer for batched attention. A fixed-size
        // stack array would overwrite the stack for models with head_dim > 256.
        #[cfg(target_arch = "aarch64")]
        let mut local = if b > 1 { vec![0.0f32; dh] } else { Vec::new() };
        for (l, lw) in layers.iter().enumerate() {
            if cancellable && util::stop_requested() {
                bail!("generation stopped");
            }
            self.rmsnorm(x, &lw.attn_norm, b, x2);
            if self.dbg_on && l == 0 {
                self.dbg.push(("attn_norm-0", b, x2[..ne * b].to_vec()));
            }
            self.matmul(&lw.attn_q, x2, b, q);
            self.matmul(&lw.attn_k, x2, b, k);
            self.matmul(&lw.attn_v, x2, b, v);
            if self.dbg_on && l == 0 {
                self.dbg.push(("Qcur-0", b, q[..qdim * b].to_vec()));
                self.dbg.push(("Kcur-0", b, k[..kdim * b].to_vec()));
                self.dbg.push(("Vcur-0", b, v[..kdim * b].to_vec()));
            }
            {
                let half = cfg.n_rot / 2;
                let cos = &self.scratch.rope_cos;
                let sin = &self.scratch.rope_sin;
                for (bi, pos) in positions.iter().enumerate() {
                    let cb = &cos[pos * half..pos * half + half];
                    let sb = &sin[pos * half..pos * half + half];
                    for h in 0..n_head {
                        for i in 0..half {
                            let i0 = (h * dh + 2 * i) * b + bi;
                            let i1 = (h * dh + 2 * i + 1) * b + bi;
                            let (x0, x1) = (q[i0], q[i1]);
                            q[i0] = x0 * cb[i] - x1 * sb[i];
                            q[i1] = x0 * sb[i] + x1 * cb[i];
                        }
                    }
                    for h in 0..n_kv {
                        for i in 0..half {
                            let i0 = (h * dh + 2 * i) * b + bi;
                            let i1 = (h * dh + 2 * i + 1) * b + bi;
                            let (x0, x1) = (k[i0], k[i1]);
                            k[i0] = x0 * cb[i] - x1 * sb[i];
                            k[i1] = x0 * sb[i] + x1 * cb[i];
                        }
                    }
                }
            }
            if self.dbg_on && l == 0 {
                self.dbg.push(("Qrope-0", b, q[..qdim * b].to_vec()));
                self.dbg.push(("Krope-0", b, k[..kdim * b].to_vec()));
            }
            for (bi, pos) in positions.iter().enumerate() {
                for h in 0..n_kv {
                    let base_k = ((l * self.kv.n_kv + h) * self.kv.n_ctx + *pos) * self.kv.dh;
                    for d in 0..dh {
                        self.kv.k[base_k + d] = k[(h * dh + d) * b + bi];
                        self.kv.v[base_k + d] = v[(h * dh + d) * b + bi];
                    }
                }
            }
            for (bi, pos) in positions.iter().enumerate() {
                for h in 0..n_head {
                    let kh = h / group;
                    let qbase = (h * dh) * b + bi;
                    let mut maxs = f32::NEG_INFINITY;
                    if b == 1 {
                        let qp = &q[qbase..qbase + dh];
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            let mut p = 0usize;
                            while p <= *pos {
                                let kr = self.kv.k_row(l, kh, p);
                                let s = crate::simd::f32_dot(qp.as_ptr(), kr, dh) * scale;
                                self.scratch.scores[p] = s;
                                if s > maxs {
                                    maxs = s;
                                }
                                p += 1;
                            }
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            for p in 0..=*pos {
                                let mut s = 0.0f32;
                                for d in 0..dh {
                                    s += qp[d] * self.kv.k[self.kv.idx(l, kh, p, d)];
                                }
                                let s = s * scale;
                                self.scratch.scores[p] = s;
                                if s > maxs {
                                    maxs = s;
                                }
                            }
                        }
                    } else {
                        for p in 0..=*pos {
                            let mut s = 0.0f32;
                            for d in 0..dh {
                                s += q[qbase + d * b] * self.kv.k[self.kv.idx(l, kh, p, d)];
                            }
                            let s = s * scale;
                            self.scratch.scores[p] = s;
                            if s > maxs {
                                maxs = s;
                            }
                        }
                    }
                    let mut sum = 0.0f32;
                    for p in 0..=*pos {
                        let e = (self.scratch.scores[p] - maxs).exp();
                        self.scratch.scores[p] = e;
                        sum += e;
                    }
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        if b == 1 {
                            let ap = &mut attn[(h * dh)..(h * dh + dh)];
                            ap.fill(0.0);
                            for p in 0..=*pos {
                                let s = self.scratch.scores[p];
                                if s != 0.0 {
                                    let vr = self.kv.v_row(l, kh, p);
                                    crate::simd::f32_axpy(ap.as_mut_ptr(), vr, dh, s);
                                }
                            }
                            for d in 0..dh {
                                ap[d] *= inv;
                            }
                        } else {
                            local.fill(0.0);
                            for p in 0..=*pos {
                                let s = self.scratch.scores[p];
                                if s != 0.0 {
                                    let vr = self.kv.v_row(l, kh, p);
                                    crate::simd::f32_axpy(local.as_mut_ptr(), vr, dh, s);
                                }
                            }
                            for d in 0..dh {
                                attn[(h * dh + d) * b + bi] = local[d] * inv;
                            }
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        for d in 0..dh {
                            let mut acc = 0.0f32;
                            for p in 0..=*pos {
                                acc += self.scratch.scores[p] * self.kv.v[self.kv.idx(l, kh, p, d)];
                            }
                            attn[(h * dh + d) * b + bi] = acc * inv;
                        }
                    }
                }
            }
            if self.dbg_on && l == 0 {
                self.dbg.push(("kqv_out-0", b, attn[..qdim * b].to_vec()));
            }
            self.matmul(&lw.attn_o, attn, b, proj);
            if self.dbg_on && l == 0 {
                self.dbg.push(("attn_out-0", b, proj[..ne * b].to_vec()));
            }
            for i in 0..ne * b {
                x[i] += proj[i];
            }
            if self.dbg_on && l == 0 {
                self.dbg.push(("ffn_inp-0", b, x[..ne * b].to_vec()));
            }
            self.rmsnorm(x, &lw.ffn_norm, b, x2);
            if self.dbg_on && l == 0 {
                self.dbg.push(("ffn_norm-0", b, x2[..ne * b].to_vec()));
            }
            self.matmul(&lw.ffn_gate, x2, b, gate);
            self.matmul(&lw.ffn_up, x2, b, up);
            if self.dbg_on && l == 0 {
                self.dbg.push(("ffn_gate-0", b, gate[..cfg.n_ff * b].to_vec()));
                self.dbg.push(("ffn_up-0", b, up[..cfg.n_ff * b].to_vec()));
            }
            for i in 0..cfg.n_ff * b {
                let g = gate[i];
                let s = g / (1.0 + (-g).exp());
                gate[i] = s * up[i];
            }
            if self.dbg_on && l == 0 {
                self.dbg.push(("ffn_swiglu-0", b, gate[..cfg.n_ff * b].to_vec()));
            }
            self.matmul(&lw.ffn_down, gate, b, proj);
            if self.dbg_on && l == 0 {
                self.dbg.push(("ffn_out-0", b, proj[..ne * b].to_vec()));
            }
            for i in 0..ne * b {
                x[i] += proj[i];
            }
            if self.dbg_on && l == 0 {
                self.dbg.push(("l_out-0", b, x[..ne * b].to_vec()));
            }
            let _ = (qdim, kdim, ctx);
        }
        if want_logits {
            if cancellable && util::stop_requested() {
                bail!("generation stopped");
            }
            let last = b - 1;
            let outn = self.model.output_norm.clone();
            if let Some(outn) = outn {
                let mut last_x = vec![0.0f32; ne];
                for i in 0..ne {
                    last_x[i] = x[i * b + last];
                }
                let mut normed = vec![0.0f32; ne];
                self.rmsnorm(&last_x, &outn, 1, &mut normed);
                let out = self.model.output.clone().unwrap();
                let n_rows = out.ne1;
                self.scratch.ensure_logits(n_rows.max(self.model.vocab_size()));
                let base = self.model.gguf.file.as_ptr();
                let ptr = unsafe { base.add(out.off) };
                let act = if self.int8_enabled && crate::quant_int::supported(out.ttype) {
                    self.q8.prepare(&normed, out.ne0, 1);
                    &self.q8 as *const crate::quant_int::Q8Act
                } else {
                    core::ptr::null()
                };
                self.pool.matmul(
                    out.ttype,
                    ptr,
                    out.row_bytes,
                    out.ne0,
                    &normed,
                    1,
                    &mut self.scratch.logits[..n_rows],
                    n_rows,
                    act,
                );
                if self.dbg_on {
                    self.dbg.push(("result_output", 1, self.scratch.logits[..n_rows].to_vec()));
                }
            }
        }
        Ok(())
    }

    pub fn debug_matmul(&mut self, name: &str, x: &[f32], b: usize, out: &mut [f32]) -> Result<()> {
        let t = self
            .model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| crate::err!("no tensor {name}"))?;
        let base = self.model.gguf.file.as_ptr();
        let ptr = unsafe { base.add(t.off) };
        if crate::quant_int::supported(t.ttype) {
            self.q8.prepare(x, t.ne0, b);
            let act = &self.q8 as *const crate::quant_int::Q8Act;
            self.pool
                .matmul(t.ttype, ptr, t.row_bytes, t.ne0, x, b, out, t.ne1, act);
            return Ok(());
        }
        self.pool
            .matmul(t.ttype, ptr, t.row_bytes, t.ne0, x, b, out, t.ne1, core::ptr::null());
        Ok(())
    }

    pub fn debug_matmul_scalar(&mut self, name: &str, x: &[f32], b: usize, out: &mut [f32]) -> Result<()> {
        let t = self
            .model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| crate::err!("no tensor {name}"))?;
        let base = self.model.gguf.file.as_ptr();
        let row_bytes = t.row_bytes;
        for r in 0..t.ne1 {
            let row = unsafe { core::slice::from_raw_parts(base.add(t.off + r * row_bytes), row_bytes) };
            for lane in 0..b {
                let xr: Vec<f32> = (0..t.ne0).map(|j| x[j * b + lane]).collect();
                out[r * b + lane] = crate::quant::dot_row(t.ttype, row, t.ne0, &xr);
            }
        }
        Ok(())
    }

    pub fn debug_matmul_f32(&mut self, name: &str, x: &[f32], b: usize, out: &mut [f32]) -> Result<()> {
        let t = self
            .model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| crate::err!("no tensor {name}"))?;
        let base = self.model.gguf.file.as_ptr();
        let ptr = unsafe { base.add(t.off) };
        self.pool
            .matmul(t.ttype, ptr, t.row_bytes, t.ne0, x, b, out, t.ne1, core::ptr::null());
        Ok(())
    }

    pub fn logits(&self) -> &[f32] {
        &self.scratch.logits
    }

    pub fn generate<F: FnMut(&[u8]) -> bool>(
        &mut self,
        prompt: &[u32],
        opts: &GenOpts,
        mut on_text: F,
    ) -> Result<GenStats> {
        let mut stats = GenStats {
            stop: "max_tokens".to_string(),
            ..Default::default()
        };
        let mut pending: Vec<u8> = Vec::new();
        let mut rng = Rng::new(opts.seed);
        if prompt.is_empty() {
            bail!("empty prompt");
        }
        let t0 = std::time::Instant::now();
        let chunk = MAX_BATCH.min(32);
        let mut off = 0usize;
        while off < prompt.len() {
            if util::stop_requested() {
                stats.stop = "user_stop".to_string();
                break;
            }
            let n = chunk.min(prompt.len() - off);
            let ids = &prompt[off..off + n];
            let positions: Vec<usize> = (off..off + n).collect();
            let last = off + n >= prompt.len();
            if let Err(error) = self.forward_inner(ids, &positions, last, true) {
                if util::stop_requested() {
                    stats.stop = "user_stop".to_string();
                    break;
                }
                return Err(error);
            }
            off += n;
        }
        stats.prefill_tokens = off;
        stats.prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if stats.stop == "user_stop" {
            return Ok(stats);
        }
        let mut pos = prompt.len();
        let gen_t0 = std::time::Instant::now();
        let max_gen = opts.max_tokens;
        for i in 0..max_gen {
            if util::stop_requested() {
                stats.stop = "user_stop".to_string();
                break;
            }
            let nvocab = self.model.vocab_size().min(self.scratch.logits.len());
            let tok = sample(&mut self.scratch.logits[..nvocab], opts.temp, opts.top_p, &mut rng);
            if self.model.tok.is_eog(tok) {
                stats.stop = "eog".to_string();
                break;
            }
            stats.gen_tokens += 1;
            stats.gen_ids.push(tok);
            pending.extend_from_slice(&self.model.tok.decode_token(tok));
            if !flush_utf8(&mut pending, &mut on_text) {
                stats.stop = "callback_error".to_string();
                break;
            }
            if i + 1 == max_gen {
                break;
            }
            if pos >= self.n_ctx {
                stats.stop = "context_full".to_string();
                break;
            }
            if let Err(error) = self.forward_inner(&[tok], &[pos], true, true) {
                if util::stop_requested() {
                    stats.stop = "user_stop".to_string();
                    break;
                }
                return Err(error);
            }
            pos += 1;
        }
        if !pending.is_empty() && stats.stop != "callback_error" {
            let text = String::from_utf8_lossy(&pending).into_owned();
            if !on_text(text.as_bytes()) {
                stats.stop = "callback_error".to_string();
            }
        }
        stats.gen_ms = gen_t0.elapsed().as_secs_f64() * 1000.0;
        Ok(stats)
    }
}

fn flush_utf8(pending: &mut Vec<u8>, on_text: &mut impl FnMut(&[u8]) -> bool) -> bool {
    loop {
        match core::str::from_utf8(pending) {
            Ok(_) => {
                if !pending.is_empty() {
                    let ok = on_text(pending);
                    pending.clear();
                    return ok;
                }
                return true;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    let ok = on_text(&pending[..valid]);
                    pending.drain(..valid);
                    if !ok {
                        return false;
                    }
                } else {
                    if let Some(len) = e.error_len() {
                        let ok = on_text("\u{FFFD}".as_bytes());
                        pending.drain(..len);
                        if !ok {
                            return false;
                        }
                    } else {
                        return true;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod stream_tests {
    use super::{check_shape, flush_utf8, KvCache, TensorRef};

    #[test]
    fn callback_failure_aborts_stream() {
        let mut pending = b"hello".to_vec();
        let mut calls = 0;
        assert!(!flush_utf8(&mut pending, &mut |_| {
            calls += 1;
            false
        }));
        assert_eq!(calls, 1);
    }

    #[test]
    fn invalid_tensor_shape_is_rejected_before_raw_matmul() {
        let tensor = TensorRef {
            name: "blk.0.attn_q.weight".into(), ttype: 0,
            ne0: 128, ne1: 256, row_bytes: 512, off: 0,
        };
        assert!(check_shape(&tensor, 256, 256).is_err());
    }

    #[test]
    fn kv_allocation_overflow_returns_error() {
        assert!(KvCache::new(usize::MAX, 2, 8192, 128).is_err());
    }
}
