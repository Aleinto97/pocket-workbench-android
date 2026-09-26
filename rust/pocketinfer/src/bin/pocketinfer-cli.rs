use pocketinfer::model::{Engine, GenOpts};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: pocketinfer-cli <model.gguf> tokens <text> | generate <text> <n> [temp]");
        std::process::exit(2);
    }
    let model = &args[1];
    let mode = args[2].as_str();
    let t0 = std::time::Instant::now();
    let threads = std::env::var("POCKET_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4usize);
    let mut engine = Engine::load(model, 512, threads).expect("load");
    eprintln!("load_ms={:.1}", t0.elapsed().as_secs_f64() * 1000.0);
    match mode {
        "genfile" => {
            let text = std::fs::read_to_string(args.get(3).cloned().unwrap_or_default()).unwrap();
            let ids = engine.model.tok.encode(&text, true);
            eprintln!("ids={ids:?}");
            let opts = GenOpts {
                n_ctx: 512,
                max_tokens: args.get(4).and_then(|v| v.parse().ok()).unwrap_or(8),
                temp: 0.0,
                top_p: 1.0,
                seed: 1,
                threads,
            };
            let stats = engine.generate(&ids, &opts, |_| true).unwrap();
            println!("{:?}", stats.gen_ids);
        }
        "tokensfile" => {
            let text = std::fs::read_to_string(args.get(3).cloned().unwrap_or_default()).unwrap();
            eprintln!("text={text:?}");
            let ids = engine.model.tok.encode(&text, true);
            println!("{ids:?} ({} tokens)", ids.len());
        }
        "tokens" => {
            let text = args.get(3).cloned().unwrap_or_default();
            let ids = engine.model.tok.encode(&text, true);
            println!("{ids:?}");
        }
        "chat" => {
            let text = args.get(3).cloned().unwrap_or_default();
            let msgs = vec![pocketinfer::chat::Message { role: "user".into(), content: text }];
            let prompt =
                pocketinfer::chat::apply_minicpm5(&msgs, true) + if args.get(4).map(|s| s == "think").unwrap_or(false) {
                    " thinking\n"
                } else {
                    " thinking\n\n\n\n"
                };
            eprintln!("prompt={prompt:?}");
            let ids = engine.model.tok.encode(&prompt, true);
            eprintln!("prompt_tokens={}", ids.len());
            let opts = GenOpts {
                n_ctx: 512,
                max_tokens: args.get(5).and_then(|s| s.parse().ok()).unwrap_or(16),
                temp: 0.0,
                top_p: 1.0,
                seed: 0xC0FFEE,
                threads: 4,
            };
            let stats = engine
                .generate(&ids, &opts, |bytes| {
                    print!("{}", String::from_utf8_lossy(bytes));
                    true
                })
                .expect("generate");
            println!();
            eprintln!(
                "prefill={} tok in {:.1}ms, gen={} tok in {:.1}ms, stop={}",
                stats.prefill_tokens, stats.prefill_ms, stats.gen_tokens, stats.gen_ms, stats.stop
            );
            eprintln!("ids={:?}", stats.gen_ids);
        }
        "deq" => {
            let name = args.get(3).cloned().unwrap_or_default();
            let n: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(8);
            let info = engine
                .model
                .gguf
                .tensor(&name)
                .cloned()
                .expect("tensor");
            let data = engine.model.gguf.tensor_data(&info).expect("data");
            let mut out = vec![0f32; info.ne[0] as usize];
            pocketinfer::quant::dequant_row(info.ttype, data, info.ne[0] as usize, &mut out);
            let vals: Vec<String> = out[..n].iter().map(|v| format!("{v:.8}")).collect();
            println!("{}", vals.join(" "));
        }
        "micro" => {
            let list = engine.model.tensor_list();
            let mut act = pocketinfer::quant_int::Q8Act::new();
            for (want_name, ttype_want) in [("blk.0.attn_q.weight", 12u32), ("blk.0.attn_v.weight", 14), ("blk.0.ffn_down.weight", 14)] {
                let (name, ttype, ne0, ne1) = list
                    .iter()
                    .find(|(n, t, _, _)| n == want_name && *t == ttype_want)
                    .cloned()
                    .unwrap();
                let info = engine.model.gguf.tensor(&name).cloned().unwrap();
                let data = engine.model.gguf.tensor_data(&info).unwrap();
                let row_bytes = pocketinfer::quant::tensor_nbytes(ttype, &[ne0 as u64]).unwrap();
                let mut x = vec![0f32; ne0];
                for j in 0..ne0 {
                    x[j] = ((j as f32) * 0.013).sin();
                }
                act.prepare(&x, ne0, 1);
                let iters = 2000usize;
                let t0 = std::time::Instant::now();
                let mut sink = 0f32;
                for i in 0..iters {
                    for r in 0..16 {
                        sink += pocketinfer::quant_int::dot_row_q8(ttype, &data[(i + r) % ne1 * row_bytes..], ne0, &act, 0);
                    }
                }
                let dt = t0.elapsed().as_secs_f64();
                let gw = (iters * 16 * ne0) as f64 / dt / 1e9;
                let t1 = std::time::Instant::now();
                for i in 0..iters / 4 {
                    for r in 0..4 {
                        sink += pocketinfer::quant::dot_row(ttype, &data[(i + r) % ne1 * row_bytes..], ne0, &x);
                    }
                }
                let dt2 = t1.elapsed().as_secs_f64();
                let gw2 = (iters / 4 * 4 * ne0) as f64 / dt2 / 1e9;
                println!("{name} type={ttype} k={ne0}: int8 {:.2} Gweight/s, f32 {:.2} Gweight/s (sink {sink:.2})", gw, gw2);
            }
        }
        "bench" => {
            let text = args.get(3).cloned().unwrap_or_else(|| "The capital of France is".into());
            let n: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(32);
            let ids = engine.model.tok.encode(&text, true);
            for round in 0..4 {
                for int8 in [true, false] {
                    engine.int8_enabled = int8;
                    let opts = GenOpts {
                        n_ctx: 512,
                        max_tokens: n,
                        temp: 0.7,
                        top_p: 0.95,
                        seed: 1,
                        threads: 4,
                    };
                    let stats = engine.generate(&ids, &opts, |_| true).unwrap();
                    println!(
                        "round {round} int8={int8}: prefill {:.0}ms ({:.2} tok/s) gen {:.0}ms ({:.2} tok/s)",
                        stats.prefill_ms,
                        stats.prefill_tokens as f64 / stats.prefill_ms * 1000.0,
                        stats.gen_ms,
                        stats.gen_tokens as f64 / stats.gen_ms * 1000.0
                    );
                }
            }
        }
        "typedecode" => {
            let text = args.get(3).cloned().unwrap_or_else(|| "The capital of France is".into());
            let ids = engine.model.tok.encode(&text, true);
            let b = ids.len();
            engine.int8_enabled = false;
            engine.dbg_on = true;
            engine.forward(&ids, &(0..b).collect::<Vec<_>>(), true).unwrap();
            let first = {
                let mut v: Vec<(usize, f32)> = engine.logits().iter().copied().enumerate().collect();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                v[0].0 as u32
            };
            engine.dbg.clear();
            engine.forward(&[first], &[b], true).unwrap();
            let find = |name: &str| -> Vec<f32> {
                for (n, nb, data) in engine.dbg.iter() {
                    if *n == name {
                        let mut x = vec![0f32; data.len() / nb];
                        for i in 0..x.len() {
                            x[i] = data[i * nb];
                        }
                        return x;
                    }
                }
                Vec::new()
            };
            let attn_norm = find("attn_norm-0");
            if let Some(path) = args.get(4) {
                let mut bytes = Vec::new();
                for v in attn_norm.iter() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(path, bytes).unwrap();
                let sw = find("ffn_swiglu-0");
                let mut bytes2 = Vec::new();
                for v in sw.iter() {
                    bytes2.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(format!("{path}.swiglu"), bytes2).unwrap();
            }
            let ffn_norm = find("ffn_norm-0");
            let swiglu = find("ffn_swiglu-0");
            for (name, x) in [
                ("blk.0.attn_q.weight", attn_norm.clone()),
                ("blk.0.attn_k.weight", attn_norm.clone()),
                ("blk.0.attn_v.weight", attn_norm.clone()),
                ("blk.0.ffn_gate.weight", ffn_norm.clone()),
                ("blk.0.ffn_up.weight", ffn_norm.clone()),
                ("blk.0.ffn_down.weight", swiglu.clone()),
                ("output.weight", attn_norm.clone()),
            ] {
                if x.is_empty() {
                    continue;
                }
                let info = engine.model.tensor_list();
                let ne1 = info
                    .iter()
                    .find(|(n, _, _, _)| n == name)
                    .map(|(_, _, _, ne1)| *ne1)
                    .unwrap_or(0);
                let mut a = vec![0f32; ne1];
                let mut b2 = vec![0f32; ne1];
                engine.debug_matmul(name, &x, 1, &mut a).unwrap();
                engine.debug_matmul_f32(name, &x, 1, &mut b2).unwrap();
                let mut md = 0f32;
                let mut mx = 0f32;
                for i in 0..ne1 {
                    md = md.max((a[i] - b2[i]).abs());
                    mx = mx.max(b2[i].abs());
                }
                println!("{name}: maxdiff={md:.5} maxval={mx:.3} rel={:.5}", if mx > 0.0 { md / mx } else { md });
            }
        }
        "cmpdecode" => {
            let text = args.get(3).cloned().unwrap_or_else(|| "The capital of France is".into());
            let ids = engine.model.tok.encode(&text, true);
            let b = ids.len();
            engine.int8_enabled = false;
            engine.forward(&ids, &(0..b).collect::<Vec<_>>(), true).unwrap();
            let first = {
                let mut v: Vec<(usize, f32)> = engine.logits().iter().copied().enumerate().collect();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                v[0].0 as u32
            };
            engine.forward(&[first], &[b], true).unwrap();
            let f32_logits = engine.logits().to_vec();
            engine.int8_enabled = true;
            engine.forward(&[first], &[b], true).unwrap();
            let i8_logits = engine.logits().to_vec();
            let mut md = 0f32;
            let mut mx = 0f32;
            for (a, b2) in f32_logits.iter().zip(i8_logits.iter()) {
                md = md.max((a - b2).abs());
                mx = mx.max(a.abs());
            }
            let top = |l: &[f32]| {
                let mut v: Vec<(usize, f32)> = l.iter().copied().enumerate().collect();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                v[..5].to_vec()
            };
            println!("decode token {first}: maxdiff={md:.6} maxval={mx:.3} rel={:.5}", md / mx);
            println!("f32 top5: {:?}", top(&f32_logits));
            println!("i8  top5: {:?}", top(&i8_logits));
        }
        "intcheck" => {
            let list = engine.model.tensor_list();
            let mut checked = 0usize;
            let mut worst = 0f32;
            for (name, ttype, ne0, ne1) in list {
                if !pocketinfer::quant_int::supported(ttype) {
                    continue;
                }
                let k = ne0;
                let mut x = vec![0f32; k];
                for j in 0..k {
                    x[j] = ((j as f32) * 0.017).sin() * 1.7 + ((j % 13) as f32) * 0.01;
                }
                let mut a = vec![0f32; ne1];
                let mut b = vec![0f32; ne1];
                engine.debug_matmul(&name, &x, 1, &mut a).unwrap();
                engine.debug_matmul_f32(&name, &x, 1, &mut b).unwrap();
                let mut md = 0f32;
                let mut refmax = 0f32;
                for i in 0..ne1 {
                    md = md.max((a[i] - b[i]).abs());
                    refmax = refmax.max(b[i].abs());
                }
                let rel = if refmax > 0.0 { md / refmax } else { md };
                if rel > worst {
                    worst = rel;
                }
                if rel > 0.02 {
                    println!("{name} type={ttype} maxdiff={md:.6} rel={rel:.4}");
                }
                checked += 1;
            }
            println!("intcheck: {checked} tensors, worst relative diff {worst:.5}");
        }
        "probe" => {
            for b in [1usize, 3] {
                let k = 256;
                let mut x = vec![0f32; k * b];
                for t in 0..b {
                    for j in 0..k {
                        x[j * b + t] = ((j as f32) * 0.001 + t as f32 * 0.1).sin();
                    }
                }
                let mut out = vec![0f32; 256 * b];
                engine.debug_matmul("blk.0.ffn_down.weight", &x, b, &mut out).unwrap();
                print!("ffn_down b={b} out[0..3]:");
                for i in 0..3 {
                    print!(" [");
                    for t in 0..b {
                        print!("{:.6} ", out[i * b + t]);
                    }
                    print!("]");
                }
                println!();
                engine.debug_matmul("blk.0.ffn_gate.weight", &x, b, &mut out).unwrap();
                print!("ffn_gate2 b={b} out[0..3]:");
                for i in 0..3 {
                    print!(" [");
                    for t in 0..b {
                        print!("{:.6} ", out[i * b + t]);
                    }
                    print!("]");
                }
                println!();
                let mut out2 = vec![0f32; 302 * b];
                engine.debug_matmul("output.weight", &x, b, &mut out2).unwrap();
                print!("output b={b} out[0..3]:");
                for i in 0..3 {
                    print!(" [");
                    for t in 0..b {
                        print!("{:.6} ", out2[i * b + t]);
                    }
                    print!("]");
                }
                println!();
                engine.debug_matmul("blk.0.ffn_gate.weight", &x, b, &mut out).unwrap();
                print!("ffn_gate b={b} out[0..3]:");
                for i in 0..3 {
                    print!(" [");
                    for t in 0..b {
                        print!("{:.6} ", out[i * b + t]);
                    }
                    print!("]");
                }
                println!();
            }
        }
        "tensor" => {
            for name in ["blk.0.ffn_down.weight", "blk.0.ffn_gate.weight", "blk.0.attn_q.weight", "blk.1.ffn_down.weight"] {
                println!("{name}: {:?}", engine.model.debug_tensor(name));
            }
        }
        "dumpfile" => {
            let text = std::fs::read_to_string(args.get(3).cloned().unwrap_or_default()).unwrap();
            let dir = args.get(4).cloned().unwrap_or_else(|| ".".into());
            let ids = engine.model.tok.encode(&text, true);
            eprintln!("ids={ids:?}");
            engine.dbg_on = true;
            let b = ids.len();
            engine.forward(&ids, &(0..b).collect::<Vec<_>>(), true).expect("forward");
            for (name, nb, data) in engine.dbg.clone() {
                let dim = data.len() / nb;
                let path = format!("{dir}/{name}.dump.{dim}x{nb}x1x1.f32");
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for bi in 0..nb {
                    for i in 0..dim {
                        bytes.extend_from_slice(&data[i * nb + bi].to_le_bytes());
                    }
                }
                std::fs::write(&path, bytes).expect("write dump");
            }
            println!("dumped");
        }
        "dump" => {
            let text = args.get(3).cloned().unwrap_or_else(|| "hello world".into());
            let dir = args.get(4).cloned().unwrap_or_else(|| ".".into());
            let ids = engine.model.tok.encode(&text, true);
            eprintln!("ids={ids:?}");
            engine.dbg_on = true;
            let b = ids.len();
            engine.forward(&ids, &(0..b).collect::<Vec<_>>(), true).expect("forward");
            for (name, nb, data) in engine.dbg.clone() {
                let dim = data.len() / nb;
                let path = format!("{dir}/{name}.dump.{dim}x{nb}x1x1.f32");
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for bi in 0..nb {
                    for i in 0..dim {
                        bytes.extend_from_slice(&data[i * nb + bi].to_le_bytes());
                    }
                }
                std::fs::write(&path, bytes).expect("write dump");
                println!("{path}");
            }
        }
        "debug" => {
            let ids = engine.model.tok.encode("hello", true);
            eprintln!("ids={ids:?}");
            engine.dbg_on = true;
            engine.forward(&ids, &[0, 1, 2], true).expect("forward");
            for (name, nb, data) in engine.dbg.clone() {
                let dim = data.len() / nb;
                let mut sum = 0.0f32;
                for v in data.iter() {
                    sum += *v;
                }
                println!(
                    "{name}: dim={dim} n={nb} first=[{:.4}, {:.4}, {:.4}] sum={sum:.6}",
                    data[0], data[nb], data[2 * nb]
                );
            }
        }
        "generate" => {
            let text = args.get(3).cloned().unwrap_or_default();
            let ids = engine.model.tok.encode(&text, true);
            eprintln!("prompt_tokens={ids:?}");
            let opts = GenOpts {
                n_ctx: 512,
                max_tokens: args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8),
                temp: args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                top_p: 1.0,
                seed: 0xC0FFEE,
                threads: 4,
            };
            let stats = engine
                .generate(&ids, &opts, |bytes| {
                    print!("{}", String::from_utf8_lossy(bytes));
                    true
                })
                .expect("generate");
            println!();
            eprintln!(
                "prefill={} tok in {:.1}ms, gen={} tok in {:.1}ms, stop={}",
                stats.prefill_tokens, stats.prefill_ms, stats.gen_tokens, stats.gen_ms, stats.stop
            );
            eprintln!("ids={:?}", stats.gen_ids);
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}
