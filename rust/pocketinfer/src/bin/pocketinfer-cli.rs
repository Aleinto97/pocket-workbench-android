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
    let n_ctx = std::env::var("POCKET_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512usize);
    let mut engine = Engine::load(model, n_ctx, threads).expect("load");
    engine.int8_enabled = std::env::var("POCKET_NO_INT8").is_err();
    eprintln!("load_ms={:.1}", t0.elapsed().as_secs_f64() * 1000.0);
    match mode {
        "prefixcheck" => {
            let base = args.get(3).cloned().unwrap_or_else(|| "The capital of France is Paris.\n".repeat(16));
            let addition = args.get(4).cloned().unwrap_or_else(|| "The capital of Germany is Berlin.\n".to_string());
            let initial = engine.model.tok.encode(&base, true);
            let extended = engine.model.tok.encode(&(base + &addition), true);
            assert!(extended.len() < n_ctx, "increase POCKET_CTX for this prompt");
            let opts = GenOpts { n_ctx, max_tokens: 4, temp: 0.0, top_p: 1.0, seed: 1, threads };
            let first = engine.generate(&initial, &opts, |_| true).expect("initial prompt");
            let cached = engine.generate(&extended, &opts, |_| true).expect("cached prompt");
            let cached_logits = engine.logits().to_vec();
            let mut fresh = Engine::load(model, n_ctx, threads).expect("fresh engine");
            fresh.int8_enabled = engine.int8_enabled;
            let uncached = fresh.generate(&extended, &opts, |_| true).expect("fresh prompt");
            assert_eq!(cached.gen_ids, uncached.gen_ids, "prefix reuse changed greedy tokens");
            let max_logit_diff = cached_logits.iter().zip(fresh.logits()).map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_logit_diff < 1e-3, "prefix reuse changed logits: {max_logit_diff}");
            println!("first: {} eval tokens in {:.1}ms", first.prefill_tokens, first.prefill_ms);
            println!("second: {} cached + {} eval tokens in {:.1}ms", cached.prefill_cached_tokens, cached.prefill_tokens, cached.prefill_ms);
            println!("fresh second: {} eval tokens in {:.1}ms", uncached.prefill_tokens, uncached.prefill_ms);
            println!("maximum logit difference: {max_logit_diff:.6}");
            println!("identical generated token ids: {:?}", cached.gen_ids);
        }
        "stoptest" => {
            let prompt = engine.model.tok.encode(&"Ciao ".repeat(120), true);
            let stop = std::thread::spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(200));
                pocketinfer::util::request_stop();
            });
            let opts = GenOpts {
                n_ctx,
                max_tokens: 8,
                temp: 0.0,
                top_p: 1.0,
                seed: 1,
                threads,
            };
            let first = engine.generate(&prompt, &opts, |_| true).expect("stopped prefill");
            stop.join().unwrap();
            println!("first: stop={} prefill={}/{} tokens", first.stop, first.prefill_tokens, prompt.len());
            assert_eq!(first.stop, "user_stop");
            pocketinfer::util::STOP_REQUESTED.store(false, std::sync::atomic::Ordering::SeqCst);
            let retry = engine.model.tok.encode("The capital of France is", true);
            let second = engine.generate(&retry, &opts, |_| true).expect("retry with cached engine");
            println!("retry: stop={} gen={} first_id={:?}", second.stop, second.gen_tokens, second.gen_ids.first());
            assert!(second.gen_tokens > 0);
        }
        "genfile" => {
            let text = std::fs::read_to_string(args.get(3).cloned().unwrap_or_default()).unwrap();
            let ids = engine.model.tok.encode(&text, true);
            eprintln!("ids={ids:?}");
            let opts = GenOpts {
                n_ctx,
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
        "multicheck" => {
            let names = ["blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.ffn_down.weight", "output.weight"];
            let k = engine.model.cfg.n_embd.max(1);
            let b = 5usize;
            for name in names {
                let Some((_, _, ne0, ne1)) = engine
                    .model
                    .tensor_list()
                    .into_iter()
                    .find(|(n, _, _, _)| n == name)
                else {
                    continue;
                };
                let mut x = vec![0f32; ne0 * b];
                for j in 0..ne0 {
                    for lane in 0..b {
                        x[j * b + lane] = ((j * 7 + lane * 13) as f32 * 0.01).sin();
                    }
                }
                let mut a = vec![0f32; ne1 * b];
                let mut c = vec![0f32; ne1 * b];
                engine.debug_matmul_f32(name, &x, b, &mut a).unwrap();
                engine.debug_matmul_scalar(name, &x, b, &mut c).unwrap();
                let mut md = 0f32;
                let mut mx = 0f32;
                for i in 0..ne1 * b {
                    md = md.max((a[i] - c[i]).abs());
                    mx = mx.max(c[i].abs());
                }
                println!("{name}: maxdiff={md:.6} maxval={mx:.3} rel={:.5}", if mx > 0.0 { md / mx } else { md });
            }
            let _ = k;
        }
        "gpu" => {
            match engine.enable_opencl() {
                Ok(()) => println!("opencl_enabled=ok"),
                Err(e) => println!("opencl_enabled=FAILED: {e}"),
            }
            let text = args.get(3).cloned().unwrap_or_else(|| "The capital of France is".into());
            let ids = engine.model.tok.encode(&text, true);
            eprintln!("ids={ids:?}");
            let opts = GenOpts {
                n_ctx,
                max_tokens: args.get(4).and_then(|v| v.parse().ok()).unwrap_or(8),
                temp: 0.0,
                top_p: 1.0,
                seed: 1,
                threads,
            };
            let stats = engine.generate(&ids, &opts, |b| {
                print!("{}", String::from_utf8_lossy(b));
                true
            }).unwrap();
            println!();
            eprintln!("ids_out={:?}", stats.gen_ids);
        }
        "dumpids" => {
            let ids: Vec<u32> = args
                .get(3)
                .cloned()
                .unwrap_or_default()
                .split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect();
            let dir = args.get(4).cloned().unwrap_or_else(|| ".".into());
            eprintln!("ids={ids:?} ({})", ids.len());
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
            let lg = engine.logits();
            let mut top: Vec<(usize, f32)> = lg.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("top5: {:?}", &top[..5]);
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
        "chat" => {
            let text = args.get(3).cloned().unwrap_or_default();
            let n: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(32);
            let temp: f32 = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let mode = args.get(6).cloned().unwrap_or_else(|| "direct".into());
            let msgs = vec![pocketinfer::chat::Message { role: "user".into(), content: text }];
            let family = pocketinfer::chat::family_of(&engine.model.tok_pre(), &engine.model.cfg.name);
            let mut prompt = pocketinfer::chat::apply_for(family, &msgs, true);
            // Thinking tags exist only in the minicpm template; other families
            // must not see them.
            if family == "minicpm" {
                match mode.as_str() {
                    "direct" => prompt.push_str(pocketinfer::chat::direct_suffix()),
                    "think" => prompt.push_str(pocketinfer::chat::thinking_on_suffix()),
                    _ => {}
                }
            }
            eprintln!("mode={mode} temp={temp} prompt={prompt:?}");
            let ids = engine.model.tok.encode(&prompt, true);
            eprintln!("prompt_tokens={}", ids.len());
            let opts = GenOpts {
                n_ctx,
                max_tokens: n,
                temp,
                top_p: 0.95,
                seed: 0xC0FFEE,
                threads,
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
        }
        "generate" => {
            let text = args.get(3).cloned().unwrap_or_default();
            let ids = engine.model.tok.encode(&text, true);
            eprintln!("prompt_tokens={ids:?}");
            let opts = GenOpts {
                n_ctx,
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
