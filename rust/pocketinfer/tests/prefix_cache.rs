use pocketinfer::model::{Engine, GenOpts};
use pocketinfer::chat::{self, Message};
use std::process::Command;

#[test]
fn reused_and_divergent_prompts_match_fresh_kv() {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_path = std::env::temp_dir()
        .join(format!("pocketinfer-prefix-cache-{}.gguf", std::process::id()));
    let generated = Command::new("python3")
        .arg(crate_dir.join("tools/make_tiny_model.py"))
        .arg(&model_path)
        .status()
        .expect("python3 is needed to generate the test GGUF");
    assert!(generated.success());

    let path = model_path.to_str().unwrap();
    let mut engine = Engine::load(path, 512, 2).unwrap();
    let base = "The capital of France is Paris.\n".repeat(12);
    let initial = engine.model.tok.encode(&base, true);
    let extended = engine.model.tok.encode(&format!("{base}The capital of Germany is Berlin.\n"), true);
    let divergent = engine.model.tok.encode("A different conversation about Rust.", true);
    let options = GenOpts {
        n_ctx: 512, max_tokens: 2, temp: 0.0, top_p: 1.0, seed: 1, threads: 2,
    };

    engine.generate(&initial, &options, |_| true).unwrap();
    let cached = engine.generate(&extended, &options, |_| true).unwrap();
    assert!(cached.prefill_cached_tokens > 0);
    assert_eq!(cached.prefill_cached_tokens + cached.prefill_tokens, extended.len());

    let mut fresh = Engine::load(path, 512, 2).unwrap();
    let uncached = fresh.generate(&extended, &options, |_| true).unwrap();
    assert_eq!(cached.gen_ids, uncached.gen_ids);

    let stochastic = GenOpts {
        n_ctx: 512, max_tokens: 2, temp: 0.7, top_p: 0.95, seed: 1, threads: 2,
    };
    let repeated = engine.generate(&extended, &stochastic, |_| true).unwrap();
    let mut fresh = Engine::load(path, 512, 2).unwrap();
    let repeated_uncached = fresh.generate(&extended, &stochastic, |_| true).unwrap();
    assert_eq!(repeated.prefill_cached_tokens, (extended.len() - 1) / 32 * 32);
    assert_eq!(repeated.gen_ids, repeated_uncached.gen_ids);

    let changed = engine.generate(&divergent, &options, |_| true).unwrap();
    let mut fresh = Engine::load(path, 512, 2).unwrap();
    let changed_uncached = fresh.generate(&divergent, &options, |_| true).unwrap();
    assert_eq!(changed.gen_ids, changed_uncached.gen_ids);

    // A public evaluation can overwrite arbitrary KV positions. It must
    // invalidate the generation cache before the next prompt is processed.
    engine.forward(&[divergent[0]], &[0], false).unwrap();
    let after_manual_forward = engine.generate(&extended, &options, |_| true).unwrap();
    assert_eq!(after_manual_forward.prefill_cached_tokens, 0);
    assert_eq!(after_manual_forward.gen_ids, uncached.gen_ids);

    pocketinfer::util::request_stop();
    let stopped = engine.generate(&extended, &options, |_| true).unwrap();
    assert_eq!(stopped.stop, "user_stop");
    pocketinfer::util::STOP_REQUESTED.store(false, std::sync::atomic::Ordering::SeqCst);
    let after_stop = engine.generate(&extended, &options, |_| true).unwrap();
    assert_eq!(after_stop.prefill_cached_tokens, 0);
    assert_eq!(after_stop.gen_ids, uncached.gen_ids);

    // The agent does not simply append raw text: it turns a completed step
    // into an assistant/tool exchange. Compare actual MiniCPM5 templates.
    let system = Message {
        role: "system".into(), content: "You can list files in the workspace. ".repeat(10),
    };
    let user = Message { role: "user".into(), content: "List the workspace.".into() };
    let mut first_chat = chat::apply(&[system.clone(), user.clone()], true, true);
    first_chat.push_str(chat::direct_suffix());
    let mut next_chat = chat::apply(&[
        system, user,
        Message { role: "assistant".into(), content: "🔧 workspace_list(path=.)".into() },
        Message { role: "user".into(), content: "[TOOL RESULT] []".into() },
    ], true, true);
    next_chat.push_str(chat::direct_suffix());
    let first_ids = engine.model.tok.encode(&first_chat, true);
    let next_ids = engine.model.tok.encode(&next_chat, true);
    engine.generate(&first_ids, &options, |_| true).unwrap();
    let next = engine.generate(&next_ids, &options, |_| true).unwrap();
    assert!(next.prefill_cached_tokens > first_ids.len() / 2);
    assert_eq!(next.prefill_cached_tokens + next.prefill_tokens, next_ids.len());
    let mut fresh = Engine::load(path, 512, 2).unwrap();
    let next_uncached = fresh.generate(&next_ids, &options, |_| true).unwrap();
    assert_eq!(next.gen_ids, next_uncached.gen_ids);
    std::fs::remove_file(model_path).unwrap();
}
