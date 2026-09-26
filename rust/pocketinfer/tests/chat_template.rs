use pocketinfer::chat::{apply_minicpm5, Message};

#[test]
fn minicpm5_template_matches_llama_cpp_goldens() {
    let msgs = vec![Message { role: "user".into(), content: "Ciao".into() }];
    let got = apply_minicpm5(&msgs, true);
    assert_eq!(got, "<s><|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\n", "user");
    let msgs = vec![Message { role: "system".into(), content: "Sei utile.".into() }, Message { role: "user".into(), content: "Ciao".into() }];
    let got = apply_minicpm5(&msgs, true);
    assert_eq!(got, "<s><|im_start|>system\nSei utile.<|im_end|>\n<|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\n", "sys_user");
    let msgs = vec![Message { role: "system".into(), content: "Sei utile.".into() }, Message { role: "user".into(), content: "Ciao".into() }, Message { role: "assistant".into(), content: "Come posso aiutarti?".into() }, Message { role: "user".into(), content: "Spiega la fotosintesi".into() }];
    let got = apply_minicpm5(&msgs, true);
    assert_eq!(got, "<s><|im_start|>system\nSei utile.<|im_end|>\n<|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nCome posso aiutarti?<|im_end|>\n<|im_start|>user\nSpiega la fotosintesi<|im_end|>\n<|im_start|>assistant\n", "multi");
    let msgs = vec![Message { role: "user".into(), content: "2+2?".into() }, Message { role: "assistant".into(), content: "Diamo la risposta.\n\n\n4".into() }, Message { role: "user".into(), content: "grazie".into() }];
    let got = apply_minicpm5(&msgs, true);
    assert_eq!(got, "<s><|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nDiamo la risposta.\n\n\n4<|im_end|>\n<|im_start|>user\ngrazie<|im_end|>\n<|im_start|>assistant\n", "reasoning");
    let msgs = vec![Message { role: "user".into(), content: "2+2?".into() }, Message { role: "assistant".into(), content: "ragiono qui</think>4".into() }, Message { role: "user".into(), content: "grazie".into() }];
    let got = apply_minicpm5(&msgs, true);
    assert_eq!(got, "<s><|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>\nragiono qui\n</think>\n\n4<|im_end|>\n<|im_start|>user\ngrazie<|im_end|>\n<|im_start|>assistant\n", "reasoning_markers");
    let msgs = vec![Message { role: "user".into(), content: "leggi file".into() }, Message { role: "tool".into(), content: "contenuto A".into() }, Message { role: "tool".into(), content: "contenuto B".into() }, Message { role: "user".into(), content: "grazie".into() }];
    let got = apply_minicpm5(&msgs, true);
    assert_eq!(got, "<s><|im_start|>user\nleggi file<|im_end|>\n<|im_start|>user\n<tool_response>\ncontenuto A\n</tool_response>\n<tool_response>\ncontenuto B\n</tool_response><|im_end|>\n<|im_start|>user\ngrazie<|im_end|>\n<|im_start|>assistant\n", "toolgroup");
}
