#!/usr/bin/env python3
"""Generate a tiny MiniCPM5-architecture GGUF for engine tests (stdlib only).

Usage: python3 tools/make_tiny_model.py out-f32.gguf
Then:  llama-quantize out-f32.gguf out-q4km.gguf Q4_K_M
"""
import struct
import sys
import random

VOCAB_EXTRA = 44
N_EMBD = 256
N_FF = 256
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 64
N_LAYER = 2
N_CTX = 128

CORPUS = (
    "hello world this is a tiny test model for pocket workbench. "
    "the quick brown fox jumps over the lazy dog. "
    "ciao mondo questo e un piccolo modello di prova. "
    "system user assistant think answer question tool call. "
    "inference engine rust gpu npu cpu snapdragon minicpm. "
)

BYTES_TO_UNICODE = {}


def byte_encoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def encode_bytes(s):
    enc = byte_encoder()
    return "".join(enc[b] for b in s.encode("utf-8"))


def train_bpe(merges_count=VOCAB_EXTRA):
    words = {}
    for w in CORPUS.split(" "):
        if not w:
            continue
        enc = list(encode_bytes(w))
        words[tuple(enc)] = words.get(tuple(enc), 0) + 1
    merges = []
    for _ in range(merges_count):
        pairs = {}
        for w, c in words.items():
            for i in range(len(w) - 1):
                pairs[(w[i], w[i + 1])] = pairs.get((w[i], w[i + 1]), 0) + c
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best)
        new_words = {}
        for w, c in words.items():
            out = []
            i = 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(w[i] + w[i + 1])
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_words[tuple(out)] = new_words.get(tuple(out), 0) + c
        words = new_words
    return merges


def str_bytes(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_str(key, value):
    b = value.encode("utf-8")
    return str_bytes(key) + struct.pack("<I", 8) + struct.pack("<Q", len(b)) + b


def kv_u32(key, value):
    return str_bytes(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def kv_i32(key, value):
    return str_bytes(key) + struct.pack("<I", 5) + struct.pack("<i", value)


def kv_f32(key, value):
    return str_bytes(key) + struct.pack("<I", 6) + struct.pack("<f", value)


def kv_bool(key, value):
    return str_bytes(key) + struct.pack("<I", 7) + struct.pack("<B", 1 if value else 0)


def kv_str_array(key, values):
    out = str_bytes(key) + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(values))
    for v in values:
        out += str_bytes(v)
    return out


def kv_i32_array(key, values):
    out = str_bytes(key) + struct.pack("<I", 9) + struct.pack("<I", 5) + struct.pack("<Q", len(values))
    for v in values:
        out += struct.pack("<i", v)
    return out


def main(out_path):
    merges = train_bpe()
    enc = byte_encoder()
    tokens = []
    types = []
    tokens.append("<s>")
    types.append(3)
    tokens.append("</s>")
    types.append(3)
    for b in range(256):
        tokens.append(enc[b])
        types.append(1)
    seen = set(tokens)
    for a, b in merges:
        t = a + b
        if t not in seen:
            seen.add(t)
            tokens.append(t)
            types.append(1)
    vocab_size = len(tokens)
    merges_list = [f"{a} {b}" for a, b in merges]

    tensors = []
    rng = random.Random(1234)

    def rand_tensor(name, ne0, ne1, scale=0.05):
        vals = [rng.gauss(0.0, scale) for _ in range(ne0 * ne1)]
        data = struct.pack("<%df" % len(vals), *vals)
        tensors.append((name, [ne0, ne1], data))

    def norm_tensor(name, n):
        vals = [1.0 + rng.gauss(0.0, 0.01) for _ in range(n)]
        tensors.append((name, [n], struct.pack("<%df" % n, *vals)))

    rand_tensor("token_embd.weight", N_EMBD, vocab_size, 0.08)
    rand_tensor("output.weight", N_EMBD, vocab_size, 0.08)
    norm_tensor("output_norm.weight", N_EMBD)
    for i in range(N_LAYER):
        norm_tensor(f"blk.{i}.attn_norm.weight", N_EMBD)
        norm_tensor(f"blk.{i}.ffn_norm.weight", N_EMBD)
        rand_tensor(f"blk.{i}.attn_q.weight", N_EMBD, N_HEAD * HEAD_DIM)
        rand_tensor(f"blk.{i}.attn_k.weight", N_EMBD, N_HEAD_KV * HEAD_DIM)
        rand_tensor(f"blk.{i}.attn_v.weight", N_EMBD, N_HEAD_KV * HEAD_DIM)
        rand_tensor(f"blk.{i}.attn_output.weight", N_HEAD * HEAD_DIM, N_EMBD)
        rand_tensor(f"blk.{i}.ffn_gate.weight", N_EMBD, N_FF)
        rand_tensor(f"blk.{i}.ffn_up.weight", N_EMBD, N_FF)
        rand_tensor(f"blk.{i}.ffn_down.weight", N_FF, N_EMBD)

    kvs_list = []
    kvs_list.append(kv_str("general.architecture", "llama"))
    kvs_list.append(kv_str("general.name", "TinyMiniCPM5"))
    kvs_list.append(kv_u32("general.file_type", 0))
    kvs_list.append(kv_u32("general.quantization_version", 2))
    kvs_list.append(kv_u32("general.alignment", 32))
    kvs_list.append(kv_u32("llama.block_count", N_LAYER))
    kvs_list.append(kv_u32("llama.context_length", N_CTX))
    kvs_list.append(kv_u32("llama.embedding_length", N_EMBD))
    kvs_list.append(kv_u32("llama.feed_forward_length", N_FF))
    kvs_list.append(kv_u32("llama.attention.head_count", N_HEAD))
    kvs_list.append(kv_u32("llama.attention.head_count_kv", N_HEAD_KV))
    kvs_list.append(kv_u32("llama.attention.key_length", HEAD_DIM))
    kvs_list.append(kv_u32("llama.attention.value_length", HEAD_DIM))
    kvs_list.append(kv_u32("llama.rope.dimension_count", HEAD_DIM))
    kvs_list.append(kv_f32("llama.rope.freq_base", 10000.0))
    kvs_list.append(kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-6))
    kvs_list.append(kv_u32("llama.vocab_size", vocab_size))
    kvs_list.append(kv_str("tokenizer.ggml.model", "gpt2"))
    kvs_list.append(kv_str("tokenizer.ggml.pre", "minicpm5"))
    kvs_list.append(kv_str_array("tokenizer.ggml.tokens", tokens))
    kvs_list.append(kv_i32_array("tokenizer.ggml.token_type", types))
    kvs_list.append(kv_str_array("tokenizer.ggml.merges", merges_list))
    kvs_list.append(kv_u32("tokenizer.ggml.bos_token_id", 0))
    kvs_list.append(kv_u32("tokenizer.ggml.eos_token_id", 1))
    kvs_list.append(kv_u32("tokenizer.ggml.unknown_token_id", 0))
    kvs_list.append(kv_u32("tokenizer.ggml.padding_token_id", 1))
    kvs_list.append(kv_bool("tokenizer.ggml.add_bos_token", False))
    kvs_list.append(kv_bool("tokenizer.ggml.add_sep_token", False))
    kvs_list.append(kv_bool("tokenizer.ggml.add_eos_token", False))

    info = b""
    offset = 0
    data = b""
    for name, dims, blob in tensors:
        info += str_bytes(name)
        info += struct.pack("<I", len(dims))
        for d in dims:
            info += struct.pack("<Q", d)
        info += struct.pack("<I", 0)
        info += struct.pack("<Q", offset)
        pad = (-len(blob)) % 32
        data += blob + b"\x00" * pad
        offset += len(blob) + pad

    kvs = b"".join(kvs_list)
    kv_count = len(kvs_list)
    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), kv_count)
    body = header + kvs + info
    align_pad = (-len(body)) % 32
    with open(out_path, "wb") as f:
        f.write(body + b"\x00" * align_pad + data)
    print(f"wrote {out_path}: vocab={vocab_size} tensors={len(tensors)} merges={len(merges)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny-f32.gguf")
