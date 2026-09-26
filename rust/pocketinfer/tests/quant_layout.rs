use pocketinfer::quant::*;

#[test]
fn q6k_block_layout() {
    let mut block = vec![0u8; 210];
    block[0] = 0x19;
    block[128] = 0x55;
    block[192] = 1;
    block[194] = 2;
    block[196] = 3;
    block[198] = 4;
    block[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
    let mut out = [0f32; 256];
    dequant_row(GGML_TYPE_Q6_K, &block, 256, &mut out);
    let q1 = ((0x19u32 & 0xF) | (((0x55u32 >> 0) & 3) << 4)) as i32 - 32;
    let q2 = ((0u32 & 0xF) | (((0x55u32 >> 2) & 3) << 4)) as i32 - 32;
    let q3 = ((0x19u32 >> 4) | (((0x55u32 >> 4) & 3) << 4)) as i32 - 32;
    let q4 = ((0u32 >> 4) | (((0x55u32 >> 6) & 3) << 4)) as i32 - 32;
    assert_eq!(out[0], q1 as f32, "q1");
    assert_eq!(out[32], 2.0 * q2 as f32, "q2");
    assert_eq!(out[64], 3.0 * q3 as f32, "q3");
    assert_eq!(out[96], 4.0 * q4 as f32, "q4");
    assert_eq!(out[1], -32.0);
}

#[test]
fn q4k_block_layout() {
    let mut block = vec![0u8; 144];
    block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    block[2..4].copy_from_slice(&0x3C00u16.to_le_bytes());
    block[4] = 1;
    block[8] = 0;
    block[16] = 0x21;
    let mut out = [0f32; 256];
    dequant_row(GGML_TYPE_Q4_K, &block, 256, &mut out);
    assert_eq!(out[0], 1.0);
    assert_eq!(out[1], 0.0);
    assert_eq!(out[32], 0.0);
}

#[test]
fn f16_roundtrip() {
    for v in [0.0f32, 1.0, -1.0, 0.5, 0.1, 65504.0, 6.1e-5, -0.333] {
        let h = pocketinfer::util::f32_to_f16(v);
        let back = pocketinfer::util::f16_to_f32(h);
        assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-6, "{v} -> {back}");
    }
}

#[test]
fn int8_q8_0_matches() {
    if !pocketinfer::quant_int::int8_available() {
        return;
    }
    let mut w = vec![0u8; 34];
    w[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    for i in 0..32 {
        w[2 + i] = i as u8;
    }
    let x = vec![1.0f32; 32];
    let mut act = pocketinfer::quant_int::Q8Act::new();
    act.prepare(&x, 32, 1);
    let got = pocketinfer::quant_int::dot_row_q8(GGML_TYPE_Q8_0, &w, 32, &act, 0);
    let want: f32 = (0..32).map(|i| i as f32).sum();
    assert!((got - want).abs() < 0.5, "got {got} want {want}");
}

#[test]
fn int8_q4_0_matches() {
    if !pocketinfer::quant_int::int8_available() {
        return;
    }
    let mut w = vec![0u8; 18];
    w[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    for i in 0..16 {
        w[2 + i] = 0x99;
    }
    let x = vec![1.0f32; 32];
    let mut act = pocketinfer::quant_int::Q8Act::new();
    act.prepare(&x, 32, 1);
    let got = pocketinfer::quant_int::dot_row_q8(GGML_TYPE_Q4_0, &w, 32, &act, 0);
    let want = 32.0f32;
    assert!((got - want).abs() < 0.5, "got {got} want {want}");
}

#[test]
fn int8_q4k_matches_scalar() {
    if !pocketinfer::quant_int::int8_available() {
        return;
    }
    let mut w = vec![0u8; 144];
    w[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    w[2..4].copy_from_slice(&0x3C00u16.to_le_bytes());
    w[4] = 1;
    w[8] = 2;
    for i in 0..128 {
        w[16 + i] = 0x21;
    }
    let mut x = vec![0f32; 256];
    for j in 0..256 {
        x[j] = 0.25 + (j % 7) as f32 * 0.01;
    }
    let mut act = pocketinfer::quant_int::Q8Act::new();
    act.prepare(&x, 256, 1);
    let got = pocketinfer::quant_int::dot_row_q8(GGML_TYPE_Q4_K, &w, 256, &act, 0);
    let want = pocketinfer::quant::dot_row(GGML_TYPE_Q4_K, &w, 256, &x);
    assert!((got - want).abs() <= want.abs() * 0.02 + 1.0, "got {got} want {want}");
}

#[test]
fn int8_q6k_matches_scalar() {
    if !pocketinfer::quant_int::int8_available() {
        return;
    }
    let mut w = vec![0u8; 210];
    for i in 0..128 {
        w[i] = 0x95;
    }
    for i in 0..64 {
        w[128 + i] = 0x1B;
    }
    for i in 0..16 {
        w[192 + i] = 2;
    }
    w[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
    let mut x = vec![0f32; 256];
    for j in 0..256 {
        x[j] = 0.5 + (j % 5) as f32 * 0.02;
    }
    let mut act = pocketinfer::quant_int::Q8Act::new();
    act.prepare(&x, 256, 1);
    let got = pocketinfer::quant_int::dot_row_q8(GGML_TYPE_Q6_K, &w, 256, &act, 0);
    let want = pocketinfer::quant::dot_row(GGML_TYPE_Q6_K, &w, 256, &x);
    assert!((got - want).abs() <= want.abs() * 0.02 + 1.0, "got {got} want {want}");
}

fn prng(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

#[test]
fn int8_q4k_random_block() {
    if !pocketinfer::quant_int::int8_available() {
        return;
    }
    let mut st = 12345u64;
    let mut w = vec![0u8; 288 * 8];
    w[0..2].copy_from_slice(&0x3200u16.to_le_bytes());
    w[2..4].copy_from_slice(&0x2C00u16.to_le_bytes());
    for i in 4..288 * 8 {
        w[i] = prng(&mut st) as u8;
    }
    for blk in 0..8 {
        w[blk * 144..blk * 144 + 2].copy_from_slice(&0x3000u16.to_le_bytes());
        w[blk * 144 + 2..blk * 144 + 4].copy_from_slice(&0x2800u16.to_le_bytes());
    }
    let mut x = vec![0f32; 2048];
    for j in 0..2048 {
        x[j] = ((prng(&mut st) % 2000) as f32 / 1000.0) - 1.0;
    }
    let mut act = pocketinfer::quant_int::Q8Act::new();
    act.prepare(&x, 2048, 1);
    let got = pocketinfer::quant_int::dot_row_q8(GGML_TYPE_Q4_K, &w, 2048, &act, 0);
    let want = pocketinfer::quant::dot_row(GGML_TYPE_Q4_K, &w, 2048, &x);
    let tol = want.abs() * 0.02 + 0.01;
    assert!((got - want).abs() <= tol, "got {got} want {want} diff {}", got - want);
}

#[test]
fn int8_q6k_random_block() {
    if !pocketinfer::quant_int::int8_available() {
        return;
    }
    let mut st = 999u64;
    let mut w = vec![0u8; 420];
    for i in 0..420 {
        w[i] = prng(&mut st) as u8;
    }
    w[208..210].copy_from_slice(&0x3400u16.to_le_bytes());
    w[418..420].copy_from_slice(&0x3400u16.to_le_bytes());
    let mut x = vec![0f32; 512];
    for j in 0..512 {
        x[j] = ((prng(&mut st) % 2000) as f32 / 1000.0) - 1.0;
    }
    let mut act = pocketinfer::quant_int::Q8Act::new();
    act.prepare(&x, 512, 1);
    let got = pocketinfer::quant_int::dot_row_q8(GGML_TYPE_Q6_K, &w, 512, &act, 0);
    let want = pocketinfer::quant::dot_row(GGML_TYPE_Q6_K, &w, 512, &x);
    let tol = want.abs() * 0.02 + 0.01;
    assert!((got - want).abs() <= tol, "got {got} want {want} diff {}", got - want);
}

#[test]
fn row_dot_multi_matches_dot_row() {
    let mut st = 777u64;
    let mut w = vec![0u8; 144 * 4];
    for i in 0..w.len() {
        w[i] = prng(&mut st) as u8;
    }
    for blk in 0..4 {
        w[blk * 144..blk * 144 + 2].copy_from_slice(&0x3000u16.to_le_bytes());
        w[blk * 144 + 2..blk * 144 + 4].copy_from_slice(&0x2800u16.to_le_bytes());
    }
    let k = 1024usize;
    let b = 5usize;
    let mut xt = vec![0f32; k * b];
    for j in 0..k {
        for bi in 0..b {
            xt[j * b + bi] = ((prng(&mut st) % 2000) as f32 / 1000.0) - 1.0;
        }
    }
    let mut multi = vec![0f32; b];
    row_dot_multi(GGML_TYPE_Q4_K, &w, k, &xt, b, &mut multi);
    let mut single = vec![0f32; b];
    for bi in 0..b {
        let x: Vec<f32> = (0..k).map(|j| xt[j * b + bi]).collect();
        single[bi] = dot_row(GGML_TYPE_Q4_K, &w, k, &x);
    }
    for bi in 0..b {
        let d = (multi[bi] - single[bi]).abs();
        assert!(d <= single[bi].abs() * 1e-4 + 1e-4, "lane {bi}: multi={} single={}", multi[bi], single[bi]);
    }
}
