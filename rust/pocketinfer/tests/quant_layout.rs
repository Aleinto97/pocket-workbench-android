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

fn prng_bytes(state: &mut u64, out: &mut [u8]) {
    for b in out.iter_mut() {
        *b = prng(state) as u8;
    }
}

fn random_act(state: &mut u64, k: usize, b: usize) -> pocketinfer::quant_int::Q8Act {
    let mut act = pocketinfer::quant_int::Q8Act::new();
    let mut x = vec![0f32; k * b];
    for v in x.iter_mut() {
        *v = ((prng(state) % 2000) as f32 / 1000.0) - 1.0;
    }
    act.prepare(&x, k, b);
    act
}

#[test]
fn int8_multi_matches_single_bitwise() {
    if !pocketinfer::quant_int::int8_available() {
        return;
    }
    let mut st = 4242u64;
    let b = 32usize;
    let k = 1024usize;
    for ttype in [GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K] {
        let (be, bb) = type_block(ttype).unwrap();
        let nblk = k / be;
        let mut row = vec![0u8; bb * nblk];
        prng_bytes(&mut st, &mut row);
        for blk in 0..nblk {
            let base = blk * bb;
            row[base..base + 2].copy_from_slice(&0x3200u16.to_le_bytes());
            match ttype {
                GGML_TYPE_Q4_K => {
                    row[base + 2..base + 4].copy_from_slice(&0x2C00u16.to_le_bytes());
                }
                GGML_TYPE_Q6_K => {
                    row[base + 208..base + 210].copy_from_slice(&0x3200u16.to_le_bytes());
                }
                _ => {}
            }
        }
        let act = random_act(&mut st, k, b);
        let mut multi = vec![0f32; b];
        pocketinfer::quant_int::dot_row_q8_lanes(ttype, &row, k, &act, &mut multi, b);
        let mut single = vec![0f32; b];
        for lane in 0..b {
            single[lane] = pocketinfer::quant_int::dot_row_q8(ttype, &row, k, &act, lane);
        }
        for lane in 0..b {
            let diff = (multi[lane] - single[lane]).abs();
            assert!(
                diff <= single[lane].abs() * 1e-5 + 1e-4,
                "type {ttype} lane {lane}: multi={} single={} diff={}",
                multi[lane],
                single[lane],
                diff
            );
        }
    }
}

#[test]
#[ignore]
fn bench_multilane_vs_single() {
    if !pocketinfer::quant_int::int8_available() {
        return;
    }
    let mut st = 7u64;
    let b = 32usize;
    let k = 2048usize;
    let rows = 512usize;
    for ttype in [GGML_TYPE_Q4_K, GGML_TYPE_Q6_K] {
        let (be, bb) = type_block(ttype).unwrap();
        let row_bytes = bb * (k / be);
        let mut row = vec![0u8; row_bytes];
        prng_bytes(&mut st, &mut row);
        for blk in 0..k / 256 {
            row[blk * row_bytes / (k / 256)] = 0;
        }
        let act = random_act(&mut st, k, b);
        let mut out = vec![0f32; b * rows];
        let t1 = std::time::Instant::now();
        for r in 0..rows {
            pocketinfer::quant_int::dot_row_q8_lanes(ttype, &row, k, &act, &mut out[r * b..r * b + b], b);
        }
        let multi = t1.elapsed().as_secs_f64() * 1000.0;
        let t2 = std::time::Instant::now();
        for r in 0..rows {
            for lane in 0..b {
                out[r * b + lane] = pocketinfer::quant_int::dot_row_q8(ttype, &row, k, &act, lane);
            }
        }
        let single = t2.elapsed().as_secs_f64() * 1000.0;
        println!(
            "type {ttype} rows={rows} b={b} k={k}: multi={multi:.1}ms single={single:.1}ms speedup={:.2}x",
            single / multi
        );
    }
}

#[test]
fn int8_multi_exact_bitwise() {
    if !pocketinfer::quant_int::int8_available() {
        return;
    }
    let mut st = 99u64;
    let b = 32usize;
    let k = 512usize;
    for ttype in [GGML_TYPE_Q4_K, GGML_TYPE_Q6_K] {
        let (be, bb) = type_block(ttype).unwrap();
        let nblk = k / be;
        let mut row = vec![0u8; bb * nblk];
        prng_bytes(&mut st, &mut row);
        for blk in 0..nblk {
            let base = blk * bb;
            row[base..base + 2].copy_from_slice(&0x3200u16.to_le_bytes());
            if ttype == GGML_TYPE_Q4_K {
                row[base + 2..base + 4].copy_from_slice(&0x2C00u16.to_le_bytes());
            } else {
                row[base + 208..base + 210].copy_from_slice(&0x3200u16.to_le_bytes());
            }
        }
        let act = random_act(&mut st, k, b);
        let mut multi = vec![0f32; b];
        pocketinfer::quant_int::dot_row_q8_lanes(ttype, &row, k, &act, &mut multi, b);
        for lane in 0..b {
            let single = pocketinfer::quant_int::dot_row_q8(ttype, &row, k, &act, lane);
            assert_eq!(multi[lane].to_bits(), single.to_bits(), "type {ttype} lane {lane}: multi={} single={}", multi[lane], single);
        }
    }
}
