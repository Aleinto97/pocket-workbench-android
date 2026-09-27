use crate::util::{f16_to_f32, Result};

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q4_1: u32 = 3;
pub const GGML_TYPE_Q5_0: u32 = 6;
pub const GGML_TYPE_Q5_1: u32 = 7;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_Q8_1: u32 = 9;
pub const GGML_TYPE_Q2_K: u32 = 10;
pub const GGML_TYPE_Q3_K: u32 = 11;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;
pub const GGML_TYPE_Q8_K: u32 = 15;
pub const GGML_TYPE_BF16: u32 = 30;

pub fn type_block(ttype: u32) -> Option<(usize, usize)> {
    Some(match ttype {
        GGML_TYPE_F32 => (1, 4),
        GGML_TYPE_F16 | GGML_TYPE_BF16 => (1, 2),
        GGML_TYPE_Q4_0 => (32, 18),
        GGML_TYPE_Q4_1 => (32, 20),
        GGML_TYPE_Q5_0 => (32, 22),
        GGML_TYPE_Q5_1 => (32, 24),
        GGML_TYPE_Q8_0 => (32, 34),
        GGML_TYPE_Q8_1 => (32, 36),
        GGML_TYPE_Q2_K => (256, 84),
        GGML_TYPE_Q3_K => (256, 110),
        GGML_TYPE_Q4_K => (256, 144),
        GGML_TYPE_Q5_K => (256, 176),
        GGML_TYPE_Q6_K => (256, 210),
        GGML_TYPE_Q8_K => (256, 292),
        _ => return None,
    })
}

pub fn type_name(ttype: u32) -> &'static str {
    match ttype {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        GGML_TYPE_Q4_0 => "Q4_0",
        GGML_TYPE_Q4_1 => "Q4_1",
        GGML_TYPE_Q5_0 => "Q5_0",
        GGML_TYPE_Q5_1 => "Q5_1",
        GGML_TYPE_Q8_0 => "Q8_0",
        GGML_TYPE_Q8_1 => "Q8_1",
        GGML_TYPE_Q2_K => "Q2_K",
        GGML_TYPE_Q3_K => "Q3_K",
        GGML_TYPE_Q4_K => "Q4_K",
        GGML_TYPE_Q5_K => "Q5_K",
        GGML_TYPE_Q6_K => "Q6_K",
        GGML_TYPE_Q8_K => "Q8_K",
        GGML_TYPE_BF16 => "BF16",
        _ => "UNKNOWN",
    }
}

pub fn tensor_nbytes(ttype: u32, ne: &[u64]) -> Result<usize> {
    let (be, bb) = type_block(ttype).ok_or_else(|| crate::err!("unsupported tensor type {}", ttype))?;
    let mut n = 1u64;
    for (i, d) in ne.iter().enumerate() {
        if i == 0 && *d % be as u64 != 0 {
            bail!("dim0 {d} not divisible by block {be}");
        }
        n = n.checked_mul(*d).ok_or_else(|| crate::err!("tensor element count overflow"))?;
    }
    let blocks = n / be as u64;
    usize::try_from(blocks).ok().and_then(|n| n.checked_mul(bb))
        .ok_or_else(|| crate::err!("tensor byte size overflow"))
}

#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

#[inline]
fn q_scale(ttype: u32, b: &[u8]) -> (f32, f32) {
    match ttype {
        GGML_TYPE_Q4_0 | GGML_TYPE_Q5_0 | GGML_TYPE_Q8_0 | GGML_TYPE_Q8_1 => {
            (f16_to_f32(u16::from_le_bytes([b[0], b[1]])), 0.0)
        }
        GGML_TYPE_Q4_1 | GGML_TYPE_Q5_1 => (
            f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
            f16_to_f32(u16::from_le_bytes([b[2], b[3]])),
        ),
        _ => (0.0, 0.0),
    }
}

pub fn dequant_row(ttype: u32, data: &[u8], k: usize, out: &mut [f32]) {
    assert!(out.len() >= k);
    let mut done = 0usize;
    let mut off = 0usize;
    match ttype {
        GGML_TYPE_F32 => {
            for i in 0..k {
                out[i] = f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
            }
            return;
        }
        GGML_TYPE_F16 => {
            for i in 0..k {
                out[i] = f16_to_f32(u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]));
            }
            return;
        }
        GGML_TYPE_BF16 => {
            for i in 0..k {
                out[i] = bf16_to_f32(u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]));
            }
            return;
        }
        _ => {}
    }
    while done < k {
        match ttype {
            GGML_TYPE_Q4_0 => {
                let (d, _) = q_scale(ttype, &data[off..]);
                for j in 0..16 {
                    let v = data[off + 2 + j];
                    out[done + j] = ((v & 0x0F) as f32 - 8.0) * d;
                    out[done + j + 16] = ((v >> 4) as f32 - 8.0) * d;
                }
                off += 18;
                done += 32;
            }
            GGML_TYPE_Q4_1 => {
                let (d, m) = q_scale(ttype, &data[off..]);
                for j in 0..16 {
                    let v = data[off + 4 + j];
                    out[done + j] = (v & 0x0F) as f32 * d + m;
                    out[done + j + 16] = (v >> 4) as f32 * d + m;
                }
                off += 20;
                done += 32;
            }
            GGML_TYPE_Q5_0 => {
                let (d, _) = q_scale(ttype, &data[off..]);
                let qh = u32::from_le_bytes(data[off + 2..off + 6].try_into().unwrap());
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    let v = data[off + 6 + j];
                    out[done + j] = (((v & 0x0F) as u32 | xh0) as f32 - 16.0) * d;
                    out[done + j + 16] = (((v >> 4) as u32 | xh1) as f32 - 16.0) * d;
                }
                off += 22;
                done += 32;
            }
            GGML_TYPE_Q5_1 => {
                let (d, m) = q_scale(ttype, &data[off..]);
                let qh = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    let v = data[off + 8 + j];
                    out[done + j] = ((v & 0x0F) as u32 | xh0) as f32 * d + m;
                    out[done + j + 16] = ((v >> 4) as u32 | xh1) as f32 * d + m;
                }
                off += 24;
                done += 32;
            }
            GGML_TYPE_Q8_0 | GGML_TYPE_Q8_1 => {
                let (d, _) = q_scale(ttype, &data[off..]);
                let qs_off = if ttype == GGML_TYPE_Q8_0 { off + 2 } else { off + 4 };
                for j in 0..32 {
                    out[done + j] = (data[qs_off + j] as i8) as f32 * d;
                }
                off += if ttype == GGML_TYPE_Q8_0 { 34 } else { 36 };
                done += 32;
            }
            GGML_TYPE_Q2_K => {
                let d = f16_to_f32(u16::from_le_bytes([data[off + 80], data[off + 81]]));
                let dmin = f16_to_f32(u16::from_le_bytes([data[off + 82], data[off + 83]]));
                let scales = &data[off..off + 16];
                let qs = &data[off + 16..off + 80];
                let mut is = 0usize;
                let mut n = 0usize;
                let mut qp = 0usize;
                while n < 256 {
                    let mut shift = 0u32;
                    for _ in 0..4 {
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = dmin * (sc >> 4) as f32;
                        for l in 0..16 {
                            out[done + n + l] = dl * ((qs[qp + l] >> shift) & 3) as f32 - ml;
                        }
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = dmin * (sc >> 4) as f32;
                        for l in 0..16 {
                            out[done + n + 16 + l] = dl * ((qs[qp + 16 + l] >> shift) & 3) as f32 - ml;
                        }
                        shift += 2;
                        n += 32;
                        qp += 32;
                    }
                }
                off += 84;
                done += 256;
            }
            GGML_TYPE_Q3_K => {
                let d_all = f16_to_f32(u16::from_le_bytes([data[off + 108], data[off + 109]]));
                let qs = &data[off + 32..off + 96];
                let hm = &data[off..off + 32];
                let mut aux = [0u32; 4];
                for i in 0..3 {
                    aux[i] = u32::from_le_bytes(data[off + 96 + i * 4..off + 100 + i * 4].try_into().unwrap());
                }
                let tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & 0x0f0f_0f0f) | (((tmp >> 4) & 0x0303_0303) << 4);
                aux[3] = ((aux[1] >> 4) & 0x0f0f_0f0f) | (((tmp >> 6) & 0x0303_0303) << 4);
                aux[0] = (aux[0] & 0x0f0f_0f0f) | (((tmp >> 0) & 0x0303_0303) << 4);
                aux[1] = (aux[1] & 0x0f0f_0f0f) | (((tmp >> 2) & 0x0303_0303) << 4);
                let scales: &[u8] = unsafe { core::slice::from_raw_parts(aux.as_ptr() as *const u8, 16) };
                let mut is = 0usize;
                let mut n = 0usize;
                let mut qp = 0usize;
                let mut m: u8 = 1;
                while n < 256 {
                    let mut shift = 0u32;
                    for _ in 0..4 {
                        let dl = d_all * (scales[is] as i8 as f32 - 32.0);
                        is += 1;
                        for l in 0..16 {
                            let hi = if hm[qp + l] & m != 0 { 0 } else { 4 };
                            out[done + n + l] = dl * ((((qs[qp + l] >> shift) & 3) as i8 - hi) as f32);
                        }
                        let dl = d_all * (scales[is] as i8 as f32 - 32.0);
                        is += 1;
                        for l in 0..16 {
                            let hi = if hm[qp + 16 + l] & m != 0 { 0 } else { 4 };
                            out[done + n + 16 + l] =
                                dl * ((((qs[qp + 16 + l] >> shift) & 3) as i8 - hi) as f32);
                        }
                        shift += 2;
                        m <<= 1;
                        n += 32;
                        qp += 32;
                    }
                }
                off += 110;
                done += 256;
            }
            GGML_TYPE_Q4_K => {
                let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));
                let scales = &data[off + 4..off + 16];
                let qs = &data[off + 16..off + 144];
                let mut is = 0usize;
                let mut n = 0usize;
                let mut qp = 0usize;
                while n < 256 {
                    let (sc, m) = get_scale_min_k4(is, scales);
                    let d1 = d * sc as f32;
                    let m1 = dmin * m as f32;
                    let (sc, m) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc as f32;
                    let m2 = dmin * m as f32;
                    for l in 0..32 {
                        out[done + n + l] = d1 * (qs[qp + l] & 0x0F) as f32 - m1;
                    }
                    for l in 0..32 {
                        out[done + n + 32 + l] = d2 * (qs[qp + l] >> 4) as f32 - m2;
                    }
                    qp += 32;
                    n += 64;
                    is += 2;
                }
                off += 144;
                done += 256;
            }
            GGML_TYPE_Q5_K => {
                let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));
                let scales = &data[off + 4..off + 16];
                let qh = &data[off + 16..off + 48];
                let ql = &data[off + 48..off + 176];
                let mut is = 0usize;
                let mut n = 0usize;
                let mut u1 = 1u8;
                let mut u2 = 2u8;
                let mut qp = 0usize;
                while n < 256 {
                    let (sc, m) = get_scale_min_k4(is, scales);
                    let d1 = d * sc as f32;
                    let m1 = dmin * m as f32;
                    let (sc, m) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * sc as f32;
                    let m2 = dmin * m as f32;
                    for l in 0..32 {
                        let q = (ql[qp + l] & 0x0F) + if qh[qp + l] & u1 != 0 { 16 } else { 0 };
                        out[done + n + l] = d1 * q as f32 - m1;
                    }
                    for l in 0..32 {
                        let q = (ql[qp + l] >> 4) + if qh[qp + l] & u2 != 0 { 16 } else { 0 };
                        out[done + n + 32 + l] = d2 * q as f32 - m2;
                    }
                    qp += 32;
                    n += 64;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
                off += 176;
                done += 256;
            }
            GGML_TYPE_Q6_K => {
                let d = f16_to_f32(u16::from_le_bytes([data[off + 208], data[off + 209]]));
                let ql = &data[off..off + 128];
                let qh = &data[off + 128..off + 192];
                let sc = &data[off + 192..off + 208];
                for half in 0..2 {
                    let qlb = half * 64;
                    let qhb = half * 32;
                    let scb = half * 8;
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[qlb + l] & 0xF) as i16 | (((qh[qhb + l] >> 0) & 3) as i16) << 4) - 32;
                        let q2 = ((ql[qlb + l + 32] & 0xF) as i16 | (((qh[qhb + l] >> 2) & 3) as i16) << 4) - 32;
                        let q3 = ((ql[qlb + l] >> 4) as i16 | (((qh[qhb + l] >> 4) & 3) as i16) << 4) - 32;
                        let q4 = ((ql[qlb + l + 32] >> 4) as i16 | (((qh[qhb + l] >> 6) & 3) as i16) << 4) - 32;
                        let base = done + half * 128;
                        out[base + l] = d * (sc[scb + is] as i8 as f32) * q1 as f32;
                        out[base + 32 + l] = d * (sc[scb + is + 2] as i8 as f32) * q2 as f32;
                        out[base + 64 + l] = d * (sc[scb + is + 4] as i8 as f32) * q3 as f32;
                        out[base + 96 + l] = d * (sc[scb + is + 6] as i8 as f32) * q4 as f32;
                    }
                }
                off += 210;
                done += 256;
            }
            GGML_TYPE_Q8_K => {
                let d = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                for j in 0..256 {
                    out[done + j] = (data[off + 4 + j] as i8) as f32 * d;
                }
                off += 292;
                done += 256;
            }
            _ => return,
        }
    }
}

pub fn dot_row(ttype: u32, data: &[u8], k: usize, x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        match ttype {
            GGML_TYPE_Q4_0 => return unsafe { neon::dot_q4_0(data, k, x) },
            GGML_TYPE_Q8_0 => return unsafe { neon::dot_q8_0(data, k, x) },
            GGML_TYPE_Q4_K => return unsafe { neon::dot_q4_k(data, k, x) },
            GGML_TYPE_Q6_K => return unsafe { neon::dot_q6_k(data, k, x) },
            GGML_TYPE_F16 => return unsafe { neon::dot_f16(data, k, x) },
            GGML_TYPE_F32 => return unsafe { neon::dot_f32(data, k, x) },
            _ => {}
        }
    }
    dot_row_scalar(ttype, data, k, x)
}

pub fn dot_row_scalar(ttype: u32, data: &[u8], k: usize, x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    match ttype {
        GGML_TYPE_F32 => {
            for i in 0..k {
                sum += f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap()) * x[i];
            }
        }
        GGML_TYPE_F16 => {
            for i in 0..k {
                sum += f16_to_f32(u16::from_le_bytes([data[i * 2], data[i * 2 + 1]])) * x[i];
            }
        }
        GGML_TYPE_BF16 => {
            for i in 0..k {
                sum += bf16_to_f32(u16::from_le_bytes([data[i * 2], data[i * 2 + 1]])) * x[i];
            }
        }
        GGML_TYPE_Q6_K => {
            let nb = k / 256;
            let mut acc = 0.0f32;
            for b in 0..nb {
                let base = b * 210;
                let d = f16_to_f32(u16::from_le_bytes([data[base + 208], data[base + 209]]));
                let ql = &data[base..base + 128];
                let qh = &data[base + 128..base + 192];
                let sc = &data[base + 192..base + 208];
                let xb = &x[b * 256..];
                for half in 0..2 {
                    let qlb = half * 64;
                    let qhb = half * 32;
                    let scb = half * 8;
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[qlb + l] & 0xF) as i32 | (((qh[qhb + l] >> 0) & 3) as i32) << 4) - 32;
                        let q2 = ((ql[qlb + l + 32] & 0xF) as i32 | (((qh[qhb + l] >> 2) & 3) as i32) << 4) - 32;
                        let q3 = ((ql[qlb + l] >> 4) as i32 | (((qh[qhb + l] >> 4) & 3) as i32) << 4) - 32;
                        let q4 = ((ql[qlb + l + 32] >> 4) as i32 | (((qh[qhb + l] >> 6) & 3) as i32) << 4) - 32;
                        let o = half * 128;
                        acc += d * (sc[scb + is] as i8 as f32) * q1 as f32 * xb[o + l];
                        acc += d * (sc[scb + is + 2] as i8 as f32) * q2 as f32 * xb[o + 32 + l];
                        acc += d * (sc[scb + is + 4] as i8 as f32) * q3 as f32 * xb[o + 64 + l];
                        acc += d * (sc[scb + is + 6] as i8 as f32) * q4 as f32 * xb[o + 96 + l];
                    }
                }
            }
            sum = acc;
        }
        _ => {
            let (be, bb) = match type_block(ttype) {
                Some(v) => v,
                None => return 0.0,
            };
            let nb = k / be;
            let mut tmp = [0f32; 256];
            for i in 0..nb {
                dequant_row(ttype, &data[i * bb..], be, &mut tmp);
                for j in 0..be {
                    sum += tmp[j] * x[i * be + j];
                }
            }
        }
    }
    sum
}

pub fn row_dot_multi(ttype: u32, row: &[u8], k: usize, xt: &[f32], b: usize, out: &mut [f32]) {
    if b == 1 {
        out[0] = dot_row(ttype, row, k, xt);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if b >= 4 && unsafe { neon::row_dot_multi_neon(ttype, row, k, xt, b, out) } {
            return;
        }
    }
    let (be, bb) = match type_block(ttype) {
        Some(v) => v,
        None => {
            for o in out[..b].iter_mut() {
                *o = 0.0;
            }
            return;
        }
    };
    let nb = k / be;
    let mut tmp = [0f32; 256];
    for i in 0..nb {
        dequant_row(ttype, &row[i * bb..], be, &mut tmp);
        let obase = i * be * b;
        for j in 0..be {
            let w = tmp[j];
            let xp = &xt[obase + j * b..obase + j * b + b];
            for bi in 0..b {
                out[bi] += w * xp[bi];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::*;
    use core::arch::aarch64::*;

    #[inline]
    unsafe fn nib8_to_s32x4(q: uint8x8_t) -> (int32x4_t, int32x4_t) {
        let w = vmovl_u8(q);
        (vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w))), vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(w))))
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn row_dot_multi_neon(
        ttype: u32,
        row: &[u8],
        k: usize,
        xt: &[f32],
        b: usize,
        out: &mut [f32],
    ) -> bool {
        let (be, bb) = match type_block(ttype) {
            Some(v) => v,
            None => return false,
        };
        let nb = k / be;
        let mut tmp = [0f32; 256];
        let op = out.as_mut_ptr();
        for i in 0..nb {
            dequant_row(ttype, &row[i * bb..], be, &mut tmp);
            let xb = xt.as_ptr().add(i * be * b);
            for j in 0..be {
                let wv = vdupq_n_f32(tmp[j]);
                let xj = xb.add(j * b);
                let mut bi = 0usize;
                while bi + 4 <= b {
                    let acc = vld1q_f32(op.add(bi));
                    let xv = vld1q_f32(xj.add(bi));
                    vst1q_f32(op.add(bi), vfmaq_f32(acc, wv, xv));
                    bi += 4;
                }
                while bi < b {
                    *op.add(bi) += tmp[j] * *xj.add(bi);
                    bi += 1;
                }
            }
        }
        true
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_f32(data: &[u8], k: usize, x: &[f32]) -> f32 {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let src = data.as_ptr() as *const f32;
        let mut i = 0;
        while i + 8 <= k {
            acc0 = vfmaq_f32(acc0, vld1q_f32(src.add(i)), vld1q_f32(x.as_ptr().add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(src.add(i + 4)), vld1q_f32(x.as_ptr().add(i + 4)));
            i += 8;
        }
        let mut s = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < k {
            s += *src.add(i) * *x.get_unchecked(i);
            i += 1;
        }
        s
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_f16(data: &[u8], k: usize, x: &[f32]) -> f32 {
        let mut sum = 0f32;
        for i in 0..k {
            let h = u16::from_le_bytes([*data.get_unchecked(i * 2), *data.get_unchecked(i * 2 + 1)]);
            sum += f16_to_f32(h) * *x.get_unchecked(i);
        }
        sum
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q4_0(data: &[u8], k: usize, x: &[f32]) -> f32 {
        let nb = k / 32;
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..nb {
            let base = i * 18;
            let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
            let qs = data.as_ptr().add(base + 2);
            let xp = x.as_ptr().add(i * 32);
            let mut s = vdupq_n_f32(0.0);
            for g in 0..2 {
                let qv = vld1_u8(qs.add(g * 8));
                let lo = vand_u8(qv, vdup_n_u8(0x0F));
                let hi = vshr_n_u8(qv, 4);
                let (l0, l1) = nib8_to_s32x4(lo);
                let (h0, h1) = nib8_to_s32x4(hi);
                let eight = vdupq_n_s32(8);
                s = vfmaq_f32(s, vcvtq_f32_s32(vsubq_s32(l0, eight)), vld1q_f32(xp.add(g * 8)));
                s = vfmaq_f32(s, vcvtq_f32_s32(vsubq_s32(l1, eight)), vld1q_f32(xp.add(g * 8 + 4)));
                s = vfmaq_f32(s, vcvtq_f32_s32(vsubq_s32(h0, eight)), vld1q_f32(xp.add(16 + g * 8)));
                s = vfmaq_f32(s, vcvtq_f32_s32(vsubq_s32(h1, eight)), vld1q_f32(xp.add(16 + g * 8 + 4)));
            }
            acc = vfmaq_f32(acc, s, vdupq_n_f32(d));
        }
        vaddvq_f32(acc)
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q8_0(data: &[u8], k: usize, x: &[f32]) -> f32 {
        let nb = k / 32;
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..nb {
            let base = i * 34;
            let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
            let qs = data.as_ptr().add(base + 2);
            let xp = x.as_ptr().add(i * 32);
            let mut s = vdupq_n_f32(0.0);
            for half in 0..2 {
                let qv = vld1q_s8(qs.add(half * 16) as *const i8);
                let lo = vmovl_s8(vget_low_s8(qv));
                let hi = vmovl_s8(vget_high_s8(qv));
                s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(xp.add(half * 16)));
                s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vld1q_f32(xp.add(half * 16 + 4)));
                s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vld1q_f32(xp.add(half * 16 + 8)));
                s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vld1q_f32(xp.add(half * 16 + 12)));
            }
            acc = vfmaq_f32(acc, s, vdupq_n_f32(d));
        }
        vaddvq_f32(acc)
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q4_k(data: &[u8], k: usize, x: &[f32]) -> f32 {
        let nb = k / 256;
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..nb {
            let base = i * 144;
            let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
            let scales = &data[base + 4..base + 16];
            let qs = data.as_ptr().add(base + 16);
            let xp = x.as_ptr().add(i * 256);
            let mut is = 0usize;
            for n in (0..8).step_by(2) {
                let (sc1, m1u) = get_scale_min_k4(is, scales);
                let (sc2, m2u) = get_scale_min_k4(is + 1, scales);
                is += 2;
                let d1 = d * sc1 as f32;
                let b1 = -dmin * m1u as f32;
                let d2 = d * sc2 as f32;
                let b2 = -dmin * m2u as f32;
                let qb = qs.add(n / 2 * 32);
                let xb = xp.add(n * 32);
                let mut s1 = vdupq_n_f32(0.0);
                let mut s2 = vdupq_n_f32(0.0);
                let mut sx1 = vdupq_n_f32(0.0);
                let mut sx2 = vdupq_n_f32(0.0);
                for g in 0..4 {
                    let qv = vld1_u8(qb.add(g * 8));
                    let (lo0, lo1) = nib8_to_s32x4(vand_u8(qv, vdup_n_u8(0x0F)));
                    let (hi0, hi1) = nib8_to_s32x4(vshr_n_u8(qv, 4));
                    let x1 = vld1q_f32(xb.add(g * 8));
                    let x2 = vld1q_f32(xb.add(g * 8 + 4));
                    let y1 = vld1q_f32(xb.add(32 + g * 8));
                    let y2 = vld1q_f32(xb.add(32 + g * 8 + 4));
                    s1 = vfmaq_f32(s1, vcvtq_f32_s32(lo0), x1);
                    s1 = vfmaq_f32(s1, vcvtq_f32_s32(lo1), x2);
                    sx1 = vaddq_f32(sx1, vaddq_f32(x1, x2));
                    s2 = vfmaq_f32(s2, vcvtq_f32_s32(hi0), y1);
                    s2 = vfmaq_f32(s2, vcvtq_f32_s32(hi1), y2);
                    sx2 = vaddq_f32(sx2, vaddq_f32(y1, y2));
                }
                acc = vfmaq_f32(acc, s1, vdupq_n_f32(d1));
                acc = vfmaq_f32(acc, sx1, vdupq_n_f32(b1));
                acc = vfmaq_f32(acc, s2, vdupq_n_f32(d2));
                acc = vfmaq_f32(acc, sx2, vdupq_n_f32(b2));
            }
        }
        vaddvq_f32(acc)
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q6_k(data: &[u8], k: usize, x: &[f32]) -> f32 {
        let nb = k / 256;
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..nb {
            let base = i * 210;
            let d = f16_to_f32(u16::from_le_bytes([data[base + 208], data[base + 209]]));
            let ql = data.as_ptr().add(base);
            let qh = data.as_ptr().add(base + 128);
            let sc = data.as_ptr().add(base + 192);
            let xp = x.as_ptr().add(i * 256);
            for half in 0..2 {
                let qlb = ql.add(half * 64);
                let qhb = qh.add(half * 32);
                let scb = sc.add(half * 8);
                let xb = xp.add(half * 128);
                for g in 0..4 {
                    let l = g * 8;
                    let ql0 = vld1_u8(qlb.add(l));
                    let ql1 = vld1_u8(qlb.add(l + 32));
                    let qh0 = vld1_u8(qhb.add(l));
                    let m3 = vdup_n_u8(3);
                    let q1 = vorr_u8(vand_u8(ql0, vdup_n_u8(0x0F)), vshl_n_u8(vand_u8(qh0, m3), 4));
                    let q2 = vorr_u8(vand_u8(ql1, vdup_n_u8(0x0F)), vshl_n_u8(vand_u8(vshr_n_u8(qh0, 2), m3), 4));
                    let q3 = vorr_u8(vshr_n_u8(ql0, 4), vshl_n_u8(vand_u8(vshr_n_u8(qh0, 4), m3), 4));
                    let q4 = vorr_u8(vshr_n_u8(ql1, 4), vshl_n_u8(vand_u8(vshr_n_u8(qh0, 6), m3), 4));
                    let s1 = d * (*scb.add((l) / 16) as i8 as f32);
                    let s2 = d * (*scb.add((l) / 16 + 2) as i8 as f32);
                    let s3 = d * (*scb.add((l) / 16 + 4) as i8 as f32);
                    let s4 = d * (*scb.add((l) / 16 + 6) as i8 as f32);
                    let sub = vdupq_n_s32(32);
                    let (a0, a1) = nib8_to_s32x4(q1);
                    let (b0, b1) = nib8_to_s32x4(q2);
                    let (c0, c1) = nib8_to_s32x4(q3);
                    let (e0, e1) = nib8_to_s32x4(q4);
                    let x1 = vld1q_f32(xb.add(l));
                    let x2 = vld1q_f32(xb.add(l + 4));
                    let y1 = vld1q_f32(xb.add(32 + l));
                    let y2 = vld1q_f32(xb.add(32 + l + 4));
                    let z1 = vld1q_f32(xb.add(64 + l));
                    let z2 = vld1q_f32(xb.add(64 + l + 4));
                    let w1 = vld1q_f32(xb.add(96 + l));
                    let w2 = vld1q_f32(xb.add(96 + l + 4));
                    let sv1 = vdupq_n_f32(s1);
                    let sv2 = vdupq_n_f32(s2);
                    let sv3 = vdupq_n_f32(s3);
                    let sv4 = vdupq_n_f32(s4);
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vsubq_s32(a0, sub)), sv1), x1);
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vsubq_s32(a1, sub)), sv1), x2);
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vsubq_s32(b0, sub)), sv2), y1);
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vsubq_s32(b1, sub)), sv2), y2);
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vsubq_s32(c0, sub)), sv3), z1);
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vsubq_s32(c1, sub)), sv3), z2);
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vsubq_s32(e0, sub)), sv4), w1);
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vsubq_s32(e1, sub)), sv4), w2);
                }
            }
        }
        vaddvq_f32(acc)
    }
}

pub fn is_supported(ttype: u32) -> bool {
    type_block(ttype).is_some()
}

#[cfg(test)]
mod size_tests {
    use super::{tensor_nbytes, GGML_TYPE_F32};

    #[test]
    fn oversized_tensor_is_rejected_instead_of_saturating() {
        assert!(tensor_nbytes(GGML_TYPE_F32, &[u64::MAX, u64::MAX]).is_err());
    }
}
