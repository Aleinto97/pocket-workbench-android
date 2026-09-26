use crate::quant::*;

pub struct Q8Act {
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    pub sum: Vec<i32>,
    pub k: usize,
    pub b: usize,
    pub nb: usize,
}

impl Q8Act {
    pub fn new() -> Self {
        Self { q: Vec::new(), d: Vec::new(), sum: Vec::new(), k: 0, b: 0, nb: 0 }
    }

    pub fn prepare(&mut self, x: &[f32], k: usize, b: usize) {
        let nb = k / 32;
        if self.k != k || self.b != b {
            self.q.resize(k * b, 0);
            self.d.resize(nb * b, 0.0);
            self.sum.resize(nb * b, 0);
            self.k = k;
            self.b = b;
            self.nb = nb;
        }
        for bi in 0..b {
            for blk in 0..nb {
                let base = blk * 32;
                let mut amax = 0f32;
                for j in 0..32 {
                    let v = x[(base + j) * b + bi].abs();
                    if v > amax {
                        amax = v;
                    }
                }
                let dd = amax / 127.0;
                let inv = if dd > 0.0 { 1.0 / dd } else { 0.0 };
                let mut s = 0i32;
                let qbase = bi * k + base;
                for j in 0..32 {
                    let v = (x[(base + j) * b + bi] * inv).round().clamp(-127.0, 127.0) as i32;
                    self.q[qbase + j] = v as i8;
                    s += v;
                }
                self.d[bi * nb + blk] = dd;
                self.sum[bi * nb + blk] = s;
            }
        }
    }

    #[inline]
    pub fn qblock(&self, lane: usize, blk: usize) -> &[i8] {
        &self.q[lane * self.k + blk * 32..lane * self.k + blk * 32 + 32]
    }
}

pub fn int8_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        return std::arch::is_aarch64_feature_detected!("dotprod");
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub fn supported(ttype: u32) -> bool {
    matches!(
        ttype,
        GGML_TYPE_Q4_K | GGML_TYPE_Q6_K | GGML_TYPE_Q4_0 | GGML_TYPE_Q8_0
    ) && int8_available()
}

pub fn dot_row_q8(ttype: u32, row: &[u8], k: usize, act: &Q8Act, lane: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        match ttype {
            GGML_TYPE_Q4_0 => return dp::q4_0(row, k, act, lane),
            GGML_TYPE_Q8_0 => return dp::q8_0(row, k, act, lane),
            GGML_TYPE_Q4_K => return dp::q4_k(row, k, act, lane),
            GGML_TYPE_Q6_K => return dp::q6_k(row, k, act, lane),
            _ => {}
        }
    }
    let _ = (ttype, row, k, act, lane);
    0.0
}

#[cfg(target_arch = "aarch64")]
mod dp {
    use super::*;
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn q8_0(w: &[u8], k: usize, act: &Q8Act, lane: usize) -> f32 {
        let nb = k / 32;
        let mut acc = 0f32;
        for blk in 0..nb {
            let base = blk * 34;
            let d = crate::util::f16_to_f32(u16::from_le_bytes([w[base], w[base + 1]]));
            let qw = w.as_ptr().add(base + 2) as *const i8;
            let a = act.qblock(lane, blk).as_ptr();
            let mut d0 = vdupq_n_s32(0);
            let mut d1 = vdupq_n_s32(0);
            d0 = vdotq_s32(d0, vld1q_s8(qw), vld1q_s8(a));
            d1 = vdotq_s32(d1, vld1q_s8(qw.add(16)), vld1q_s8(a.add(16)));
            acc += d * act.d[lane * nb + blk] * (vaddvq_s32(vaddq_s32(d0, d1)) as f32);
        }
        acc
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn q4_0(w: &[u8], k: usize, act: &Q8Act, lane: usize) -> f32 {
        let nb = k / 32;
        let mut acc = 0f32;
        for blk in 0..nb {
            let base = blk * 18;
            let d = crate::util::f16_to_f32(u16::from_le_bytes([w[base], w[base + 1]]));
            let qw = w.as_ptr().add(base + 2);
            let a = act.qblock(lane, blk).as_ptr();
            let v = vld1q_u8(qw);
            let lo = vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(0x0F)));
            let hi = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
            let mut s1 = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(a));
            s1 = vdotq_s32(s1, hi, vld1q_s8(a.add(16)));
            let dq = act.d[lane * nb + blk];
            let sq = act.sum[lane * nb + blk];
            acc += d * dq * (vaddvq_s32(s1) as f32 - 8.0 * sq as f32);
        }
        acc
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn q4_k(w: &[u8], k: usize, act: &Q8Act, lane: usize) -> f32 {
        let nb = k / 256;
        let mut acc = 0f32;
        for blk in 0..nb {
            let base = blk * 144;
            let d = crate::util::f16_to_f32(u16::from_le_bytes([w[base], w[base + 1]]));
            let dmin = crate::util::f16_to_f32(u16::from_le_bytes([w[base + 2], w[base + 3]]));
            let scales = &w[base + 4..base + 16];
            let qs = w.as_ptr().add(base + 16);
            let abl = lane * act.nb + blk * 8;
            let abase = lane * act.k + blk * 256;
            let mut gcount = 0usize;
            for g in 0..4 {
                let (s1, m1) = get_scale_min_k4_pub(gcount, scales);
                let (s2, m2) = get_scale_min_k4_pub(gcount + 1, scales);
                gcount += 2;
                let q = qs.add(g * 32);
                let a = act.q.as_ptr().add(abase + g * 64);
                let mut lo = vdupq_n_s32(0);
                let mut hi = vdupq_n_s32(0);
                for h in 0..2 {
                    let v = vld1q_u8(q.add(h * 16));
                    let l = vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(0x0F)));
                    let hh = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
                    lo = vdotq_s32(lo, l, vld1q_s8(a.add(h * 16)));
                    hi = vdotq_s32(hi, hh, vld1q_s8(a.add(32 + h * 16)));
                }
                let da = act.d[abl + 2 * g];
                let db = act.d[abl + 2 * g + 1];
                let sa = act.sum[abl + 2 * g];
                let sb = act.sum[abl + 2 * g + 1];
                let s_lo = vaddvq_s32(lo) as f32;
                let s_hi = vaddvq_s32(hi) as f32;
                acc += d * da * s1 as f32 * s_lo - da * dmin * m1 as f32 * sa as f32;
                acc += d * db * s2 as f32 * s_hi - db * dmin * m2 as f32 * sb as f32;
            }
        }
        acc
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn q6_k(w: &[u8], k: usize, act: &Q8Act, lane: usize) -> f32 {
        let nb = k / 256;
        let mut acc = 0f32;
        for blk in 0..nb {
            let base = blk * 210;
            let d = crate::util::f16_to_f32(u16::from_le_bytes([w[base + 208], w[base + 209]]));
            let ql = w.as_ptr().add(base);
            let qh = w.as_ptr().add(base + 128);
            let sc = w.as_ptr().add(base + 192);
            let abase = lane * act.k + blk * 256;
            let abl = lane * act.nb + blk * 8;
            for half in 0..2 {
                let qlb = ql.add(half * 64);
                let qhb = qh.add(half * 32);
                let scb = sc.add(half * 8);
                let a0 = act.q.as_ptr().add(abase + half * 128);
                for g in 0..2 {
                    let l = g * 16;
                    let q1 = build_q6_stream(qlb.add(l), qhb.add(l), 0, false);
                    let q2 = build_q6_stream(qlb.add(l + 32), qhb.add(l), 2, false);
                    let q3 = build_q6_stream(qlb.add(l), qhb.add(l), 4, true);
                    let q4 = build_q6_stream(qlb.add(l + 32), qhb.add(l), 6, true);
                    let s1 = *scb.add(g) as i8 as f32;
                    let s2 = *scb.add(g + 2) as i8 as f32;
                    let s3 = *scb.add(g + 4) as i8 as f32;
                    let s4 = *scb.add(g + 6) as i8 as f32;
                    let x1 = vld1q_s8(a0.add(l));
                    let x2 = vld1q_s8(a0.add(32 + l));
                    let x3 = vld1q_s8(a0.add(64 + l));
                    let x4 = vld1q_s8(a0.add(96 + l));
                    let d1 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q1, x1)) as f32;
                    let d2 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q2, x2)) as f32;
                    let d3 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q3, x3)) as f32;
                    let d4 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q4, x4)) as f32;
                    let da = act.d[abl + 4 * half + 0];
                    let db = act.d[abl + 4 * half + 1];
                    let dc = act.d[abl + 4 * half + 2];
                    let de = act.d[abl + 4 * half + 3];
                    acc += d * s1 * da * d1;
                    acc += d * s2 * db * d2;
                    acc += d * s3 * dc * d3;
                    acc += d * s4 * de * d4;
                }
            }
        }
        acc
    }

    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn build_q6_stream(ql: *const u8, qh: *const u8, shift: u8, high: bool) -> int8x16_t {
        let mut out_ptr = [0u8; 16];
        for l in 0..16usize {
            let b = *ql.add(l);
            let lo = if high { (b >> 4) as u16 } else { (b & 0xF) as u16 };
            let hi = ((*qh.add(l) >> shift) & 3) as u16;
            let v = ((lo | (hi << 4)) as i16 - 32) as i8;
            out_ptr[l] = v as u8;
        }
        vld1q_s8(out_ptr.as_ptr() as *const i8)
    }

    #[inline]
    pub fn get_scale_min_k4_pub(j: usize, q: &[u8]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
        }
    }
}
