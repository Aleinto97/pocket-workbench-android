//! Fast paths for Q8 activation quantization (b == 1, i.e. decode).

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn prepare_lane0(
    x: &[f32],
    _k: usize,
    nb: usize,
    q: &mut [i8],
    d: &mut [f32],
    sum: &mut [i32],
) {
    let xp = x.as_ptr();
    let mut blk = 0usize;
    while blk < nb {
        let base = blk * 32;
        let mut amax = 0f32;
        let mut j = 0usize;
        while j + 4 <= 32 {
            let v = vabsq_f32(vld1q_f32(xp.add(base + j)));
            amax = amax.max(vmaxvq_f32(v));
            j += 4;
        }
        let dd = amax / 127.0;
        let inv = if dd > 0.0 { 1.0 / dd } else { 0.0 };
        let mut s = 0i32;
        let qbase = base;
        let mut j = 0usize;
        while j + 4 <= 32 {
            let xv = vld1q_f32(xp.add(base + j));
            let scaled = vmulq_n_f32(xv, inv);
            // round-half-away-from-zero (matches f32::round), then clamp
            let rounded = vrndaq_f32(scaled);
            let clamped = vmaxq_f32(vminq_f32(rounded, vdupq_n_f32(127.0)), vdupq_n_f32(-127.0));
            let iv = vcvtq_s32_f32(clamped);
            let mut buf = [0i32; 4];
            vst1q_s32(buf.as_mut_ptr(), iv);
            for t in 0..4 {
                q[qbase + j + t] = buf[t] as i8;
                s += buf[t];
            }
            j += 4;
        }
        d[blk] = dd;
        sum[blk] = s;
        blk += 1;
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use crate::quant_int::Q8Act;

    #[test]
    fn decode_quantization_matches_scalar_including_halfway_values() {
        let k = 512;
        let mut x: Vec<f32> = (0..k).map(|i| (i as f32 % 73.0 - 36.0) * 0.0625).collect();
        x[0] = 127.0;
        x[1] = 0.5;
        x[2] = -0.5;
        let mut act = Q8Act::new();
        act.prepare(&x, k, 1);
        for blk in 0..k / 32 {
            let start = blk * 32;
            let amax = x[start..start + 32].iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut expected_sum = 0;
            for j in start..start + 32 {
                let q = (x[j] * inv).round().clamp(-127.0, 127.0) as i32;
                assert_eq!(act.q[j], q as i8, "block {blk}, index {j}");
                expected_sum += q;
            }
            assert_eq!(act.d[blk].to_bits(), d.to_bits());
            assert_eq!(act.sum[blk], expected_sum);
        }
    }
}
