//! aarch64 SIMD helpers for the hot loops. Pure `core::arch::aarch64`, no crates.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn prefetch_read(p: *const u8, ahead: usize) {
    let addr = p.add(ahead);
    core::arch::asm!(
        "prfm pldl1keep, [{p}]",
        p = in(reg) addr,
        options(nostack, preserves_flags)
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn f32_dot(a: *const f32, b: *const f32, n: usize) -> f32 {
    // Avoid fused multiply-add: accumulation order changes can flip near-tied
    // greedy tokens. Numerical agreement is tested with a tolerance, not exact
    // bit-for-bit identity across all models and compilers.
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let a0 = vld1q_f32(a.add(i));
        let b0 = vld1q_f32(b.add(i));
        let a1 = vld1q_f32(a.add(i + 4));
        let b1 = vld1q_f32(b.add(i + 4));
        acc0 = vaddq_f32(acc0, vmulq_f32(a0, b0));
        acc1 = vaddq_f32(acc1, vmulq_f32(a1, b1));
        i += 8;
    }
    // Fold the vector accumulators, then handle the scalar tail.
    let mut s = vaddvq_f32(acc0);
    s += vaddvq_f32(acc1);
    while i < n {
        s += *a.add(i) * *b.add(i);
        i += 1;
    }
    s
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn f32_sum_sq(x: *const f32, n: usize) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let a0 = vld1q_f32(x.add(i));
        let a1 = vld1q_f32(x.add(i + 4));
        acc0 = vfmaq_f32(acc0, a0, a0);
        acc1 = vfmaq_f32(acc1, a1, a1);
        i += 8;
    }
    let mut s = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        s += *x.add(i) * *x.add(i);
        i += 1;
    }
    s
}

/// out[i] += scale * a[i]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn f32_axpy(out: *mut f32, a: *const f32, n: usize, scale: f32) {
    // mul+add rather than FMA: keeps the rounding order identical to the
    // scalar kernel so greedy near-ties resolve the same way as llama.cpp.
    let sv = vdupq_n_f32(scale);
    let mut i = 0usize;
    while i + 8 <= n {
        vst1q_f32(
            out.add(i),
            vaddq_f32(vld1q_f32(out.add(i)), vmulq_f32(sv, vld1q_f32(a.add(i)))),
        );
        vst1q_f32(
            out.add(i + 4),
            vaddq_f32(
                vld1q_f32(out.add(i + 4)),
                vmulq_f32(sv, vld1q_f32(a.add(i + 4))),
            ),
        );
        i += 8;
    }
    while i < n {
        *out.add(i) += scale * *a.add(i);
        i += 1;
    }
}

/// out[i] = out[i] * s * g[i]   (rmsnorm apply)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn f32_rms_apply(x: *const f32, w: *const f32, out: *mut f32, n: usize, s: f32) {
    let sv = vdupq_n_f32(s);
    let mut i = 0usize;
    while i + 4 <= n {
        let xv = vld1q_f32(x.add(i));
        let wv = vld1q_f32(w.add(i));
        vst1q_f32(out.add(i), vmulq_f32(vmulq_f32(xv, sv), wv));
        i += 4;
    }
    while i < n {
        *out.add(i) = *x.add(i) * s * *w.add(i);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn f32_dot_matches_scalar() {
        let n = 128;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.3).collect();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        #[cfg(target_arch = "aarch64")]
        let got = unsafe { super::f32_dot(a.as_ptr(), b.as_ptr(), n) };
        #[cfg(not(target_arch = "aarch64"))]
        let got = want;
        assert!((got - want).abs() <= want.abs() * 1e-4 + 1e-4, "got {got} want {want}");
    }

    #[test]
    fn f32_axpy_matches_scalar() {
        let n = 128;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let mut out = vec![1.0f32; n];
        #[cfg(target_arch = "aarch64")]
        unsafe { super::f32_axpy(out.as_mut_ptr(), a.as_ptr(), n, 2.0); }
        #[cfg(not(target_arch = "aarch64"))]
        for i in 0..n { out[i] += 2.0 * a[i]; }
        for i in 0..n {
            let want = 1.0 + 2.0 * a[i];
            assert!((out[i] - want).abs() < 1e-5, "i={i} got {} want {want}", out[i]);
        }
    }
}
