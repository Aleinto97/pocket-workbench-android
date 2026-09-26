pub struct Rng {
    s: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { s: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut z = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        self.s = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

pub fn sample(logits: &mut [f32], temp: f32, top_p: f32, rng: &mut Rng) -> u32 {
    let n = logits.len();
    if n == 0 {
        return 0;
    }
    if temp <= 0.0 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, v) in logits.iter().enumerate() {
            if *v > bv {
                bv = *v;
                best = i;
            }
        }
        return best as u32;
    }
    let inv = 1.0 / temp;
    let mut max = f32::NEG_INFINITY;
    for v in logits.iter() {
        if *v > max {
            max = *v;
        }
    }
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        let e = ((*v - max) * inv).exp();
        *v = e;
        sum += e;
    }
    if !(sum > 0.0) {
        return 0;
    }
    if top_p > 0.0 && top_p < 1.0 {
        let mut idx: Vec<u32> = (0..n as u32).collect();
        idx.sort_unstable_by(|a, b| logits[*b as usize].partial_cmp(&logits[*a as usize]).unwrap());
        let mut acc = 0.0f32;
        let mut cut = n;
        for (k, i) in idx.iter().enumerate() {
            acc += logits[*i as usize] / sum;
            if acc >= top_p {
                cut = k + 1;
                break;
            }
        }
        let mut keep = vec![false; n];
        for i in idx.iter().take(cut) {
            keep[*i as usize] = true;
        }
        let mut new_sum = 0.0f32;
        for i in 0..n {
            if !keep[i] {
                logits[i] = 0.0;
            } else {
                new_sum += logits[i];
            }
        }
        sum = new_sum;
    }
    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for (i, v) in logits.iter().enumerate() {
        acc += *v;
        if acc >= r {
            return i as u32;
        }
    }
    (n - 1) as u32
}
