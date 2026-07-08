//! Portable SIMD elementwise kernels for Burn-backed models.
//!
//! These routines operate on contiguous `f32` slices, use the `wide` crate for
//! portable 8-lane vector math, and fall back to scalar loops for tails. The
//! `wide::f32x8` type maps to AVX/AVX2 when the binary is compiled with those
//! target features and to two SSE2 `f32x4`s otherwise, so the same code is
//! portable across x86_64 builds.
//!
//! These kernels are intentionally model-agnostic: any transformer-style model
//! running on a Burn CPU backend can use softmax, layer/RMS norm, GELU, GLU,
//! and bias/residual helpers from here.

use wide::f32x8;

const LANES: usize = 8;

#[inline]
unsafe fn load_f32x8(ptr: *const f32) -> f32x8 {
    f32x8::new(unsafe { std::ptr::read_unaligned(ptr as *const [f32; LANES]) })
}

#[inline]
unsafe fn store_f32x8(ptr: *mut f32, v: f32x8) {
    unsafe { std::ptr::write_unaligned(ptr as *mut [f32; LANES], v.to_array()) };
}

/// In-place softmax over a single contiguous row.
pub fn softmax_in_place(x: &mut [f32]) {
    let n = x.len();
    if n < 2 * LANES {
        scalar_softmax(x);
        return;
    }

    let mut max = f32::NEG_INFINITY;
    let mut i = 0;
    while i + LANES <= n {
        let v = unsafe { load_f32x8(x.as_ptr().add(i)) };
        let arr = v.to_array();
        for k in 0..LANES {
            if arr[k] > max {
                max = arr[k];
            }
        }
        i += LANES;
    }
    for j in i..n {
        if x[j] > max {
            max = x[j];
        }
    }

    let max_v = f32x8::splat(max);
    let mut sum = 0.0f32;
    i = 0;
    while i + LANES <= n {
        let v = unsafe { load_f32x8(x.as_ptr().add(i)) };
        let e = (v - max_v).exp();
        sum += e.to_array().iter().sum::<f32>();
        unsafe { store_f32x8(x.as_mut_ptr().add(i), e) };
        i += LANES;
    }
    for j in i..n {
        let e = (x[j] - max).exp();
        x[j] = e;
        sum += e;
    }

    let inv_sum = 1.0f32 / sum;
    let inv_v = f32x8::splat(inv_sum);
    i = 0;
    while i + LANES <= n {
        let v = unsafe { load_f32x8(x.as_ptr().add(i)) };
        unsafe { store_f32x8(x.as_mut_ptr().add(i), v * inv_v) };
        i += LANES;
    }
    for j in i..n {
        x[j] *= inv_sum;
    }
}

fn scalar_softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = x[0];
    for &v in x.iter().skip(1) {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    let inv = 1.0f32 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Apply layer normalization to a single row: `out = (x - mean) / sqrt(var+eps) * gamma + beta`.
pub fn layer_norm_row(
    x: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    eps: f32,
    out: &mut [f32],
) {
    let n = x.len();
    debug_assert_eq!(gamma.len(), n);
    debug_assert_eq!(out.len(), n);
    if let Some(b) = beta {
        debug_assert_eq!(b.len(), n);
    }

    if n < 2 * LANES {
        scalar_layer_norm_row(x, gamma, beta, eps, out);
        return;
    }

    let mut sum = 0.0f32;
    let mut i = 0;
    while i + LANES <= n {
        let v = unsafe { load_f32x8(x.as_ptr().add(i)) };
        sum += v.to_array().iter().sum::<f32>();
        i += LANES;
    }
    for j in i..n {
        sum += x[j];
    }
    let mean = sum / n as f32;

    let mean_v = f32x8::splat(mean);
    let mut var_sum = 0.0f32;
    i = 0;
    while i + LANES <= n {
        let v = unsafe { load_f32x8(x.as_ptr().add(i)) };
        let d = v - mean_v;
        let d2 = d * d;
        var_sum += d2.to_array().iter().sum::<f32>();
        i += LANES;
    }
    for j in i..n {
        let d = x[j] - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0f32 / ((var_sum / n as f32) + eps).sqrt();

    let inv_std_v = f32x8::splat(inv_std);
    i = 0;
    while i + LANES <= n {
        let xv = unsafe { load_f32x8(x.as_ptr().add(i)) };
        let gv = unsafe { load_f32x8(gamma.as_ptr().add(i)) };
        let mut y = (xv - mean_v) * inv_std_v * gv;
        if let Some(b) = beta {
            let bv = unsafe { load_f32x8(b.as_ptr().add(i)) };
            y = y + bv;
        }
        unsafe { store_f32x8(out.as_mut_ptr().add(i), y) };
        i += LANES;
    }
    for j in i..n {
        out[j] = (x[j] - mean) * inv_std * gamma[j];
        if let Some(b) = beta {
            out[j] += b[j];
        }
    }
}

fn scalar_layer_norm_row(
    x: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    eps: f32,
    out: &mut [f32],
) {
    let n = x.len();
    let mean = x.iter().copied().sum::<f32>() / n as f32;
    let var = x
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / n as f32;
    let inv_std = 1.0f32 / (var + eps).sqrt();
    for j in 0..n {
        out[j] = (x[j] - mean) * inv_std * gamma[j];
        if let Some(b) = beta {
            out[j] += b[j];
        }
    }
}

/// Apply RMS normalization to a single row: `out = x / sqrt(mean(x^2)+eps) * gamma`.
#[allow(dead_code)]
pub fn rms_norm_row(x: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(gamma.len(), n);
    debug_assert_eq!(out.len(), n);

    if n < 2 * LANES {
        scalar_rms_norm_row(x, gamma, eps, out);
        return;
    }

    let mut sumsq = 0.0f32;
    let mut i = 0;
    while i + LANES <= n {
        let v = unsafe { load_f32x8(x.as_ptr().add(i)) };
        let sq = v * v;
        sumsq += sq.to_array().iter().sum::<f32>();
        i += LANES;
    }
    for j in i..n {
        sumsq += x[j] * x[j];
    }
    let scale = 1.0f32 / ((sumsq / n as f32) + eps).sqrt();
    let scale_v = f32x8::splat(scale);

    i = 0;
    while i + LANES <= n {
        let xv = unsafe { load_f32x8(x.as_ptr().add(i)) };
        let gv = unsafe { load_f32x8(gamma.as_ptr().add(i)) };
        let y = xv * scale_v * gv;
        unsafe { store_f32x8(out.as_mut_ptr().add(i), y) };
        i += LANES;
    }
    for j in i..n {
        out[j] = x[j] * scale * gamma[j];
    }
}

#[allow(dead_code)]
fn scalar_rms_norm_row(x: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let sumsq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0f32 / (sumsq + eps).sqrt();
    for j in 0..n {
        out[j] = x[j] * scale * gamma[j];
    }
}

/// In-place GELU tanh approximation: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu_approx_tanh_in_place(x: &mut [f32]) {
    let n = x.len();
    if n < 2 * LANES {
        for v in x.iter_mut() {
            *v = scalar_gelu_approx_tanh_f32(*v);
        }
        return;
    }

    let sqrt_2_over_pi = f32x8::splat((2.0f32 / std::f32::consts::PI).sqrt());
    let c = f32x8::splat(0.044715f32);
    let half = f32x8::splat(0.5f32);
    let one = f32x8::splat(1.0f32);
    let clamp_lo = f32x8::splat(-9.0f32);
    let clamp_hi = f32x8::splat(9.0f32);

    let mut i = 0;
    while i + LANES <= n {
        let v = unsafe { load_f32x8(x.as_ptr().add(i)) };
        let v2 = v * v;
        let v3 = v2 * v;
        let arg = sqrt_2_over_pi * (v + c * v3);
        let clamped = arg.max(clamp_lo).min(clamp_hi);
        let t = (-(clamped + clamped)).exp();
        let tanh_val = (one - t) / (one + t);
        let y = half * v * (one + tanh_val);
        unsafe { store_f32x8(x.as_mut_ptr().add(i), y) };
        i += LANES;
    }
    for j in i..n {
        x[j] = scalar_gelu_approx_tanh_f32(x[j]);
    }
}

fn scalar_gelu_approx_tanh_f32(x: f32) -> f32 {
    let c = 0.044715f32;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let x3 = x * x * x;
    let tanh_arg = sqrt_2_over_pi * (x + c * x3);
    let tanh_val = tanh_arg.tanh();
    0.5f32 * x * (1.0f32 + tanh_val)
}

/// GLU softplus activation: `out = softplus(gate) * up`.
#[allow(dead_code)]
pub fn glu_softplus_in_place(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let n = gate.len();
    debug_assert_eq!(up.len(), n);
    debug_assert_eq!(out.len(), n);

    if n < 2 * LANES {
        for j in 0..n {
            out[j] = scalar_softplus_f32(gate[j]) * up[j];
        }
        return;
    }

    let zero = f32x8::splat(0.0f32);
    let one = f32x8::splat(1.0f32);

    let mut i = 0;
    while i + LANES <= n {
        let g = unsafe { load_f32x8(gate.as_ptr().add(i)) };
        let u = unsafe { load_f32x8(up.as_ptr().add(i)) };
        let abs_g = g.abs();
        let z = (-abs_g).exp();
        let sp = g.max(zero) + (one + z).ln();
        unsafe { store_f32x8(out.as_mut_ptr().add(i), sp * u) };
        i += LANES;
    }
    for j in i..n {
        out[j] = scalar_softplus_f32(gate[j]) * up[j];
    }
}

/// In-place GLU softplus activation on an interleaved `[gate..., up...]` buffer.
/// Overwrites the first `glu_out_dim` elements with `softplus(gate) * up`.
pub fn glu_softplus_in_place_interleaved(buf: &mut [f32], glu_out_dim: usize) {
    debug_assert_eq!(buf.len(), 2 * glu_out_dim);

    if glu_out_dim < 2 * LANES {
        for j in 0..glu_out_dim {
            buf[j] = scalar_softplus_f32(buf[j]) * buf[glu_out_dim + j];
        }
        return;
    }

    let zero = f32x8::splat(0.0f32);
    let one = f32x8::splat(1.0f32);

    let mut i = 0;
    while i + LANES <= glu_out_dim {
        let g = unsafe { load_f32x8(buf.as_ptr().add(i)) };
        let u = unsafe { load_f32x8(buf.as_ptr().add(glu_out_dim + i)) };
        let abs_g = g.abs();
        let z = (-abs_g).exp();
        let sp = g.max(zero) + (one + z).ln();
        unsafe { store_f32x8(buf.as_mut_ptr().add(i), sp * u) };
        i += LANES;
    }
    for j in i..glu_out_dim {
        buf[j] = scalar_softplus_f32(buf[j]) * buf[glu_out_dim + j];
    }
}

fn scalar_softplus_f32(x: f32) -> f32 {
    x.exp().ln_1p()
}

/// `out = a + b` (fused two-input add)
pub fn add2_in_place(out: &mut [f32], a: &[f32], b: &[f32]) {
    let n = out.len();
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(b.len(), n);
    let mut i = 0;
    while i + LANES <= n {
        let av = unsafe { load_f32x8(a.as_ptr().add(i)) };
        let bv = unsafe { load_f32x8(b.as_ptr().add(i)) };
        unsafe { store_f32x8(out.as_mut_ptr().add(i), av + bv) };
        i += LANES;
    }
    for j in i..n {
        out[j] = a[j] + b[j];
    }
}

/// `out = out + bias + residual`
#[allow(dead_code)]
pub fn add_bias_and_residual_in_place(out: &mut [f32], bias: &[f32], residual: &[f32]) {
    let n = out.len();
    debug_assert_eq!(bias.len(), n);
    debug_assert_eq!(residual.len(), n);
    let mut i = 0;
    while i + LANES <= n {
        let o = unsafe { load_f32x8(out.as_ptr().add(i)) };
        let b = unsafe { load_f32x8(bias.as_ptr().add(i)) };
        let r = unsafe { load_f32x8(residual.as_ptr().add(i)) };
        unsafe { store_f32x8(out.as_mut_ptr().add(i), o + b + r) };
        i += LANES;
    }
    for j in i..n {
        out[j] = out[j] + bias[j] + residual[j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < eps, "{} vs {} (eps {})", x, y, eps);
        }
    }

    #[test]
    fn softmax_matches_scalar() {
        let mut a = vec![0.1f32, 0.5, -0.2, 1.0, -1.0, 0.0];
        let mut b = a.clone();
        softmax_in_place(&mut a);
        scalar_softmax(&mut b);
        approx_eq(&a, &b, 1e-5);
    }

    #[test]
    fn layer_norm_matches_scalar() {
        let x = vec![0.1f32, 0.5, -0.2, 1.0, -1.0, 0.0, 0.3, -0.4];
        let gamma = vec![1.0f32; 8];
        let beta = vec![0.1f32; 8];
        let mut out = vec![0.0f32; 8];
        let mut out_scalar = vec![0.0f32; 8];
        layer_norm_row(&x, &gamma, Some(&beta), 1e-5, &mut out);
        scalar_layer_norm_row(&x, &gamma, Some(&beta), 1e-5, &mut out_scalar);
        approx_eq(&out, &out_scalar, 1e-4);
    }

    #[test]
    fn rms_norm_matches_scalar() {
        let x = vec![0.1f32, 0.5, -0.2, 1.0, -1.0, 0.0, 0.3, -0.4];
        let gamma = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 8];
        let mut out_scalar = vec![0.0f32; 8];
        rms_norm_row(&x, &gamma, 1e-5, &mut out);
        scalar_rms_norm_row(&x, &gamma, 1e-5, &mut out_scalar);
        approx_eq(&out, &out_scalar, 1e-4);
    }

    #[test]
    fn gelu_matches_scalar() {
        let mut a = vec![0.1f32, 0.5, -0.2, 1.0, -1.0, 0.0, 0.3, -0.4];
        let mut b = a.clone();
        gelu_approx_tanh_in_place(&mut a);
        for v in b.iter_mut() {
            *v = scalar_gelu_approx_tanh_f32(*v);
        }
        approx_eq(&a, &b, 1e-4);
    }

    #[test]
    fn glu_softplus_matches_scalar() {
        let gate = vec![0.1f32, 0.5, -0.2, 1.0, -1.0, 0.0, 0.3, -0.4];
        let up = vec![0.2f32, -0.1, 0.4, 0.5, -0.3, 0.6, -0.5, 0.7];
        let mut out = vec![0.0f32; 8];
        glu_softplus_in_place(&gate, &up, &mut out);
        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| scalar_softplus_f32(*g) * u)
            .collect();
        approx_eq(&out, &expected, 1e-4);
    }

    #[test]
    fn add2_in_place_is_correct() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut out = vec![0.0f32; 8];
        add2_in_place(&mut out, &a, &b);
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        approx_eq(&out, &expected, 1e-5);
    }

    #[test]
    fn add_bias_and_residual_is_correct() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bias = vec![0.1f32; 8];
        let residual = vec![0.5f32; 8];
        let mut out = original.clone();
        add_bias_and_residual_in_place(&mut out, &bias, &residual);
        let expected: Vec<f32> = original
            .iter()
            .zip(bias.iter().zip(residual.iter()))
            .map(|(o, (b, r))| o + b + r)
            .collect();
        approx_eq(&out, &expected, 1e-5);
    }
}
