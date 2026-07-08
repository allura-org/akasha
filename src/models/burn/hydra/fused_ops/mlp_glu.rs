#![allow(unsafe_op_in_unsafe_fn)]

//! Custom fused MLP / GLU kernel for Hydra-3.5 CPU inference.
//!
//! This module implements a hand-tiled `linear -> activation -> linear` kernel
//! with pre-packed static weights. It targets the three hot feed-forward paths:
//!
//! - NaFlex MLP: `fc1 -> GELU -> fc2`
//! - Hydra pool/mid FF: `glu_proj -> softplus-gate -> proj_out`
//!
//! The kernel is intentionally isolated behind a runtime switch and is **disabled
//! by default**. If benchmarks show it is not faster than the existing
//! `best_row_major` + SIMD activation path, the fallback remains the release path.

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Output columns processed per microkernel tile.  8 matches one AVX2 vector.
const NR: usize = 8;

/// Hidden/projection dimension chunk.  Each microkernel iteration computes one
/// row's worth of `PC` hidden activations, applies the non-linearity, and
/// accumulates into the output tile.
const PC: usize = 64;

/// Input/feature dimension chunk used for packing; matches the second-level
/// tile size in the packed weight layouts.
const KC: usize = 64;

// ---------------------------------------------------------------------------
// Runtime switch
// ---------------------------------------------------------------------------

const DEFAULT_USE_CUSTOM_MLP_GLU: bool = false;

/// Return `true` if the custom fused MLP/GLU kernel should be used.
fn use_custom_mlp_glu() -> bool {
    if let Ok(v) = std::env::var("AKASHA_USE_CUSTOM_MLP_GLU") {
        return v == "1" || v.eq_ignore_ascii_case("true");
    }
    DEFAULT_USE_CUSTOM_MLP_GLU
}

// ---------------------------------------------------------------------------
// Packed weights
// ---------------------------------------------------------------------------

/// Pre-packed weights for the NaFlex MLP (`fc1 -> GELU -> fc2`).
///
/// Layout:
/// - `fc1_w`: `[h_block, k_block, KC, PC]` where the original weight is
///   `[k, hidden]` row-major.  `h_block = hidden / PC`.
/// - `fc2_w`: `[n_block, h_block, PC, NR]` where the original weight is
///   `[hidden, n]` row-major.  `n_block = n / NR`.
#[derive(Debug, Clone)]
pub struct PackedMlpWeights {
    pub fc1_w: Vec<f32>,
    pub fc1_b: Option<Vec<f32>>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Option<Vec<f32>>,
    pub k: usize,
    pub hidden: usize,
    pub n: usize,
}

/// Pre-packed weights for the Hydra feed-forward (`glu_proj -> softplus -> proj_out`).
///
/// Layout:
/// - `glu_w`: `[h_block, k_block, KC, 2*PC]` where the original weight is
///   `[k, 2*hidden]` row-major.  Each `PC` slice holds the gate, the next `PC`
///   slice holds the up projection.
/// - `proj_w`: `[n_block, h_block, PC, NR]` where the original weight is
///   `[hidden, n]` row-major.
#[derive(Debug, Clone)]
pub struct PackedGluWeights {
    pub glu_w: Vec<f32>,
    pub proj_w: Vec<f32>,
    pub proj_b: Option<Vec<f32>>,
    pub k: usize,
    pub hidden: usize,
    pub n: usize,
}

/// Pack a row-major `[k, hidden]` weight for the first MLP linear.
pub fn pack_mlp_fc1_w(fc1_w: &[f32], k: usize, hidden: usize) -> Vec<f32> {
    debug_assert_eq!(fc1_w.len(), k * hidden);
    let n_h_blocks = div_ceil(hidden, PC);
    let n_k_blocks = div_ceil(k, KC);
    let mut packed = vec![0.0f32; n_h_blocks * n_k_blocks * KC * PC];
    for h in 0..hidden {
        let h_block = h / PC;
        let h_offset = h % PC;
        for k_idx in 0..k {
            let k_block = k_idx / KC;
            let k_offset = k_idx % KC;
            let idx = ((h_block * n_k_blocks + k_block) * KC + k_offset) * PC + h_offset;
            packed[idx] = fc1_w[k_idx * hidden + h];
        }
    }
    packed
}

/// Pack a row-major `[hidden, n]` weight for the second MLP linear.
pub fn pack_mlp_fc2_w(fc2_w: &[f32], hidden: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(fc2_w.len(), hidden * n);
    let n_h_blocks = div_ceil(hidden, PC);
    let n_n_blocks = div_ceil(n, NR);
    let mut packed = vec![0.0f32; n_n_blocks * n_h_blocks * PC * NR];
    for h in 0..hidden {
        let h_block = h / PC;
        let h_offset = h % PC;
        for n_idx in 0..n {
            let n_block = n_idx / NR;
            let n_offset = n_idx % NR;
            let idx = ((n_block * n_h_blocks + h_block) * PC + h_offset) * NR + n_offset;
            packed[idx] = fc2_w[h * n + n_idx];
        }
    }
    packed
}

/// Pack a row-major `[k, 2*hidden]` GLU weight.  Gate and up are interleaved
/// per `PC` slice.
pub fn pack_glu_w(glu_w: &[f32], k: usize, hidden2: usize) -> Vec<f32> {
    debug_assert_eq!(glu_w.len(), k * hidden2);
    debug_assert_eq!(hidden2 % 2, 0);
    let hidden = hidden2 / 2;
    let n_h_blocks = div_ceil(hidden, PC);
    let n_k_blocks = div_ceil(k, KC);
    let mut packed = vec![0.0f32; n_h_blocks * n_k_blocks * KC * (2 * PC)];
    for h in 0..hidden {
        let h_block = h / PC;
        let h_offset = h % PC;
        for k_idx in 0..k {
            let k_block = k_idx / KC;
            let k_offset = k_idx % KC;
            let gate_idx =
                ((h_block * n_k_blocks + k_block) * KC + k_offset) * (2 * PC) + h_offset;
            let up_idx = gate_idx + PC;
            packed[gate_idx] = glu_w[k_idx * hidden2 + h];
            packed[up_idx] = glu_w[k_idx * hidden2 + hidden + h];
        }
    }
    packed
}

/// Pack a row-major `[hidden, n]` weight for the GLU output projection.
pub fn pack_proj_w(proj_w: &[f32], hidden: usize, n: usize) -> Vec<f32> {
    pack_mlp_fc2_w(proj_w, hidden, n)
}

#[inline]
fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ---------------------------------------------------------------------------
// Scalar fallbacks
// ---------------------------------------------------------------------------

/// Scalar fused MLP kernel.  `out` is overwritten.
pub fn fused_mlp_scalar(
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    hidden: usize,
    n: usize,
    packed: &PackedMlpWeights,
    residual: Option<&[f32]>,
) {
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(out.len(), m * n);
    debug_assert_eq!(packed.k, k);
    debug_assert_eq!(packed.hidden, hidden);
    debug_assert_eq!(packed.n, n);

    let n_h_blocks = div_ceil(hidden, PC);
    let n_k_blocks = div_ceil(k, KC);
    let n_n_blocks = div_ceil(n, NR);

    let mut hidden_tile = [0.0f32; PC];

    let mut x_row_local = vec![0.0f32; k];

    for i in 0..m {
        x_row_local.copy_from_slice(&x[i * k..(i + 1) * k]);
        let x_row = &x_row_local[..];
        let out_row = &mut out[i * n..(i + 1) * n];

        // Initialise output row with fc2 bias + residual.
        for j in 0..n {
            let mut v = 0.0f32;
            if let Some(b) = packed.fc2_b.as_deref() {
                v += b[j];
            }
            if let Some(r) = residual {
                v += r[i * n + j];
            }
            out_row[j] = v;
        }

        for h_block in 0..n_h_blocks {
            let h_start = h_block * PC;

            // Compute activated hidden tile for this h_block.
            for h_offset in 0..PC {
                let h_idx = h_start + h_offset;
                let bias = if h_idx < hidden {
                    packed.fc1_b.as_deref().map(|b| b[h_idx]).unwrap_or(0.0)
                } else {
                    0.0
                };
                let mut acc = bias;
                for k_block in 0..n_k_blocks {
                    let k_start = k_block * KC;
                    let k_end = (k_start + KC).min(k);
                    let base = ((h_block * n_k_blocks + k_block) * KC) * PC + h_offset;
                    for (kk, k_idx) in (k_start..k_end).enumerate() {
                        acc += x_row[k_idx] * packed.fc1_w[base + kk * PC];
                    }
                }
                hidden_tile[h_offset] = if h_idx < hidden {
                    gelu_approx_tanh_f32(acc)
                } else {
                    0.0
                };
            }

            // Accumulate hidden tile into all output column blocks.
            for n_block in 0..n_n_blocks {
                let n_start = n_block * NR;
                for j in 0..NR {
                    let n_idx = n_start + j;
                    if n_idx >= n {
                        continue;
                    }
                    let mut acc = out_row[n_idx];
                    let base = ((n_block * n_h_blocks + h_block) * PC) * NR + j;
                    for h_offset in 0..PC {
                        let h_idx = h_start + h_offset;
                        if h_idx >= hidden {
                            continue;
                        }
                        acc += hidden_tile[h_offset] * packed.fc2_w[base + h_offset * NR];
                    }
                    out_row[n_idx] = acc;
                }
            }
        }
    }
}

/// Scalar fused GLU kernel.  `out` is overwritten.
pub fn fused_glu_scalar(
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    hidden: usize,
    n: usize,
    packed: &PackedGluWeights,
    residual: Option<&[f32]>,
) {
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(out.len(), m * n);
    debug_assert_eq!(packed.k, k);
    debug_assert_eq!(packed.hidden, hidden);
    debug_assert_eq!(packed.n, n);

    let n_h_blocks = div_ceil(hidden, PC);
    let n_k_blocks = div_ceil(k, KC);
    let n_n_blocks = div_ceil(n, NR);

    let mut gate_tile = [0.0f32; PC];
    let mut up_tile = [0.0f32; PC];

    let mut x_row_local = vec![0.0f32; k];

    for i in 0..m {
        x_row_local.copy_from_slice(&x[i * k..(i + 1) * k]);
        let x_row = &x_row_local[..];
        let out_row = &mut out[i * n..(i + 1) * n];

        for j in 0..n {
            let mut v = 0.0f32;
            if let Some(b) = packed.proj_b.as_deref() {
                v += b[j];
            }
            if let Some(r) = residual {
                v += r[i * n + j];
            }
            out_row[j] = v;
        }

        for h_block in 0..n_h_blocks {
            let h_start = h_block * PC;

            for h_offset in 0..PC {
                let h_idx = h_start + h_offset;
                let mut gate_acc = 0.0f32;
                let mut up_acc = 0.0f32;
                for k_block in 0..n_k_blocks {
                    let k_start = k_block * KC;
                    let k_end = (k_start + KC).min(k);
                    let base = ((h_block * n_k_blocks + k_block) * KC) * (2 * PC) + h_offset;
                    for (kk, k_idx) in (k_start..k_end).enumerate() {
                        gate_acc += x_row[k_idx] * packed.glu_w[base + kk * (2 * PC)];
                        up_acc += x_row[k_idx] * packed.glu_w[base + kk * (2 * PC) + PC];
                    }
                }
                if h_idx < hidden {
                    gate_tile[h_offset] = softplus_f32(gate_acc);
                    up_tile[h_offset] = up_acc;
                } else {
                    gate_tile[h_offset] = 0.0;
                    up_tile[h_offset] = 0.0;
                }
            }

            for n_block in 0..n_n_blocks {
                let n_start = n_block * NR;
                for j in 0..NR {
                    let n_idx = n_start + j;
                    if n_idx >= n {
                        continue;
                    }
                    let mut acc = out_row[n_idx];
                    let base = ((n_block * n_h_blocks + h_block) * PC) * NR + j;
                    for h_offset in 0..PC {
                        let h_idx = h_start + h_offset;
                        if h_idx >= hidden {
                            continue;
                        }
                        acc += gate_tile[h_offset] * up_tile[h_offset]
                            * packed.proj_w[base + h_offset * NR];
                    }
                    out_row[n_idx] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2/FMA path
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn has_avx2_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
unsafe fn has_avx2_fma() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_mlp_avx2(
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    hidden: usize,
    n: usize,
    packed: &PackedMlpWeights,
    residual: Option<&[f32]>,
) {
    use std::arch::x86_64::*;

    let n_h_blocks = div_ceil(hidden, PC);
    let n_k_blocks = div_ceil(k, KC);
    let n_n_blocks = div_ceil(n, NR);

    let mut hidden_tile = [0.0f32; PC];

    let mut x_row_local = vec![0.0f32; k];

    for i in 0..m {
        x_row_local.copy_from_slice(&x[i * k..(i + 1) * k]);
        let x_row = &x_row_local[..];
        let out_row = &mut out[i * n..(i + 1) * n];

        // Initialise output row with bias + residual.
        for j in 0..n {
            let mut v = 0.0f32;
            if let Some(b) = packed.fc2_b.as_deref() {
                v += b[j];
            }
            if let Some(r) = residual {
                v += r[i * n + j];
            }
            out_row[j] = v;
        }

        for h_block in 0..n_h_blocks {
            let h_start = h_block * PC;

            // Compute activated hidden tile.
            for h_offset in 0..PC {
                let h_idx = h_start + h_offset;
                let bias = if h_idx < hidden {
                    packed.fc1_b.as_deref().map(|b| b[h_idx]).unwrap_or(0.0)
                } else {
                    0.0
                };

                let mut total = bias;
                for k_block in 0..n_k_blocks {
                    let k_start = k_block * KC;
                    let k_end = (k_start + KC).min(k);
                    let base = ((h_block * n_k_blocks + k_block) * KC) * PC + h_offset;
                    for k_idx in k_start..k_end {
                        total += x_row[k_idx] * packed.fc1_w[base + (k_idx - k_start) * PC];
                    }
                }

                hidden_tile[h_offset] = if h_idx < hidden {
                    gelu_approx_tanh_f32(total)
                } else {
                    0.0
                };
            }

            // Accumulate into output tiles.
            for n_block in 0..n_n_blocks {
                let n_start = n_block * NR;
                if n_start + NR <= n {
                    let mut acc_v = _mm256_loadu_ps(out_row.as_ptr().add(n_start));
                    let base = ((n_block * n_h_blocks + h_block) * PC) * NR;
                    for h_offset in 0..PC {
                        let h_idx = h_start + h_offset;
                        if h_idx >= hidden {
                            continue;
                        }
                        let h_v = _mm256_set1_ps(hidden_tile[h_offset]);
                        let w_v = _mm256_loadu_ps(packed.fc2_w.as_ptr().add(base + h_offset * NR));
                        acc_v = _mm256_fmadd_ps(h_v, w_v, acc_v);
                    }
                    _mm256_storeu_ps(out_row.as_mut_ptr().add(n_start), acc_v);
                } else {
                    for j in 0..NR {
                        let n_idx = n_start + j;
                        if n_idx >= n {
                            continue;
                        }
                        let mut acc = out_row[n_idx];
                        let base = ((n_block * n_h_blocks + h_block) * PC) * NR + j;
                        for h_offset in 0..PC {
                            let h_idx = h_start + h_offset;
                            if h_idx >= hidden {
                                continue;
                            }
                            acc += hidden_tile[h_offset] * packed.fc2_w[base + h_offset * NR];
                        }
                        out_row[n_idx] = acc;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_glu_avx2(
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    hidden: usize,
    n: usize,
    packed: &PackedGluWeights,
    residual: Option<&[f32]>,
) {
    use std::arch::x86_64::*;

    let n_h_blocks = div_ceil(hidden, PC);
    let n_k_blocks = div_ceil(k, KC);
    let n_n_blocks = div_ceil(n, NR);

    let mut gate_tile = [0.0f32; PC];
    let mut up_tile = [0.0f32; PC];

    let mut x_row_local = vec![0.0f32; k];

    for i in 0..m {
        x_row_local.copy_from_slice(&x[i * k..(i + 1) * k]);
        let x_row = &x_row_local[..];
        let out_row = &mut out[i * n..(i + 1) * n];

        for j in 0..n {
            let mut v = 0.0f32;
            if let Some(b) = packed.proj_b.as_deref() {
                v += b[j];
            }
            if let Some(r) = residual {
                v += r[i * n + j];
            }
            out_row[j] = v;
        }

        for h_block in 0..n_h_blocks {
            let h_start = h_block * PC;

            for h_offset in 0..PC {
                let h_idx = h_start + h_offset;
                let mut gate_total = 0.0f32;
                let mut up_total = 0.0f32;

                for k_block in 0..n_k_blocks {
                    let k_start = k_block * KC;
                    let k_end = (k_start + KC).min(k);
                    let base = ((h_block * n_k_blocks + k_block) * KC) * (2 * PC) + h_offset;
                    for k_idx in k_start..k_end {
                        let off = base + (k_idx - k_start) * (2 * PC);
                        gate_total += x_row[k_idx] * packed.glu_w[off];
                        up_total += x_row[k_idx] * packed.glu_w[off + PC];
                    }
                }

                if h_idx < hidden {
                    gate_tile[h_offset] = softplus_f32(gate_total);
                    up_tile[h_offset] = up_total;
                } else {
                    gate_tile[h_offset] = 0.0;
                    up_tile[h_offset] = 0.0;
                }
            }

            for n_block in 0..n_n_blocks {
                let n_start = n_block * NR;
                if n_start + NR <= n {
                    let mut acc_v = _mm256_loadu_ps(out_row.as_ptr().add(n_start));
                    let base = ((n_block * n_h_blocks + h_block) * PC) * NR;
                    for h_offset in 0..PC {
                        let h_idx = h_start + h_offset;
                        if h_idx >= hidden {
                            continue;
                        }
                        let a_v = _mm256_set1_ps(gate_tile[h_offset] * up_tile[h_offset]);
                        let w_v = _mm256_loadu_ps(packed.proj_w.as_ptr().add(base + h_offset * NR));
                        acc_v = _mm256_fmadd_ps(a_v, w_v, acc_v);
                    }
                    _mm256_storeu_ps(out_row.as_mut_ptr().add(n_start), acc_v);
                } else {
                    for j in 0..NR {
                        let n_idx = n_start + j;
                        if n_idx >= n {
                            continue;
                        }
                        let mut acc = out_row[n_idx];
                        let base = ((n_block * n_h_blocks + h_block) * PC) * NR + j;
                        for h_offset in 0..PC {
                            let h_idx = h_start + h_offset;
                            if h_idx >= hidden {
                                continue;
                            }
                            acc += gate_tile[h_offset]
                                * up_tile[h_offset]
                                * packed.proj_w[base + h_offset * NR];
                        }
                        out_row[n_idx] = acc;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch
// ---------------------------------------------------------------------------

/// Fused MLP: `out = fc2(gelu(fc1(x))) + fc2_bias + residual?`.
///
/// If `use_custom_mlp_glu()` is false, returns without doing anything so callers
/// can fall back to the existing path.
pub fn fused_mlp_custom(
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    hidden: usize,
    n: usize,
    packed: &PackedMlpWeights,
    residual: Option<&[f32]>,
) -> bool {
    if !use_custom_mlp_glu() {
        return false;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        if has_avx2_fma() {
            fused_mlp_avx2(x, out, m, k, hidden, n, packed, residual);
            return true;
        }
    }

    fused_mlp_scalar(x, out, m, k, hidden, n, packed, residual);
    true
}

/// Fused GLU + projection: `out = proj(softplus(gate) * up) + proj_bias + residual?`.
pub fn fused_glu_custom(
    x: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    hidden: usize,
    n: usize,
    packed: &PackedGluWeights,
    residual: Option<&[f32]>,
) -> bool {
    if !use_custom_mlp_glu() {
        return false;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        if has_avx2_fma() {
            fused_glu_avx2(x, out, m, k, hidden, n, packed, residual);
            return true;
        }
    }

    fused_glu_scalar(x, out, m, k, hidden, n, packed, residual);
    true
}

// ---------------------------------------------------------------------------
// Helpers (duplicated here to avoid making this module depend on private fns)
// ---------------------------------------------------------------------------

#[inline]
fn gelu_approx_tanh_f32(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608028654f32;
    let coeff = 0.044715f32;
    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (x + coeff * x3);
    0.5f32 * x * (1.0f32 + inner.tanh())
}

#[inline]
fn softplus_f32(x: f32) -> f32 {
    x.exp().ln_1p()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        let mut max_diff = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(d < eps, "{x} vs {y} (max diff {max_diff}, eps {eps})");
        }
    }

    fn naive_mlp(
        x: &[f32],
        fc1_w: &[f32],
        fc1_b: Option<&[f32]>,
        fc2_w: &[f32],
        fc2_b: Option<&[f32]>,
        residual: Option<&[f32]>,
        m: usize,
        k: usize,
        hidden: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut tmp = vec![0.0f32; m * hidden];
        for i in 0..m {
            for h in 0..hidden {
                let mut acc = fc1_b.map(|b| b[h]).unwrap_or(0.0);
                for kk in 0..k {
                    acc += x[i * k + kk] * fc1_w[kk * hidden + h];
                }
                tmp[i * hidden + h] = gelu_approx_tanh_f32(acc);
            }
        }
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = fc2_b.map(|b| b[j]).unwrap_or(0.0)
                    + residual.map(|r| r[i * n + j]).unwrap_or(0.0);
                for h in 0..hidden {
                    acc += tmp[i * hidden + h] * fc2_w[h * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn naive_glu(
        x: &[f32],
        glu_w: &[f32],
        proj_w: &[f32],
        proj_b: Option<&[f32]>,
        residual: Option<&[f32]>,
        m: usize,
        k: usize,
        hidden: usize,
        n: usize,
    ) -> Vec<f32> {
        let hidden2 = hidden * 2;
        let mut tmp = vec![0.0f32; m * hidden2];
        for i in 0..m {
            for h in 0..hidden2 {
                let mut acc = 0.0;
                for kk in 0..k {
                    acc += x[i * k + kk] * glu_w[kk * hidden2 + h];
                }
                tmp[i * hidden2 + h] = acc;
            }
        }
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = proj_b.map(|b| b[j]).unwrap_or(0.0)
                    + residual.map(|r| r[i * n + j]).unwrap_or(0.0);
                for h in 0..hidden {
                    let gate = softplus_f32(tmp[i * hidden2 + h]);
                    let up = tmp[i * hidden2 + hidden + h];
                    acc += gate * up * proj_w[h * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn run_mlp_test(
        m: usize,
        k: usize,
        hidden: usize,
        n: usize,
        with_bias: bool,
        with_residual: bool,
        run_scalar: bool,
    ) {
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let fc1_w: Vec<f32> = (0..k * hidden).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let fc1_b: Vec<f32> = (0..hidden).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();
        let fc2_w: Vec<f32> = (0..hidden * n).map(|i| ((i % 11) as f32 - 5.0) * 0.04).collect();
        let fc2_b: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.03).collect();
        let residual: Vec<f32> = (0..m * n).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect();

        let packed = PackedMlpWeights {
            fc1_w: pack_mlp_fc1_w(&fc1_w, k, hidden),
            fc1_b: with_bias.then(|| fc1_b.clone()),
            fc2_w: pack_mlp_fc2_w(&fc2_w, hidden, n),
            fc2_b: with_bias.then(|| fc2_b.clone()),
            k,
            hidden,
            n,
        };

        let expected = naive_mlp(
            &x,
            &fc1_w,
            with_bias.then_some(&fc1_b[..]),
            &fc2_w,
            with_bias.then_some(&fc2_b[..]),
            with_residual.then_some(&residual[..]),
            m,
            k,
            hidden,
            n,
        );

        let mut out = vec![0.0f32; m * n];
        if run_scalar {
            fused_mlp_scalar(
                &x,
                &mut out,
                m,
                k,
                hidden,
                n,
                &packed,
                with_residual.then_some(&residual[..]),
            );
            approx_eq(&out, &expected, 1e-3);
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            if has_avx2_fma() {
                out.fill(0.0);
                fused_mlp_avx2(
                    &x,
                    &mut out,
                    m,
                    k,
                    hidden,
                    n,
                    &packed,
                    with_residual.then_some(&residual[..]),
                );
                approx_eq(&out, &expected, 1e-3);
            }
        }
    }

    fn run_glu_test(
        m: usize,
        k: usize,
        hidden: usize,
        n: usize,
        with_bias: bool,
        with_residual: bool,
        run_scalar: bool,
    ) {
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let glu_w: Vec<f32> =
            (0..k * hidden * 2).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let proj_w: Vec<f32> =
            (0..hidden * n).map(|i| ((i % 11) as f32 - 5.0) * 0.04).collect();
        let proj_b: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.03).collect();
        let residual: Vec<f32> = (0..m * n).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect();

        let packed = PackedGluWeights {
            glu_w: pack_glu_w(&glu_w, k, hidden * 2),
            proj_w: pack_proj_w(&proj_w, hidden, n),
            proj_b: with_bias.then(|| proj_b.clone()),
            k,
            hidden,
            n,
        };

        let expected = naive_glu(
            &x,
            &glu_w,
            &proj_w,
            with_bias.then_some(&proj_b[..]),
            with_residual.then_some(&residual[..]),
            m,
            k,
            hidden,
            n,
        );

        let mut out = vec![0.0f32; m * n];
        if run_scalar {
            fused_glu_scalar(
                &x,
                &mut out,
                m,
                k,
                hidden,
                n,
                &packed,
                with_residual.then_some(&residual[..]),
            );
            approx_eq(&out, &expected, 1e-3);
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            if has_avx2_fma() {
                out.fill(0.0);
                fused_glu_avx2(
                    &x,
                    &mut out,
                    m,
                    k,
                    hidden,
                    n,
                    &packed,
                    with_residual.then_some(&residual[..]),
                );
                approx_eq(&out, &expected, 1e-3);
            }
        }
    }

    // Correctness on the exact weight shapes, with a small batch so the unit
    // test finishes quickly.  Full-batch benchmarks are exercised via
    // `hydra_burn_runs` and the ignored tests below.
    #[test]
    fn mlp_naflex_shape() {
        run_mlp_test(8, 2048, 8192, 2048, true, true, false);
    }

    #[test]
    fn mlp_naflex_no_bias_no_residual() {
        run_mlp_test(8, 2048, 8192, 2048, false, false, false);
    }

    #[test]
    fn glu_pool_shape() {
        run_glu_test(8, 2048, 5120, 2048, true, true, false);
    }

    #[test]
    fn glu_mid_shape() {
        run_glu_test(8, 2048, 5120, 2048, true, true, false);
    }

    #[test]
    fn glu_no_bias_no_residual() {
        run_glu_test(8, 2048, 5120, 2048, false, false, false);
    }

    #[test]
    fn mlp_tiny() {
        run_mlp_test(3, 5, 7, 4, true, true, true);
    }

    #[test]
    fn glu_tiny() {
        run_glu_test(3, 5, 7, 4, true, true, true);
    }

    // Full-size benchmarks for the custom kernel.  These are ignored by default
    // because the scalar fallback would be far too slow; run manually with
    // `--ignored --nocapture`.
    #[test]
    #[ignore = "manual benchmark"]
    fn bench_mlp_naflex_full() {
        run_mlp_test(1024, 2048, 8192, 2048, true, true, false);
    }

    #[test]
    #[ignore = "manual benchmark"]
    fn bench_glu_pool_full() {
        run_glu_test(8886, 2048, 5120, 2048, true, true, false);
    }

    #[test]
    #[ignore = "manual benchmark"]
    fn bench_glu_mid_full() {
        run_glu_test(1024, 2048, 5120, 2048, true, true, false);
    }
}
