//! Online-softmax fused attention kernel.
//!
//! Replaces the two-GEMM pattern (`Q @ K^T` → softmax → `scores @ V`) with a
//! single tiled kernel that streams key/value chunks and updates the softmax
//! normalizer on the fly. This avoids materializing the full `[seq_q, seq_kv]`
//! `scores` matrix and eliminates the LHS packing cost of the second GEMM.
//!
//! The kernel is intentionally layout-agnostic: it accepts arbitrary row
//! strides for Q, K, V, and the output so it can be dropped into the packed
//! NaFlex QKV buffer, the strided HydraPool K/V buffer, and the contiguous
//! HydraMid per-head buffer without copies.

use super::gemm_f32_ex;

/// Default switch for the online-softmax attention path.
///
/// The current implementation is correct but slightly slower than the existing
/// two-GEMM path on the tested Hydra-3.5 shapes, so it is disabled by default.
/// Set the `AKASHA_USE_ONLINE_SOFTMAX_ATTENTION` env var to `1`/`true` to
/// enable it for benchmarking or further tuning.
pub const USE_ONLINE_SOFTMAX_ATTENTION: bool = false;

/// Environment variable that overrides [`USE_ONLINE_SOFTMAX_ATTENTION`].
const ENV_USE_ONLINE_SOFTMAX: &str = "AKASHA_USE_ONLINE_SOFTMAX_ATTENTION";

/// Query tile size. Chosen so a tile of accumulators fits in L1/L2:
/// 64 queries × 72 dims × 4 bytes ≈ 18 KiB.
const QUERY_TILE: usize = 64;

/// Key/value chunk size. Kept small so the per-tile scores buffer stays
/// cache-resident: 64 × 64 × 4 bytes ≈ 16 KiB.
const KV_CHUNK: usize = 64;

/// Whether the online-softmax kernel should be used at runtime.
///
/// The const default can be overridden with the environment variable
/// `AKASHA_USE_ONLINE_SOFTMAX_ATTENTION=1` (or `true`) for quick A/B testing or
/// bisecting accuracy issues.
#[inline]
pub fn use_online_softmax() -> bool {
    std::env::var(ENV_USE_ONLINE_SOFTMAX)
        .map(|v| v == "1" || v == "true")
        .unwrap_or(USE_ONLINE_SOFTMAX_ATTENTION)
}

/// Compute multi-head attention for a single (batch, head) using online softmax.
///
/// Layout:
/// * `q`: `[seq_q, head_dim]` with row stride `q_stride`
/// * `k`: `[seq_kv, head_dim]` with row stride `kv_stride`
/// * `v`: `[seq_kv, head_dim]` with row stride `kv_stride`
/// * `out`: `[seq_q, head_dim]` with row stride `out_stride`
/// * `n_valid`: prefix-valid KV length (positions `n_valid..seq_kv` are ignored)
/// * `scale`: typically `1.0 / sqrt(head_dim as f32)`
///
/// All slices are assumed to be large enough for the described indexing.
pub fn fused_attention_online_softmax(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    n_valid: usize,
    scale: f32,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return fused_attention_online_softmax_avx2(
                q, k, v, out, seq_q, seq_kv, head_dim, q_stride, kv_stride, out_stride, n_valid,
                scale,
            );
        }
    }

    fused_attention_online_softmax_scalar(
        q, k, v, out, seq_q, seq_kv, head_dim, q_stride, kv_stride, out_stride, n_valid, scale,
    )
}

/// Pure scalar implementation. This is the portability fallback and the
/// reference for the vectorized path.
#[allow(clippy::too_many_arguments)]
fn fused_attention_online_softmax_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_q: usize,
    _seq_kv: usize,
    head_dim: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    n_valid: usize,
    scale: f32,
) {
    let seq_kv_eff = n_valid;

    let n_tiles = (seq_q + QUERY_TILE - 1) / QUERY_TILE;
    for t in 0..n_tiles {
        let tile_start = t * QUERY_TILE;
        let tile_q = (tile_start + QUERY_TILE).min(seq_q) - tile_start;

        // Contiguous per-tile accumulator and online-softmax state.
        let mut acc = vec![0.0f32; tile_q * head_dim];
        let mut m = vec![f32::NEG_INFINITY; tile_q];
        let mut l = vec![0.0f32; tile_q];

        // Reusable scratch buffers for one KV chunk.
        let mut scores_chunk = vec![0.0f32; tile_q * KV_CHUNK];
        let mut weights = vec![0.0f32; KV_CHUNK];

        let n_kv_chunks = (seq_kv_eff + KV_CHUNK - 1) / KV_CHUNK;
        for c in 0..n_kv_chunks {
            let chunk_start = c * KV_CHUNK;
            let chunk_len = (chunk_start + KV_CHUNK).min(seq_kv_eff) - chunk_start;

            // 1. Compute scores_chunk[tile_q, chunk_len] = q_tile @ k_chunk^T * scale.
            for i in 0..tile_q {
                let q_row_off = (tile_start + i) * q_stride;
                let q_row = &q[q_row_off..q_row_off + head_dim];
                for j in 0..chunk_len {
                    let k_row_off = (chunk_start + j) * kv_stride;
                    let k_row = &k[k_row_off..k_row_off + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_row[d] * k_row[d];
                    }
                    scores_chunk[i * chunk_len + j] = dot * scale;
                }
            }

            // 2. Update online softmax and accumulate weighted V values.
            for i in 0..tile_q {
                let row_scores = &scores_chunk[i * chunk_len..(i + 1) * chunk_len];

                // Max over the new chunk.
                let mut m_new = f32::NEG_INFINITY;
                for &s in row_scores.iter() {
                    if s > m_new {
                        m_new = s;
                    }
                }

                let m_old = m[i];
                let m_upd = m_old.max(m_new);
                let alpha_old = (m_old - m_upd).exp();
                let alpha_new = (m_new - m_upd).exp();

                // exp(row_scores - m_new) and sum.
                let mut l_new = 0.0f32;
                for j in 0..chunk_len {
                    let e = (row_scores[j] - m_new).exp();
                    weights[j] = e;
                    l_new += e;
                }

                // acc[i] = alpha_old * acc[i] + alpha_new * (weights @ v_chunk)
                let acc_row = &mut acc[i * head_dim..(i + 1) * head_dim];
                for d in 0..head_dim {
                    acc_row[d] *= alpha_old;
                }
                for j in 0..chunk_len {
                    let v_row_off = (chunk_start + j) * kv_stride;
                    let v_row = &v[v_row_off..v_row_off + head_dim];
                    let w = weights[j] * alpha_new;
                    for d in 0..head_dim {
                        acc_row[d] += w * v_row[d];
                    }
                }

                l[i] = alpha_old * l[i] + alpha_new * l_new;
                m[i] = m_upd;
            }
        }

        // 3. Normalize and write out.
        for i in 0..tile_q {
            let inv_l = 1.0f32 / l[i];
            let out_off = (tile_start + i) * out_stride;
            let out_row = &mut out[out_off..out_off + head_dim];
            let acc_row = &acc[i * head_dim..(i + 1) * head_dim];
            for d in 0..head_dim {
                out_row[d] = acc_row[d] * inv_l;
            }
        }
    }
}

/// AVX2/FMA path. The outer structure is identical to the scalar version;
/// only the per-element dot products and weighted accumulations are
/// vectorized across `head_dim`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_attention_online_softmax_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_q: usize,
    _seq_kv: usize,
    head_dim: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    n_valid: usize,
    scale: f32,
) {
    use std::arch::x86_64::*;

    let seq_kv_eff = n_valid;

    // For the typical Hydra head dims (64, 72) we can unroll the dot-product
    // loop by 8 floats at a time with a small tail.
    let dim_vec = head_dim / 8;

    let n_tiles = (seq_q + QUERY_TILE - 1) / QUERY_TILE;
    for t in 0..n_tiles {
        let tile_start = t * QUERY_TILE;
        let tile_q = (tile_start + QUERY_TILE).min(seq_q) - tile_start;

        let mut acc = vec![0.0f32; tile_q * head_dim];
        let mut m = vec![f32::NEG_INFINITY; tile_q];
        let mut l = vec![0.0f32; tile_q];

        let mut scores_chunk = vec![0.0f32; tile_q * KV_CHUNK];
        let mut weights = vec![0.0f32; KV_CHUNK];

        let n_kv_chunks = (seq_kv_eff + KV_CHUNK - 1) / KV_CHUNK;
        for c in 0..n_kv_chunks {
            let chunk_start = c * KV_CHUNK;
            let chunk_len = (chunk_start + KV_CHUNK).min(seq_kv_eff) - chunk_start;

            // scores_chunk = q_tile @ k_chunk^T * scale.
            for i in 0..tile_q {
                let q_row_off = (tile_start + i) * q_stride;
                let q_row = q.as_ptr().add(q_row_off);
                for j in 0..chunk_len {
                    let k_row_off = (chunk_start + j) * kv_stride;
                    let k_row = k.as_ptr().add(k_row_off);

                    let mut sum_vec = _mm256_setzero_ps();
                    for vd in 0..dim_vec {
                        let qv = _mm256_loadu_ps(q_row.add(vd * 8));
                        let kv = _mm256_loadu_ps(k_row.add(vd * 8));
                        sum_vec = _mm256_fmadd_ps(qv, kv, sum_vec);
                    }
                    // Horizontal reduce.
                    let mut sum = hsum256_ps(sum_vec);
                    for d in (dim_vec * 8)..head_dim {
                        sum += (*q_row.add(d)) * (*k_row.add(d));
                    }
                    *scores_chunk.as_mut_ptr().add(i * chunk_len + j) = sum * scale;
                }
            }

            // Online softmax update and weighted V sum.
            for i in 0..tile_q {
                let row_scores = &scores_chunk[i * chunk_len..(i + 1) * chunk_len];

                let mut m_new = f32::NEG_INFINITY;
                for &s in row_scores.iter() {
                    if s > m_new {
                        m_new = s;
                    }
                }

                let m_old = m[i];
                let m_upd = m_old.max(m_new);
                let alpha_old = (m_old - m_upd).exp();
                let alpha_new = (m_new - m_upd).exp();

                let mut l_new = 0.0f32;
                for j in 0..chunk_len {
                    let e = (row_scores[j] - m_new).exp();
                    weights[j] = e;
                    l_new += e;
                }

                let acc_row = acc.as_mut_ptr().add(i * head_dim);
                // acc *= alpha_old
                let alpha_old_vec = _mm256_set1_ps(alpha_old);
                for vd in 0..dim_vec {
                    let av = _mm256_loadu_ps(acc_row.add(vd * 8));
                    _mm256_storeu_ps(acc_row.add(vd * 8), _mm256_mul_ps(av, alpha_old_vec));
                }
                for d in (dim_vec * 8)..head_dim {
                    *acc_row.add(d) *= alpha_old;
                }

                // acc += alpha_new * weights[j] * v_row
                for j in 0..chunk_len {
                    let v_row_off = (chunk_start + j) * kv_stride;
                    let v_row = v.as_ptr().add(v_row_off);
                    let w_vec = _mm256_set1_ps(weights[j] * alpha_new);
                    for vd in 0..dim_vec {
                        let vv = _mm256_loadu_ps(v_row.add(vd * 8));
                        let av = _mm256_loadu_ps(acc_row.add(vd * 8));
                        _mm256_storeu_ps(
                            acc_row.add(vd * 8),
                            _mm256_fmadd_ps(vv, w_vec, av),
                        );
                    }
                    let w_scalar = weights[j] * alpha_new;
                    for d in (dim_vec * 8)..head_dim {
                        *acc_row.add(d) += w_scalar * (*v_row.add(d));
                    }
                }

                l[i] = alpha_old * l[i] + alpha_new * l_new;
                m[i] = m_upd;
            }
        }

        // Normalize and write out.
        for i in 0..tile_q {
            let inv_l = 1.0f32 / l[i];
            let out_off = (tile_start + i) * out_stride;
            let out_row = out.as_mut_ptr().add(out_off);
            let acc_row = acc.as_ptr().add(i * head_dim);
            let inv_l_vec = _mm256_set1_ps(inv_l);
            for vd in 0..dim_vec {
                let av = _mm256_loadu_ps(acc_row.add(vd * 8));
                _mm256_storeu_ps(out_row.add(vd * 8), _mm256_mul_ps(av, inv_l_vec));
            }
            for d in (dim_vec * 8)..head_dim {
                *out_row.add(d) = (*acc_row.add(d)) * inv_l;
            }
        }
    }
}

/// Horizontal sum of an AVX2 register.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    let mut arr = [0.0f32; 8];
    std::arch::x86_64::_mm256_storeu_ps(arr.as_mut_ptr(), v);
    arr.iter().sum()
}

/// Two-GEMM attention fallback that matches the original implementation.
///
/// This is kept for correctness tests and as a safety fallback when the
/// online-softmax path is disabled. It expects `scores_buf` to hold at least
/// `seq_q * n_valid` elements.
pub fn fused_attention_two_gemm_fallback(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_q: usize,
    _seq_kv: usize,
    head_dim: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    n_valid: usize,
    scale: f32,
    scores_buf: &mut [f32],
) {
    use crate::models::burn::kernels as simd_ops;

    let seq_kv_eff = n_valid;
    debug_assert!(scores_buf.len() >= seq_q * seq_kv_eff);
    let scores = &mut scores_buf[..seq_q * seq_kv_eff];

    // scores = q @ k^T * scale.
    unsafe {
        gemm_f32_ex(
            seq_q,
            seq_kv_eff,
            head_dim,
            q,
            q_stride as isize,
            1,
            k,
            1,
            kv_stride as isize,
            scores,
            seq_kv_eff as isize,
            1,
            scale,
            gemm::Parallelism::None,
        );
    }

    // Softmax per query row.
    for i in 0..seq_q {
        let row_start = i * seq_kv_eff;
        simd_ops::softmax_in_place(&mut scores[row_start..row_start + seq_kv_eff]);
    }

    // out = scores @ v.
    unsafe {
        gemm_f32_ex(
            seq_q,
            head_dim,
            seq_kv_eff,
            scores,
            seq_kv_eff as isize,
            1,
            v,
            kv_stride as isize,
            1,
            out,
            out_stride as isize,
            1,
            1.0,
            gemm::Parallelism::None,
        );
    }
}

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
            assert!(d < eps, "{x} vs {y} (diff {d}, eps {eps})");
        }
    }

    fn make_qkv(seq_q: usize, seq_kv: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut q = vec![0.0f32; seq_q * head_dim];
        let mut k = vec![0.0f32; seq_kv * head_dim];
        let mut v = vec![0.0f32; seq_kv * head_dim];
        for i in 0..q.len() {
            q[i] = ((i * 7) % 17) as f32 * 0.1 - 0.8;
        }
        for i in 0..k.len() {
            k[i] = ((i * 13) % 19) as f32 * 0.1 - 0.9;
        }
        for i in 0..v.len() {
            v[i] = ((i * 11) % 23) as f32 * 0.1 - 1.0;
        }
        (q, k, v)
    }

    #[test]
    fn online_softmax_matches_two_gemm_square() {
        let seq = 128;
        let head_dim = 64;
        let (q, k, v) = make_qkv(seq, seq, head_dim);
        let mut out_online = vec![0.0f32; seq * head_dim];
        let mut out_fallback = vec![0.0f32; seq * head_dim];
        let mut scores = vec![0.0f32; seq * seq];

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        fused_attention_online_softmax(
            &q, &k, &v, &mut out_online, seq, seq, head_dim, head_dim, head_dim, head_dim, seq, scale,
        );
        fused_attention_two_gemm_fallback(
            &q, &k, &v, &mut out_fallback, seq, seq, head_dim, head_dim, head_dim, head_dim, seq,
            scale, &mut scores,
        );

        approx_eq(&out_online, &out_fallback, 1e-4);
    }

    #[test]
    fn online_softmax_matches_two_gemm_non_square() {
        let seq_q = 64;
        let seq_kv = 256;
        let head_dim = 72;
        let (q, k, v) = make_qkv(seq_q, seq_kv, head_dim);
        let mut out_online = vec![0.0f32; seq_q * head_dim];
        let mut out_fallback = vec![0.0f32; seq_q * head_dim];
        let mut scores = vec![0.0f32; seq_q * seq_kv];

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        fused_attention_online_softmax(
            &q, &k, &v, &mut out_online, seq_q, seq_kv, head_dim, head_dim, head_dim, head_dim,
            seq_kv, scale,
        );
        fused_attention_two_gemm_fallback(
            &q, &k, &v, &mut out_fallback, seq_q, seq_kv, head_dim, head_dim, head_dim, head_dim,
            seq_kv, scale, &mut scores,
        );

        approx_eq(&out_online, &out_fallback, 1e-4);
    }

    #[test]
    fn online_softmax_matches_two_gemm_prefix_mask() {
        let seq_q = 64;
        let seq_kv = 256;
        let n_valid = 123;
        let head_dim = 64;
        let (q, k, v) = make_qkv(seq_q, seq_kv, head_dim);
        let mut out_online = vec![0.0f32; seq_q * head_dim];
        let mut out_fallback = vec![0.0f32; seq_q * head_dim];
        let mut scores = vec![0.0f32; seq_q * n_valid];

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        fused_attention_online_softmax(
            &q, &k, &v, &mut out_online, seq_q, seq_kv, head_dim, head_dim, head_dim, head_dim,
            n_valid, scale,
        );
        fused_attention_two_gemm_fallback(
            &q, &k, &v, &mut out_fallback, seq_q, seq_kv, head_dim, head_dim, head_dim, head_dim,
            n_valid, scale, &mut scores,
        );

        approx_eq(&out_online, &out_fallback, 1e-4);
    }

    #[test]
    fn online_softmax_matches_two_gemm_strided_q() {
        // Simulate the packed NaFlex QKV layout: q rows are strided by 3*hidden.
        let seq_q = 48;
        let seq_kv = 160;
        let head_dim = 72;
        let hidden = 16 * head_dim;
        let qkv_out = 3 * hidden;

        let (q_dense, k, v) = make_qkv(seq_q, seq_kv, head_dim);
        let mut q_strided = vec![0.0f32; seq_q * qkv_out];
        for i in 0..seq_q {
            q_strided[i * qkv_out..i * qkv_out + head_dim].copy_from_slice(&q_dense[i * head_dim..(i + 1) * head_dim]);
        }

        let mut out_online = vec![0.0f32; seq_q * head_dim];
        let mut out_fallback = vec![0.0f32; seq_q * head_dim];
        let mut scores = vec![0.0f32; seq_q * seq_kv];

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        fused_attention_online_softmax(
            &q_strided, &k, &v, &mut out_online, seq_q, seq_kv, head_dim, qkv_out, head_dim,
            head_dim, seq_kv, scale,
        );
        fused_attention_two_gemm_fallback(
            &q_strided, &k, &v, &mut out_fallback, seq_q, seq_kv, head_dim, qkv_out, head_dim,
            head_dim, seq_kv, scale, &mut scores,
        );

        approx_eq(&out_online, &out_fallback, 1e-4);
    }

    #[test]
    fn online_softmax_small_head_dim() {
        let seq_q = 16;
        let seq_kv = 32;
        let head_dim = 8;
        let (q, k, v) = make_qkv(seq_q, seq_kv, head_dim);
        let mut out_online = vec![0.0f32; seq_q * head_dim];
        let mut out_fallback = vec![0.0f32; seq_q * head_dim];
        let mut scores = vec![0.0f32; seq_q * seq_kv];

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        fused_attention_online_softmax(
            &q, &k, &v, &mut out_online, seq_q, seq_kv, head_dim, head_dim, head_dim, head_dim,
            seq_kv, scale,
        );
        fused_attention_two_gemm_fallback(
            &q, &k, &v, &mut out_fallback, seq_q, seq_kv, head_dim, head_dim, head_dim, head_dim,
            seq_kv, scale, &mut scores,
        );

        approx_eq(&out_online, &out_fallback, 1e-4);
    }

    #[test]
    fn online_softmax_tiny_deterministic() {
        // Hand-calculable case to debug the kernel.
        let head_dim = 4;
        let seq_q = 2;
        let seq_kv = 3;
        let q = vec![
            1.0, 0.0, 0.0, 0.0, // query 0
            0.0, 1.0, 0.0, 0.0, // query 1
        ];
        let k = vec![
            1.0, 0.0, 0.0, 0.0, // key 0
            0.0, 1.0, 0.0, 0.0, // key 1
            0.0, 0.0, 1.0, 0.0, // key 2
        ];
        let v = vec![
            1.0, 2.0, 3.0, 4.0, // value 0
            5.0, 6.0, 7.0, 8.0, // value 1
            9.0, 10.0, 11.0, 12.0, // value 2
        ];

        let mut out_online = vec![0.0f32; seq_q * head_dim];
        let mut out_fallback = vec![0.0f32; seq_q * head_dim];
        let mut scores = vec![0.0f32; seq_q * seq_kv];

        fused_attention_online_softmax(
            &q, &k, &v, &mut out_online, seq_q, seq_kv, head_dim, head_dim, head_dim, head_dim,
            seq_kv, 1.0,
        );
        fused_attention_two_gemm_fallback(
            &q, &k, &v, &mut out_fallback, seq_q, seq_kv, head_dim, head_dim, head_dim, head_dim,
            seq_kv, 1.0, &mut scores,
        );

        eprintln!("online: {:?}", out_online);
        eprintln!("fallback: {:?}", out_fallback);
        approx_eq(&out_online, &out_fallback, 1e-4);
    }
}
