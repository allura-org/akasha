use burn::prelude::*;
use burn::tensor::ops::{BoolTensor, FloatTensor};
use burn::tensor::{DType, TensorPrimitive};

use crate::models::burn::kernels as simd_ops;

use super::{
    best_row_major, best_row_major_accum, best_row_major_accum_bf16, bf16_gemm,
    fused_attention_online_softmax, fused_attention_two_gemm_fallback, fused_mlp_custom,
    gemm_a_bt_scaled, gemm_row_major, resize_buf, use_online_softmax, BlockWorkspace, QUERY_TILE,
};

pub trait FusedNaFlexBlockBackend: Backend {
    /// Compute one NaFlexBlock forward pass: norm1, self-attention, residual,
    /// norm2, MLP, residual. The `mask` argument is accepted for parity with
    /// the generic forward path; the fast implementation currently requires an
    /// all-valid mask and will fall back to the high-level module if a mask is
    /// supplied.
    fn fused_na_flex_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self>;

    /// Compute one NaFlexBlock forward pass directly on F32 buffers.
    ///
    /// `x` is a contiguous `[batch, seq, hidden]` F32 buffer and `out` is the
    /// same-shaped buffer to write the result into. The default implementation
    /// reconstructs a tensor from `x`, calls `fused_na_flex_block`, and copies
    /// the result back into `out`. Backends that can operate on buffers
    /// directly should override this to avoid the per-call tensor round-trip.
    fn fused_na_flex_block_to_buffer(
        x: &[f32],
        out: &mut [f32],
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
        batch: usize,
        seq: usize,
        hidden: usize,
    ) {
        let device = block.norm1.gamma.val().device();
        let x_t =
            Tensor::<Self, 1>::from_data(x, (&device, DType::F32)).reshape([batch, seq, hidden]);
        let out_t = Self::fused_na_flex_block(
            match x_t.into_primitive() {
                TensorPrimitive::Float(t) => t,
                _ => unreachable!("fused_na_flex_block_to_buffer input is a float tensor"),
            },
            block,
            mask,
            workspace,
        );
        let out_data = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(out_t)).to_data();
        out.copy_from_slice(
            out_data
                .as_slice::<f32>()
                .expect("fused_na_flex_block_to_buffer output is contiguous F32"),
        );
    }
}

/// Tensor-level entry point for the fused NaFlex block path.
pub fn fused_na_flex_block<B: FusedNaFlexBlockBackend>(
    x: Tensor<B, 3>,
    block: &crate::models::burn::hydra::modules::NaFlexBlock<B>,
    mask: Option<Tensor<B, 4, Bool>>,
    workspace: &mut BlockWorkspace,
) -> Tensor<B, 3> {
    let prim = match x.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("fused_na_flex_block input is a float tensor"),
    };

    Tensor::from_primitive(TensorPrimitive::Float(B::fused_na_flex_block(
        prim,
        block,
        mask.map(|m| m.into_primitive()),
        workspace,
    )))
}

/// Fused multi-head self-attention for NaFlex blocks, operating directly on the
/// packed QKV buffer produced by the block's QKV linear projection.
///
/// Reads `qkv` as `[batch, seq, 3 * hidden]` with Q/K/V interleaved per head:
///   Q offset = h * head_dim, K offset = hidden + h * head_dim,
///   V offset = 2 * hidden + h * head_dim, row stride = 3 * hidden.
///
/// Writes the merged attention output into `attn_out` (`[batch, seq, hidden]`).
/// `scores_tile_all` and `head_out_tile_all` must be large enough for
/// `batch * heads * QUERY_TILE * max_n_valid` and
/// `batch * heads * QUERY_TILE * head_dim` elements respectively. This avoids
/// the per-head Q/K/V copies that the previous implementation performed.
fn fused_na_flex_attn_buffer(
    qkv: &[f32],
    n_valids: &[usize],
    attn_out: &mut [f32],
    scores_tile_all: &mut [f32],
    head_out_tile_all: &mut [f32],
    batch: usize,
    seq: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
) {
    use rayon::prelude::*;

    let m = batch * seq;
    let qkv_out = 3 * hidden;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let max_n_valid = n_valids.iter().copied().max().unwrap_or(seq);

    debug_assert_eq!(attn_out.len(), m * hidden);
    debug_assert_eq!(scores_tile_all.len(), batch * heads * QUERY_TILE * max_n_valid);
    debug_assert_eq!(head_out_tile_all.len(), batch * heads * QUERY_TILE * head_dim);

    // The raw output pointer is passed as an integer so the parallel closure
    // can capture it; each head writes to disjoint positions.
    let attn_addr = attn_out.as_mut_ptr() as usize;
    let online = use_online_softmax();

    scores_tile_all
        .par_chunks_exact_mut(QUERY_TILE * max_n_valid)
        .zip(head_out_tile_all.par_chunks_exact_mut(QUERY_TILE * head_dim))
        .enumerate()
        .for_each(|(flat, (scores_tile, head_out_tile))| {
            let b = flat / heads;
            let h = flat % heads;
            let seq_kv_eff = n_valids[b];

            let q_base = b * seq * qkv_out + h * head_dim;
            let k_base = q_base + hidden;
            let v_base = q_base + 2 * hidden;

            let n_tiles = (seq + QUERY_TILE - 1) / QUERY_TILE;
            for t in 0..n_tiles {
                let tile_start = t * QUERY_TILE;
                let tile_q = (tile_start + QUERY_TILE).min(seq) - tile_start;

                let q_tile_off = q_base + tile_start * qkv_out;
                let q_tile_len = if tile_q > 0 {
                    (tile_q - 1) * qkv_out + head_dim
                } else {
                    0
                };
                let q_tile = &qkv[q_tile_off..q_tile_off + q_tile_len];

                let kv_len = if seq_kv_eff > 0 {
                    (seq_kv_eff - 1) * qkv_out + head_dim
                } else {
                    0
                };
                let k_slice = &qkv[k_base..k_base + kv_len];
                let v_slice = &qkv[v_base..v_base + kv_len];

                let head_out = &mut head_out_tile[..tile_q * head_dim];

                if online {
                    fused_attention_online_softmax(
                        q_tile,
                        k_slice,
                        v_slice,
                        head_out,
                        tile_q,
                        seq_kv_eff,
                        head_dim,
                        qkv_out,
                        qkv_out,
                        head_dim,
                        seq_kv_eff,
                        scale,
                    );
                } else {
                    let scores = &mut scores_tile[..tile_q * seq_kv_eff];
                    fused_attention_two_gemm_fallback(
                        q_tile,
                        k_slice,
                        v_slice,
                        head_out,
                        tile_q,
                        seq_kv_eff,
                        head_dim,
                        qkv_out,
                        qkv_out,
                        head_dim,
                        seq_kv_eff,
                        scale,
                        scores,
                    );
                }

                unsafe {
                    let attn_ptr = attn_addr as *mut f32;
                    for p in 0..tile_q {
                        let out_offset =
                            ((b * seq + tile_start + p) * hidden + h * head_dim) as usize;
                        std::ptr::copy_nonoverlapping(
                            head_out.as_ptr().add(p * head_dim),
                            attn_ptr.add(out_offset),
                            head_dim,
                        );
                    }
                }
            }
        });
}

#[cfg(feature = "burn-candle")]
impl FusedNaFlexBlockBackend for burn::backend::candle::Candle {
    fn fused_na_flex_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let [batch, seq, hidden] = x_t.dims();

        // Fast path only supports F32; fall back to the high-level module for
        // other dtypes.
        if x_t.dtype() != DType::F32 {
            let mask_t = mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m));
            let x_clone = x_t.clone();
            let attn_out = block.attn.forward(block.norm1.forward(x_t), mask_t);
            let post_attn = attn_out.clone() + x_clone;
            let out = block
                .mlp
                .forward_fused_norm(post_attn.clone(), post_attn, &block.norm2);
            return match out.into_primitive() {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("NaFlexBlock returns a float tensor"),
            };
        }

        let x_data = x_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("NaFlexBlock input is contiguous F32");
        let mut out = vec![0.0f32; batch * seq * hidden];
        Self::fused_na_flex_block_to_buffer(
            x_slice, &mut out, block, mask, workspace, batch, seq, hidden,
        );

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(out.as_slice(), (&device, DType::F32))
            .reshape([batch, seq, hidden])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_na_flex_block returns a float tensor"),
        }
    }

    fn fused_na_flex_block_to_buffer(
        x: &[f32],
        out: &mut [f32],
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
        batch: usize,
        seq: usize,
        hidden: usize,
    ) {
        use rayon::prelude::*;
        use std::time::Instant;

        let m = batch * seq;
        let heads = crate::models::burn::hydra::modules::NAFLEX_HEADS;
        let head_dim = crate::models::burn::hydra::modules::NAFLEX_HEAD_DIM;
        debug_assert_eq!(hidden, heads * head_dim);

        // Validate mask shape and prefix-validity (Burn uses `true` to mean
        // "mask out", so the valid prefix is the leading run of falses).
        // Anything else falls back to the high-level module implementation
        // reconstructed from the input buffer.
        let n_valids: Vec<usize> = if let Some(mask) = mask {
            let mask_t = Tensor::<Self, 4, Bool>::from_primitive(mask);
            let [mb, mh, mw, ms] = mask_t.dims();
            if mb != batch || mh != 1 || mw != 1 || ms != seq {
                let device = block.norm1.gamma.val().device();
                let x_t = Tensor::<Self, 1>::from_data(x, (&device, DType::F32))
                    .reshape([batch, seq, hidden]);
                let x_clone = x_t.clone();
                let attn_out = block.attn.forward(block.norm1.forward(x_t), Some(mask_t));
                let post_attn = attn_out.clone() + x_clone;
                let out_t =
                    block
                        .mlp
                        .forward_fused_norm(post_attn.clone(), post_attn, &block.norm2);
                out.copy_from_slice(
                    out_t
                        .to_data()
                        .as_slice::<f32>()
                        .expect("NaFlexBlock fallback output is contiguous F32"),
                );
                return;
            }
            let mask_data = mask_t.to_data();
            let mask_slice = mask_data
                .as_slice::<bool>()
                .expect("mask is contiguous bool");
            let mut n_valids = vec![seq; batch];
            let mut is_prefix = true;
            for b in 0..batch {
                let row = &mask_slice[b * seq..(b + 1) * seq];
                let first_true = row.iter().position(|&v| v).unwrap_or(seq);
                n_valids[b] = first_true;
                if row[first_true..].iter().any(|&v| !v) {
                    is_prefix = false;
                    break;
                }
            }
            if !is_prefix || n_valids.iter().any(|&v| v == 0) {
                let device = block.norm1.gamma.val().device();
                let x_t = Tensor::<Self, 1>::from_data(x, (&device, DType::F32))
                    .reshape([batch, seq, hidden]);
                let x_clone = x_t.clone();
                let attn_out = block.attn.forward(block.norm1.forward(x_t), Some(mask_t));
                let post_attn = attn_out.clone() + x_clone;
                let out_t =
                    block
                        .mlp
                        .forward_fused_norm(post_attn.clone(), post_attn, &block.norm2);
                out.copy_from_slice(
                    out_t
                        .to_data()
                        .as_slice::<f32>()
                        .expect("NaFlexBlock fallback output is contiguous F32"),
                );
                return;
            }
            n_valids
        } else {
            vec![seq; batch]
        };

        let t0 = Instant::now();

        // Use cached contiguous weight/bias slices instead of copying every call.
        let qkv_w = block.attn.qkv_w_cache.as_slice();
        let qkv_b = block.attn.qkv_b_cache.as_deref();
        let qkv_out = 3 * hidden;
        debug_assert_eq!(qkv_w.len(), hidden * qkv_out);

        let proj_w = block.attn.proj_w_cache.as_slice();
        let proj_b = block.attn.proj_b_cache.as_deref();
        debug_assert_eq!(proj_w.len(), hidden * hidden);

        let fc1_w = block.mlp.fc1_w_cache.as_slice();
        let fc1_b = block.mlp.fc1_b_cache.as_deref();
        let fc1_hidden = block.mlp.fc1_w_cache.len() / hidden;
        debug_assert_eq!(block.mlp.fc1_w_cache.len(), hidden * fc1_hidden);

        let fc2_w = block.mlp.fc2_w_cache.as_slice();
        let fc2_b = block.mlp.fc2_b_cache.as_deref();
        debug_assert_eq!(block.mlp.fc2_w_cache.len(), fc1_hidden * hidden);

        let norm1_gamma_data = block.norm1.gamma.val().to_data();
        let norm1_gamma = norm1_gamma_data
            .as_slice::<f32>()
            .expect("NaFlexBlock norm1 gamma is contiguous F32");
        let norm1_beta_data = block.norm1.beta.as_ref().map(|b| b.val().to_data());
        let norm1_beta = norm1_beta_data.as_ref().map(|d| {
            d.as_slice::<f32>()
                .expect("NaFlexBlock norm1 beta is contiguous F32")
        });

        let norm2_gamma_data = block.norm2.gamma.val().to_data();
        let norm2_gamma = norm2_gamma_data
            .as_slice::<f32>()
            .expect("NaFlexBlock norm2 gamma is contiguous F32");
        let norm2_beta_data = block.norm2.beta.as_ref().map(|b| b.val().to_data());
        let norm2_beta = norm2_beta_data.as_ref().map(|d| {
            d.as_slice::<f32>()
                .expect("NaFlexBlock norm2 beta is contiguous F32")
        });

        // Reusable main buffers from the shared workspace. `norm1_buf` holds
        // norm1, then the attention projection fused with the first residual.
        // `mlp_hidden_buf` holds the MLP hidden activation. The attention output
        // is written into `workspace.b` and returned by the fused attention
        // kernel. The final output is written directly into `out`.
        let mut norm1_buf = resize_buf(&mut workspace.a, m * hidden);
        let mut mlp_hidden_buf = resize_buf(&mut workspace.c, m * fc1_hidden);

        // ---- 1. LayerNorm1 into norm1_buf. ----
        x.par_chunks_exact(hidden)
            .zip(norm1_buf.par_chunks_exact_mut(hidden))
            .for_each(|(row_x, row_n)| {
                simd_ops::layer_norm_row(row_x, norm1_gamma, norm1_beta, 1e-5f32, row_n);
            });

        // ---- 2. QKV projection. ----
        // Pre-initialise the output with the bias (if any) and accumulate the
        // matrix product into it, saving a separate bias-add pass. Prefer the
        // bf16 kernel when the packed weight is available; the pre-fill is
        // idempotent, so falling back to the F32 GEMM on a shape mismatch is
        // safe.
        let mut qkv = resize_buf(&mut workspace.d, m * qkv_out);
        let bf16_qkv = if let Some(packed) = block.attn.qkv_w_bf16.as_ref() {
            if let Some(ref b) = qkv_b {
                qkv.par_chunks_exact_mut(qkv_out)
                    .for_each(|row| row.copy_from_slice(b));
                bf16_gemm::linear_accum_from_f32(
                    &norm1_buf,
                    hidden,
                    m,
                    packed,
                    &mut qkv,
                    &mut workspace.bf16_scratch,
                )
            } else {
                bf16_gemm::linear_replace_from_f32(
                    &norm1_buf,
                    hidden,
                    m,
                    packed,
                    &mut qkv,
                    &mut workspace.bf16_scratch,
                )
            }
        } else {
            false
        };
        if !bf16_qkv {
            if let Some(ref b) = qkv_b {
                qkv.par_chunks_exact_mut(qkv_out)
                    .for_each(|row| row.copy_from_slice(b));
                best_row_major_accum(m, qkv_out, hidden, &norm1_buf, qkv_w, &mut qkv);
            } else {
                best_row_major(m, qkv_out, hidden, &norm1_buf, qkv_w, &mut qkv);
            }
        }

        // ---- 3. Attention from packed QKV, writing merged output into workspace.b. ----
        // Reborrow QKV immutably so the workspace score/output tiles can be
        // borrowed independently.
        let qkv_ref = &*qkv;
        let max_n_valid = n_valids.iter().copied().max().unwrap_or(seq);
        let mut attn_buf = resize_buf(&mut workspace.b, m * hidden);
        let scores_tile_all = resize_buf(
            &mut workspace.f,
            batch * heads * QUERY_TILE * max_n_valid,
        );
        let head_out_tile_all =
            resize_buf(&mut workspace.g, batch * heads * QUERY_TILE * head_dim);
        fused_na_flex_attn_buffer(
            qkv_ref,
            &n_valids,
            &mut attn_buf,
            scores_tile_all,
            head_out_tile_all,
            batch,
            seq,
            hidden,
            heads,
            head_dim,
        );

        // ---- 4. Output projection + first residual into norm1_buf. ----
        // norm1_buf = attn_buf @ W_proj + b_proj + x
        // Initialise norm1_buf with the residual (+ bias), then accumulate the
        // projection to avoid a separate elementwise pass.
        if let Some(ref b) = proj_b {
            norm1_buf
                .par_chunks_exact_mut(hidden)
                .zip(x.par_chunks_exact(hidden))
                .for_each(|(out_row, x_row)| {
                    simd_ops::add2_in_place(out_row, b, x_row);
                });
        } else {
            norm1_buf
                .par_chunks_exact_mut(hidden)
                .zip(x.par_chunks_exact(hidden))
                .for_each(|(out_row, x_row)| {
                    out_row.copy_from_slice(x_row);
                });
        }
        best_row_major_accum_bf16(
            m,
            hidden,
            hidden,
            &attn_buf,
            hidden,
            proj_w,
            block.attn.proj_w_bf16.as_ref(),
            &mut norm1_buf,
            &mut workspace.bf16_scratch,
        );

        // ---- 5. LayerNorm2 into workspace.b. ----
        let attn_buf = resize_buf(&mut workspace.b, m * hidden);
        norm1_buf
            .par_chunks_exact(hidden)
            .zip(attn_buf.par_chunks_exact_mut(hidden))
            .for_each(|(row_x, row_n)| {
                simd_ops::layer_norm_row(row_x, norm2_gamma, norm2_beta, 1e-5f32, row_n);
            });

        // ---- 6. MLP (fc1 -> GELU -> fc2) + second residual into `out`. ----
        // Try the custom fused kernel first; if it is disabled or unavailable,
        // fall back to the two-GEMM + SIMD activation path.
        let used_custom_mlp = if let Some(packed) = block.mlp.fc1_w_packed.as_ref() {
            fused_mlp_custom(&attn_buf, out, m, hidden, fc1_hidden, hidden, packed, Some(&norm1_buf))
        } else {
            false
        };

        if !used_custom_mlp {
            // Initialise the MLP hidden buffer with the fc1 bias and accumulate the
            // fc1 projection, saving a separate bias-add pass. Prefer the bf16
            // kernel when the packed weight is available.
            let bf16_fc1 = if let Some(packed) = block.mlp.fc1_w_bf16.as_ref() {
                if let Some(ref b) = fc1_b {
                    mlp_hidden_buf
                        .par_chunks_exact_mut(fc1_hidden)
                        .for_each(|row| row.copy_from_slice(b));
                    bf16_gemm::linear_accum_from_f32(
                        &attn_buf,
                        hidden,
                        m,
                        packed,
                        &mut mlp_hidden_buf,
                        &mut workspace.bf16_scratch,
                    )
                } else {
                    bf16_gemm::linear_replace_from_f32(
                        &attn_buf,
                        hidden,
                        m,
                        packed,
                        &mut mlp_hidden_buf,
                        &mut workspace.bf16_scratch,
                    )
                }
            } else {
                false
            };
            if !bf16_fc1 {
                if let Some(ref b) = fc1_b {
                    mlp_hidden_buf
                        .par_chunks_exact_mut(fc1_hidden)
                        .for_each(|row| row.copy_from_slice(b));
                    best_row_major_accum(
                        m,
                        fc1_hidden,
                        hidden,
                        &attn_buf,
                        fc1_w,
                        &mut mlp_hidden_buf,
                    );
                } else {
                    best_row_major(m, fc1_hidden, hidden, &attn_buf, fc1_w, &mut mlp_hidden_buf);
                }
            }
            simd_ops::gelu_approx_tanh_in_place(&mut mlp_hidden_buf);

            // Write the fc2 output into `out` and fuse the bias and the residual
            // from norm1_buf by initialising `out` with the residual (+ bias) and
            // accumulating the fc2 projection.
            if let Some(ref b) = fc2_b {
                out.par_chunks_exact_mut(hidden)
                    .zip(norm1_buf.par_chunks_exact(hidden))
                    .for_each(|(out_row, post_row)| {
                        simd_ops::add2_in_place(out_row, b, post_row);
                    });
            } else {
                out.par_chunks_exact_mut(hidden)
                    .zip(norm1_buf.par_chunks_exact(hidden))
                    .for_each(|(out_row, post_row)| {
                        out_row.copy_from_slice(post_row);
                    });
            }
            best_row_major_accum_bf16(
                m,
                hidden,
                fc1_hidden,
                &mlp_hidden_buf,
                fc1_hidden,
                fc2_w,
                block.mlp.fc2_w_bf16.as_ref(),
                out,
                &mut workspace.bf16_scratch,
            );
        }

        tracing::debug!(
            "fused_na_flex_block_to_buffer total={:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }
}

#[cfg(feature = "burn-flex")]
impl FusedNaFlexBlockBackend for burn::backend::flex::Flex {
    fn fused_na_flex_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        _workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        // Generic tensor-op path. Note: this must NOT call `block.forward`
        // (that dispatches back to this trait method and recurses forever).
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let mask_t = mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m));
        let attn = block.attn.forward(block.norm1.forward(x_t.clone()), mask_t);
        let x_t = x_t + attn;
        let out = x_t.clone() + block.mlp.forward(block.norm2.forward(x_t));
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("NaFlexBlock returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-wgpu")]
impl FusedNaFlexBlockBackend for burn::backend::Wgpu {
    fn fused_na_flex_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        _workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        // Generic tensor-op path. Note: this must NOT call `block.forward`
        // (that dispatches back to this trait method and recurses forever).
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let mask_t = mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m));
        let attn = block.attn.forward(block.norm1.forward(x_t.clone()), mask_t);
        let x_t = x_t + attn;
        let out = x_t.clone() + block.mlp.forward(block.norm2.forward(x_t));
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("NaFlexBlock returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-cuda")]
impl FusedNaFlexBlockBackend for burn::backend::Cuda {
    fn fused_na_flex_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        _workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        // Generic tensor-op path. Note: this must NOT call `block.forward`
        // (that dispatches back to this trait method and recurses forever).
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let mask_t = mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m));
        let attn = block.attn.forward(block.norm1.forward(x_t.clone()), mask_t);
        let x_t = x_t + attn;
        let out = x_t.clone() + block.mlp.forward(block.norm2.forward(x_t));
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("NaFlexBlock returns a float tensor"),
        }
    }
}

#[cfg(not(any(feature = "burn-candle", feature = "burn-flex")))]
impl FusedNaFlexBlockBackend for burn::backend::NdArray {
    fn fused_na_flex_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        _workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        // Generic tensor-op path. Note: this must NOT call `block.forward`
        // (that dispatches back to this trait method and recurses forever).
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let mask_t = mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m));
        let attn = block.attn.forward(block.norm1.forward(x_t.clone()), mask_t);
        let x_t = x_t + attn;
        let out = x_t.clone() + block.mlp.forward(block.norm2.forward(x_t));
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("NaFlexBlock returns a float tensor"),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend-specific fused HydraMidBlock dispatch
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub trait FusedNaFlexAttnBackend: Backend {
    /// Compute `x + proj(attention(norm1(x)))` as a single dispatch.
    fn fused_na_flex_attn(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self>;
}

#[cfg(feature = "burn-candle")]
impl FusedNaFlexAttnBackend for burn::backend::candle::Candle {
    fn fused_na_flex_attn(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        use rayon::prelude::*;

        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));

        if x_t.dtype() != DType::F32 {
            let out = block.attn.forward(
                block.norm1.forward(x_t),
                mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
            );
            return match out.into_primitive() {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("NaFlexAttn returns a float tensor"),
            };
        }

        let [batch, seq, hidden] = x_t.dims();
        let m = batch * seq;
        let heads = crate::models::burn::hydra::modules::NAFLEX_HEADS;
        let head_dim = crate::models::burn::hydra::modules::NAFLEX_HEAD_DIM;
        debug_assert_eq!(hidden, heads * head_dim);

        let qkv_w = block.attn.qkv_w_cache.as_slice();
        let qkv_out = 3 * hidden;
        debug_assert_eq!(qkv_w.len(), hidden * qkv_out);
        let qkv_b = block.attn.qkv_b_cache.as_deref();

        let proj_w = block.attn.proj_w_cache.as_slice();
        debug_assert_eq!(proj_w.len(), hidden * hidden);
        let proj_b = block.attn.proj_b_cache.as_deref();

        let norm1_gamma_data = block.norm1.gamma.val().to_data();
        let norm1_gamma = norm1_gamma_data
            .as_slice::<f32>()
            .expect("NaFlexAttn norm1 gamma is contiguous F32");
        let norm1_beta_data = block.norm1.beta.as_ref().map(|b| b.val().to_data());
        let norm1_beta = norm1_beta_data.as_ref().map(|d| {
            d.as_slice::<f32>()
                .expect("NaFlexAttn norm1 beta is contiguous F32")
        });

        let eps = 1e-5f32; // Burn LayerNorm default; field is private

        let x_data = x_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("NaFlexAttn input is contiguous F32");

        // Hydra pads images to max_seq_len with a contiguous suffix of invalid
        // positions. Burn's attention mask uses true = mask out, so the valid
        // prefix is the leading run of false values.
        let n_valids: Vec<usize> = if let Some(mask) = mask {
            let mask_t = Tensor::<Self, 4, Bool>::from_primitive(mask);
            let [mb, mh, mw, ms] = mask_t.dims();
            if mb != batch || mh != 1 || mw != 1 || ms != seq {
                let out = block.attn.forward(block.norm1.forward(x_t), Some(mask_t));
                return match out.into_primitive() {
                    TensorPrimitive::Float(tensor) => tensor,
                    _ => unreachable!("NaFlexAttn returns a float tensor"),
                };
            }
            let mask_data = mask_t.to_data();
            let mask_slice = mask_data
                .as_slice::<bool>()
                .expect("mask is contiguous bool");
            let mut n_valids = vec![seq; batch];
            let mut is_prefix = true;
            for b in 0..batch {
                let row = &mask_slice[b * seq..(b + 1) * seq];
                let first_true = row.iter().position(|&v| v).unwrap_or(seq);
                n_valids[b] = first_true;
                if row[first_true..].iter().any(|&v| !v) {
                    is_prefix = false;
                    break;
                }
            }
            if !is_prefix || n_valids.iter().any(|&v| v == 0) {
                let out = block.attn.forward(block.norm1.forward(x_t), Some(mask_t));
                return match out.into_primitive() {
                    TensorPrimitive::Float(tensor) => tensor,
                    _ => unreachable!("NaFlexAttn returns a float tensor"),
                };
            }
            n_valids
        } else {
            vec![seq; batch]
        };

        // LayerNorm1.
        let mut norm1 = vec![0.0f32; m * hidden];
        x_slice
            .par_chunks_exact(hidden)
            .zip(norm1.par_chunks_exact_mut(hidden))
            .for_each(|(row_x, row_n)| {
                let mean = row_x.iter().copied().sum::<f32>() / hidden as f32;
                let var = row_x
                    .iter()
                    .map(|v| {
                        let d = *v - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / hidden as f32;
                let inv_std = 1.0f32 / (var + eps).sqrt();
                for j in 0..hidden {
                    row_n[j] = (row_x[j] - mean) * inv_std * norm1_gamma[j]
                        + norm1_beta.as_ref().map(|b| b[j]).unwrap_or(0.0f32);
                }
            });

        // QKV projection.
        let mut qkv = vec![0.0f32; m * qkv_out];
        best_row_major(m, qkv_out, hidden, norm1.as_slice(), qkv_w, &mut qkv);
        if let Some(ref b) = qkv_b {
            qkv.par_chunks_exact_mut(qkv_out).for_each(|row| {
                for j in 0..qkv_out {
                    row[j] += b[j];
                }
            });
        }

        // Attention, parallel over (batch, head).
        let mut attn_out = vec![0.0f32; m * hidden];
        let attn_out_addr = attn_out.as_mut_ptr() as usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        (0..batch * heads).into_par_iter().for_each(|flat| {
            let b_idx = flat / heads;
            let h = flat % heads;
            let seq_kv_eff = n_valids[b_idx];

            let mut q_buf = vec![0.0f32; seq * head_dim];
            let mut k_buf = vec![0.0f32; seq_kv_eff * head_dim];
            let mut v_buf = vec![0.0f32; seq_kv_eff * head_dim];

            for p in 0..seq {
                let row = b_idx * seq + p;
                let qkv_base = row * qkv_out;
                let q_off = qkv_base + h * head_dim;
                let buf_base = p * head_dim;
                q_buf[buf_base..buf_base + head_dim].copy_from_slice(&qkv[q_off..q_off + head_dim]);
            }
            for p in 0..seq_kv_eff {
                let row = b_idx * seq + p;
                let qkv_base = row * qkv_out;
                let k_off = qkv_base + hidden + h * head_dim;
                let v_off = qkv_base + 2 * hidden + h * head_dim;
                let buf_base = p * head_dim;
                k_buf[buf_base..buf_base + head_dim].copy_from_slice(&qkv[k_off..k_off + head_dim]);
                v_buf[buf_base..buf_base + head_dim].copy_from_slice(&qkv[v_off..v_off + head_dim]);
            }

            let mut scores = vec![0.0f32; seq * seq_kv_eff];
            gemm_a_bt_scaled(
                seq,
                seq_kv_eff,
                head_dim,
                q_buf.as_slice(),
                k_buf.as_slice(),
                &mut scores,
                scale,
                gemm::Parallelism::None,
            );

            for i in 0..seq {
                let row_start = i * seq_kv_eff;
                let mut max = f32::NEG_INFINITY;
                for j in 0..seq_kv_eff {
                    max = max.max(scores[row_start + j]);
                }
                let mut sum = 0.0f32;
                for j in 0..seq_kv_eff {
                    let e = (scores[row_start + j] - max).exp();
                    scores[row_start + j] = e;
                    sum += e;
                }
                let inv_sum = 1.0f32 / sum;
                for j in 0..seq_kv_eff {
                    scores[row_start + j] *= inv_sum;
                }
            }

            let mut head_out = vec![0.0f32; seq * head_dim];
            gemm_row_major(
                seq,
                head_dim,
                seq_kv_eff,
                scores.as_slice(),
                v_buf.as_slice(),
                &mut head_out,
                gemm::Parallelism::None,
            );

            unsafe {
                let attn_out_ptr = attn_out_addr as *mut f32;
                for p in 0..seq {
                    let row = b_idx * seq + p;
                    let out_base = row * hidden + h * head_dim;
                    let buf_base = p * head_dim;
                    std::ptr::copy_nonoverlapping(
                        head_out.as_ptr().add(buf_base),
                        attn_out_ptr.add(out_base),
                        head_dim,
                    );
                }
            }
        });

        // Output projection + residual.
        let mut output = vec![0.0f32; m * hidden];
        best_row_major(m, hidden, hidden, attn_out.as_slice(), proj_w, &mut output);
        if let Some(ref b) = proj_b {
            output.par_chunks_exact_mut(hidden).for_each(|row| {
                for j in 0..hidden {
                    row[j] += b[j];
                }
            });
        }
        output
            .par_chunks_exact_mut(hidden)
            .zip(x_slice.par_chunks_exact(hidden))
            .for_each(|(out_row, x_row)| {
                for j in 0..hidden {
                    out_row[j] += x_row[j];
                }
            });

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(output.as_slice(), (&device, DType::F32))
            .reshape([batch, seq, hidden])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_na_flex_attn returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-flex")]
impl FusedNaFlexAttnBackend for burn::backend::flex::Flex {
    fn fused_na_flex_attn(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        let out = block.attn.forward(
            block
                .norm1
                .forward(Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x))),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
        );
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("NaFlexAttn returns a float tensor"),
        }
    }
}

#[cfg(not(any(feature = "burn-candle", feature = "burn-flex")))]
impl FusedNaFlexAttnBackend for burn::backend::NdArray {
    fn fused_na_flex_attn(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        let out = block.attn.forward(
            block
                .norm1
                .forward(Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x))),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
        );
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("NaFlexAttn returns a float tensor"),
        }
    }
}
