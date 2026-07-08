use burn::prelude::*;
use burn::tensor::ops::{BoolTensor, FloatTensor};
use burn::tensor::{DType, TensorPrimitive};

use crate::models::burn::kernels as simd_ops;

use super::{
    best_row_major, fused_attention, gemm_a_bt_scaled, gemm_row_major, gemm_row_major_accum,
    resize_buf, BlockWorkspace, FastRmsNormBackend, FusedAttentionBackend,
};
#[cfg(feature = "burn-candle")]
use super::hydra_mid::fused_hydra_mid_block_to_buffer;

pub trait FusedHydraPoolTailBackend: Backend {
    /// Compute `x + HydraFeedForward(x)` followed by all mid blocks, as a single
    /// dispatch. This eliminates the intermediate tensor and residual-add
    /// round-trip between the pool feed-forward and the first mid block.
    ///
    /// * `x`:    `[batch, seq_q, hidden]` (post-cross-attention merged output)
    /// * `pool`: the HydraPool module (provides `ff` and `mid_blocks`)
    /// * `k`:    `[batch, heads, seq_kv, head_dim]`
    /// * `v`:    `[batch, heads, seq_kv, head_dim]`
    /// * returns:`[batch, seq_q, hidden]`
    fn fused_hydra_pool_tail(
        x: FloatTensor<Self>,
        pool: &crate::models::burn::hydra::modules::HydraPool<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self>;
}

/// Tensor-level entry point for the fused pool tail path.
pub fn fused_hydra_pool_tail<B: FusedHydraPoolTailBackend>(
    x: Tensor<B, 3>,
    pool: &crate::models::burn::hydra::modules::HydraPool<B>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
    workspace: &mut BlockWorkspace,
) -> Tensor<B, 3> {
    Tensor::from_primitive(TensorPrimitive::Float(B::fused_hydra_pool_tail(
        match x.into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("fused_hydra_pool_tail input is a float tensor"),
        },
        pool,
        match k.into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("fused_hydra_pool_tail k is a float tensor"),
        },
        match v.into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("fused_hydra_pool_tail v is a float tensor"),
        },
        mask.map(|m| m.into_primitive()),
        workspace,
    )))
}

#[cfg(feature = "burn-candle")]
fn fused_hydra_pool_tail_to_buffer(
    x_slice: &[f32],
    pool: &crate::models::burn::hydra::modules::HydraPool<burn::backend::candle::Candle>,
    k_slice: &[f32],
    v_slice: &[f32],
    n_valids: &[usize],
    batch: usize,
    seq_q: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    seq_kv: usize,
    workspace: &mut BlockWorkspace,
) {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, MatMut, MatRef, Par};
    use rayon::prelude::*;
    use std::time::Instant;

    let t_tail0 = Instant::now();

    // ---- 1. Pool FF: post_ff = x + HydraFeedForward(x). ----
    let t_ff0 = Instant::now();

    let eps = 1e-5f32;
    let ff_gamma_data = pool.ff.norm.gamma.val().to_data();
    let ff_gamma = ff_gamma_data
        .as_slice::<f32>()
        .expect("pool ff norm gamma is contiguous F32");
    let ff_beta_data = pool.ff.norm.beta.as_ref().map(|b| b.val().to_data());
    let ff_beta = ff_beta_data.as_ref().map(|d| {
        d.as_slice::<f32>()
            .expect("pool ff norm beta is contiguous F32")
    });

    let glu_w = pool.ff.glu_w_cache.as_slice();
    let glu_out2 = glu_w.len() / hidden;
    let glu_out_dim = glu_out2 / 2;

    let proj_out_w = pool.ff.proj_out_w_cache.as_slice();
    let proj_out_dim = proj_out_w.len() / glu_out_dim;
    let proj_out_b = pool.ff.proj_out_b_cache.as_deref();
    debug_assert_eq!(proj_out_dim, hidden);

    let m = batch * seq_q;

    // Reusable workspace buffers. The pool FF writes into workspace.a,
    // uses workspace.c for the FF norm input and workspace.d for the GLU
    // projection. The mid-block helper reuses c/d/e/f for its internal
    // buffers and workspace.b as the scratch buffer for the mid-block loop.
    let mut current = {
        {
            let mut post_ff = resize_buf(&mut workspace.a, m * hidden);
            let normed = resize_buf(&mut workspace.c, m * hidden);
            let mut glu_proj = resize_buf(&mut workspace.d, m * glu_out2);

            // Layer norm.
            x_slice
                .par_chunks_exact(hidden)
                .zip(normed.par_chunks_exact_mut(hidden))
                .for_each(|(row_x, row_n)| {
                    simd_ops::layer_norm_row(row_x, ff_gamma, ff_beta, eps, row_n);
                });

            // GLU projection.
            best_row_major(m, glu_out2, hidden, &normed, glu_w, &mut glu_proj);

            // In-place GLU activation.
            glu_proj
                .par_chunks_exact_mut(glu_out2)
                .for_each(|row| simd_ops::glu_softplus_in_place_interleaved(row, glu_out_dim));

            // Initialise post_ff with the residual (+ bias) and accumulate the
            // output projection, avoiding a separate elementwise pass.
            if let Some(ref b) = proj_out_b {
                post_ff
                    .par_chunks_exact_mut(hidden)
                    .zip(x_slice.par_chunks_exact(hidden))
                    .for_each(|(out_row, x_row)| {
                        simd_ops::add2_in_place(out_row, b, x_row);
                    });
            } else {
                post_ff
                    .par_chunks_exact_mut(hidden)
                    .zip(x_slice.par_chunks_exact(hidden))
                    .for_each(|(out_row, x_row)| {
                        out_row.copy_from_slice(x_row);
                    });
            }

            // Output projection into post_ff.
            if m <= 1024 {
                // For moderate batch sizes the strided faer path loses to a
                // contiguous copy + gemm because the latter has a fast pure-Rust
                // implementation for these shapes.
                let glu_contig = resize_buf(&mut workspace.e, m * glu_out_dim);
                for i in 0..m {
                    let src = &glu_proj[i * glu_out2..i * glu_out2 + glu_out_dim];
                    let dst = &mut glu_contig[i * glu_out_dim..(i + 1) * glu_out_dim];
                    dst.copy_from_slice(src);
                }
                gemm_row_major_accum(
                    m,
                    hidden,
                    glu_out_dim,
                    &glu_contig,
                    proj_out_w,
                    &mut post_ff,
                    gemm::Parallelism::Rayon(0),
                );
            } else {
                let a =
                    MatRef::from_row_major_slice_with_stride(&glu_proj, m, glu_out_dim, glu_out2);
                let b = MatRef::from_row_major_slice(proj_out_w, glu_out_dim, hidden);
                let mut c = MatMut::from_row_major_slice_mut(&mut post_ff, m, hidden);
                matmul(c.as_mut(), Accum::Add, a, b, 1.0f32, Par::rayon(0));
            }
        }
        std::mem::take(&mut workspace.a)
    };
    let mut next = std::mem::take(&mut workspace.b);
    next.resize(m * hidden, 0.0);

    let t_ff = t_ff0.elapsed();

    // ---- 2. Mid blocks. ----
    let t_mid0 = Instant::now();

    for block in &pool.mid_blocks {
        fused_hydra_mid_block_to_buffer(
            &current, &mut next, block, k_slice, v_slice, &n_valids, batch, seq_q, hidden, heads,
            head_dim, seq_kv, workspace,
        );
        std::mem::swap(&mut current, &mut next);
    }

    // Ensure the final result lives in workspace.a so callers always know where
    // to find it.
    workspace.a = current;
    workspace.b = next;

    let t_mid = t_mid0.elapsed();

    let t_tail = t_tail0.elapsed();
    tracing::debug!(
        "fused_hydra_pool_tail total={:.3}s ff={:.3}s mid_blocks={:.3}s",
        t_tail.as_secs_f64(),
        t_ff.as_secs_f64(),
        t_mid.as_secs_f64()
    );
}

#[cfg(feature = "burn-candle")]
impl FusedHydraPoolTailBackend for burn::backend::candle::Candle {
    fn fused_hydra_pool_tail(
        x: FloatTensor<Self>,
        pool: &crate::models::burn::hydra::modules::HydraPool<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let k_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k));
        let v_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v));

        if x_t.dtype() != DType::F32 {
            let mut out = x_t.clone() + pool.ff.forward(x_t);
            for block in &pool.mid_blocks {
                out = block.forward_fused(
                    out,
                    &k_t,
                    &v_t,
                    mask.as_ref()
                        .map(|m| Tensor::<Self, 4, Bool>::from_primitive(m.clone())),
                );
            }
            return match out.into_primitive() {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("HydraPool tail returns a float tensor"),
            };
        }

        let [batch, seq_q, hidden] = x_t.dims();
        let [k_batch, heads, seq_kv, head_dim] = k_t.dims();
        let [v_batch, v_heads, v_seq, v_head_dim] = v_t.dims();
        assert_eq!(k_batch, v_batch, "k and v batch sizes must match");
        assert_eq!(heads, v_heads, "k and v head counts must match");
        assert_eq!(seq_kv, v_seq, "k and v seq lengths must match");
        assert_eq!(head_dim, v_head_dim, "k and v head dims must match");
        assert_eq!(batch, k_batch, "x and k batch sizes must match");
        assert_eq!(
            hidden,
            heads * head_dim,
            "x hidden dim must match heads*head_dim"
        );

        let k_data = k_t.to_data();
        let v_data = v_t.to_data();
        let k_slice = k_data
            .as_slice::<f32>()
            .expect("HydraPool tail k is contiguous F32");
        let v_slice = v_data
            .as_slice::<f32>()
            .expect("HydraPool tail v is contiguous F32");

        // Hydra pads images to max_seq_len with a contiguous suffix of invalid
        // positions. Burn's attention mask uses true = mask out, so the valid
        // prefix is the leading run of false values.
        let n_valids: Vec<usize> = if let Some(mask) = mask {
            let mask_t = Tensor::<Self, 4, Bool>::from_primitive(mask);
            let [mb, mh, mw, ms] = mask_t.dims();
            if mb != batch || mh != 1 || mw != 1 || ms != seq_kv {
                let mut out = x_t.clone() + pool.ff.forward(x_t);
                for block in &pool.mid_blocks {
                    out = block.forward_fused(out, &k_t, &v_t, Some(mask_t.clone()));
                }
                return match out.into_primitive() {
                    TensorPrimitive::Float(tensor) => tensor,
                    _ => unreachable!("HydraPool tail returns a float tensor"),
                };
            }
            let mask_data = mask_t.to_data();
            let mask_slice = mask_data
                .as_slice::<bool>()
                .expect("mask is contiguous bool");
            let mut n_valids = vec![seq_kv; batch];
            let mut is_prefix = true;
            for b in 0..batch {
                let row = &mask_slice[b * seq_kv..(b + 1) * seq_kv];
                let first_true = row.iter().position(|&v| v).unwrap_or(seq_kv);
                n_valids[b] = first_true;
                if row[first_true..].iter().any(|&v| !v) {
                    is_prefix = false;
                    break;
                }
            }
            if !is_prefix || n_valids.iter().any(|&v| v == 0) {
                let mut out = x_t.clone() + pool.ff.forward(x_t);
                for block in &pool.mid_blocks {
                    out = block.forward_fused(out, &k_t, &v_t, Some(mask_t.clone()));
                }
                return match out.into_primitive() {
                    TensorPrimitive::Float(tensor) => tensor,
                    _ => unreachable!("HydraPool tail returns a float tensor"),
                };
            }
            n_valids
        } else {
            vec![seq_kv; batch]
        };

        let x_data = x_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("HydraPool tail input is contiguous F32");

        fused_hydra_pool_tail_to_buffer(
            x_slice, pool, k_slice, v_slice, &n_valids, batch, seq_q, hidden, heads, head_dim,
            seq_kv, workspace,
        );

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(&*workspace.a, (&device, DType::F32))
            .reshape([batch, seq_q, hidden])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_hydra_pool_tail returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-flex")]
impl FusedHydraPoolTailBackend for burn::backend::flex::Flex {
    fn fused_hydra_pool_tail(
        x: FloatTensor<Self>,
        pool: &crate::models::burn::hydra::modules::HydraPool<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        _workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let k_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k));
        let v_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v));
        let mask_t = mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m));
        let mut out = x_t.clone() + pool.ff.forward(x_t);
        for block in &pool.mid_blocks {
            out = block.forward_fused(out, &k_t, &v_t, mask_t.clone());
        }
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("HydraPool tail returns a float tensor"),
        }
    }
}

#[cfg(not(any(feature = "burn-candle", feature = "burn-flex")))]
impl FusedHydraPoolTailBackend for burn::backend::NdArray {
    fn fused_hydra_pool_tail(
        x: FloatTensor<Self>,
        pool: &crate::models::burn::hydra::modules::HydraPool<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        _workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let k_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k));
        let v_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v));
        let mask_t = mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m));
        let mut out = x_t.clone() + pool.ff.forward(x_t);
        for block in &pool.mid_blocks {
            out = block.forward_fused(out, &k_t, &v_t, mask_t.clone());
        }
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("HydraPool tail returns a float tensor"),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend-specific fused HydraPool dispatch (kv + cross-attention + tail)
// ---------------------------------------------------------------------------

/// Generic fallback for the fused pool kernel: builds `k`/`v`/`q` as tensors,
/// runs fused attention, merges heads, then calls the fused tail. Used by the
/// trait default and by the Candle fast path when the input is not F32 or the
/// mask is not a simple prefix-valid mask.
fn fused_hydra_pool_fallback_tensor<
    B: FusedAttentionBackend + FusedHydraPoolTailBackend + FastRmsNormBackend,
>(
    x_t: Tensor<B, 3>,
    pool: &crate::models::burn::hydra::modules::HydraPool<B>,
    mask: Option<Tensor<B, 4, Bool>>,
    workspace: &mut BlockWorkspace,
) -> FloatTensor<B> {
    let [batch, seq, _x_hidden] = x_t.dims();

    let kv = pool.kv.forward(x_t);
    let kv = kv.reshape([
        batch,
        seq,
        2,
        crate::models::burn::hydra::modules::HYDRA_HEADS,
        crate::models::burn::hydra::modules::HYDRA_HEAD_DIM,
    ]);
    let kv = kv.permute([2, 0, 3, 1, 4]);
    let mut chunks: Vec<Tensor<B, 5>> = kv.split_with_sizes(vec![1, 1], 0);
    let v = chunks.swap_remove(1).reshape([
        batch,
        crate::models::burn::hydra::modules::HYDRA_HEADS,
        seq,
        crate::models::burn::hydra::modules::HYDRA_HEAD_DIM,
    ]);
    let k = pool.qk_norm.forward_fast(chunks.swap_remove(0).reshape([
        batch,
        crate::models::burn::hydra::modules::HYDRA_HEADS,
        seq,
        crate::models::burn::hydra::modules::HYDRA_HEAD_DIM,
    ]));

    let [heads, n_classes, head_dim] = pool.q.dims();
    let attn_dim = heads * head_dim;
    let q = pool
        .q
        .val()
        .reshape([1, heads, n_classes, head_dim])
        .expand([batch, heads, n_classes, head_dim]);

    let out = fused_attention(q, k.clone(), v.clone(), mask.clone());
    let out = out
        .permute([0, 2, 1, 3])
        .reshape([batch, n_classes, attn_dim]);
    let out = fused_hydra_pool_tail(out, pool, k, v, mask, workspace);

    match out.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor,
        _ => unreachable!("fused_hydra_pool returns a float tensor"),
    }
}

/// Backends that provide a fully fused HydraPool path.
pub trait FusedHydraPoolBackend:
    FusedAttentionBackend + FusedHydraPoolTailBackend + FastRmsNormBackend
{
    /// Compute the full HydraPool forward pass in a single kernel:
    /// kv projection, query expansion, cross-attention, head merge, pool FF +
    /// residual, and mid blocks.
    ///
    /// * `x`:    `[batch, seq_kv, hidden]` (post-final-layer-norm, trimmed to
    ///           the valid prefix)
    /// * `pool`: the HydraPool module
    /// * `mask`: optional `[batch, 1, 1, seq_kv]` attention mask where `true`
    ///           means "mask out" (Burn semantics)
    /// * returns:`[batch, n_classes, hidden]`
    fn fused_hydra_pool(
        x: FloatTensor<Self>,
        pool: &crate::models::burn::hydra::modules::HydraPool<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        fused_hydra_pool_fallback_tensor(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)),
            pool,
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
            workspace,
        )
    }
}

/// Tensor-level entry point for the fused HydraPool path.
pub fn fused_hydra_pool<B: FusedHydraPoolBackend>(
    x: Tensor<B, 3>,
    pool: &crate::models::burn::hydra::modules::HydraPool<B>,
    mask: Option<Tensor<B, 4, Bool>>,
    workspace: &mut BlockWorkspace,
) -> Tensor<B, 3> {
    Tensor::from_primitive(TensorPrimitive::Float(B::fused_hydra_pool(
        match x.into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("fused_hydra_pool input is a float tensor"),
        },
        pool,
        mask.map(|m| m.into_primitive()),
        workspace,
    )))
}

#[cfg(feature = "burn-candle")]
fn fused_hydra_pool_kv_to_buffer(
    x_slice: &[f32],
    pool: &crate::models::burn::hydra::modules::HydraPool<burn::backend::candle::Candle>,
    k_buf: &mut [f32],
    v_buf: &mut [f32],
    batch: usize,
    seq_kv: usize,
    x_hidden: usize,
    heads: usize,
    head_dim: usize,
    attn_dim: usize,
    kv_proj_buf: &mut Vec<f32>,
) {
    use rayon::prelude::*;

    let kv_w = pool.kv_w_cache.as_slice();
    let kv_b = pool.kv_b_cache.as_deref();
    let m = batch * seq_kv;
    let kv_out = attn_dim * 2;

    // Project x to the packed [batch, seq_kv, 2*attn_dim] kv buffer.
    let kv_proj = resize_buf(kv_proj_buf, m * kv_out);
    best_row_major(m, kv_out, x_hidden, x_slice, kv_w, kv_proj);
    if let Some(b) = kv_b {
        kv_proj.par_chunks_exact_mut(kv_out).for_each(|row| {
            for j in 0..kv_out {
                row[j] += b[j];
            }
        });
    }

    // Unpack into [batch, heads, seq_kv, head_dim] k/v buffers.
    // Raw pointers are not Sync; pass their addresses as usize values.
    let k_addr = k_buf.as_mut_ptr() as usize;
    let v_addr = v_buf.as_mut_ptr() as usize;
    kv_proj
        .par_chunks_exact(kv_out)
        .enumerate()
        .for_each(|(row_idx, kv_row)| {
            let b = row_idx / seq_kv;
            let p = row_idx % seq_kv;
            for h in 0..heads {
                let src_off = h * head_dim;
                let dst_off = ((b * heads + h) * seq_kv + p) * head_dim;
                unsafe {
                    let k_ptr = k_addr as *mut f32;
                    let v_ptr = v_addr as *mut f32;
                    std::ptr::copy_nonoverlapping(
                        kv_row.as_ptr().add(src_off),
                        k_ptr.add(dst_off),
                        head_dim,
                    );
                    std::ptr::copy_nonoverlapping(
                        kv_row.as_ptr().add(attn_dim + src_off),
                        v_ptr.add(dst_off),
                        head_dim,
                    );
                }
            }
        });

    // Apply RMS norm to k over the last dimension, matching forward_kv.
    let qk_norm_eps = pool.qk_norm.eps;
    k_buf.par_chunks_exact_mut(head_dim).for_each(|slice| {
        let mut sum2 = 0.0f32;
        for &v in slice.iter() {
            sum2 += v * v;
        }
        let scale = 1.0f32 / ((sum2 / head_dim as f32) + qk_norm_eps).sqrt();
        for v in slice.iter_mut() {
            *v *= scale;
        }
    });
}

#[cfg(feature = "burn-candle")]
fn fused_hydra_pool_cross_attn_to_buffer(
    q_slice: &[f32], // [heads, n_classes, head_dim]
    k_slice: &[f32], // [batch, heads, seq_kv, head_dim]
    v_slice: &[f32],
    out_buf: &mut [f32], // [batch, n_classes, hidden]
    n_valids: &[usize],
    batch: usize,
    n_classes: usize,
    heads: usize,
    head_dim: usize,
    seq_kv: usize,
    scores_buf: &mut Vec<f32>,
    head_out_buf: &mut Vec<f32>,
) {
    use rayon::prelude::*;

    const QUERY_TILE: usize = 64;
    let max_n_valid = n_valids.iter().copied().max().unwrap_or(seq_kv);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q_stride_head = n_classes * head_dim;

    let scores_tile_all = resize_buf(scores_buf, batch * heads * QUERY_TILE * max_n_valid);
    let head_out_tile_all = resize_buf(head_out_buf, batch * heads * QUERY_TILE * head_dim);

    let out_buf_addr = out_buf.as_mut_ptr() as usize;

    (0..batch * heads)
        .into_par_iter()
        .zip(scores_tile_all.par_chunks_exact_mut(QUERY_TILE * max_n_valid))
        .zip(head_out_tile_all.par_chunks_exact_mut(QUERY_TILE * head_dim))
        .for_each(|((flat, scores_tile), head_out_tile)| {
            let b = flat / heads;
            let h = flat % heads;
            let seq_kv_eff = n_valids[b];

            let q_head = &q_slice[h * q_stride_head..(h + 1) * q_stride_head];
            let kv_head_off = flat * seq_kv * head_dim;
            let k_head = &k_slice[kv_head_off..kv_head_off + seq_kv_eff * head_dim];
            let v_head = &v_slice[kv_head_off..kv_head_off + seq_kv_eff * head_dim];

            let n_tiles = (n_classes + QUERY_TILE - 1) / QUERY_TILE;
            for t in 0..n_tiles {
                let tile_start = t * QUERY_TILE;
                let tile_q = (tile_start + QUERY_TILE).min(n_classes) - tile_start;

                let q_tile = &q_head[tile_start * head_dim..(tile_start + tile_q) * head_dim];
                let scores = &mut scores_tile[..tile_q * seq_kv_eff];
                let head_out = &mut head_out_tile[..tile_q * head_dim];

                gemm_a_bt_scaled(
                    tile_q,
                    seq_kv_eff,
                    head_dim,
                    q_tile,
                    k_head,
                    scores,
                    scale,
                    gemm::Parallelism::None,
                );

                for i in 0..tile_q {
                    let row_start = i * seq_kv_eff;
                    simd_ops::softmax_in_place(&mut scores[row_start..row_start + seq_kv_eff]);
                }

                gemm_row_major(
                    tile_q,
                    head_dim,
                    seq_kv_eff,
                    scores,
                    v_head,
                    head_out,
                    gemm::Parallelism::None,
                );

                unsafe {
                    let out_ptr = out_buf_addr as *mut f32;
                    for p in 0..tile_q {
                        let out_base =
                            (b * n_classes + tile_start + p) * heads * head_dim + h * head_dim;
                        let buf_base = p * head_dim;
                        std::ptr::copy_nonoverlapping(
                            head_out.as_ptr().add(buf_base),
                            out_ptr.add(out_base),
                            head_dim,
                        );
                    }
                }
            }
        });
}

#[cfg(feature = "burn-candle")]
impl FusedHydraPoolBackend for burn::backend::candle::Candle {
    fn fused_hydra_pool(
        x: FloatTensor<Self>,
        pool: &crate::models::burn::hydra::modules::HydraPool<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        use std::time::Instant;

        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));

        if x_t.dtype() != DType::F32 {
            return fused_hydra_pool_fallback_tensor(
                x_t,
                pool,
                mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
                workspace,
            );
        }

        let [batch, seq_kv, x_hidden] = x_t.dims();
        let [heads, n_classes, head_dim] = pool.q.dims();
        let attn_dim = heads * head_dim;

        let q_t = pool.q.val();
        let q_data = q_t.to_data();
        let q_slice = q_data
            .as_slice::<f32>()
            .expect("HydraPool q is contiguous F32");

        // Hydra pads images to max_seq_len with a contiguous suffix of invalid
        // positions. Burn's attention mask uses true = mask out, so the valid
        // prefix is the leading run of false values.
        let n_valids: Vec<usize> = if let Some(mask) = mask {
            let mask_t = Tensor::<Self, 4, Bool>::from_primitive(mask);
            let [mb, mh, mw, ms] = mask_t.dims();
            if mb != batch || mh != 1 || mw != 1 || ms != seq_kv {
                return fused_hydra_pool_fallback_tensor(x_t, pool, Some(mask_t), workspace);
            }
            let mask_data = mask_t.to_data();
            let mask_slice = mask_data
                .as_slice::<bool>()
                .expect("mask is contiguous bool");
            let mut n_valids = vec![seq_kv; batch];
            let mut is_prefix = true;
            for b in 0..batch {
                let row = &mask_slice[b * seq_kv..(b + 1) * seq_kv];
                let first_true = row.iter().position(|&v| v).unwrap_or(seq_kv);
                n_valids[b] = first_true;
                if row[first_true..].iter().any(|&v| !v) {
                    is_prefix = false;
                    break;
                }
            }
            if !is_prefix || n_valids.iter().any(|&v| v == 0) {
                return fused_hydra_pool_fallback_tensor(x_t, pool, Some(mask_t), workspace);
            }
            n_valids
        } else {
            vec![seq_kv; batch]
        };

        let t0 = Instant::now();

        let x_data = x_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("HydraPool input is contiguous F32");

        // ---- 1. KV projection into workspace buffers. ----
        let mut k_vec = std::mem::take(&mut workspace.h);
        let mut v_vec = std::mem::take(&mut workspace.i);
        let mut kv_proj = std::mem::take(&mut workspace.g);
        k_vec.resize(batch * heads * seq_kv * head_dim, 0.0);
        v_vec.resize(batch * heads * seq_kv * head_dim, 0.0);
        fused_hydra_pool_kv_to_buffer(
            x_slice,
            pool,
            &mut k_vec,
            &mut v_vec,
            batch,
            seq_kv,
            x_hidden,
            heads,
            head_dim,
            attn_dim,
            &mut kv_proj,
        );
        workspace.g = kv_proj;
        let t_kv = t0.elapsed();

        // ---- 2. Cross-attention: merged output into workspace.l. ----
        let t_attn0 = Instant::now();
        let mut merged_vec = std::mem::take(&mut workspace.l);
        let mut scores_buf = std::mem::take(&mut workspace.j);
        let mut head_out_buf = std::mem::take(&mut workspace.k);
        merged_vec.resize(batch * n_classes * attn_dim, 0.0);
        fused_hydra_pool_cross_attn_to_buffer(
            q_slice,
            &k_vec,
            &v_vec,
            &mut merged_vec,
            &n_valids,
            batch,
            n_classes,
            heads,
            head_dim,
            seq_kv,
            &mut scores_buf,
            &mut head_out_buf,
        );
        workspace.j = scores_buf;
        workspace.k = head_out_buf;
        let t_attn = t_attn0.elapsed();

        // ---- 3. Pool tail (FF + mid blocks) on the merged buffer. ----
        let t_tail0 = Instant::now();
        fused_hydra_pool_tail_to_buffer(
            merged_vec.as_slice(),
            pool,
            k_vec.as_slice(),
            v_vec.as_slice(),
            &n_valids,
            batch,
            n_classes,
            attn_dim,
            heads,
            head_dim,
            seq_kv,
            workspace,
        );
        workspace.h = k_vec;
        workspace.i = v_vec;
        workspace.l = merged_vec;
        let t_tail = t_tail0.elapsed();

        tracing::debug!(
            "fused_hydra_pool total={:.3}s kv={:.3}s attn={:.3}s tail={:.3}s",
            t0.elapsed().as_secs_f64(),
            t_kv.as_secs_f64(),
            t_attn.as_secs_f64(),
            t_tail.as_secs_f64()
        );

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(&*workspace.a, (&device, DType::F32))
            .reshape([batch, n_classes, attn_dim])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_hydra_pool returns a float tensor"),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend-specific fused NaFlex attention dispatch (norm1 + attn + residual)
// ---------------------------------------------------------------------------

