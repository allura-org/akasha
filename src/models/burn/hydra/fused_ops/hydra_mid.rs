use burn::prelude::*;
use burn::tensor::ops::{BoolTensor, FloatTensor};
use burn::tensor::{DType, TensorPrimitive};

use crate::models::burn::kernels as simd_ops;

use super::{
    best_row_major, best_row_major_accum, best_row_major_accum_bf16, bf16_gemm,
    fused_attention_online_softmax, fused_attention_two_gemm_fallback, fused_glu_custom,
    gemm_row_major_accum, resize_buf, use_online_softmax, BlockWorkspace,
};

pub trait FusedHydraMidBlockBackend: Backend {
    /// Compute one HydraMidBlock forward pass: q_proj, q_norm, cross-attention
    /// with the supplied `k`/`v`, output projection + residual, then FF + residual.
    ///
    /// The fast path currently requires an all-valid mask and falls back to the
    /// high-level module if a mask is supplied.
    fn fused_hydra_mid_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::HydraMidBlock<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self>;
}

#[cfg(feature = "burn-candle")]
#[inline(always)]
pub(crate) fn fused_hydra_mid_block_to_buffer(
    x_slice: &[f32],
    out_buf: &mut [f32],
    block: &crate::models::burn::hydra::modules::HydraMidBlock<burn::backend::candle::Candle>,
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

    let m = batch * seq_q;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q_proj_w = block.q_proj_w_cache.as_slice();
    let q_proj_b = block.q_proj_b_cache.as_deref();
    debug_assert_eq!(q_proj_w.len(), hidden * hidden);

    let o_proj_w = block.o_proj_w_cache.as_slice();
    let o_proj_b = block.o_proj_b_cache.as_deref();
    debug_assert_eq!(o_proj_w.len(), hidden * hidden);

    let q_norm_eps = block.q_norm.eps;

    let ff_gamma_data = block.ff.norm.gamma.val().to_data();
    let ff_gamma = ff_gamma_data
        .as_slice::<f32>()
        .expect("ff norm gamma is contiguous F32");
    let ff_beta_data = block.ff.norm.beta.as_ref().map(|b| b.val().to_data());
    let ff_beta = ff_beta_data
        .as_ref()
        .map(|d| d.as_slice::<f32>().expect("ff norm beta is contiguous F32"));

    let glu_w = block.ff.glu_w_cache.as_slice();
    let glu_out2 = glu_w.len() / hidden;
    let glu_out_dim = glu_out2 / 2;

    let proj_out_w = block.ff.proj_out_w_cache.as_slice();
    let proj_out_dim = proj_out_w.len() / glu_out_dim;
    let proj_out_b = block.ff.proj_out_b_cache.as_deref();
    debug_assert_eq!(proj_out_dim, hidden);

    let t_mid0 = Instant::now();

    // ---- 1. Q projection. ----
    // Initialise the output with the bias and accumulate the projection,
    // saving a separate bias-add pass. Prefer the bf16 kernel when the packed
    // weight is available; the pre-fill is idempotent, so falling back to the
    // F32 GEMM on a shape mismatch is safe.
    let t_q_proj0 = Instant::now();
    let bf16_q = if let Some(packed) = block.q_proj_w_bf16.as_ref() {
        if let Some(ref b) = q_proj_b {
            out_buf
                .par_chunks_exact_mut(hidden)
                .for_each(|row| row.copy_from_slice(b));
            bf16_gemm::linear_accum_from_f32(
                x_slice,
                hidden,
                m,
                packed,
                out_buf,
                &mut workspace.bf16_scratch,
            )
        } else {
            bf16_gemm::linear_replace_from_f32(
                x_slice,
                hidden,
                m,
                packed,
                out_buf,
                &mut workspace.bf16_scratch,
            )
        }
    } else {
        false
    };
    if !bf16_q {
        if let Some(ref b) = q_proj_b {
            out_buf
                .par_chunks_exact_mut(hidden)
                .for_each(|row| row.copy_from_slice(b));
            best_row_major_accum(m, hidden, hidden, x_slice, q_proj_w, out_buf);
        } else {
            best_row_major(m, hidden, hidden, x_slice, q_proj_w, out_buf);
        }
    }
    let t_q_proj = t_q_proj0.elapsed();

    // ---- 2. Apply RMS norm per (batch, head, query) over the last dim.
    // out_buf is row-major [batch, seq_q, heads, head_dim]; each head slice is
    // contiguous and independent.
    let t_q_norm0 = Instant::now();
    out_buf.par_chunks_exact_mut(head_dim).for_each(|slice| {
        let mut sum2 = 0.0f32;
        for &v in slice.iter() {
            sum2 += v * v;
        }
        let scale_norm = 1.0f32 / ((sum2 / head_dim as f32) + q_norm_eps).sqrt();
        for v in slice.iter_mut() {
            *v *= scale_norm;
        }
    });
    let t_q_norm = t_q_norm0.elapsed();

    // ---- 3. Cross-attention, writing merged output back into out_buf. ----
    // Use a tiled attention kernel: process queries in small tiles so the
    // per-tile scores matrix stays in cache and we never materialise the
    // full [seq_q, seq_kv] attention scores for every head at once.
    let t_attn0 = Instant::now();
    let out_buf_addr = out_buf.as_mut_ptr() as usize;

    const QUERY_TILE: usize = 64;
    let max_n_valid = n_valids.iter().copied().max().unwrap_or(seq_kv);
    let online = use_online_softmax();

    // Pull per-head working buffers from the shared workspace. `q_head_all`,
    // `scores_tile_all`, and `head_out_tile_all` are only needed during the
    // attention loop; `post_attn` and `glu_proj` are allocated afterwards by
    // reusing the same slots.
    {
        let q_head_all = resize_buf(&mut workspace.c, batch * heads * seq_q * head_dim);
        let scores_tile_all =
            resize_buf(&mut workspace.d, batch * heads * QUERY_TILE * max_n_valid);
        let head_out_tile_all = resize_buf(&mut workspace.e, batch * heads * QUERY_TILE * head_dim);

        q_head_all
            .par_chunks_exact_mut(seq_q * head_dim)
            .zip(scores_tile_all.par_chunks_exact_mut(QUERY_TILE * max_n_valid))
            .zip(head_out_tile_all.par_chunks_exact_mut(QUERY_TILE * head_dim))
            .enumerate()
            .for_each(|(flat, ((q_head, scores_tile), head_out_tile))| {
                let b_idx = flat / heads;
                let h = flat % heads;
                let seq_kv_eff = n_valids[b_idx];

                // Gather contiguous q for this head.
                for p in 0..seq_q {
                    let row = b_idx * seq_q + p;
                    let src_off = row * hidden + h * head_dim;
                    let dst_off = p * head_dim;
                    q_head[dst_off..dst_off + head_dim]
                        .copy_from_slice(&out_buf[src_off..src_off + head_dim]);
                }

                let kv_stride_head = seq_kv * head_dim;
                let kv_off = (b_idx * heads + h) * kv_stride_head;
                let k_head = &k_slice[kv_off..kv_off + seq_kv_eff * head_dim];
                let v_head = &v_slice[kv_off..kv_off + seq_kv_eff * head_dim];

                // Process queries in tiles to keep working set cache-resident.
                let n_tiles = (seq_q + QUERY_TILE - 1) / QUERY_TILE;
                for t in 0..n_tiles {
                    let tile_start = t * QUERY_TILE;
                    let tile_q = (tile_start + QUERY_TILE).min(seq_q) - tile_start;

                    let q_tile = &q_head[tile_start * head_dim..(tile_start + tile_q) * head_dim];
                    let head_out = &mut head_out_tile[..tile_q * head_dim];

                    if online {
                        fused_attention_online_softmax(
                            q_tile,
                            k_head,
                            v_head,
                            head_out,
                            tile_q,
                            seq_kv_eff,
                            head_dim,
                            head_dim,
                            head_dim,
                            head_dim,
                            seq_kv_eff,
                            scale,
                        );
                    } else {
                        let scores = &mut scores_tile[..tile_q * seq_kv_eff];
                        fused_attention_two_gemm_fallback(
                            q_tile,
                            k_head,
                            v_head,
                            head_out,
                            tile_q,
                            seq_kv_eff,
                            head_dim,
                            head_dim,
                            head_dim,
                            head_dim,
                            seq_kv_eff,
                            scale,
                            scores,
                        );
                    }

                    unsafe {
                        let out_buf_ptr = out_buf_addr as *mut f32;
                        for p in 0..tile_q {
                            let row = b_idx * seq_q + tile_start + p;
                            let out_base = row * hidden + h * head_dim;
                            let buf_base = p * head_dim;
                            std::ptr::copy_nonoverlapping(
                                head_out.as_ptr().add(buf_base),
                                out_buf_ptr.add(out_base),
                                head_dim,
                            );
                        }
                    }
                }
            });
    }
    let t_attn = t_attn0.elapsed();

    // Reuse the attention workspace slots for post_attn and the FF glu_proj.
    let mut post_attn = resize_buf(&mut workspace.f, m * hidden);
    let mut glu_proj = resize_buf(&mut workspace.c, m * glu_out2);

    // ---- 4. Output projection + first residual. ----
    // Initialise post_attn with the residual (+ bias) and accumulate the
    // projection, avoiding a separate elementwise pass.
    // post_attn = out_buf @ W_o + b_o + x
    let t_o_proj0 = Instant::now();
    if let Some(ref b) = o_proj_b {
        post_attn
            .par_chunks_exact_mut(hidden)
            .zip(x_slice.par_chunks_exact(hidden))
            .for_each(|(post_row, x_row)| {
                simd_ops::add2_in_place(post_row, b, x_row);
            });
    } else {
        post_attn
            .par_chunks_exact_mut(hidden)
            .zip(x_slice.par_chunks_exact(hidden))
            .for_each(|(post_row, x_row)| {
                post_row.copy_from_slice(x_row);
            });
    }
    best_row_major_accum_bf16(
        m,
        hidden,
        hidden,
        out_buf,
        hidden,
        o_proj_w,
        block.o_proj_w_bf16.as_ref(),
        &mut post_attn,
        &mut workspace.bf16_scratch,
    );
    let t_o_proj = t_o_proj0.elapsed();

    // ---- 5. FF: norm + GLU + projection + second residual. ----
    let t_ff0 = Instant::now();

    // Norm post_attn into out_buf (overwriting the attention output).
    post_attn
        .par_chunks_exact(hidden)
        .zip(out_buf.par_chunks_exact_mut(hidden))
        .for_each(|(row_x, row_n)| {
            simd_ops::layer_norm_row(row_x, ff_gamma, ff_beta, 1e-5f32, row_n);
        });

    // Try the custom fused GLU kernel first.  The kernel copies each input
    // row before overwriting the corresponding output row, so in-place
    // input/output aliasing is sound.
    let used_custom_glu = if let Some(packed) = block.ff.glu_w_packed.as_ref() {
        let len = out_buf.len();
        let ptr = out_buf.as_mut_ptr();
        unsafe {
            fused_glu_custom(
                std::slice::from_raw_parts(ptr, len),
                std::slice::from_raw_parts_mut(ptr, len),
                m,
                hidden,
                glu_out_dim,
                hidden,
                packed,
                Some(&post_attn),
            )
        }
    } else {
        false
    };

    if !used_custom_glu {
        // GLU projection (no bias), preferring the bf16 kernel when available.
        let bf16_glu = if let Some(packed) = block.ff.glu_w_bf16.as_ref() {
            if packed.k == hidden && packed.n == glu_out2 {
                bf16_gemm::linear_replace_from_f32(
                    out_buf,
                    hidden,
                    m,
                    packed,
                    &mut glu_proj,
                    &mut workspace.bf16_scratch,
                )
            } else {
                false
            }
        } else {
            false
        };
        if !bf16_glu {
            best_row_major(m, glu_out2, hidden, &out_buf, glu_w, &mut glu_proj);
        }

        // In-place GLU activation: overwrite the gate half with softplus(gate) * up.
        // The activated values remain packed as [activated_gate, up], which the
        // strided faer matmul below reads directly without a separate copy.
        glu_proj
            .par_chunks_exact_mut(glu_out2)
            .for_each(|row| simd_ops::glu_softplus_in_place_interleaved(row, glu_out_dim));

        // Output projection from the activated half of glu_proj back into out_buf.
        // Initialise out_buf with the residual (+ bias) and accumulate the
        // strided projection, avoiding a separate elementwise pass.
        // out_buf = glu_proj(strided) @ W_out + b_out + post_attn
        if let Some(ref b) = proj_out_b {
            out_buf
                .par_chunks_exact_mut(hidden)
                .zip(post_attn.par_chunks_exact(hidden))
                .for_each(|(out_row, post_row)| {
                    simd_ops::add2_in_place(out_row, b, post_row);
                });
        } else {
            out_buf
                .par_chunks_exact_mut(hidden)
                .zip(post_attn.par_chunks_exact(hidden))
                .for_each(|(out_row, post_row)| {
                    out_row.copy_from_slice(post_row);
                });
        }
        // The bf16 kernel converts the strided activation rows directly into
        // its scratch buffer, so it replaces both F32 branches below.
        let bf16_proj = if let Some(packed) = block.ff.proj_out_w_bf16.as_ref() {
            if packed.k == glu_out_dim && packed.n == hidden {
                bf16_gemm::linear_accum_from_f32(
                    &glu_proj,
                    glu_out2,
                    m,
                    packed,
                    out_buf,
                    &mut workspace.bf16_scratch,
                )
            } else {
                false
            }
        } else {
            false
        };
        if !bf16_proj {
            if m <= 64 {
                // For tiny batch sizes the strided faer path has high threading overhead.
                // Copy the activated gate half to a contiguous buffer and use gemm with
                // no parallelism.
                let glu_contig = resize_buf(&mut workspace.d, m * glu_out_dim);
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
                    out_buf,
                    gemm::Parallelism::None,
                );
            } else {
                let a = MatRef::from_row_major_slice_with_stride(&glu_proj, m, glu_out_dim, glu_out2);
                let b = MatRef::from_row_major_slice(proj_out_w, glu_out_dim, hidden);
                let mut c = MatMut::from_row_major_slice_mut(out_buf, m, hidden);
                matmul(c.as_mut(), Accum::Add, a, b, 1.0f32, Par::rayon(0));
            }
        }
    }
    let t_ff = t_ff0.elapsed();

    let t_mid = t_mid0.elapsed();
    tracing::debug!(
        "fused_hydra_mid_block total={:.3}s q_proj={:.3}s q_norm={:.3}s attn={:.3}s o_proj={:.3}s ff={:.3}s",
        t_mid.as_secs_f64(),
        t_q_proj.as_secs_f64(),
        t_q_norm.as_secs_f64(),
        t_attn.as_secs_f64(),
        t_o_proj.as_secs_f64(),
        t_ff.as_secs_f64()
    );
}

#[cfg(feature = "burn-candle")]
impl FusedHydraMidBlockBackend for burn::backend::candle::Candle {
    fn fused_hydra_mid_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::HydraMidBlock<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let k_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k));
        let v_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v));

        if x_t.dtype() != DType::F32 {
            let out = block.forward(
                x_t,
                &k_t,
                &v_t,
                mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
            );
            return match out.into_primitive() {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("HydraMidBlock returns a float tensor"),
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
            .expect("HydraMidBlock k is contiguous F32");
        let v_slice = v_data
            .as_slice::<f32>()
            .expect("HydraMidBlock v is contiguous F32");

        // Hydra pads images to max_seq_len with a contiguous suffix of invalid
        // positions. Burn's attention mask uses true = mask out, so the valid
        // prefix is the leading run of false values.
        let n_valids: Vec<usize> = if let Some(mask) = mask {
            let mask_t = Tensor::<Self, 4, Bool>::from_primitive(mask);
            let [mb, mh, mw, ms] = mask_t.dims();
            if mb != batch || mh != 1 || mw != 1 || ms != seq_kv {
                let out = block.forward(x_t, &k_t, &v_t, Some(mask_t));
                return match out.into_primitive() {
                    TensorPrimitive::Float(tensor) => tensor,
                    _ => unreachable!("HydraMidBlock returns a float tensor"),
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
                let out = block.forward(x_t, &k_t, &v_t, Some(mask_t));
                return match out.into_primitive() {
                    TensorPrimitive::Float(tensor) => tensor,
                    _ => unreachable!("HydraMidBlock returns a float tensor"),
                };
            }
            n_valids
        } else {
            vec![seq_kv; batch]
        };

        let x_data = x_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("HydraMidBlock input is contiguous F32");
        let mut out_buf = vec![0.0f32; batch * seq_q * hidden];
        let mut workspace = BlockWorkspace::new();

        fused_hydra_mid_block_to_buffer(
            x_slice,
            &mut out_buf,
            block,
            k_slice,
            v_slice,
            &n_valids,
            batch,
            seq_q,
            hidden,
            heads,
            head_dim,
            seq_kv,
            &mut workspace,
        );

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(out_buf.as_slice(), (&device, DType::F32))
            .reshape([batch, seq_q, hidden])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_hydra_mid_block returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-flex")]
impl FusedHydraMidBlockBackend for burn::backend::flex::Flex {
    fn fused_hydra_mid_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::HydraMidBlock<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        let out = block.forward(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)),
            &Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k)),
            &Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v)),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
        );
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("HydraMidBlock returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-wgpu")]
impl FusedHydraMidBlockBackend for burn::backend::Wgpu {
    fn fused_hydra_mid_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::HydraMidBlock<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        let out = block.forward(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)),
            &Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k)),
            &Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v)),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
        );
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("HydraMidBlock returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-cuda")]
impl FusedHydraMidBlockBackend for burn::backend::Cuda {
    fn fused_hydra_mid_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::HydraMidBlock<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        let out = block.forward(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)),
            &Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k)),
            &Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v)),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
        );
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("HydraMidBlock returns a float tensor"),
        }
    }
}

#[cfg(not(any(feature = "burn-candle", feature = "burn-flex")))]
impl FusedHydraMidBlockBackend for burn::backend::NdArray {
    fn fused_hydra_mid_block(
        x: FloatTensor<Self>,
        block: &crate::models::burn::hydra::modules::HydraMidBlock<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        let out = block.forward(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)),
            &Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k)),
            &Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v)),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
        );
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("HydraMidBlock returns a float tensor"),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend-specific fused HydraPool tail dispatch (FF + mid blocks)
// ---------------------------------------------------------------------------

