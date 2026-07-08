//! Burn modules for Hydra-3.5.

use std::time::Instant;

use burn::module::Param;
use burn::nn::{LayerNorm, Linear};
use burn::prelude::*;
use burn::tensor::TensorPrimitive;

use super::fused_ops::{
    FastLinearBackend, FastRmsNormBackend, FusedAttentionBackend, FusedGluBackend,
    FusedHydraMidBlockBackend, FusedHydraPoolTailBackend, FusedMlpBackend, FusedNaFlexAttnBackend,
    FusedNaFlexBlockBackend, fused_attention, fused_hydra_pool_tail, fused_linear_glu_proj,
    fused_mlp, fused_norm_linear_glu_proj, fused_norm_mlp,
};
use super::ops::{merge_heads, rms_norm, split_qkv, vecdot};

pub const NAFLEX_HEADS: usize = 16;
pub const NAFLEX_HEAD_DIM: usize = 72; // 1152 / 16

pub const HYDRA_HEADS: usize = 32;
pub const HYDRA_HEAD_DIM: usize = 64; // 2048 / 32

fn fast_linear<B: FastLinearBackend>(x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 3> {
    let x_prim = match x.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("fast_linear input is a float tensor"),
    };
    let w_prim = match linear.weight.val().into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("fast_linear weight is a float tensor"),
    };
    let b_prim = linear
        .bias
        .as_ref()
        .map(|b| match b.val().into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("fast_linear bias is a float tensor"),
        });
    Tensor::from_primitive(TensorPrimitive::Float(B::fast_linear(
        x_prim, w_prim, b_prim,
    )))
}

/// Full Hydra-3.5 model.
#[derive(Module, Debug)]
pub struct Hydra<B: Backend> {
    pub embeds: HydraEmbeds<B>,
    pub blocks: Vec<NaFlexBlock<B>>,
    pub norm: LayerNorm<B>,
    pub attn_pool: HydraPool<B>,
    pub head: LinearHead<B>,
}

impl<
    B: FusedGluBackend
        + FusedMlpBackend
        + FusedAttentionBackend
        + FastLinearBackend
        + FastRmsNormBackend
        + FusedHydraMidBlockBackend
        + FusedHydraPoolTailBackend
        + FusedNaFlexBlockBackend,
> Hydra<B>
{
    pub fn forward(
        &self,
        patches: Tensor<B, 3>,
        pos_embed: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2> {
        let t0 = Instant::now();
        let [batch, seq, _patch_dim] = patches.dims();
        let pos_embed = pos_embed.reshape([batch, seq, 1152]);

        let mut x = self.embeds.forward(patches, pos_embed, mask.clone());
        let t_embeds = t0.elapsed();

        // Determine the actual number of valid patch positions. Hydra pads to
        // `max_seq_len` with a contiguous suffix of invalid positions, so the
        // valid prefix length is all we need for the rest of the forward pass.
        let n_valid: usize = mask
            .as_ref()
            .map(|m| {
                let data = m.to_data();
                let slice = data.as_slice::<bool>().expect("mask is contiguous bool");
                slice.iter().take(seq).filter(|&&b| b).count()
            })
            .unwrap_or(seq);

        // Burn's attention treats `true` as "mask out"; our mask semantics are
        // `true` = attend, so invert the mask.
        let attn_mask: Option<Tensor<B, 4, Bool>> =
            mask.map(|m| m.reshape([batch, 1, 1, seq]).bool_not());

        let t1 = Instant::now();
        for block in &self.blocks {
            x = block.forward(x, attn_mask.clone());
        }
        let t_blocks = t1.elapsed();

        let t2 = Instant::now();
        x = self.norm.forward(x);
        let t_norm = t2.elapsed();

        // Slice to the valid prefix before the pool. Padded positions are masked
        // out of attention anyway, so dropping them avoids wasted work in the
        // large kv projection and cross-attention without changing semantics.
        let hidden = x.dims()[2];
        let mut x = x.slice([0..batch, 0..n_valid, 0..hidden]);

        let t3 = Instant::now();
        x = self.attn_pool.forward(x, None);
        let t_pool = t3.elapsed();

        let t4 = Instant::now();
        let out = self.head.forward(x);
        let t_head = t4.elapsed();

        tracing::debug!(
            "Hydra::forward total={:.3}s embeds={:.3}s blocks={:.3}s norm={:.3}s pool={:.3}s head={:.3}s",
            t0.elapsed().as_secs_f64(),
            t_embeds.as_secs_f64(),
            t_blocks.as_secs_f64(),
            t_norm.as_secs_f64(),
            t_pool.as_secs_f64(),
            t_head.as_secs_f64()
        );

        out
    }
}

/// Patch embedding + position embedding add.
#[derive(Module, Debug)]
pub struct HydraEmbeds<B: Backend> {
    pub proj: Linear<B>,
}

impl<B: FastLinearBackend + Backend> HydraEmbeds<B> {
    pub fn forward(
        &self,
        patches: Tensor<B, 3>,
        pos_embed: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let projected = fast_linear(patches, &self.proj);
        let projected = match mask {
            Some(mask) => {
                // Zero out padded patch positions before projection, matching Hydra's
                // _apply_pos_embed_padded behaviour.
                let [batch, seq] = mask.dims();
                let mask = mask.reshape([batch, seq, 1]);
                projected.mask_fill(mask.bool_not(), 0.0)
            }
            None => projected,
        };
        projected + pos_embed
    }
}

#[derive(Module, Debug)]
pub struct NaFlexBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub attn: NaFlexAttn<B>,
    pub mlp: NaFlexMlp<B>,
}

impl<B: FusedNaFlexBlockBackend> NaFlexBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        self.forward_fused(x, mask)
    }
}

impl<B: FusedNaFlexAttnBackend> NaFlexBlock<B> {
    pub fn forward_fused_attn(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let prim = match x.into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("NaFlexAttn input is a float tensor"),
        };
        Tensor::from_primitive(TensorPrimitive::Float(B::fused_na_flex_attn(
            prim,
            self,
            mask.map(|m| m.into_primitive()),
        )))
    }
}

impl<B: FusedNaFlexBlockBackend> NaFlexBlock<B> {
    /// Backend-fused block path. Falls back to the high-level module forward
    /// for unsupported backends (via the trait default for those backends).
    pub fn forward_fused(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        let prim = match x.into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("NaFlexBlock input is a float tensor"),
        };
        Tensor::from_primitive(TensorPrimitive::Float(B::fused_na_flex_block(
            prim,
            self,
            mask.map(|m| m.into_primitive()),
        )))
    }
}

#[derive(Module, Debug)]
pub struct NaFlexAttn<B: Backend> {
    pub qkv: Linear<B>,
    pub proj: Linear<B>,

    /// Cached contiguous F32 weight/bias slices for the fused kernels.
    /// These are derived from `qkv`/`proj` at load time and skipped by Burn's
    /// `Module` derive (they are not learnable parameters).
    #[module(skip)]
    pub qkv_w_cache: Vec<f32>,
    #[module(skip)]
    pub qkv_b_cache: Option<Vec<f32>>,
    #[module(skip)]
    pub proj_w_cache: Vec<f32>,
    #[module(skip)]
    pub proj_b_cache: Option<Vec<f32>>,
}

impl<B: FastLinearBackend + FusedAttentionBackend> NaFlexAttn<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        let t0 = Instant::now();
        let qkv = fast_linear(x, &self.qkv);
        let t_qkv = t0.elapsed();

        let t1 = Instant::now();
        let (q, k, v) = split_qkv(qkv, NAFLEX_HEADS, NAFLEX_HEAD_DIM);
        let t_split = t1.elapsed();

        let t2 = Instant::now();
        let out = fused_attention(q, k, v, mask);
        let t_attn = t2.elapsed();

        let t3 = Instant::now();
        let out = merge_heads(out);
        let t_merge = t3.elapsed();

        let t4 = Instant::now();
        let out = fast_linear(out, &self.proj);
        let t_proj = t4.elapsed();

        tracing::debug!(
            "NaFlexAttn qkv={:.3}s split={:.3}s attn={:.3}s merge={:.3}s proj={:.3}s",
            t_qkv.as_secs_f64(),
            t_split.as_secs_f64(),
            t_attn.as_secs_f64(),
            t_merge.as_secs_f64(),
            t_proj.as_secs_f64()
        );
        out
    }
}

#[derive(Module, Debug)]
pub struct NaFlexMlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,

    /// Cached contiguous F32 weight/bias slices for the fused kernels.
    #[module(skip)]
    pub fc1_w_cache: Vec<f32>,
    #[module(skip)]
    pub fc1_b_cache: Option<Vec<f32>>,
    #[module(skip)]
    pub fc2_w_cache: Vec<f32>,
    #[module(skip)]
    pub fc2_b_cache: Option<Vec<f32>>,
}

impl<B: FusedMlpBackend + Backend> NaFlexMlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let t0 = Instant::now();
        let out = fused_mlp(x, self);
        tracing::debug!("NaFlexMlp fused_mlp={:.3}s", t0.elapsed().as_secs_f64());
        out
    }

    pub fn forward_fused_norm(
        &self,
        x: Tensor<B, 3>,
        residual: Tensor<B, 3>,
        norm: &burn::nn::LayerNorm<B>,
    ) -> Tensor<B, 3> {
        let t0 = Instant::now();
        let out = fused_norm_mlp(x, residual, norm, self);
        tracing::debug!(
            "NaFlexMlp fused_norm_mlp={:.3}s",
            t0.elapsed().as_secs_f64()
        );
        out
    }
}

#[derive(Module, Debug)]
pub struct HydraPool<B: Backend> {
    pub kv: Linear<B>,
    pub q: Param<Tensor<B, 3>>, // [heads, n_classes, head_dim]
    pub qk_norm: HydraRmsNorm,
    pub ff: HydraFeedForward<B>,
    pub mid_blocks: Vec<HydraMidBlock<B>>,
}

impl<
    B: FusedGluBackend
        + FusedAttentionBackend
        + FastLinearBackend
        + FastRmsNormBackend
        + FusedHydraMidBlockBackend
        + FusedHydraPoolTailBackend,
> HydraPool<B>
{
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        let t0 = Instant::now();
        let batch = x.dims()[0];
        let (k, v) = self.forward_kv(x);
        let t_kv = t0.elapsed();

        let [heads, n_classes, head_dim] = self.q.dims();
        // q: [heads, n_classes, head_dim] -> [batch, heads, n_classes, head_dim]
        let t1 = Instant::now();
        let q = self
            .q
            .val()
            .reshape([1, heads, n_classes, head_dim])
            .expand([batch, heads, n_classes, head_dim]);
        let t_q = t1.elapsed();

        let t2 = Instant::now();
        let out = fused_attention(q, k.clone(), v.clone(), mask.clone());
        let t_attn = t2.elapsed();

        let t3 = Instant::now();
        let mut out = merge_heads_pool(out); // [batch, n_classes, attn_dim]
        let t_merge = t3.elapsed();

        let t4 = Instant::now();
        out = fused_hydra_pool_tail(out, self, k, v, mask);
        let t_tail = t4.elapsed();

        tracing::debug!(
            "HydraPool kv={:.3}s q={:.3}s attn={:.3}s merge={:.3}s tail={:.3}s",
            t_kv.as_secs_f64(),
            t_q.as_secs_f64(),
            t_attn.as_secs_f64(),
            t_merge.as_secs_f64(),
            t_tail.as_secs_f64()
        );

        out
    }

    fn forward_kv(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq, _] = x.dims();
        let kv = fast_linear(x, &self.kv); // [batch, seq, attn_dim*2]
        // reshape to [batch, seq, 2, heads, head_dim]
        let kv = kv.reshape([batch, seq, 2, HYDRA_HEADS, HYDRA_HEAD_DIM]);
        // permute to [2, batch, heads, seq, head_dim]
        let kv = kv.permute([2, 0, 3, 1, 4]);
        // split on the leading singleton dimension, then reshape explicitly so
        // batch=1 doesn't squeeze away the batch dimension.
        let mut chunks: Vec<Tensor<B, 5>> = kv.split_with_sizes(vec![1, 1], 0);
        let reshape = |t: Tensor<B, 5>| t.reshape([batch, HYDRA_HEADS, seq, HYDRA_HEAD_DIM]);
        let v = reshape(chunks.swap_remove(1));
        let k = self.qk_norm.forward_fast(reshape(chunks.swap_remove(0)));
        (k, v)
    }
}

#[derive(Module, Debug)]
pub struct HydraMidBlock<B: Backend> {
    pub q_proj: Linear<B>,
    pub q_norm: HydraRmsNorm,
    pub o_proj: Linear<B>,
    pub ff: HydraFeedForward<B>,

    /// Cached contiguous F32 slices for the fused kernels.
    #[module(skip)]
    pub q_proj_w_cache: Vec<f32>,
    #[module(skip)]
    pub q_proj_b_cache: Option<Vec<f32>>,
    #[module(skip)]
    pub o_proj_w_cache: Vec<f32>,
    #[module(skip)]
    pub o_proj_b_cache: Option<Vec<f32>>,
}

impl<B: FusedGluBackend + FusedAttentionBackend + FastLinearBackend + FastRmsNormBackend>
    HydraMidBlock<B>
{
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        k: &Tensor<B, 4>,
        v: &Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let t0 = Instant::now();
        let residual = x.clone();
        let q = fast_linear(x, &self.q_proj);
        let t_q_proj = t0.elapsed();

        let t1 = Instant::now();
        let q = split_heads(q, HYDRA_HEADS, HYDRA_HEAD_DIM);
        let q = self.q_norm.forward_fast(q);
        let t_split_norm = t1.elapsed();

        let t2 = Instant::now();
        let attn = fused_attention(q, k.clone(), v.clone(), mask.clone());
        let t_attn = t2.elapsed();

        let t3 = Instant::now();
        let attn = merge_heads_pool(attn);
        let t_merge = t3.elapsed();

        let t4 = Instant::now();
        let x = residual + fast_linear(attn, &self.o_proj);
        let t_o_proj = t4.elapsed();

        let t5 = Instant::now();
        let out = x.clone() + self.ff.forward(x);
        let t_ff = t5.elapsed();

        tracing::debug!(
            "HydraMidBlock q_proj={:.3}s split_norm={:.3}s attn={:.3}s merge={:.3}s o_proj={:.3}s ff={:.3}s",
            t_q_proj.as_secs_f64(),
            t_split_norm.as_secs_f64(),
            t_attn.as_secs_f64(),
            t_merge.as_secs_f64(),
            t_o_proj.as_secs_f64(),
            t_ff.as_secs_f64()
        );

        out
    }
}

impl<B: FusedHydraMidBlockBackend> HydraMidBlock<B> {
    pub fn forward_fused(
        &self,
        x: Tensor<B, 3>,
        k: &Tensor<B, 4>,
        v: &Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let prim = match x.into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("HydraMidBlock input is a float tensor"),
        };
        let k_prim = match k.clone().into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("HydraMidBlock k is a float tensor"),
        };
        let v_prim = match v.clone().into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("HydraMidBlock v is a float tensor"),
        };
        Tensor::from_primitive(TensorPrimitive::Float(B::fused_hydra_mid_block(
            prim,
            self,
            k_prim,
            v_prim,
            mask.map(|m| m.into_primitive()),
        )))
    }
}

#[derive(Module, Debug)]
pub struct HydraFeedForward<B: Backend> {
    pub norm: LayerNorm<B>,
    pub fused_glu_weight: Param<Tensor<B, 2>>, // [in, 2*out]
    pub proj_out: Linear<B>,

    /// Cached contiguous F32 slices for the fused GLU kernels.
    #[module(skip)]
    pub glu_w_cache: Vec<f32>,
    #[module(skip)]
    pub proj_out_w_cache: Vec<f32>,
    #[module(skip)]
    pub proj_out_b_cache: Option<Vec<f32>>,
}

impl<B: FusedGluBackend + FastLinearBackend> HydraFeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        fused_norm_linear_glu_proj(x, &self.norm, self)
    }
}

#[derive(Module, Debug, Clone)]
pub struct HydraRmsNorm {
    pub eps: f32,
}

impl HydraRmsNorm {
    pub fn forward<B: Backend>(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        rms_norm(x, self.eps)
    }

    /// Fast backend-specific RMS normalization; falls back to the generic
    /// tensor implementation for unsupported backends.
    pub fn forward_fast<B: FastRmsNormBackend>(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match x.into_primitive() {
            TensorPrimitive::Float(t) => {
                Tensor::from_primitive(TensorPrimitive::Float(B::fast_rms_norm(t, self.eps)))
            }
            _ => unreachable!("rms_norm input is a float tensor"),
        }
    }
}

#[derive(Module, Debug)]
pub struct LinearHead<B: Backend> {
    pub weight: Param<Tensor<B, 2>>, // [n_classes, input_dim]
}

impl<B: Backend> LinearHead<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // x: [batch, n_classes, input_dim]
        // weight: [n_classes, input_dim]
        // vecdot over last dim -> [batch, n_classes]
        let [_, n_classes, input_dim] = x.dims();
        let weight = self.weight.val().reshape([1, n_classes, input_dim]);
        vecdot(x, weight)
    }
}

/// Reshape `[batch, seq, heads*head_dim]` -> `[batch, heads, seq, head_dim]`.
pub fn split_heads<B: Backend>(x: Tensor<B, 3>, n_heads: usize, head_dim: usize) -> Tensor<B, 4> {
    let [batch, seq, _] = x.dims();
    x.reshape([batch, seq, n_heads, head_dim])
        .permute([0, 2, 1, 3])
}

/// Merge `[batch, heads, seq, head_dim]` -> `[batch, seq, heads*head_dim]`.
fn merge_heads_pool<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, heads, seq, head_dim] = x.dims();
    x.permute([0, 2, 1, 3])
        .reshape([batch, seq, heads * head_dim])
}
