//! Burn modules for Hydra-3.5.

use burn::module::Param;
use burn::nn::{LayerNorm, Linear};
use burn::prelude::*;

use super::ops::{
    gelu_approx_tanh, glu, merge_heads, rms_norm, scaled_dot_product_attention, split_qkv, softplus,
    vecdot,
};

const NAFLEX_HEADS: usize = 16;
const NAFLEX_HEAD_DIM: usize = 72; // 1152 / 16

const HYDRA_HEADS: usize = 32;
const HYDRA_HEAD_DIM: usize = 64; // 2048 / 32

/// Full Hydra-3.5 model.
#[derive(Module, Debug)]
pub struct Hydra<B: Backend> {
    pub embeds: HydraEmbeds<B>,
    pub blocks: Vec<NaFlexBlock<B>>,
    pub norm: LayerNorm<B>,
    pub attn_pool: HydraPool<B>,
    pub head: LinearHead<B>,
}

impl<B: Backend> Hydra<B> {
    pub fn forward(
        &self,
        patches: Tensor<B, 3>,
        pos_embed: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 2> {
        let [batch, seq, _patch_dim] = patches.dims();
        let pos_embed = pos_embed.reshape([batch, seq, 1152]);
        let mask = mask.reshape([batch, seq]);

        let mut x = self.embeds.forward(patches, pos_embed, mask.clone());

        // attention mask for Burn: [batch, 1, 1, seq]
        let attn_mask = mask.reshape([batch, 1, 1, seq]);

        for block in &self.blocks {
            x = block.forward(x, attn_mask.clone());
        }

        x = self.norm.forward(x);
        x = self.attn_pool.forward(x, attn_mask);
        self.head.forward(x)
    }
}

/// Patch embedding + position embedding add.
#[derive(Module, Debug)]
pub struct HydraEmbeds<B: Backend> {
    pub proj: Linear<B>,
}

impl<B: Backend> HydraEmbeds<B> {
    pub fn forward(
        &self,
        patches: Tensor<B, 3>,
        pos_embed: Tensor<B, 3>,
        mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        // Zero out padded patch positions before projection, matching Hydra's
        // _apply_pos_embed_padded behaviour.
        let [batch, seq] = mask.dims();
        let mask = mask.reshape([batch, seq, 1]);
        let patches = patches.mask_fill(mask.bool_not(), 0.0);
        self.proj.forward(patches) + pos_embed
    }
}

#[derive(Module, Debug)]
pub struct NaFlexBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub attn: NaFlexAttn<B>,
    pub mlp: NaFlexMlp<B>,
}

impl<B: Backend> NaFlexBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.norm1.forward(x), mask);
        x.clone() + self.mlp.forward(self.norm2.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct NaFlexAttn<B: Backend> {
    pub qkv: Linear<B>,
    pub proj: Linear<B>,
}

impl<B: Backend> NaFlexAttn<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let qkv = self.qkv.forward(x);
        let (q, k, v) = split_qkv(qkv, NAFLEX_HEADS, NAFLEX_HEAD_DIM);
        let out = scaled_dot_product_attention(q, k, v, Some(mask));
        let out = merge_heads(out);
        self.proj.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct NaFlexMlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> NaFlexMlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = gelu_approx_tanh(x);
        self.fc2.forward(x)
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

impl<B: Backend> HydraPool<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let batch = x.dims()[0];
        let (k, v) = self.forward_kv(x.clone());

        let [heads, n_classes, head_dim] = self.q.dims();
        // q: [heads, n_classes, head_dim] -> [batch, heads, n_classes, head_dim]
        let q = self
            .q
            .val()
            .reshape([1, heads, n_classes, head_dim])
            .expand([batch, heads, n_classes, head_dim]);

        let out = scaled_dot_product_attention(q, k.clone(), v.clone(), Some(mask.clone()));
        let mut out = merge_heads_pool(out); // [batch, n_classes, attn_dim]

        out = out.clone() + self.ff.forward(out);

        for block in &self.mid_blocks {
            out = block.forward(out, k.clone(), v.clone(), mask.clone());
        }

        out
    }

    fn forward_kv(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq, _] = x.dims();
        let kv = self.kv.forward(x); // [batch, seq, attn_dim*2]
        // reshape to [batch, seq, 2, heads, head_dim]
        let kv = kv.reshape([batch, seq, 2, HYDRA_HEADS, HYDRA_HEAD_DIM]);
        // permute to [2, batch, heads, seq, head_dim]
        let kv = kv.permute([2, 0, 3, 1, 4]);
        // split on the leading singleton dimension, then reshape explicitly so
        // batch=1 doesn't squeeze away the batch dimension.
        let chunks: Vec<Tensor<B, 5>> = kv.split_with_sizes(vec![1, 1], 0);
        let reshape = |t: Tensor<B, 5>| t.reshape([batch, HYDRA_HEADS, seq, HYDRA_HEAD_DIM]);
        let k = self.qk_norm.forward(reshape(chunks[0].clone()));
        (k, reshape(chunks[1].clone()))
    }
}

#[derive(Module, Debug)]
pub struct HydraMidBlock<B: Backend> {
    pub q_proj: Linear<B>,
    pub q_norm: HydraRmsNorm,
    pub o_proj: Linear<B>,
    pub ff: HydraFeedForward<B>,
}

impl<B: Backend> HydraMidBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Tensor<B, 4, Bool>,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let q = self.q_proj.forward(x);
        let q = split_heads(q, HYDRA_HEADS, HYDRA_HEAD_DIM);
        let q = self.q_norm.forward(q);
        let attn = scaled_dot_product_attention(q, k, v, Some(mask));
        let attn = merge_heads_pool(attn);
        let x = residual + self.o_proj.forward(attn);
        x.clone() + self.ff.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct HydraFeedForward<B: Backend> {
    pub norm: LayerNorm<B>,
    pub proj_in_weight: Param<Tensor<B, 2>>, // [out*2, in]
    pub proj_out: Linear<B>,
}

impl<B: Backend> HydraFeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.norm.forward(x);
        let x = glu(x, self.proj_in_weight.val(), softplus);
        self.proj_out.forward(x)
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
        let [batch, n_classes, input_dim] = x.dims();
        let weight = self
            .weight
            .val()
            .reshape([1, n_classes, input_dim])
            .expand([batch, n_classes, input_dim]);
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
