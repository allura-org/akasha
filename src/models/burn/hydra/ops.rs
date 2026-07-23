//! Tensor operations for the Hydra-3.5 Burn port.
//!
//! These replace einops rearrangements and PyTorch-specific primitives.

use burn::prelude::*;

/// Apply the GELU tanh approximation used by PyTorch's
/// `F.gelu(x, approximate="tanh")`.
pub fn gelu_approx_tanh<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = 0.7978845608028654;
    let coeff = 0.044715;

    let x3 = x.clone().powf_scalar(3.0);
    let inner = (x3 * coeff + x.clone()) * sqrt_2_over_pi;
    let tanh = inner.tanh();
    x * 0.5 * (tanh + 1.0)
}

/// RMS normalization over the last dimension.
pub fn rms_norm<B: Backend, const D: usize>(x: Tensor<B, D>, eps: f32) -> Tensor<B, D> {
    let var = x.clone().powf_scalar(2.0).mean_dim(D - 1);
    let norm = (var + eps).sqrt();
    x / norm
}

/// Split a fused QKV tensor into q, k, v of shape
/// `[batch, seq, n_heads * head_dim]` -> `[batch, n_heads, seq, head_dim]`.
pub fn split_qkv<B: Backend>(
    x: Tensor<B, 3>,
    n_heads: usize,
    head_dim: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, seq, _] = x.dims();
    // reshape to [batch, seq, 3, n_heads, head_dim]
    let x = x.reshape([batch, seq, 3, n_heads, head_dim]);
    // permute to [3, batch, n_heads, seq, head_dim]
    let x = x.permute([2, 0, 3, 1, 4]);
    // split on first dim, then drop the singleton dimension with an explicit
    // reshape so batch=1 doesn't accidentally squeeze the batch dimension too.
    let mut chunks: Vec<Tensor<B, 5>> = x.split_with_sizes(vec![1, 1, 1], 0);
    let reshape = |t: Tensor<B, 5>| t.reshape([batch, n_heads, seq, head_dim]);
    let v = reshape(chunks.swap_remove(2));
    let k = reshape(chunks.swap_remove(1));
    let q = reshape(chunks.swap_remove(0));
    (q, k, v)
}

/// Merge `[batch, n_heads, seq, head_dim]` -> `[batch, seq, n_heads * head_dim]`.
pub fn merge_heads<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, heads, seq, head_dim] = x.dims();
    x.permute([0, 2, 1, 3])
        .reshape([batch, seq, heads * head_dim])
}

/// Vector dot product over the last dimension, with broadcasting.
pub fn vecdot<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch, n_classes, _] = a.dims();
    (a * b).sum_dim(2).reshape([batch, n_classes])
}
