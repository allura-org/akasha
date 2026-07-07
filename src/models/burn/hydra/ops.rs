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

/// Softplus with default beta=1, matching PyTorch's `F.softplus`.
pub fn softplus<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // softplus(x) = (1/beta) * log(1 + exp(beta * x))
    // For beta=1.
    let exp = x.exp();
    (exp + 1.0).log()
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
    let chunks: Vec<Tensor<B, 5>> = x.split_with_sizes(vec![1, 1, 1], 0);
    let reshape = |t: Tensor<B, 5>| t.reshape([batch, n_heads, seq, head_dim]);
    (reshape(chunks[0].clone()), reshape(chunks[1].clone()), reshape(chunks[2].clone()))
}

/// Merge `[batch, n_heads, seq, head_dim]` -> `[batch, seq, n_heads * head_dim]`.
pub fn merge_heads<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, heads, seq, head_dim] = x.dims();
    x.permute([0, 2, 1, 3])
        .reshape([batch, seq, heads * head_dim])
}

/// Compute scaled dot-product attention with a bool mask over the key dimension.
///
/// - q, k, v: `[batch, heads, seq, head_dim]`
/// - mask: broadcastable to `[batch, 1, 1, seq]` where `true` means attend.
pub fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
) -> Tensor<B, 4> {
    let [_batch, _heads, _seq_q, head_dim] = q.dims();
    let scale = (head_dim as f32).sqrt();

    // scores: [batch, heads, seq_q, seq_k]
    let k_t = k.swap_dims(2, 3);
    let mut scores = q.matmul(k_t) / scale;

    if let Some(mask) = mask {
        // mask=true => keep; mask=false => -inf
        let neg_inf = -1e9f32;
        scores = scores.mask_fill(mask.bool_not(), neg_inf);
    }

    let weights = softmax(scores, 3);
    weights.matmul(v)
}

/// Softmax over a dimension.
pub fn softmax<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    burn::tensor::activation::softmax(x, dim)
}

/// Vector dot product over the last dimension, with broadcasting.
pub fn vecdot<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch, n_classes, _] = a.dims();
    (a * b).sum_dim(2).reshape([batch, n_classes])
}

/// GLU variant used by Hydra.
///
/// `weight` has shape `[out_features * 2, in_features]`.
/// Splits the projected output into two halves, applies activation to the first,
/// and multiplies elementwise with the second.
pub fn glu<B: Backend>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 2>,
    activation: impl Fn(Tensor<B, 3>) -> Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch, _seq, _in_features] = x.dims();
    let [out2, in_features] = weight.dims();
    let out_features = out2 / 2;

    // x @ weight.T -> [batch, seq, out*2]
    let w_t = weight.swap_dims(0, 1);
    let w_t = w_t.reshape([1, in_features, out2]).expand([batch, in_features, out2]);
    let proj = x.matmul(w_t);

    let chunks: Vec<Tensor<B, 3>> = proj.split_with_sizes(vec![out_features, out_features], 2);
    let a = activation(chunks[0].clone());
    a * chunks[1].clone()
}
