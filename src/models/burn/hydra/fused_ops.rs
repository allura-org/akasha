//! Fused GLU and attention operations for Hydra-3.5.
//!
//! This module is the CPU-optimized backend layer for Hydra. It defines a set
//! of backend traits (e.g. `FusedNaFlexBlockBackend`, `FusedHydraPoolBackend`)
//! and implements them for `burn-candle` with hand-fused, buffer-to-buffer
//! kernels. Generic building blocks — GEMM dispatch, SIMD elementwise kernels,
//! and reusable sub-kernels like `fused_attention` — live in this file so other
//! models can reuse them via `crate::models::burn::kernels` and the helpers
//! exported here.
//!
//! Hydra-specific orchestration is split into submodules to keep each file
//! focused:
//!
//! - `na_flex.rs` — `FusedNaFlexBlockBackend` and the NaFlex attention fallback.
//! - `hydra_mid.rs` — `FusedHydraMidBlockBackend`.
//! - `hydra_pool.rs` — `FusedHydraPoolBackend` and `FusedHydraPoolTailBackend`.
//!
//! # Backend / GPU portability
//!
//! The model architecture in `modules.rs` remains generic over any Burn
//! `Backend`. The traits in this module provide *optional* fast paths; if a
//! backend does not implement them, the model falls back to the generic Burn
//! tensor ops. That means a GPU backend (e.g. `burn-cubecl`, `burn-wgpu`) will
//! run correctly without any code here — it just won't take the CPU fast paths.
//! Adding a GPU fast path is a matter of implementing the same traits with
//! backend-specific kernels.
//!
//! # Adding a new fast path
//!
//! 1. Define a backend trait with the operation signature.
//! 2. Provide a generic fallback implementation using Burn's high-level ops.
//! 3. Add an optimized implementation under `#[cfg(feature = "burn-candle")]`
//!    (or the relevant backend feature).
//! 4. Register the trait bound on `HydraModel<B>` in `hydra/mod.rs`.
//! 5. Add an end-to-end test in `hydra/mod.rs` to verify accuracy.

use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::module::attention;
use burn::tensor::ops::{AttentionModuleOptions, BoolTensor, FloatTensor};
use burn::tensor::{DType, TensorPrimitive};

#[cfg(feature = "burn-flex")]
use burn::tensor::ops::ModuleOps;

#[cfg(all(feature = "burn-candle", feature = "burn-flex"))]
use crate::models::burn::kernels as simd_ops;

#[cfg(feature = "burn-flex")]
use burn::backend::flex::FlexDevice;

// Hydra-specific fused kernels. Generic building blocks (GEMM helpers, SIMD
// elementwise kernels, and reusable backend traits) stay in this module; the
// NaFlex block, HydraMidBlock, and HydraPool kernels live in submodules so each
// file stays focused and easier to navigate.
mod attention;
pub mod bf16_gemm;
#[cfg(feature = "gpu-kernels")]
pub(crate) mod gpu_kernels;
mod hydra_mid;
mod hydra_pool;
mod mlp_glu;
mod na_flex;

pub use attention::{
    fused_attention_online_softmax, fused_attention_two_gemm_fallback, use_online_softmax,
};
pub use bf16_gemm::PackedBf16Weight;
pub use hydra_mid::FusedHydraMidBlockBackend;
pub use hydra_pool::{
    FusedHydraPoolBackend, FusedHydraPoolTailBackend, fused_hydra_pool,
};
pub use mlp_glu::{
    fused_glu_custom, fused_mlp_custom, pack_glu_w, pack_mlp_fc1_w, pack_mlp_fc2_w, pack_proj_w,
    PackedGluWeights, PackedMlpWeights,
};
pub use na_flex::{FusedNaFlexAttnBackend, FusedNaFlexBlockBackend, fused_na_flex_block};

// ---------------------------------------------------------------------------
// Reusable workspace buffers for the fused Hydra kernels
// ---------------------------------------------------------------------------

/// Reusable scratch buffers for the fused Hydra kernels.
///
/// A single `BlockWorkspace` is created per forward pass and shared across all
/// NaFlex blocks and the pool tail, eliminating per-block `Vec` allocations.
pub struct BlockWorkspace {
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
    e: Vec<f32>,
    f: Vec<f32>,
    g: Vec<f32>,
    h: Vec<f32>,
    i: Vec<f32>,
    j: Vec<f32>,
    k: Vec<f32>,
    l: Vec<f32>,
    /// Reusable bf16 scratch buffer for the BF16 GEMM activation conversion.
    bf16_scratch: Vec<half::bf16>,
}

impl BlockWorkspace {
    pub fn new() -> Self {
        Self {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            e: Vec::new(),
            f: Vec::new(),
            g: Vec::new(),
            h: Vec::new(),
            i: Vec::new(),
            j: Vec::new(),
            k: Vec::new(),
            l: Vec::new(),
            bf16_scratch: Vec::new(),
        }
    }

}

impl Default for BlockWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Tile size for query rows in the fused NaFlex attention kernel.
const QUERY_TILE: usize = 64;

/// Resize a single workspace buffer and return a mutable slice of the requested
/// length. Using the raw `Vec` field allows the borrow checker to see that
/// distinct workspace slots are borrowed independently.
#[inline]
fn resize_buf(buf: &mut Vec<f32>, len: usize) -> &mut [f32] {
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
    &mut buf[..len]
}

// ---------------------------------------------------------------------------
// Low-level GEMM helper (pure Rust, no C/C++)
// ---------------------------------------------------------------------------

/// Wrapper around the `gemm` crate's pure-Rust GEMM.
///
/// All strides are in units of elements. Row-major storage uses
/// `row_stride = ncols`, `col_stride = 1`. For `C = A @ B` set
/// `read_dst = false`; existing `c` contents are then ignored.
#[inline]
unsafe fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    rsa: isize,
    csa: isize,
    b: &[f32],
    rsb: isize,
    csb: isize,
    c: &mut [f32],
    rsc: isize,
    csc: isize,
    par: gemm::Parallelism,
) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            csc,
            rsc,
            false,
            a.as_ptr(),
            csa,
            rsa,
            b.as_ptr(),
            csb,
            rsb,
            0.0,
            1.0,
            false,
            false,
            false,
            par,
        );
    }
}

/// General GEMM with arbitrary strides and an `A @ B` scale factor.
///
/// The `gemm` crate names its scalar arguments as `(dst_scale, ab_scale)` in
/// this order: `dst = ab_scale * (A @ B) + dst_scale * dst` (with `read_dst`
/// always false here, so `dst_scale` is ignored).
#[inline]
pub(super) unsafe fn gemm_f32_ex(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    rsa: isize,
    csa: isize,
    b: &[f32],
    rsb: isize,
    csb: isize,
    c: &mut [f32],
    rsc: isize,
    csc: isize,
    ab_scale: f32,
    par: gemm::Parallelism,
) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            csc,
            rsc,
            false,
            a.as_ptr(),
            csa,
            rsa,
            b.as_ptr(),
            csb,
            rsb,
            0.0,
            ab_scale,
            false,
            false,
            false,
            par,
        );
    }
}

/// `C = A @ B` with row-major A, B, C.
#[inline]
fn gemm_row_major(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    par: gemm::Parallelism,
) {
    unsafe {
        gemm_f32(
            m, n, k, a, k as isize, 1, b, n as isize, 1, c, n as isize, 1, par,
        );
    }
}

/// `C = scale * (A @ B^T)` where A is row-major `[m, k]` and `b_t` is row-major `[n, k]`.
#[inline]
fn gemm_a_bt_scaled(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    scale: f32,
    par: gemm::Parallelism,
) {
    unsafe {
        gemm_f32_ex(
            m, n, k, a, k as isize, 1, b_t, 1, k as isize, c, n as isize, 1, scale, par,
        );
    }
}

/// General GEMM accumulation: `C += A @ B` with arbitrary strides.
///
/// Existing contents of `C` are read and accumulated into.
#[inline]
unsafe fn gemm_f32_accum(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    rsa: isize,
    csa: isize,
    b: &[f32],
    rsb: isize,
    csb: isize,
    c: &mut [f32],
    rsc: isize,
    csc: isize,
    par: gemm::Parallelism,
) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            csc,
            rsc,
            true, // read_dst
            a.as_ptr(),
            csa,
            rsa,
            b.as_ptr(),
            csb,
            rsb,
            1.0, // dst_scale
            1.0, // ab_scale
            false,
            false,
            false,
            par,
        );
    }
}

/// `C += A @ B` with row-major A, B, C. Existing C contents are read and accumulated.
#[inline]
fn gemm_row_major_accum(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    par: gemm::Parallelism,
) {
    unsafe {
        gemm_f32_accum(
            m, n, k, a, k as isize, 1, b, n as isize, 1, c, n as isize, 1, par,
        );
    }
}

/// Dispatch to the fastest pure-Rust GEMM for `C += A @ B`.
///
/// The caller must initialize `c` before calling; this routine only accumulates
/// the matrix product into the existing contents.
#[inline]
fn best_row_major_accum(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if (m as u64) * (n as u64) * (k as u64) >= 20_000_000_000u64 {
        use faer::linalg::matmul::matmul;
        use faer::{Accum, MatMut, MatRef, Par};
        let a_ref = MatRef::from_row_major_slice(a, m, k);
        let b_ref = MatRef::from_row_major_slice(b, k, n);
        let mut c_mut = MatMut::from_row_major_slice_mut(c, m, n);
        matmul(
            c_mut.as_mut(),
            Accum::Add,
            a_ref,
            b_ref,
            1.0f32,
            Par::rayon(0),
        );
    } else {
        gemm_row_major_accum(m, n, k, a, b, c, gemm::Parallelism::Rayon(0));
    }
}

/// `C += A @ W`, preferring the BF16 kernel when a packed bf16 weight is
/// available for this layer.
///
/// `a` is an `[m, k]` F32 activation matrix with row stride `a_stride`
/// (contiguous when `a_stride == k`). `w_f32` is the fallback row-major
/// `[k, n]` F32 weight used when there is no packed bf16 weight or the shapes
/// do not match; the fallback requires `a_stride == k`. The caller must
/// pre-fill `c` (bias/residual epilogue) before calling.
#[inline]
#[allow(clippy::too_many_arguments)]
fn best_row_major_accum_bf16(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    a_stride: usize,
    w_f32: &[f32],
    w_bf16: Option<&PackedBf16Weight>,
    c: &mut [f32],
    scratch: &mut Vec<half::bf16>,
) {
    if let Some(packed) = w_bf16 {
        if packed.k == k
            && packed.n == n
            && bf16_gemm::linear_accum_from_f32(a, a_stride, m, packed, c, scratch)
        {
            return;
        }
    }
    debug_assert_eq!(a_stride, k, "F32 GEMM fallback requires contiguous A");
    best_row_major_accum(m, n, k, a, w_f32, c);
}

/// Dispatch to the fastest pure-Rust GEMM for the given shape.
///
/// Benchmarks showed `faer` wins on very large square-ish matmuls while `gemm`
/// is faster for small/medium and attention-sized shapes.
#[inline]
fn best_row_major(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if (m as u64) * (n as u64) * (k as u64) >= 20_000_000_000u64 {
        use faer::linalg::matmul::matmul;
        use faer::{Accum, MatMut, MatRef, Par};
        let a_ref = MatRef::from_row_major_slice(a, m, k);
        let b_ref = MatRef::from_row_major_slice(b, k, n);
        let mut c_mut = MatMut::from_row_major_slice_mut(c, m, n);
        matmul(
            c_mut.as_mut(),
            Accum::Replace,
            a_ref,
            b_ref,
            1.0f32,
            Par::rayon(0),
        );
    } else {
        gemm_row_major(m, n, k, a, b, c, gemm::Parallelism::Rayon(0));
    }
}

// ---------------------------------------------------------------------------
// Backend-specific fast linear dispatch
// ---------------------------------------------------------------------------

/// Backends that provide a fast linear path with optional pure-Rust GEMM acceleration.
pub trait FastLinearBackend: Backend {
    /// Compute `x @ weight + bias`.
    ///
    /// * `x`:      `[batch, seq, in_features]`
    /// * `weight`: `[in_features, out_features]`
    /// * `bias`:   optional `[out_features]`
    /// * returns:  `[batch, seq, out_features]`
    fn fast_linear(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        match fast_linear_fallback_tensor(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)),
            Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(weight)),
            bias.map(|b| Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(b))),
        )
        .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fast_linear returns a float tensor"),
        }
    }
}

/// Generic fallback implementation using Burn's high-level tensor API.
fn fast_linear_fallback_tensor<B: Backend>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Tensor<B, 3> {
    let [in_features, out_features] = weight.dims();
    let weight = weight.reshape([1, in_features, out_features]);
    let out = x.matmul(weight);
    match bias {
        Some(bias) => out + bias.reshape([1, 1, out_features]),
        None => out,
    }
}

#[cfg(feature = "burn-candle")]
impl FastLinearBackend for burn::backend::candle::Candle {
    fn fast_linear(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let w_t = Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(weight));

        let [batch, seq, k] = x_t.dims();
        let [k2, n] = w_t.dims();
        assert_eq!(k, k2, "weight in_features must match input");
        let m = batch * seq;

        // Only accelerate large F32 matmuls; fall back to Burn for small or non-F32 shapes.
        if x_t.dtype() != DType::F32 || m * n * k <= 20_000_000_000 {
            return match fast_linear_fallback_tensor(
                x_t,
                w_t,
                bias.map(|b| Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(b))),
            )
            .into_primitive()
            {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("fast_linear returns a float tensor"),
            };
        }

        let x_data = x_t.to_data();
        let w_data = w_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("fast_linear input is contiguous F32");
        let w_slice = w_data
            .as_slice::<f32>()
            .expect("fast_linear weight is contiguous F32");

        let mut out = vec![0.0f32; m * n];
        best_row_major(m, n, k, x_slice, w_slice, &mut out);

        if let Some(bias) = bias {
            let bias_t = Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(bias));
            let bias_data = bias_t.to_data();
            let bias_slice = bias_data
                .as_slice::<f32>()
                .expect("fast_linear bias is contiguous F32");
            for i in 0..m {
                let base = i * n;
                for j in 0..n {
                    out[base + j] += bias_slice[j];
                }
            }
        }

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(out.as_slice(), (&device, DType::F32))
            .reshape([batch, seq, n])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fast_linear returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-flex")]
impl FastLinearBackend for burn::backend::flex::Flex {}

#[cfg(feature = "burn-wgpu")]
impl FastLinearBackend for burn::backend::Wgpu {}

#[cfg(feature = "burn-cuda")]
impl FastLinearBackend for burn::backend::Cuda {}

#[cfg(not(any(feature = "burn-candle", feature = "burn-flex")))]
impl FastLinearBackend for burn::backend::NdArray {}

// ---------------------------------------------------------------------------
// Backend-specific fused MLP dispatch (fc1 -> gelu -> fc2)
// ---------------------------------------------------------------------------

/// Backends that provide a fused MLP path.
pub trait FusedMlpBackend: Backend {
    /// Compute `fc2(gelu(fc1(x)))` as a single dispatch.
    ///
    /// * `x`:  `[batch, seq, in_features]`
    /// * `mlp`: NaFlexMlp module (cached weights are used by fast backends)
    /// * returns: `[batch, seq, out_features]`
    fn fused_mlp(x: FloatTensor<Self>, mlp: &super::modules::NaFlexMlp<Self>) -> FloatTensor<Self> {
        let fc1_weight = match mlp.fc1.weight.val().into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("fc1 weight is a float tensor"),
        };
        let fc1_bias = mlp
            .fc1
            .bias
            .as_ref()
            .map(|b| match b.val().into_primitive() {
                TensorPrimitive::Float(t) => t,
                _ => unreachable!("fc1 bias is a float tensor"),
            });
        let fc2_weight = match mlp.fc2.weight.val().into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("fc2 weight is a float tensor"),
        };
        let fc2_bias = mlp
            .fc2
            .bias
            .as_ref()
            .map(|b| match b.val().into_primitive() {
                TensorPrimitive::Float(t) => t,
                _ => unreachable!("fc2 bias is a float tensor"),
            });
        match fused_mlp_fallback_tensor(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)),
            Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(fc1_weight)),
            fc1_bias.map(|b| Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(b))),
            Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(fc2_weight)),
            fc2_bias.map(|b| Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(b))),
        )
        .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_mlp returns a float tensor"),
        }
    }

    /// Compute `residual + fc2(gelu(fc1(layer_norm(x, norm))))` as a single dispatch.
    ///
    /// * `x`:       `[batch, seq, in_features]` (input to layer norm)
    /// * `residual`:`[batch, seq, out_features]` (added to MLP output)
    /// * `norm`:    LayerNorm module (gamma + optional beta)
    /// * `mlp`:     NaFlexMlp module (cached weights are used by fast backends)
    /// * returns:   `[batch, seq, out_features]`
    fn fused_norm_mlp(
        x: FloatTensor<Self>,
        residual: FloatTensor<Self>,
        norm: &burn::nn::LayerNorm<Self>,
        mlp: &super::modules::NaFlexMlp<Self>,
    ) -> FloatTensor<Self> {
        let normalized = norm.forward(Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)));
        let mlp_out = Self::fused_mlp(
            match normalized.into_primitive() {
                TensorPrimitive::Float(t) => t,
                _ => unreachable!("layer_norm returns a float tensor"),
            },
            mlp,
        );
        match (Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(mlp_out))
            + Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(residual)))
        .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_norm_mlp returns a float tensor"),
        }
    }
}

/// Tensor-level entry point for the fused MLP path.
#[allow(dead_code)]
pub fn fused_mlp<B: FusedMlpBackend>(
    x: Tensor<B, 3>,
    mlp: &super::modules::NaFlexMlp<B>,
) -> Tensor<B, 3> {
    let x_prim = match x.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("fused_mlp input is a float tensor"),
    };

    Tensor::from_primitive(TensorPrimitive::Float(B::fused_mlp(x_prim, mlp)))
}

/// Tensor-level entry point for the fused norm + MLP + residual path.
pub fn fused_norm_mlp<B: FusedMlpBackend>(
    x: Tensor<B, 3>,
    residual: Tensor<B, 3>,
    norm: &burn::nn::LayerNorm<B>,
    mlp: &super::modules::NaFlexMlp<B>,
) -> Tensor<B, 3> {
    let x_prim = match x.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("fused_norm_mlp input is a float tensor"),
    };
    let residual_prim = match residual.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("fused_norm_mlp residual is a float tensor"),
    };

    Tensor::from_primitive(TensorPrimitive::Float(B::fused_norm_mlp(
        x_prim,
        residual_prim,
        norm,
        mlp,
    )))
}

/// Generic fallback implementation using Burn's high-level tensor API.
fn fused_mlp_fallback_tensor<B: Backend>(
    x: Tensor<B, 3>,
    fc1_weight: Tensor<B, 2>,
    fc1_bias: Option<Tensor<B, 1>>,
    fc2_weight: Tensor<B, 2>,
    fc2_bias: Option<Tensor<B, 1>>,
) -> Tensor<B, 3> {
    use super::ops::gelu_approx_tanh;

    let [in_features, hidden_features] = fc1_weight.dims();
    let [hidden2, out_features] = fc2_weight.dims();
    assert_eq!(
        hidden_features, hidden2,
        "fc1 hidden must match fc2 in features"
    );

    let fc1_weight = fc1_weight.reshape([1, in_features, hidden_features]);
    let mut x = x.matmul(fc1_weight);
    if let Some(bias) = fc1_bias {
        x = x + bias.reshape([1, 1, hidden_features]);
    }
    x = gelu_approx_tanh(x);

    let fc2_weight = fc2_weight.reshape([1, hidden_features, out_features]);
    let mut out = x.matmul(fc2_weight);
    if let Some(bias) = fc2_bias {
        out = out + bias.reshape([1, 1, out_features]);
    }
    out
}

#[cfg(feature = "burn-candle")]
impl FusedMlpBackend for burn::backend::candle::Candle {
    fn fused_mlp(x: FloatTensor<Self>, mlp: &super::modules::NaFlexMlp<Self>) -> FloatTensor<Self> {
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));

        let [batch, seq, k] = x_t.dims();
        let hidden = mlp.fc1_w_cache.len() / k;
        let n = mlp.fc2_w_cache.len() / hidden;
        assert_eq!(mlp.fc1_w_cache.len(), k * hidden);
        assert_eq!(mlp.fc2_w_cache.len(), hidden * n);
        let m = batch * seq;

        // Only accelerate F32 matmuls; fall back to Burn for other dtypes.
        if x_t.dtype() != DType::F32 {
            let fc1_w_t = match mlp.fc1.weight.val().into_primitive() {
                TensorPrimitive::Float(t) => t,
                _ => unreachable!("fc1 weight is a float tensor"),
            };
            let fc1_b_t = mlp
                .fc1
                .bias
                .as_ref()
                .map(|b| match b.val().into_primitive() {
                    TensorPrimitive::Float(t) => t,
                    _ => unreachable!("fc1 bias is a float tensor"),
                });
            let fc2_w_t = match mlp.fc2.weight.val().into_primitive() {
                TensorPrimitive::Float(t) => t,
                _ => unreachable!("fc2 weight is a float tensor"),
            };
            let fc2_b_t = mlp
                .fc2
                .bias
                .as_ref()
                .map(|b| match b.val().into_primitive() {
                    TensorPrimitive::Float(t) => t,
                    _ => unreachable!("fc2 bias is a float tensor"),
                });
            return match fused_mlp_fallback_tensor(
                x_t,
                Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(fc1_w_t)),
                fc1_b_t.map(|b| Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(b))),
                Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(fc2_w_t)),
                fc2_b_t.map(|b| Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(b))),
            )
            .into_primitive()
            {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("fused_mlp returns a float tensor"),
            };
        }

        let x_data = x_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("fused_mlp input is contiguous F32");
        let fc1_w_slice = mlp.fc1_w_cache.as_slice();
        let fc2_w_slice = mlp.fc2_w_cache.as_slice();
        let fc1_b_slice = mlp.fc1_b_cache.as_deref();
        let fc2_b_slice = mlp.fc2_b_cache.as_deref();

        let mut tmp = vec![0.0f32; m * hidden];
        best_row_major(m, hidden, k, x_slice, fc1_w_slice, &mut tmp);

        if let Some(bias) = fc1_b_slice {
            for i in 0..m {
                let base = i * hidden;
                for j in 0..hidden {
                    tmp[base + j] += bias[j];
                }
            }
        }

        // Apply GELU tanh approximation in-place.
        for v in tmp.iter_mut() {
            *v = gelu_approx_tanh_f32(*v);
        }

        let mut out = vec![0.0f32; m * n];
        best_row_major(m, n, hidden, &tmp, fc2_w_slice, &mut out);

        if let Some(bias) = fc2_b_slice {
            for i in 0..m {
                let base = i * n;
                for j in 0..n {
                    out[base + j] += bias[j];
                }
            }
        }

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(out.as_slice(), (&device, DType::F32))
            .reshape([batch, seq, n])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_mlp returns a float tensor"),
        }
    }

    fn fused_norm_mlp(
        x: FloatTensor<Self>,
        residual: FloatTensor<Self>,
        norm: &burn::nn::LayerNorm<Self>,
        mlp: &super::modules::NaFlexMlp<Self>,
    ) -> FloatTensor<Self> {
        use rayon::prelude::*;

        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let residual_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(residual));

        let [batch, seq, k] = x_t.dims();
        let [res_batch, res_seq, n] = residual_t.dims();
        assert_eq!(batch, res_batch, "x and residual batch sizes must match");
        assert_eq!(seq, res_seq, "x and residual seq lengths must match");
        let hidden = mlp.fc1_w_cache.len() / k;
        let n2 = mlp.fc2_w_cache.len() / hidden;
        assert_eq!(mlp.fc1_w_cache.len(), k * hidden);
        assert_eq!(mlp.fc2_w_cache.len(), hidden * n2);
        assert_eq!(n, n2, "fc2 weight out_features must match residual");
        let m = batch * seq;

        if x_t.dtype() != DType::F32 {
            let normalized = norm.forward(x_t);
            return Self::fused_norm_mlp(
                match normalized.into_primitive() {
                    TensorPrimitive::Float(t) => t,
                    _ => unreachable!("layer_norm returns a float tensor"),
                },
                match residual_t.into_primitive() {
                    TensorPrimitive::Float(t) => t,
                    _ => unreachable!("residual is a float tensor"),
                },
                norm,
                mlp,
            );
        }

        let eps = 1e-5f32;
        let gamma_data = norm.gamma.val().to_data();
        let gamma = gamma_data
            .as_slice::<f32>()
            .expect("norm gamma is contiguous F32");
        let beta_data = norm.beta.as_ref().map(|b| b.val().to_data());
        let beta = beta_data
            .as_ref()
            .map(|d| d.as_slice::<f32>().expect("norm beta is contiguous F32"));

        let x_data = x_t.to_data();
        let residual_data = residual_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("fused_norm_mlp input is contiguous F32");
        let residual_slice = residual_data
            .as_slice::<f32>()
            .expect("fused_norm_mlp residual is contiguous F32");
        let fc1_w_slice = mlp.fc1_w_cache.as_slice();
        let fc2_w_slice = mlp.fc2_w_cache.as_slice();
        let fc1_b_slice = mlp.fc1_b_cache.as_deref();
        let fc2_b_slice = mlp.fc2_b_cache.as_deref();

        // Normalize x into a temporary buffer.
        let mut normed = vec![0.0f32; m * k];
        x_slice
            .par_chunks_exact(k)
            .zip(normed.par_chunks_exact_mut(k))
            .for_each(|(row_x, row_n)| {
                let mean = row_x.iter().copied().sum::<f32>() / k as f32;
                let var = row_x
                    .iter()
                    .map(|v| {
                        let d = *v - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / k as f32;
                let inv_std = 1.0f32 / (var + eps).sqrt();
                for j in 0..k {
                    row_n[j] = (row_x[j] - mean) * inv_std * gamma[j]
                        + beta.map(|b| b[j]).unwrap_or(0.0f32);
                }
            });

        let mut tmp = vec![0.0f32; m * hidden];
        best_row_major(m, hidden, k, &normed, fc1_w_slice, &mut tmp);

        if let Some(bias) = fc1_b_slice {
            tmp.par_chunks_exact_mut(hidden).for_each(|row| {
                for j in 0..hidden {
                    row[j] += bias[j];
                }
            });
        }

        tmp.par_iter_mut().for_each(|v| {
            *v = gelu_approx_tanh_f32(*v);
        });

        let device = x_t.device();

        let mut out = vec![0.0f32; m * n];
        best_row_major(m, n, hidden, tmp.as_slice(), fc2_w_slice, &mut out);

        if let Some(bias) = fc2_b_slice {
            out.par_chunks_exact_mut(n).for_each(|row| {
                for j in 0..n {
                    row[j] += bias[j];
                }
            });
        }

        // Add residual.
        out.par_chunks_exact_mut(n)
            .zip(residual_slice.par_chunks_exact(n))
            .for_each(|(out_row, res_row)| {
                for j in 0..n {
                    out_row[j] += res_row[j];
                }
            });

        match Tensor::<Self, 1>::from_data(out.as_slice(), (&device, DType::F32))
            .reshape([batch, seq, n])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_norm_mlp returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-flex")]
impl FusedMlpBackend for burn::backend::flex::Flex {}

#[cfg(feature = "burn-wgpu")]
impl FusedMlpBackend for burn::backend::Wgpu {}

#[cfg(feature = "burn-cuda")]
impl FusedMlpBackend for burn::backend::Cuda {}

#[cfg(not(any(feature = "burn-candle", feature = "burn-flex")))]
impl FusedMlpBackend for burn::backend::NdArray {}

// ---------------------------------------------------------------------------
// Backend-specific fused linear + GLU path
// ---------------------------------------------------------------------------

/// Backends that provide a fused linear + GLU path.
pub trait FusedGluBackend: FastLinearBackend {
    /// Compute `softplus(x @ weight[..., :out]) * (x @ weight[..., out:])`.
    ///
    /// * `x`:      `[batch, seq, in_features]`
    /// * `weight`: `[in_features, 2 * out_features]`
    /// * returns:  `[batch, seq, out_features]`
    fn fused_linear_glu(x: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self> {
        match fused_linear_glu_fallback_tensor(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)),
            Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(weight)),
        )
        .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_linear_glu returns a float tensor"),
        }
    }

    /// Compute `softplus(x @ glu_weight[..., :out]) * (x @ glu_weight[..., out:]) @ proj_weight + proj_bias`.
    ///
    /// * `x`:           `[batch, seq, in_features]`
    /// * `glu_weight`:  `[in_features, 2 * out_features]`
    /// * `proj_weight`: `[out_features, out_proj_features]`
    /// * `proj_bias`:   optional `[out_proj_features]`
    /// * returns:       `[batch, seq, out_proj_features]`
    fn fused_linear_glu_proj(
        x: FloatTensor<Self>,
        glu_weight: FloatTensor<Self>,
        proj_weight: FloatTensor<Self>,
        proj_bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        // Dispatch the GLU part through the backend-specific override (e.g. Flex's
        // manual loop) and only do the output projection with generic Burn ops.
        let glu = Self::fused_linear_glu(x, glu_weight);
        match fused_linear_glu_proj_matmul_tensor(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(glu)),
            Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(proj_weight)),
            proj_bias.map(|b| Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(b))),
        )
        .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_linear_glu_proj returns a float tensor"),
        }
    }

    /// Compute `fused_linear_glu_proj(layer_norm(x, norm), ...)` in one dispatch.
    ///
    /// * `x`:    `[batch, seq, in_features]`
    /// * `norm`: LayerNorm module (gamma + optional beta)
    /// * `ff`:   HydraFeedForward module (cached weights are used by fast backends)
    /// * returns:`[batch, seq, out_proj_features]`
    fn fused_norm_linear_glu_proj(
        x: FloatTensor<Self>,
        norm: &burn::nn::LayerNorm<Self>,
        ff: &super::modules::HydraFeedForward<Self>,
    ) -> FloatTensor<Self> {
        let glu_weight = match ff.fused_glu_weight.val().into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("glu_weight is a float tensor"),
        };
        let proj_weight = match ff.proj_out.weight.val().into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!("proj weight is a float tensor"),
        };
        let proj_bias = ff
            .proj_out
            .bias
            .as_ref()
            .map(|b| match b.val().into_primitive() {
                TensorPrimitive::Float(t) => t,
                _ => unreachable!("proj bias is a float tensor"),
            });
        let normalized = norm.forward(Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)));
        Self::fused_linear_glu_proj(
            match normalized.into_primitive() {
                TensorPrimitive::Float(t) => t,
                _ => unreachable!("layer_norm returns a float tensor"),
            },
            glu_weight,
            proj_weight,
            proj_bias,
        )
    }
}

/// Tensor-level entry point for the fused norm + GLU + output projection path.
pub fn fused_norm_linear_glu_proj<B: FusedGluBackend>(
    x: Tensor<B, 3>,
    norm: &burn::nn::LayerNorm<B>,
    ff: &super::modules::HydraFeedForward<B>,
) -> Tensor<B, 3> {
    let x_prim = match x.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("fused_norm_linear_glu_proj input is a float tensor"),
    };

    Tensor::from_primitive(TensorPrimitive::Float(B::fused_norm_linear_glu_proj(
        x_prim, norm, ff,
    )))
}

/// Generic fallback implementation using Burn's high-level tensor API.
fn fused_linear_glu_fallback_tensor<B: Backend>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [in_features, out2] = weight.dims();
    let out = out2 / 2;

    let weight = weight.reshape([1, in_features, out2]);
    let proj = x.matmul(weight);

    let mut chunks = proj.split_with_sizes(vec![out, out], 2);
    let up = chunks.swap_remove(1);
    let gate = activation::softplus(chunks.swap_remove(0), 1.0);

    gate * up
}

/// Output projection half of fused GLU + output projection using generic Burn ops.
fn fused_linear_glu_proj_matmul_tensor<B: Backend>(
    glu: Tensor<B, 3>,
    proj_weight: Tensor<B, 2>,
    proj_bias: Option<Tensor<B, 1>>,
) -> Tensor<B, 3> {
    let [proj_in, proj_out] = proj_weight.dims();

    let proj_weight = proj_weight.reshape([1, proj_in, proj_out]);
    let mut out = glu.matmul(proj_weight);
    if let Some(bias) = proj_bias {
        out = out + bias.reshape([1, 1, proj_out]);
    }
    out
}

/// Generic fallback for fused GLU + output projection.
fn fused_linear_glu_proj_fallback_tensor<B: Backend>(
    x: Tensor<B, 3>,
    glu_weight: Tensor<B, 2>,
    proj_weight: Tensor<B, 2>,
    proj_bias: Option<Tensor<B, 1>>,
) -> Tensor<B, 3> {
    let glu = fused_linear_glu_fallback_tensor(x, glu_weight);
    fused_linear_glu_proj_matmul_tensor(glu, proj_weight, proj_bias)
}

#[cfg(feature = "burn-candle")]
impl FusedGluBackend for burn::backend::candle::Candle {
    fn fused_linear_glu(x: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self> {
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let w_t = Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(weight));

        let [batch, seq, k] = x_t.dims();
        let [k2, out2] = w_t.dims();
        assert_eq!(k, k2, "weight in_features must match input");
        let out = out2 / 2;
        let m = batch * seq;

        // Only accelerate large F32 matmuls; fall back to Burn for small or non-F32 shapes.
        if x_t.dtype() != DType::F32 || m * out2 * k <= 20_000_000_000 {
            return match fused_linear_glu_fallback_tensor(x_t, w_t).into_primitive() {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("fused_linear_glu returns a float tensor"),
            };
        }

        let x_data = x_t.to_data();
        let w_data = w_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("fused_linear_glu input is contiguous F32");
        let w_slice = w_data
            .as_slice::<f32>()
            .expect("fused_linear_glu weight is contiguous F32");

        let mut proj = vec![0.0f32; m * out2];
        best_row_major(m, out2, k, x_slice, w_slice, &mut proj);

        // In-place convert the full projection to GLU output:
        // output[i, j] = softplus(proj[i, j]) * proj[i, out + j]
        let mut output = vec![0.0f32; m * out];
        for i in 0..m {
            let base = i * out2;
            let dst_base = i * out;
            for j in 0..out {
                let gate = softplus_f32(proj[base + j]);
                output[dst_base + j] = gate * proj[base + out + j];
            }
        }

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(output.as_slice(), (&device, DType::F32))
            .reshape([batch, seq, out])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_linear_glu returns a float tensor"),
        }
    }

    fn fused_linear_glu_proj(
        x: FloatTensor<Self>,
        glu_weight: FloatTensor<Self>,
        proj_weight: FloatTensor<Self>,
        proj_bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));
        let glu_w_t = Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(glu_weight));
        let proj_w_t = Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(proj_weight));

        let [batch, seq, k] = x_t.dims();
        let [k2, out2] = glu_w_t.dims();
        let [proj_in, proj_out_dim] = proj_w_t.dims();
        assert_eq!(k, k2, "glu_weight in_features must match input");
        let out = out2 / 2;
        assert_eq!(out, proj_in, "proj_weight in_features must match glu out");
        let m = batch * seq;

        // Only accelerate F32 matmuls; fall back to Burn for other dtypes.
        if x_t.dtype() != DType::F32 {
            return match fused_linear_glu_proj_fallback_tensor(
                x_t,
                glu_w_t,
                proj_w_t,
                proj_bias.map(|b| Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(b))),
            )
            .into_primitive()
            {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("fused_linear_glu_proj returns a float tensor"),
            };
        }

        let x_data = x_t.to_data();
        let glu_w_data = glu_w_t.to_data();
        let proj_w_data = proj_w_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("fused_linear_glu_proj input is contiguous F32");
        let glu_w_slice = glu_w_data
            .as_slice::<f32>()
            .expect("glu_weight is contiguous F32");
        let proj_w_slice = proj_w_data
            .as_slice::<f32>()
            .expect("proj_weight is contiguous F32");

        let mut glu_proj = vec![0.0f32; m * out2];
        best_row_major(m, out2, k, x_slice, glu_w_slice, &mut glu_proj);

        let mut glu_out = vec![0.0f32; m * out];
        for i in 0..m {
            let base = i * out2;
            let dst_base = i * out;
            for j in 0..out {
                let gate = softplus_f32(glu_proj[base + j]);
                glu_out[dst_base + j] = gate * glu_proj[base + out + j];
            }
        }

        let mut output = vec![0.0f32; m * proj_out_dim];
        best_row_major(m, proj_out_dim, out, &glu_out, proj_w_slice, &mut output);

        if let Some(bias) = proj_bias {
            let bias_t = Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(bias));
            let bias_data = bias_t.to_data();
            let bias_slice = bias_data
                .as_slice::<f32>()
                .expect("proj_bias is contiguous F32");
            for i in 0..m {
                let base = i * proj_out_dim;
                for j in 0..proj_out_dim {
                    output[base + j] += bias_slice[j];
                }
            }
        }

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(output.as_slice(), (&device, DType::F32))
            .reshape([batch, seq, proj_out_dim])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_linear_glu_proj returns a float tensor"),
        }
    }

    fn fused_norm_linear_glu_proj(
        x: FloatTensor<Self>,
        norm: &burn::nn::LayerNorm<Self>,
        ff: &super::modules::HydraFeedForward<Self>,
    ) -> FloatTensor<Self> {
        use faer::linalg::matmul::matmul;
        use faer::{Accum, MatMut, MatRef, Par};
        use rayon::prelude::*;

        let x_t = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x));

        let [batch, seq, k] = x_t.dims();
        let out2 = ff.glu_w_cache.len() / k;
        let out = out2 / 2;
        let proj_out_dim = ff.proj_out_w_cache.len() / out;
        assert_eq!(ff.glu_w_cache.len(), k * out2);
        assert_eq!(ff.proj_out_w_cache.len(), out * proj_out_dim);
        let m = batch * seq;

        if x_t.dtype() != DType::F32 {
            let normalized = norm.forward(x_t);
            return Self::fused_linear_glu_proj(
                match normalized.into_primitive() {
                    TensorPrimitive::Float(t) => t,
                    _ => unreachable!("layer_norm returns a float tensor"),
                },
                match ff.fused_glu_weight.val().into_primitive() {
                    TensorPrimitive::Float(t) => t,
                    _ => unreachable!("glu_weight is a float tensor"),
                },
                match ff.proj_out.weight.val().into_primitive() {
                    TensorPrimitive::Float(t) => t,
                    _ => unreachable!("proj weight is a float tensor"),
                },
                ff.proj_out
                    .bias
                    .as_ref()
                    .map(|b| match b.val().into_primitive() {
                        TensorPrimitive::Float(t) => t,
                        _ => unreachable!("proj bias is a float tensor"),
                    }),
            );
        }

        // Extract norm parameters (Burn's LayerNorm epsilon is private but the
        // Hydra checkpoints use the default 1e-5).
        let eps = 1e-5f32;
        let gamma_data = norm.gamma.val().to_data();
        let gamma = gamma_data
            .as_slice::<f32>()
            .expect("norm gamma is contiguous F32");
        let beta_data = norm.beta.as_ref().map(|b| b.val().to_data());
        let beta = beta_data
            .as_ref()
            .map(|d| d.as_slice::<f32>().expect("norm beta is contiguous F32"));

        let x_data = x_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("fused_norm_linear_glu_proj input is contiguous F32");
        let glu_w_slice = ff.glu_w_cache.as_slice();
        let proj_w_slice = ff.proj_out_w_cache.as_slice();
        let proj_b_slice = ff.proj_out_b_cache.as_deref();

        // Apply layer norm over the last dimension into a temporary buffer.
        let mut normed = vec![0.0f32; m * k];
        x_slice
            .par_chunks_exact(k)
            .zip(normed.par_chunks_exact_mut(k))
            .for_each(|(row_x, row_n)| {
                let mean = row_x.iter().copied().sum::<f32>() / k as f32;
                let var = row_x
                    .iter()
                    .map(|v| {
                        let d = *v - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / k as f32;
                let inv_std = 1.0f32 / (var + eps).sqrt();
                for j in 0..k {
                    row_n[j] = (row_x[j] - mean) * inv_std * gamma[j]
                        + beta.map(|b| b[j]).unwrap_or(0.0f32);
                }
            });

        let mut glu_proj = vec![0.0f32; m * out2];
        best_row_major(m, out2, k, &normed, glu_w_slice, &mut glu_proj);

        // In-place GLU activation: overwrite the gate half with softplus(gate) * up.
        glu_proj.par_chunks_exact_mut(out2).for_each(|row| {
            for j in 0..out {
                let gate = softplus_f32(row[j]);
                row[j] = gate * row[out + j];
            }
        });

        // Output projection from the activated half of glu_proj with row stride out2.
        let mut output = vec![0.0f32; m * proj_out_dim];
        {
            let a = MatRef::from_row_major_slice_with_stride(&glu_proj, m, out, out2);
            let b = MatRef::from_row_major_slice(proj_w_slice, out, proj_out_dim);
            let mut c = MatMut::from_row_major_slice_mut(&mut output, m, proj_out_dim);
            matmul(c.as_mut(), Accum::Replace, a, b, 1.0f32, Par::rayon(0));
        }

        if let Some(bias) = proj_b_slice {
            output.par_chunks_exact_mut(proj_out_dim).for_each(|row| {
                for j in 0..proj_out_dim {
                    row[j] += bias[j];
                }
            });
        }

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(output.as_slice(), (&device, DType::F32))
            .reshape([batch, seq, proj_out_dim])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fused_norm_linear_glu_proj returns a float tensor"),
        }
    }
}

#[cfg(not(any(feature = "burn-candle", feature = "burn-flex")))]
impl FusedGluBackend for burn::backend::NdArray {}

// GPU backends use the trait's generic tensor-op fallback; the Flex override
// below relies on Flex-specific raw storage access and does not apply.
#[cfg(feature = "burn-wgpu")]
impl FusedGluBackend for burn::backend::Wgpu {}

#[cfg(feature = "burn-cuda")]
impl FusedGluBackend for burn::backend::Cuda {}

#[cfg(feature = "burn-flex")]
impl FusedGluBackend for burn::backend::flex::Flex {
    fn fused_linear_glu(x: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self> {
        use burn::backend::flex::{Flex, FlexTensor};

        // Single fused linear projection [batch, seq, 2*out].
        // Inputs are cloned so the original primitives remain available for the
        // non-F32 fallback path; `FlexTensor` clone is an Arc refcount bump.
        let proj = Flex::linear(x.clone(), weight.clone(), None);
        let proj = proj.to_contiguous();

        // The optimized raw-storage path is implemented for F32, which is what
        // the Hydra model uses. Fall back to the generic tensor path for other
        // dtypes so behavior stays correct regardless of runtime dtype.
        if proj.dtype() != DType::F32 {
            return match fused_linear_glu_fallback_tensor(
                Tensor::<Flex, 3>::from_primitive(TensorPrimitive::Float(x)),
                Tensor::<Flex, 2>::from_primitive(TensorPrimitive::Float(weight)),
            )
            .into_primitive()
            {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("fused_linear_glu returns a float tensor"),
            };
        }

        let shape = proj.layout().shape().clone();
        let [batch, seq, out2]: [usize; 3] = shape.dims();
        let out = out2 / 2;

        // Allocate the output tensor and get its raw storage.
        let device = <FlexDevice as Default>::default();
        let mut output: FlexTensor =
            match Tensor::<Flex, 3>::zeros([batch, seq, out], (&device, DType::F32))
                .into_primitive()
            {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("zeros returns a float tensor"),
            };

        let src = proj.storage::<f32>();
        let dst = output.storage_mut::<f32>();

        // The linear output is row-major [batch, seq, 2*out]; within each
        // (batch, seq) position the first `out` values are the gate and the
        // next `out` values are the up projection.
        for flat in 0..batch * seq {
            let base = flat * out2;
            let dst_base = flat * out;
            for j in 0..out {
                let gate = softplus_f32(src[base + j]);
                dst[dst_base + j] = gate * src[base + out + j];
            }
        }

        output
    }
}

#[cfg(any(feature = "burn-flex", feature = "burn-candle"))]
fn softplus_f32(x: f32) -> f32 {
    // softplus(x) = log(1 + exp(x)); ln_1p is more stable for large inputs.
    x.exp().ln_1p()
}

#[cfg(any(feature = "burn-flex", feature = "burn-candle"))]
fn gelu_approx_tanh_f32(x: f32) -> f32 {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = 0.7978845608028654f32;
    let coeff = 0.044715f32;
    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (x + coeff * x3);
    0.5f32 * x * (1.0f32 + inner.tanh())
}

// ---------------------------------------------------------------------------
// Backend-specific fast RMS normalization (last dimension)
// ---------------------------------------------------------------------------

/// Backends that provide a fast RMS-normalization path over the last axis.
pub trait FastRmsNormBackend: Backend {
    /// Compute `x / sqrt(mean(x^2) + eps)` over the last dimension.
    fn fast_rms_norm(x: FloatTensor<Self>, eps: f32) -> FloatTensor<Self> {
        match super::ops::rms_norm(
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(x)),
            eps,
        )
        .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("rms_norm returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-candle")]
impl FastRmsNormBackend for burn::backend::candle::Candle {
    fn fast_rms_norm(x: FloatTensor<Self>, eps: f32) -> FloatTensor<Self> {
        use rayon::prelude::*;

        let x_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(x));
        if x_t.dtype() != DType::F32 {
            return match super::ops::rms_norm(x_t, eps).into_primitive() {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("rms_norm returns a float tensor"),
            };
        }

        let [batch, heads, seq, dim] = x_t.dims();
        let data = x_t.to_data();
        let src = data
            .as_slice::<f32>()
            .expect("fast_rms_norm input is contiguous F32");
        let mut dst = src.to_vec();

        dst.par_chunks_exact_mut(dim).for_each(|row| {
            let mut sum2 = 0.0f32;
            for &v in row.iter() {
                sum2 += v * v;
            }
            let scale = 1.0f32 / ((sum2 / dim as f32) + eps).sqrt();
            for v in row.iter_mut() {
                *v *= scale;
            }
        });

        let device = x_t.device();
        match Tensor::<Self, 1>::from_data(dst.as_slice(), (&device, DType::F32))
            .reshape([batch, heads, seq, dim])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("fast_rms_norm returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-flex")]
impl FastRmsNormBackend for burn::backend::flex::Flex {}

#[cfg(feature = "burn-wgpu")]
impl FastRmsNormBackend for burn::backend::Wgpu {}

#[cfg(feature = "burn-cuda")]
impl FastRmsNormBackend for burn::backend::Cuda {}

#[cfg(not(any(feature = "burn-candle", feature = "burn-flex")))]
impl FastRmsNormBackend for burn::backend::NdArray {}

// ---------------------------------------------------------------------------
// Backend-specific fused attention dispatch
// ---------------------------------------------------------------------------

/// Backends that provide a fused attention path.
pub trait FusedAttentionBackend: Backend {
    /// Compute multi-head attention for 4-D Q/K/V tensors.
    ///
    /// * `q`:   `[batch, heads, seq_q, head_dim]`
    /// * `k`:   `[batch, heads, seq_kv, head_dim]`
    /// * `v`:   `[batch, heads, seq_kv, head_dim]`
    /// * returns: `[batch, heads, seq_q, head_dim]`
    fn fused_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        match attention(
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(q)),
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k)),
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v)),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
            None,
            AttentionModuleOptions::default(),
        )
        .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("attention returns a float tensor"),
        }
    }
}

/// Tensor-level entry point for the fused attention path.
pub fn fused_attention<B: FusedAttentionBackend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
) -> Tensor<B, 4> {
    Tensor::from_primitive(TensorPrimitive::Float(B::fused_attention(
        match q.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("input is a float tensor"),
        },
        match k.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("input is a float tensor"),
        },
        match v.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("input is a float tensor"),
        },
        mask.map(|m| m.into_primitive()),
    )))
}

#[cfg(feature = "burn-flex")]
impl FusedAttentionBackend for burn::backend::flex::Flex {
    fn fused_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        match attention(
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(q)),
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k)),
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v)),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
            None,
            AttentionModuleOptions::default(),
        )
        .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("attention returns a float tensor"),
        }
    }
}

// GPU backends tile masked attention over the query dimension (see below).
#[cfg(feature = "burn-wgpu")]
impl FusedAttentionBackend for burn::backend::Wgpu {
    fn fused_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        match chunked_masked_attention(
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(q)),
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k)),
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v)),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
        )
        .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("attention returns a float tensor"),
        }
    }
}

#[cfg(feature = "burn-cuda")]
impl FusedAttentionBackend for burn::backend::Cuda {
    fn fused_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        match chunked_masked_attention(
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(q)),
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k)),
            Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v)),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
        )
        .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("attention returns a float tensor"),
        }
    }
}

/// Target head_dim for GPU attention padding, from `AKASHA_HYDRA_ATTN_PAD`
/// (default 0 = disabled). Zero-pads NaFlex's head_dim 72 up to a multiple
/// of 32 so the batched scores/AV matmuls run on aligned tiles. Measured on
/// an RTX 4090 as a wash (96: ~0.098, 80: ~0.099, off: ~0.095 s/img) — the
/// pad/slice copies eat the tiling gain — so it stays off by default.
#[cfg(any(feature = "burn-cuda", feature = "burn-wgpu"))]
fn attn_pad_target() -> usize {
    static TARGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TARGET.get_or_init(|| {
        std::env::var("AKASHA_HYDRA_ATTN_PAD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

/// GPU attention entry point. Zero-pads head_dim up to [`attn_pad_target`]
/// before delegating to the (query-tiled) attention implementation. Padded
/// lanes are zero, so they contribute nothing to the scores dot product; the
/// softmax scale is pinned to the *original* head_dim (Burn would otherwise
/// derive `1/sqrt(head_dim)` from the padded dim), and the output is sliced
/// back. Results are unchanged apart from float noise from different kernel
/// tiling.
#[cfg(any(feature = "burn-cuda", feature = "burn-wgpu"))]
fn chunked_masked_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
) -> Tensor<B, 4> {
    let [batch, heads, seq_q, head_dim] = q.dims();
    let target = attn_pad_target();
    if target <= head_dim {
        return chunked_masked_attention_inner(q, k, v, mask, None);
    }

    let pad = target - head_dim;
    let device = q.device();
    let dtype = q.dtype();
    let seq_kv = k.dims()[2];
    let zeros = |rows: usize| Tensor::<B, 4>::zeros([batch, heads, rows, pad], &device).cast(dtype);
    let q = Tensor::cat(vec![q, zeros(seq_q)], 3);
    let k = Tensor::cat(vec![k, zeros(seq_kv)], 3);
    let v = Tensor::cat(vec![v, zeros(seq_kv)], 3);
    let scale = Some(1.0 / (head_dim as f64).sqrt());

    chunked_masked_attention_inner(q, k, v, mask, scale)
        .slice([0..batch, 0..heads, 0..seq_q, 0..head_dim])
}

/// Attention tiled over the query dimension when the scores tensor would be
/// large.
///
/// Burn's attention materializes the full `[batch, heads, seq_q, seq_kv]`
/// scores tensor when a mask is present — and even without a mask its fused
/// kernel may fall back to a scores-materializing path (with seq_kv padded to
/// a tile multiple) for shapes a flash kernel rejects, such as unaligned
/// seq_kv. For Hydra's pool and mid-block cross-attention (`[batch, 32, 8886,
/// seq_kv]`) that is multiple GiB at batch sizes above ~4, which exceeds
/// CubeCL's max pool page (total VRAM / 4) and panics with "can't allocate
/// buffer" even when plenty of VRAM is free. Tiling over `seq_q` keeps each
/// chunk small; small attentions pass through untouched.
#[cfg(any(feature = "burn-cuda", feature = "burn-wgpu"))]
fn chunked_masked_attention_inner<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
    scale: Option<f64>,
) -> Tensor<B, 4> {
    /// Target ceiling for one scores chunk, in bytes.
    const MAX_SCORES_BYTES: usize = 1 << 30; // 1 GiB

    let [batch, heads, seq_q, head_dim] = q.dims();
    let seq_kv = k.dims()[2];
    let options = AttentionModuleOptions {
        scale,
        ..Default::default()
    };

    let scores_bytes = batch * heads * seq_q * seq_kv * 4;
    if scores_bytes <= MAX_SCORES_BYTES {
        tracing::debug!(
            "attention: q={:?} kv={seq_kv} mask={:?} scores={} MiB -> full (under ceiling)",
            [batch, heads, seq_q, head_dim],
            mask.as_ref().map(|m| m.dims()),
            scores_bytes / (1024 * 1024),
        );
        return attention(q, k, v, mask, None, options);
    }
    // Even without a mask, Burn's "fused" attention may not take a flash
    // kernel (e.g. unaligned seq_kv) and instead materializes a scores
    // workspace padded to a tile multiple — which can exceed CubeCL's max
    // pool page. Tile over queries regardless of masking.
    if let Some(mask_t) = &mask {
        // Only the broadcastable `[batch, 1, 1, seq_kv]` mask shape can be
        // tiled without also slicing the mask; anything else goes through
        // the full path.
        if mask_t.dims() != [batch, 1, 1, seq_kv] {
            tracing::warn!(
                "chunked_masked_attention: non-broadcast mask {:?} (scores would be {} MiB); \
                 falling back to full attention, which may exceed CubeCL's max pool page",
                mask_t.dims(),
                scores_bytes / (1024 * 1024),
            );
            return attention(q, k, v, mask, None, options);
        }
    }

    let chunk = (MAX_SCORES_BYTES / (batch * heads * seq_kv * 4)).max(1);
    tracing::debug!(
        "attention: q={:?} kv={seq_kv} mask={:?} scores={} MiB -> tiled chunk={chunk}",
        [batch, heads, seq_q, head_dim],
        mask.as_ref().map(|m| m.dims()),
        scores_bytes / (1024 * 1024)
    );
    let mut outs = Vec::new();
    let mut start = 0;
    while start < seq_q {
        let end = (start + chunk).min(seq_q);
        let q_c = q.clone().slice([0..batch, 0..heads, start..end, 0..head_dim]);
        outs.push(attention(
            q_c,
            k.clone(),
            v.clone(),
            mask.clone(),
            None,
            options.clone(),
        ));
        start = end;
    }
    Tensor::cat(outs, 2)
}

#[cfg(all(feature = "burn-candle", feature = "burn-flex"))]
impl FusedAttentionBackend for burn::backend::candle::Candle {
    fn fused_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        use rayon::prelude::*;

        let q_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(q));
        let k_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(k));
        let v_t = Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(v));

        // Only accelerate the F32 path; fall back to Burn's attention for other dtypes.
        if q_t.dtype() != DType::F32 {
            return match attention(
                q_t,
                k_t,
                v_t,
                mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
                None,
                AttentionModuleOptions::default(),
            )
            .into_primitive()
            {
                TensorPrimitive::Float(tensor) => tensor,
                _ => unreachable!("attention returns a float tensor"),
            };
        }

        let q_shape = q_t.dims();
        let k_shape = k_t.dims();
        let v_shape = v_t.dims();
        assert_eq!(q_shape[0], k_shape[0], "q and k batch sizes must match");
        assert_eq!(q_shape[1], k_shape[1], "q and k head counts must match");
        assert_eq!(q_shape[3], k_shape[3], "q and k head dims must match");
        assert_eq!(k_shape, v_shape, "k and v shapes must match");
        let [batch, heads, seq_q, head_dim] = q_shape;
        let seq_kv = k_shape[2];

        // For masked attention, try to use a prefix-valid fast path. Hydra pads images
        // to max_seq_len with a contiguous block of invalid positions at the end, so
        // we only need to attend to the first n_valid key/value positions.
        // Burn's attention treats `true` as "mask out"; our preprocess produces
        // `true` = attend, and Hydra::forward inverts that before passing the mask here.
        let n_valids: Vec<usize> = if let Some(mask) = mask {
            let mask_t = Tensor::<Self, 4, Bool>::from_primitive(mask);
            let [mb, mh, mw, ms] = mask_t.dims();
            if mb != batch || mh != 1 || mw != 1 || ms != seq_kv {
                return match attention(
                    q_t,
                    k_t,
                    v_t,
                    Some(mask_t),
                    None,
                    AttentionModuleOptions::default(),
                )
                .into_primitive()
                {
                    TensorPrimitive::Float(tensor) => tensor,
                    _ => unreachable!("attention returns a float tensor"),
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
                return match attention(
                    q_t,
                    k_t,
                    v_t,
                    Some(mask_t),
                    None,
                    AttentionModuleOptions::default(),
                )
                .into_primitive()
                {
                    TensorPrimitive::Float(tensor) => tensor,
                    _ => unreachable!("attention returns a float tensor"),
                };
            }
            n_valids
        } else {
            vec![seq_kv; batch]
        };

        let q_data = q_t.to_data();
        let k_data = k_t.to_data();
        let v_data = v_t.to_data();
        let q_slice_all = q_data.as_slice::<f32>().expect("q is contiguous F32");
        let k_slice_all = k_data.as_slice::<f32>().expect("k is contiguous F32");
        let v_slice_all = v_data.as_slice::<f32>().expect("v is contiguous F32");

        let device = q_t.device();

        let q_stride_head = seq_q * head_dim;
        let q_stride_batch = heads * q_stride_head;
        let kv_stride_head = seq_kv * head_dim;
        let kv_stride_batch = heads * kv_stride_head;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut output = vec![0.0f32; batch * heads * seq_q * head_dim];

        // Tiled attention: keep per-tile scores cache-resident and avoid the
        // full [seq_q, seq_kv] allocation for every head.
        const QUERY_TILE: usize = 64;
        let max_n_valid = n_valids.iter().copied().max().unwrap_or(seq_kv);
        let mut scores_tile_all = vec![0.0f32; batch * heads * QUERY_TILE * max_n_valid];
        let mut head_out_tile_all = vec![0.0f32; batch * heads * QUERY_TILE * head_dim];

        output
            .par_chunks_exact_mut(q_stride_head)
            .zip(scores_tile_all.par_chunks_exact_mut(QUERY_TILE * max_n_valid))
            .zip(head_out_tile_all.par_chunks_exact_mut(QUERY_TILE * head_dim))
            .enumerate()
            .for_each(|(flat, ((out_head, scores_tile), head_out_tile))| {
                let b = flat / heads;
                let h = flat % heads;
                let seq_kv_eff = n_valids[b];

                let q_offset = b * q_stride_batch + h * q_stride_head;
                let kv_offset = b * kv_stride_batch + h * kv_stride_head;

                let q_slice = &q_slice_all[q_offset..q_offset + q_stride_head];
                let k_slice = &k_slice_all[kv_offset..kv_offset + seq_kv_eff * head_dim];
                let v_slice = &v_slice_all[kv_offset..kv_offset + seq_kv_eff * head_dim];

                let n_tiles = (seq_q + QUERY_TILE - 1) / QUERY_TILE;
                for t in 0..n_tiles {
                    let tile_start = t * QUERY_TILE;
                    let tile_q = (tile_start + QUERY_TILE).min(seq_q) - tile_start;

                    let q_tile = &q_slice[tile_start * head_dim..(tile_start + tile_q) * head_dim];
                    let scores = &mut scores_tile[..tile_q * seq_kv_eff];
                    let head_out = &mut head_out_tile[..tile_q * head_dim];

                    gemm_a_bt_scaled(
                        tile_q,
                        seq_kv_eff,
                        head_dim,
                        q_tile,
                        k_slice,
                        scores,
                        scale,
                        gemm::Parallelism::None,
                    );

                    for i in 0..tile_q {
                        let row_start = i * seq_kv_eff;
                        simd_ops::softmax_in_place(
                            &mut scores[row_start..row_start + seq_kv_eff],
                        );
                    }

                    gemm_row_major(
                        tile_q,
                        head_dim,
                        seq_kv_eff,
                        scores,
                        v_slice,
                        head_out,
                        gemm::Parallelism::None,
                    );

                    for p in 0..tile_q {
                        let out_base = (tile_start + p) * head_dim;
                        let buf_base = p * head_dim;
                        out_head[out_base..out_base + head_dim]
                            .copy_from_slice(&head_out[buf_base..buf_base + head_dim]);
                    }
                }
            });

        match Tensor::<Self, 1>::from_data(output.as_slice(), (&device, DType::F32))
            .reshape([batch, heads, seq_q, head_dim])
            .into_primitive()
        {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("attention returns a float tensor"),
        }
    }
}

#[cfg(all(feature = "burn-candle", not(feature = "burn-flex")))]
impl FusedAttentionBackend for burn::backend::candle::Candle {}

#[cfg(not(any(feature = "burn-candle", feature = "burn-flex")))]
impl FusedAttentionBackend for burn::backend::NdArray {}

#[cfg(all(test, feature = "burn-candle"))]
#[allow(deprecated)]
mod tests {
    use super::*;
    use burn::backend::candle::Candle;
    use burn::tensor::Tensor;

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < eps, "{x} vs {y} (eps {eps})");
        }
    }

    fn device() -> burn::backend::candle::CandleDevice {
        <burn::backend::candle::CandleDevice as Default>::default()
    }

    fn tensor3(shape: [usize; 3], values: &[f32]) -> Tensor<Candle, 3> {
        Tensor::<Candle, 1>::from_data(values, (&device(), DType::F32)).reshape(shape)
    }

    fn tensor4(shape: [usize; 4], values: &[f32]) -> Tensor<Candle, 4> {
        Tensor::<Candle, 1>::from_data(values, (&device(), DType::F32)).reshape(shape)
    }

    fn tensor2(shape: [usize; 2], values: &[f32]) -> Tensor<Candle, 2> {
        Tensor::<Candle, 1>::from_data(values, (&device(), DType::F32)).reshape(shape)
    }

    fn tensor1(values: &[f32]) -> Tensor<Candle, 1> {
        Tensor::<Candle, 1>::from_data(values, (&device(), DType::F32))
    }

    fn to_float<const D: usize>(t: Tensor<Candle, D>) -> FloatTensor<Candle> {
        match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!("expected float tensor"),
        }
    }

    fn from_float<const D: usize>(t: FloatTensor<Candle>) -> Tensor<Candle, D> {
        Tensor::<Candle, D>::from_primitive(TensorPrimitive::Float(t))
    }

    #[test]
    fn fast_linear_matches_fallback() {
        let x = tensor3([2, 3, 4], &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2,
        ]);
        let weight = tensor2([4, 3], &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        ]);
        let bias = Some(tensor1(&[0.1, 0.2, 0.3]));

        let fused = <Candle as FastLinearBackend>::fast_linear(
            to_float(x.clone()),
            to_float(weight.clone()),
            bias.clone().map(to_float),
        );
        let fused_t = from_float::<3>(fused);
        let fallback = fast_linear_fallback_tensor(x, weight, bias);

        approx_eq(
            fused_t.to_data().as_slice::<f32>().unwrap(),
            fallback.to_data().as_slice::<f32>().unwrap(),
            1e-4,
        );
    }

    #[test]
    fn fast_linear_no_bias_matches_fallback() {
        let x = tensor3([1, 2, 4], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let weight = tensor2([4, 3], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);

        let fused = <Candle as FastLinearBackend>::fast_linear(
            to_float(x.clone()),
            to_float(weight.clone()),
            None,
        );
        let fused_t = from_float::<3>(fused);
        let fallback = fast_linear_fallback_tensor(x, weight, None);

        approx_eq(
            fused_t.to_data().as_slice::<f32>().unwrap(),
            fallback.to_data().as_slice::<f32>().unwrap(),
            1e-4,
        );
    }

    #[test]
    fn fused_glu_matches_fallback() {
        let x = tensor3([1, 2, 4], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let weight = tensor2([4, 6], &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
            1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
        ]);

        let fused = <Candle as FusedGluBackend>::fused_linear_glu(
            to_float(x.clone()),
            to_float(weight.clone()),
        );
        let fused_t = from_float::<3>(fused);
        let fallback = fused_linear_glu_fallback_tensor(x, weight);

        approx_eq(
            fused_t.to_data().as_slice::<f32>().unwrap(),
            fallback.to_data().as_slice::<f32>().unwrap(),
            1e-4,
        );
    }

    #[test]
    fn fused_linear_glu_proj_matches_fallback() {
        let x = tensor3([1, 2, 4], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let glu_weight = tensor2([4, 6], &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
            1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
        ]);
        let proj_weight = tensor2([3, 2], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let proj_bias = Some(tensor1(&[0.1, 0.2]));

        let fused = <Candle as FusedGluBackend>::fused_linear_glu_proj(
            to_float(x.clone()),
            to_float(glu_weight.clone()),
            to_float(proj_weight.clone()),
            proj_bias.clone().map(to_float),
        );
        let fused_t = from_float::<3>(fused);
        let fallback = fused_linear_glu_proj_fallback_tensor(x, glu_weight, proj_weight, proj_bias);

        approx_eq(
            fused_t.to_data().as_slice::<f32>().unwrap(),
            fallback.to_data().as_slice::<f32>().unwrap(),
            1e-4,
        );
    }

    #[test]
    fn fast_rms_norm_matches_fallback() {
        let x = tensor4([1, 2, 3, 4], &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
            1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
        ]);

        let fused = <Candle as FastRmsNormBackend>::fast_rms_norm(to_float(x.clone()), 1e-5);
        let fused_t = from_float::<4>(fused);

        // Manual fallback: x / sqrt(mean(x^2) + eps).
        let mut fallback_data: Vec<f32> = x.to_data().as_slice::<f32>().unwrap().to_vec();
        let len = 4;
        for row in fallback_data.chunks_exact_mut(len) {
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / len as f32;
            let scale = 1.0 / (mean_sq + 1e-5).sqrt();
            for v in row.iter_mut() {
                *v *= scale;
            }
        }
        let fallback = Tensor::<Candle, 1>::from_data(fallback_data.as_slice(), (&device(), DType::F32))
            .reshape([1, 2, 3, 4]);

        approx_eq(
            fused_t.to_data().as_slice::<f32>().unwrap(),
            fallback.to_data().as_slice::<f32>().unwrap(),
            1e-4,
        );
    }

    #[test]
    #[cfg(feature = "burn-flex")]
    fn fused_attention_matches_burn_fallback() {
        let q = tensor4([1, 2, 3, 4], &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
            1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
        ]);
        let k = tensor4([1, 2, 3, 4], &[
            0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7,
            1.0, 0.9, 1.2, 1.1, 1.4, 1.3, 1.6, 1.5,
            1.8, 1.7, 2.0, 1.9, 2.2, 2.1, 2.4, 2.3,
        ]);
        let v = tensor4([1, 2, 3, 4], &[
            0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4,
            0.5, -0.5, 0.6, -0.6, 0.7, -0.7, 0.8, -0.8,
            0.9, -0.9, 1.0, -1.0, 1.1, -1.1, 1.2, -1.2,
        ]);

        let fused = <Candle as FusedAttentionBackend>::fused_attention(
            to_float(q.clone()),
            to_float(k.clone()),
            to_float(v.clone()),
            None,
        );
        let fused_t = from_float::<4>(fused);
        let fallback = attention(q, k, v, None, None, AttentionModuleOptions::default());

        approx_eq(
            fused_t.to_data().as_slice::<f32>().unwrap(),
            fallback.to_data().as_slice::<f32>().unwrap(),
            1e-3,
        );
    }

    #[test]
    fn best_row_major_matches_naive() {
        let m = 4;
        let n = 3;
        let k = 5;
        let a: Vec<f32> = (1..=m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (1..=k * n).map(|i| i as f32 * 0.05).collect();
        let mut c = vec![0.0f32; m * n];

        best_row_major(m, n, k, &a, &b, &mut c);

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                expected[i * n + j] = sum;
            }
        }

        approx_eq(&c, &expected, 1e-4);
    }

    #[test]
    fn gemm_a_bt_scaled_matches_naive() {
        let m = 3;
        let n = 4;
        let k = 5;
        let scale = 0.5f32;
        let a: Vec<f32> = (1..=m * k).map(|i| i as f32 * 0.1).collect();
        // b_t is row-major [n, k], i.e. the transpose of the desired B.
        let b_t: Vec<f32> = (1..=n * k).map(|i| i as f32 * 0.05).collect();
        let mut c = vec![0.0f32; m * n];

        gemm_a_bt_scaled(m, n, k, &a, &b_t, &mut c, scale, gemm::Parallelism::None);

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b_t[j * k + l];
                }
                expected[i * n + j] = scale * sum;
            }
        }

        approx_eq(&c, &expected, 1e-4);
    }

    #[test]
    fn gemm_row_major_matches_naive() {
        let m = 3;
        let n = 4;
        let k = 5;
        let a: Vec<f32> = (1..=m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (1..=k * n).map(|i| i as f32 * 0.05).collect();
        let mut c = vec![0.0f32; m * n];

        gemm_row_major(m, n, k, &a, &b, &mut c, gemm::Parallelism::None);

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                expected[i * n + j] = sum;
            }
        }

        approx_eq(&c, &expected, 1e-4);
    }

    // -----------------------------------------------------------------------
    // Online-softmax attention correctness tests
    // -----------------------------------------------------------------------

    fn make_attention_values(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 17 + 31) % 101) as f32 * 0.03 - 1.5)
            .collect()
    }

    #[test]
    fn online_softmax_attention_matches_two_gemm_multi_head_batch() {
        let batch = 2;
        let heads = 4;
        let seq_q = 32;
        let seq_kv = 48;
        let head_dim = 16;

        let q = make_attention_values(batch * heads * seq_q * head_dim);
        let k = make_attention_values(batch * heads * seq_kv * head_dim);
        let v = make_attention_values(batch * heads * seq_kv * head_dim);

        let mut out_online = vec![0.0f32; batch * heads * seq_q * head_dim];
        let mut out_fallback = vec![0.0f32; batch * heads * seq_q * head_dim];

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        for b in 0..batch {
            for h in 0..heads {
                let off = (b * heads + h) * seq_q * head_dim;
                let q_head = &q[off..off + seq_q * head_dim];
                let kv_off = (b * heads + h) * seq_kv * head_dim;
                let k_head = &k[kv_off..kv_off + seq_kv * head_dim];
                let v_head = &v[kv_off..kv_off + seq_kv * head_dim];

                let out_online_head = &mut out_online[off..off + seq_q * head_dim];
                let out_fallback_head = &mut out_fallback[off..off + seq_q * head_dim];
                let mut scores = vec![0.0f32; seq_q * seq_kv];

                fused_attention_online_softmax(
                    q_head, k_head, v_head, out_online_head, seq_q, seq_kv, head_dim, head_dim,
                    head_dim, head_dim, seq_kv, scale,
                );
                fused_attention_two_gemm_fallback(
                    q_head, k_head, v_head, out_fallback_head, seq_q, seq_kv, head_dim, head_dim,
                    head_dim, head_dim, seq_kv, scale, &mut scores,
                );
            }
        }

        approx_eq(&out_online, &out_fallback, 1e-4);
    }

    #[test]
    fn online_softmax_attention_matches_two_gemm_naflex_shape() {
        // NaFlex-like: batch=1, heads=16, head_dim=72, seq=128 (scaled down
        // from 1024 to keep unit-test runtime reasonable).
        let batch = 1;
        let heads = 16;
        let seq = 128;
        let head_dim = 72;

        let q = make_attention_values(batch * heads * seq * head_dim);
        let k = make_attention_values(batch * heads * seq * head_dim);
        let v = make_attention_values(batch * heads * seq * head_dim);

        let mut out_online = vec![0.0f32; batch * heads * seq * head_dim];
        let mut out_fallback = vec![0.0f32; batch * heads * seq * head_dim];

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        for b in 0..batch {
            for h in 0..heads {
                let off = (b * heads + h) * seq * head_dim;
                let q_head = &q[off..off + seq * head_dim];
                let kv_off = (b * heads + h) * seq * head_dim;
                let k_head = &k[kv_off..kv_off + seq * head_dim];
                let v_head = &v[kv_off..kv_off + seq * head_dim];

                let out_online_head = &mut out_online[off..off + seq * head_dim];
                let out_fallback_head = &mut out_fallback[off..off + seq * head_dim];
                let mut scores = vec![0.0f32; seq * seq];

                fused_attention_online_softmax(
                    q_head, k_head, v_head, out_online_head, seq, seq, head_dim, head_dim,
                    head_dim, head_dim, seq, scale,
                );
                fused_attention_two_gemm_fallback(
                    q_head, k_head, v_head, out_fallback_head, seq, seq, head_dim, head_dim,
                    head_dim, head_dim, seq, scale, &mut scores,
                );
            }
        }

        approx_eq(&out_online, &out_fallback, 1e-4);
    }

    #[test]
    fn online_softmax_attention_matches_two_gemm_pool_shape() {
        // HydraPool cross-attention-like: batch=1, heads=32, n_classes=256
        // (scaled down from 8886), seq_kv=128 (scaled down from 1024),
        // head_dim=64.
        let batch = 1;
        let heads = 32;
        let n_classes = 256;
        let seq_kv = 128;
        let head_dim = 64;

        let q = make_attention_values(heads * n_classes * head_dim);
        let k = make_attention_values(batch * heads * seq_kv * head_dim);
        let v = make_attention_values(batch * heads * seq_kv * head_dim);

        let mut out_online = vec![0.0f32; batch * n_classes * heads * head_dim];
        let mut out_fallback = vec![0.0f32; batch * n_classes * heads * head_dim];

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        for b in 0..batch {
            for h in 0..heads {
                let q_head = &q[h * n_classes * head_dim..(h + 1) * n_classes * head_dim];
                let kv_off = (b * heads + h) * seq_kv * head_dim;
                let k_head = &k[kv_off..kv_off + seq_kv * head_dim];
                let v_head = &v[kv_off..kv_off + seq_kv * head_dim];

                for t in 0..n_classes {
                    let out_off = (b * n_classes + t) * heads * head_dim + h * head_dim;
                    let mut scores = vec![0.0f32; seq_kv];

                    let mut online_val = [0.0f32; 64];
                    let mut fallback_val = [0.0f32; 64];

                    fused_attention_online_softmax(
                        &q_head[t * head_dim..(t + 1) * head_dim],
                        k_head,
                        v_head,
                        &mut online_val,
                        1,
                        seq_kv,
                        head_dim,
                        head_dim,
                        head_dim,
                        head_dim,
                        seq_kv,
                        scale,
                    );
                    fused_attention_two_gemm_fallback(
                        &q_head[t * head_dim..(t + 1) * head_dim],
                        k_head,
                        v_head,
                        &mut fallback_val,
                        1,
                        seq_kv,
                        head_dim,
                        head_dim,
                        head_dim,
                        head_dim,
                        seq_kv,
                        scale,
                        &mut scores,
                    );

                    out_online[out_off..out_off + head_dim].copy_from_slice(&online_val);
                    out_fallback[out_off..out_off + head_dim].copy_from_slice(&fallback_val);
                }
            }
        }

        approx_eq(&out_online, &out_fallback, 1e-4);
    }

    #[test]
    fn online_softmax_attention_matches_two_gemm_prefix_valid() {
        let seq_q = 64;
        let seq_kv = 128;
        let n_valid = 77;
        let head_dim = 64;

        let q = make_attention_values(seq_q * head_dim);
        let k = make_attention_values(seq_kv * head_dim);
        let v = make_attention_values(seq_kv * head_dim);

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
}
