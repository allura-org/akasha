//! Fused GLU and attention operations for Hydra-3.5.
//!
//! Hydra's `proj_in` is a single linear layer whose output is split in half to
//! form the GLU gate and up projections. Fusing the split, softplus, and
//! element-wise multiply into one kernel avoids an extra round-trip through the
//! backend and reduces peak memory traffic for the pool and mid-block FFs.
//!
//! Attention is dispatched backend-specifically: `burn-flex` uses Burn's fused
//! attention directly, while `burn-candle` (when `burn-flex` is also enabled)
//! copies Q/K/V to Flex for the fast no-mask path and copies the result back.

use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::module::attention;
use burn::tensor::ops::{AttentionModuleOptions, BoolTensor, FloatTensor, ModuleOps};
use burn::tensor::{DType, TensorPrimitive};

use super::simd_ops;

#[cfg(feature = "burn-flex")]
use burn::backend::flex::FlexDevice;

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
        }
    }

    pub fn get_a(&mut self, len: usize) -> &mut [f32] {
        if self.a.len() < len {
            self.a.resize(len, 0.0);
        }
        &mut self.a[..len]
    }

    pub fn get_b(&mut self, len: usize) -> &mut [f32] {
        if self.b.len() < len {
            self.b.resize(len, 0.0);
        }
        &mut self.b[..len]
    }

    pub fn get_c(&mut self, len: usize) -> &mut [f32] {
        if self.c.len() < len {
            self.c.resize(len, 0.0);
        }
        &mut self.c[..len]
    }

    pub fn get_d(&mut self, len: usize) -> &mut [f32] {
        if self.d.len() < len {
            self.d.resize(len, 0.0);
        }
        &mut self.d[..len]
    }

    pub fn get_e(&mut self, len: usize) -> &mut [f32] {
        if self.e.len() < len {
            self.e.resize(len, 0.0);
        }
        &mut self.e[..len]
    }

    pub fn get_f(&mut self, len: usize) -> &mut [f32] {
        if self.f.len() < len {
            self.f.resize(len, 0.0);
        }
        &mut self.f[..len]
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
unsafe fn gemm_f32_ex(
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

/// `C = A @ B^T` where A is row-major `[m, k]` and `b_t` is row-major `[n, k]`
/// (i.e. the transpose of the desired B).
#[inline]
fn gemm_a_bt(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    par: gemm::Parallelism,
) {
    unsafe {
        gemm_f32(
            m, n, k, a, k as isize, 1, b_t, 1, k as isize, c, n as isize, 1, par,
        );
    }
}

/// `C = scale * (A @ B)` with row-major A, B, C.
#[inline]
fn gemm_row_major_scaled(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    scale: f32,
    par: gemm::Parallelism,
) {
    unsafe {
        gemm_f32_ex(
            m, n, k, a, k as isize, 1, b, n as isize, 1, c, n as isize, 1, scale, par,
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

/// `C += A @ B^T` with row-major A, B^T, C. Existing C contents are read and accumulated.
#[inline]
fn gemm_a_bt_accum(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    par: gemm::Parallelism,
) {
    unsafe {
        gemm_f32_accum(
            m, n, k, a, k as isize, 1, b_t, 1, k as isize, c, n as isize, 1, par,
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

/// `C = A @ B` where A is row-major with arbitrary leading row stride `rsa`.
///
/// Useful for reading only the first `k` columns of a wider matrix without a
/// separate copy.
#[inline]
fn gemm_a_strided(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    rsa: isize,
    b: &[f32],
    c: &mut [f32],
    par: gemm::Parallelism,
) {
    unsafe {
        gemm_f32(m, n, k, a, rsa, 1, b, n as isize, 1, c, n as isize, 1, par);
    }
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

/// Tensor-level entry point for the fused GLU path.
pub fn fused_linear_glu<B: FusedGluBackend>(x: Tensor<B, 3>, weight: Tensor<B, 2>) -> Tensor<B, 3> {
    Tensor::from_primitive(TensorPrimitive::Float(B::fused_linear_glu(
        match x.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("input is a float tensor"),
        },
        match weight.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("weight is a float tensor"),
        },
    )))
}

/// Tensor-level entry point for the fused GLU + output projection path.
pub fn fused_linear_glu_proj<B: FusedGluBackend>(
    x: Tensor<B, 3>,
    glu_weight: Tensor<B, 2>,
    proj: &burn::nn::Linear<B>,
) -> Tensor<B, 3> {
    let x_prim = match x.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("fused_linear_glu_proj input is a float tensor"),
    };
    let glu_w_prim = match glu_weight.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("glu_weight is a float tensor"),
    };
    let proj_w_prim = match proj.weight.val().into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("proj weight is a float tensor"),
    };
    let proj_b_prim = proj.bias.as_ref().map(|b| match b.val().into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!("proj bias is a float tensor"),
    });

    Tensor::from_primitive(TensorPrimitive::Float(B::fused_linear_glu_proj(
        x_prim,
        glu_w_prim,
        proj_w_prim,
        proj_b_prim,
    )))
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

#[cfg(all(feature = "burn-candle", feature = "burn-flex"))]
impl FusedAttentionBackend for burn::backend::candle::Candle {
    fn fused_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        use faer::linalg::matmul::matmul;
        use faer::{Accum, MatMut, MatRef, Par};
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

// ---------------------------------------------------------------------------
// Backend-specific fused NaFlexBlock dispatch
// ---------------------------------------------------------------------------

/// Backends that provide a fused NaFlexBlock path.
pub trait FusedNaFlexBlockBackend: Backend {
    /// Compute one NaFlexBlock forward pass: norm1, self-attention, residual,
    /// norm2, MLP, residual. The `mask` argument is accepted for parity with
    /// the generic forward path; the fast implementation currently requires an
    /// all-valid mask and will fall back to the high-level module if a mask is
    /// supplied.
    fn fused_na_flex_block(
        x: FloatTensor<Self>,
        block: &super::modules::NaFlexBlock<Self>,
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
        block: &super::modules::NaFlexBlock<Self>,
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
    block: &super::modules::NaFlexBlock<B>,
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

    let qkv_ptr = qkv.as_ptr();
    // The raw output pointer is passed as an integer so the parallel closure
    // can capture it; each head writes to disjoint regions.
    let attn_addr = attn_out.as_mut_ptr() as usize;

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

                let scores = &mut scores_tile[..tile_q * seq_kv_eff];

                unsafe {
                    gemm_f32_ex(
                        tile_q,
                        seq_kv_eff,
                        head_dim,
                        q_tile,
                        qkv_out as isize,
                        1,
                        k_slice,
                        1,
                        qkv_out as isize,
                        scores,
                        seq_kv_eff as isize,
                        1,
                        scale,
                        gemm::Parallelism::None,
                    );
                }

                for i in 0..tile_q {
                    let row_start = i * seq_kv_eff;
                    simd_ops::softmax_in_place(&mut scores[row_start..row_start + seq_kv_eff]);
                }

                let head_out = &mut head_out_tile[..tile_q * head_dim];
                unsafe {
                    gemm_f32_ex(
                        tile_q,
                        head_dim,
                        seq_kv_eff,
                        scores,
                        seq_kv_eff as isize,
                        1,
                        v_slice,
                        qkv_out as isize,
                        1,
                        head_out,
                        head_dim as isize,
                        1,
                        1.0,
                        gemm::Parallelism::None,
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
        block: &super::modules::NaFlexBlock<Self>,
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
        block: &super::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
        batch: usize,
        seq: usize,
        hidden: usize,
    ) {
        use rayon::prelude::*;
        use std::time::Instant;

        let m = batch * seq;
        let heads = super::modules::NAFLEX_HEADS;
        let head_dim = super::modules::NAFLEX_HEAD_DIM;
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
        // matrix product into it, saving a separate bias-add pass.
        let mut qkv = resize_buf(&mut workspace.d, m * qkv_out);
        if let Some(ref b) = qkv_b {
            qkv.par_chunks_exact_mut(qkv_out)
                .for_each(|row| row.copy_from_slice(b));
            best_row_major_accum(m, qkv_out, hidden, &norm1_buf, qkv_w, &mut qkv);
        } else {
            best_row_major(m, qkv_out, hidden, &norm1_buf, qkv_w, &mut qkv);
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
        best_row_major_accum(m, hidden, hidden, &attn_buf, proj_w, &mut norm1_buf);

        // ---- 5. LayerNorm2 into workspace.b. ----
        drop(attn_buf);
        let mut attn_buf = resize_buf(&mut workspace.b, m * hidden);
        norm1_buf
            .par_chunks_exact(hidden)
            .zip(attn_buf.par_chunks_exact_mut(hidden))
            .for_each(|(row_x, row_n)| {
                simd_ops::layer_norm_row(row_x, norm2_gamma, norm2_beta, 1e-5f32, row_n);
            });

        // ---- 6. MLP (fc1 -> GELU -> fc2) + second residual into `out`. ----
        // Initialise the MLP hidden buffer with the fc1 bias and accumulate the
        // fc1 projection, saving a separate bias-add pass.
        if let Some(ref b) = fc1_b {
            mlp_hidden_buf
                .par_chunks_exact_mut(fc1_hidden)
                .for_each(|row| row.copy_from_slice(b));
            best_row_major_accum(m, fc1_hidden, hidden, &attn_buf, fc1_w, &mut mlp_hidden_buf);
        } else {
            best_row_major(m, fc1_hidden, hidden, &attn_buf, fc1_w, &mut mlp_hidden_buf);
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
        best_row_major_accum(m, hidden, fc1_hidden, &mlp_hidden_buf, fc2_w, out);

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
        block: &super::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        let out = block.forward(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
            workspace,
        );
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
        block: &super::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        let out = block.forward(
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(x)),
            mask.map(|m| Tensor::<Self, 4, Bool>::from_primitive(m)),
            workspace,
        );
        match out.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            _ => unreachable!("NaFlexBlock returns a float tensor"),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend-specific fused HydraMidBlock dispatch
// ---------------------------------------------------------------------------

/// Backends that provide a fused HydraMidBlock path.
pub trait FusedHydraMidBlockBackend: Backend {
    /// Compute one HydraMidBlock forward pass: q_proj, q_norm, cross-attention
    /// with the supplied `k`/`v`, output projection + residual, then FF + residual.
    ///
    /// The fast path currently requires an all-valid mask and falls back to the
    /// high-level module if a mask is supplied.
    fn fused_hydra_mid_block(
        x: FloatTensor<Self>,
        block: &super::modules::HydraMidBlock<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self>;
}

#[cfg(feature = "burn-candle")]
#[inline(always)]
fn fused_hydra_mid_block_to_buffer(
    x_slice: &[f32],
    out_buf: &mut [f32],
    block: &super::modules::HydraMidBlock<burn::backend::candle::Candle>,
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
    // saving a separate bias-add pass.
    let t_q_proj0 = Instant::now();
    if let Some(ref b) = q_proj_b {
        out_buf
            .par_chunks_exact_mut(hidden)
            .for_each(|row| row.copy_from_slice(b));
        best_row_major_accum(m, hidden, hidden, x_slice, q_proj_w, out_buf);
    } else {
        best_row_major(m, hidden, hidden, x_slice, q_proj_w, out_buf);
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
                        simd_ops::softmax_in_place(
                            &mut scores[row_start..row_start + seq_kv_eff],
                        );
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
    best_row_major_accum(m, hidden, hidden, out_buf, o_proj_w, &mut post_attn);
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

    best_row_major(m, glu_out2, hidden, &out_buf, glu_w, &mut glu_proj);

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
    if m <= 64 {
        // For tiny batch sizes the strided faer path has high threading overhead.
        // Copy the activated gate half to a contiguous buffer and use gemm with
        // no parallelism.
        let mut glu_contig = resize_buf(&mut workspace.d, m * glu_out_dim);
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
        block: &super::modules::HydraMidBlock<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self> {
        use faer::linalg::matmul::matmul;
        use faer::{Accum, MatMut, MatRef, Par};
        use rayon::prelude::*;

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
        let m = batch * seq_q;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // ---- Convert inputs and weights to contiguous F32 slices. ----
        let x_data = x_t.to_data();
        let x_slice = x_data
            .as_slice::<f32>()
            .expect("HydraMidBlock input is contiguous F32");

        let q_proj_w = block.q_proj_w_cache.as_slice();
        let q_proj_b = block.q_proj_b_cache.as_deref();
        debug_assert_eq!(q_proj_w.len(), hidden * hidden);

        let o_proj_w = block.o_proj_w_cache.as_slice();
        let o_proj_b = block.o_proj_b_cache.as_deref();
        debug_assert_eq!(o_proj_w.len(), hidden * hidden);

        // Norm parameters: Hydra uses no-affine LayerNorm (gamma=1, beta=0, eps=1e-5).
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
        block: &super::modules::HydraMidBlock<Self>,
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
        block: &super::modules::HydraMidBlock<Self>,
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

/// Backends that provide a fused HydraPool tail path.
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
        pool: &super::modules::HydraPool<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self>;
}

/// Tensor-level entry point for the fused pool tail path.
pub fn fused_hydra_pool_tail<B: FusedHydraPoolTailBackend>(
    x: Tensor<B, 3>,
    pool: &super::modules::HydraPool<B>,
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
    pool: &super::modules::HydraPool<burn::backend::candle::Candle>,
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
                let mut glu_contig = resize_buf(&mut workspace.e, m * glu_out_dim);
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
        pool: &super::modules::HydraPool<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        workspace: &mut BlockWorkspace,
    ) -> FloatTensor<Self> {
        use rayon::prelude::*;

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
        pool: &super::modules::HydraPool<Self>,
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
        pool: &super::modules::HydraPool<Self>,
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
    pool: &super::modules::HydraPool<B>,
    mask: Option<Tensor<B, 4, Bool>>,
    workspace: &mut BlockWorkspace,
) -> FloatTensor<B> {
    let [batch, seq, _x_hidden] = x_t.dims();

    let kv = pool.kv.forward(x_t);
    let kv = kv.reshape([
        batch,
        seq,
        2,
        super::modules::HYDRA_HEADS,
        super::modules::HYDRA_HEAD_DIM,
    ]);
    let kv = kv.permute([2, 0, 3, 1, 4]);
    let mut chunks: Vec<Tensor<B, 5>> = kv.split_with_sizes(vec![1, 1], 0);
    let v = chunks.swap_remove(1).reshape([
        batch,
        super::modules::HYDRA_HEADS,
        seq,
        super::modules::HYDRA_HEAD_DIM,
    ]);
    let k = pool.qk_norm.forward_fast(chunks.swap_remove(0).reshape([
        batch,
        super::modules::HYDRA_HEADS,
        seq,
        super::modules::HYDRA_HEAD_DIM,
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
        pool: &super::modules::HydraPool<Self>,
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
    pool: &super::modules::HydraPool<B>,
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
    pool: &super::modules::HydraPool<burn::backend::candle::Candle>,
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
        pool: &super::modules::HydraPool<Self>,
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

/// Backends that provide a fused NaFlex attention path.
pub trait FusedNaFlexAttnBackend: Backend {
    /// Compute `x + proj(attention(norm1(x)))` as a single dispatch.
    fn fused_na_flex_attn(
        x: FloatTensor<Self>,
        block: &super::modules::NaFlexBlock<Self>,
        mask: Option<BoolTensor<Self>>,
    ) -> FloatTensor<Self>;
}

#[cfg(feature = "burn-candle")]
impl FusedNaFlexAttnBackend for burn::backend::candle::Candle {
    fn fused_na_flex_attn(
        x: FloatTensor<Self>,
        block: &super::modules::NaFlexBlock<Self>,
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
        let heads = super::modules::NAFLEX_HEADS;
        let head_dim = super::modules::NAFLEX_HEAD_DIM;
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
        block: &super::modules::NaFlexBlock<Self>,
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
        block: &super::modules::NaFlexBlock<Self>,
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
