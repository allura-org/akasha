//! Custom fused CubeCL kernels for Hydra GPU backends (CUDA, and later wgpu).
//!
//! Each kernel is written against pure cubecl abstractions (no PTX or vendor
//! intrinsics), so it JIT-compiles for any CubeCL runtime. Kernels are
//! registered with the burn-fusion stream as `CustomOpIr` operations: inputs
//! are resolved to concrete `CubeTensor`s when the stream reaches the op, the
//! kernel launches onto the same CubeCL runtime/stream burn already uses, and
//! outputs are registered back as lazy fusion tensors. No host round-trips.
//!
//! Kernel 1: LayerNorm. Burn's default LayerNorm decomposes into ~7 kernel
//! launches (two mean reductions, sub/mul/add chains); Hydra runs 54 of them
//! per image. `gpu_layernorm` does it in one launch, and `gpu_add_layernorm`
//! additionally fuses the preceding residual add, emitting both the sum
//! (still needed as the next residual) and the normalized tensor.
//!
//! IMPORTANT (cubecl 0.10): the runtime allocator row-pitches tensors when a
//! row's byte size is not a multiple of 512 (e.g. BF16 [*, 1152] gets row
//! stride 1280, not 1152). Kernels must therefore index through `LinearView`
//! (layout-aware), never through raw packed-row arithmetic. This was found
//! the hard way: F32 [37, 1152] is naturally aligned and works either way,
//! BF16 is silently corrupted by packed indexing.

use std::marker::PhantomData;

use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, CubeRuntime};
use burn_fusion::stream::{Operation, OperationStreams};
use burn_fusion::{FusionRuntime, FusionTensor};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use cubecl::prelude::*;
use cubecl::std::tensor::layout::linear::LinearView;

use burn::tensor::TensorMetadata;

/// Burn's LayerNorm default epsilon (its config field is private); the CPU
/// fused path hardcodes the same value.
pub(crate) const LN_EPS: f32 = 1e-5;

/// Whether the custom fused LayerNorm kernels are active (default on).
/// `AKASHA_HYDRA_GPU_FUSED_LN=0` falls back to burn's decomposed LayerNorm.
pub(crate) fn fused_ln_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("AKASHA_HYDRA_GPU_FUSED_LN").as_deref() != Ok("0"))
}

/// Whether the custom fused bias+GELU kernel is active (default on).
/// `AKASHA_HYDRA_GPU_FUSED_GELU=0` falls back to burn's decomposed epilogue.
pub(crate) fn fused_gelu_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("AKASHA_HYDRA_GPU_FUSED_GELU").as_deref() != Ok("0"))
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// One cube (= one plane of 32 units) per row. Pass 1 computes the row mean,
/// pass 2 the variance, pass 3 normalizes — all stats in F32 regardless of
/// the storage dtype. `beta` may alias `gamma` when `has_beta == 0` (it is
/// never read then). All indexing goes through `LinearView` so row-pitched
/// layouts are handled correctly (see module docs).
///
/// cubecl note: `Vector::vector_sum(plane_sum(v))` yields a value *typed*
/// `Vector<f32,N>` that is a scalar at IR level; mixed Vector/f32 arithmetic
/// does not compile, so stats are extracted with `f32::cast_from` and
/// broadcast back with `Vector::empty().fill(..)` (the cubek-reduce pattern).
#[cube(launch, address_type = "dynamic")]
fn layernorm_kernel<F: Float, N: Size>(
    x: &LinearView<Vector<F, N>>,
    gamma: &LinearView<Vector<F, N>>,
    beta: &LinearView<Vector<F, N>>,
    out: &mut LinearView<Vector<F, N>, ReadWrite>,
    n_vec: u32,
    has_beta: u32,
    inv_hidden: f32,
    eps: f32,
    #[define(F)] _dtype: StorageType,
) {
    let n_vec = n_vec as usize;
    let row = CUBE_POS_X as usize;
    let base = row * n_vec;
    if !x.is_in_bounds(base) {
        terminate!();
    }
    let lane = UNIT_POS_X as usize;
    let stride = CUBE_DIM_X as usize;

    let mut sum = Vector::<f32, N>::zeroed();
    let mut i = lane;
    while i < n_vec {
        let v: Vector<f32, N> = Vector::cast_from(x[base + i]);
        sum += v;
        i += stride;
    }
    let mean = f32::cast_from(Vector::vector_sum(plane_sum(sum))) * inv_hidden;
    let mean_v = Vector::<f32, N>::empty().fill(mean);

    let mut var_acc = Vector::<f32, N>::zeroed();
    let mut i = lane;
    while i < n_vec {
        let v: Vector<f32, N> = Vector::cast_from(x[base + i]);
        let d = v - mean_v;
        var_acc += d * d;
        i += stride;
    }
    let var = f32::cast_from(Vector::vector_sum(plane_sum(var_acc))) * inv_hidden;
    let inv_std = (var + eps).inverse_sqrt();
    let inv_v = Vector::<f32, N>::empty().fill(inv_std);

    let mut j = lane;
    while j < n_vec {
        let v: Vector<f32, N> = Vector::cast_from(x[base + j]);
        let g: Vector<f32, N> = Vector::cast_from(gamma[j]);
        let b: Vector<f32, N> = if has_beta == 1u32 {
            Vector::cast_from(beta[j])
        } else {
            Vector::<f32, N>::zeroed()
        };
        out[base + j] = Vector::cast_from((v - mean_v) * inv_v * g + b);
        j += stride;
    }
}

/// Fused `sum = x + residual; norm = layer_norm(sum)`. One cube per row; the
/// sum is written in pass 1 and read back (L2-hot) for the stats/normalize
/// passes. Same stats discipline as `layernorm_kernel`.
#[cube(launch, address_type = "dynamic")]
fn add_layernorm_kernel<F: Float, N: Size>(
    x: &LinearView<Vector<F, N>>,
    residual: &LinearView<Vector<F, N>>,
    gamma: &LinearView<Vector<F, N>>,
    beta: &LinearView<Vector<F, N>>,
    out_sum: &mut LinearView<Vector<F, N>, ReadWrite>,
    out_norm: &mut LinearView<Vector<F, N>, ReadWrite>,
    n_vec: u32,
    has_beta: u32,
    inv_hidden: f32,
    eps: f32,
    #[define(F)] _dtype: StorageType,
) {
    let n_vec = n_vec as usize;
    let row = CUBE_POS_X as usize;
    let base = row * n_vec;
    if !x.is_in_bounds(base) {
        terminate!();
    }
    let lane = UNIT_POS_X as usize;
    let stride = CUBE_DIM_X as usize;

    let mut sum = Vector::<f32, N>::zeroed();
    let mut i = lane;
    while i < n_vec {
        let v: Vector<f32, N> = Vector::cast_from(x[base + i])
            + Vector::cast_from(residual[base + i]);
        out_sum[base + i] = Vector::cast_from(v);
        sum += v;
        i += stride;
    }
    let mean = f32::cast_from(Vector::vector_sum(plane_sum(sum))) * inv_hidden;
    let mean_v = Vector::<f32, N>::empty().fill(mean);

    let mut var_acc = Vector::<f32, N>::zeroed();
    let mut i = lane;
    while i < n_vec {
        let v: Vector<f32, N> = Vector::cast_from(out_sum[base + i]);
        let d = v - mean_v;
        var_acc += d * d;
        i += stride;
    }
    let var = f32::cast_from(Vector::vector_sum(plane_sum(var_acc))) * inv_hidden;
    let inv_std = (var + eps).inverse_sqrt();
    let inv_v = Vector::<f32, N>::empty().fill(inv_std);

    let mut j = lane;
    while j < n_vec {
        let v: Vector<f32, N> = Vector::cast_from(out_sum[base + j]);
        let g: Vector<f32, N> = Vector::cast_from(gamma[j]);
        let b: Vector<f32, N> = if has_beta == 1u32 {
            Vector::cast_from(beta[j])
        } else {
            Vector::<f32, N>::zeroed()
        };
        out_norm[base + j] = Vector::cast_from((v - mean_v) * inv_v * g + b);
        j += stride;
    }
}

// ---------------------------------------------------------------------------
// Host-side launch helpers (generic over the CubeCL runtime)
// ---------------------------------------------------------------------------

/// Fused `gelu_tanh(x + bias)` epilogue for the NaFlex MLP fc1 projection.
/// One elementwise launch; GELU is the PyTorch tanh approximation
/// `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`, computed in F32.
#[cube(launch, address_type = "dynamic")]
fn bias_gelu_kernel<F: Float, N: Size>(
    x: &LinearView<Vector<F, N>>,
    bias: &LinearView<Vector<F, N>>,
    out: &mut LinearView<Vector<F, N>, ReadWrite>,
    n_vec: u32,
    #[define(F)] _dtype: StorageType,
) {
    if !out.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    let col = ABSOLUTE_POS % (n_vec as usize);
    let v: Vector<f32, N> =
        Vector::cast_from(x[ABSOLUTE_POS]) + Vector::cast_from(bias[col]);
    let sqrt_2_over_pi = Vector::<f32, N>::empty().fill(0.7978845608028654);
    let coeff = Vector::<f32, N>::empty().fill(0.044715);
    let half = Vector::<f32, N>::empty().fill(0.5);
    let one = Vector::<f32, N>::empty().fill(1.0);
    let inner = (v * coeff * (v * v) + v) * sqrt_2_over_pi;
    out[ABSOLUTE_POS] = Vector::cast_from(v * half * (inner.tanh() + one));
}

fn launch_bias_gelu<R: CubeRuntime>(
    x: CubeTensor<R>,
    bias: CubeTensor<R>,
) -> CubeTensor<R> {
    assert_eq!(x.dtype, bias.dtype, "bias_gelu dtype mismatch");
    let client = x.client.clone();
    let device = x.device.clone();
    let shape = x.shape();
    let dtype = x.dtype;
    let rank = shape.num_dims();
    let hidden = shape[rank - 1];
    let vec = vector_size(&x, hidden).min(vector_size(&bias, hidden));
    let n_vec = hidden / vec;
    let working_units = shape.num_elements() / vec;
    let out = empty_device_dtype(client.clone(), device, shape.clone(), dtype);
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = cubecl::calculate_cube_count_elemwise(&client, working_units, cube_dim);
    bias_gelu_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        AddressType::U32,
        vec,
        x.into_linear_view(),
        bias.into_linear_view(),
        out.clone().into_linear_view(),
        n_vec as u32,
        dtype.into(),
    );
    out
}


/// Widest I/O vector size the runtime supports that also divides `hidden`.
/// Row pitch is always a multiple of 512 bytes, so it never constrains the
/// vector size for the element types we use.
fn vector_size<R: CubeRuntime>(t: &CubeTensor<R>, hidden: usize) -> usize {
    for v in t.client.io_optimized_vector_sizes(t.dtype.size()) {
        if hidden.is_multiple_of(v) {
            return v;
        }
    }
    1
}

fn launch_layernorm<R: CubeRuntime>(
    x: CubeTensor<R>,
    gamma: CubeTensor<R>,
    beta: Option<CubeTensor<R>>,
    eps: f32,
) -> CubeTensor<R> {
    assert_eq!(x.dtype, gamma.dtype, "layernorm dtype mismatch");
    let client = x.client.clone();
    let device = x.device.clone();
    let shape = x.shape();
    let dtype = x.dtype;
    let rank = shape.num_dims();
    let hidden = shape[rank - 1];
    let rows = shape.num_elements() / hidden;
    let vec = vector_size(&x, hidden);
    let n_vec = hidden / vec;
    let has_beta = beta.is_some();
    let beta_t = beta.unwrap_or_else(|| gamma.clone());
    let out = empty_device_dtype(client.clone(), device, shape.clone(), dtype);
    layernorm_kernel::launch::<R>(
        &client,
        CubeCount::new_1d(rows as u32),
        CubeDim::new_1d(32),
        AddressType::U32,
        vec,
        x.into_linear_view(),
        gamma.into_linear_view(),
        beta_t.into_linear_view(),
        out.clone().into_linear_view(),
        n_vec as u32,
        u32::from(has_beta),
        1.0f32 / hidden as f32,
        eps,
        dtype.into(),
    );
    out
}

fn launch_add_layernorm<R: CubeRuntime>(
    x: CubeTensor<R>,
    residual: CubeTensor<R>,
    gamma: CubeTensor<R>,
    beta: Option<CubeTensor<R>>,
    eps: f32,
) -> (CubeTensor<R>, CubeTensor<R>) {
    assert_eq!(x.dtype, gamma.dtype, "add_layernorm dtype mismatch");
    assert_eq!(x.shape(), residual.shape(), "add_layernorm shape mismatch");
    let client = x.client.clone();
    let device = x.device.clone();
    let shape = x.shape();
    let dtype = x.dtype;
    let rank = shape.num_dims();
    let hidden = shape[rank - 1];
    let rows = shape.num_elements() / hidden;
    let vec = vector_size(&x, hidden).min(vector_size(&residual, hidden));
    let n_vec = hidden / vec;
    let has_beta = beta.is_some();
    let beta_t = beta.unwrap_or_else(|| gamma.clone());
    let out_sum = empty_device_dtype(client.clone(), device.clone(), shape.clone(), dtype);
    let out_norm = empty_device_dtype(client.clone(), device, shape.clone(), dtype);
    add_layernorm_kernel::launch::<R>(
        &client,
        CubeCount::new_1d(rows as u32),
        CubeDim::new_1d(32),
        AddressType::U32,
        vec,
        x.into_linear_view(),
        residual.into_linear_view(),
        gamma.into_linear_view(),
        beta_t.into_linear_view(),
        out_sum.clone().into_linear_view(),
        out_norm.clone().into_linear_view(),
        n_vec as u32,
        u32::from(has_beta),
        1.0f32 / hidden as f32,
        eps,
        dtype.into(),
    );
    (out_sum, out_norm)
}

// ---------------------------------------------------------------------------
// burn-fusion custom-op wrappers
// ---------------------------------------------------------------------------

/// The inner (non-fusion) cube backend used to resolve/register tensors in
/// the fusion handle container. The float/int/bool element type parameters do
/// not affect handle resolution, and burn's default `Cuda`/`Wgpu` aliases all
/// use `CubeBackend<R, f32, i32, u8>`.
type InnerB<R> = CubeBackend<R, f32, i32, u8>;

#[derive(Clone, Debug)]
struct LayerNormOp<R: CubeRuntime> {
    desc: CustomOpIr,
    has_beta: bool,
    eps: f32,
    _r: PhantomData<R>,
}

impl<R: CubeRuntime> Operation<FusionCubeRuntime<R>> for LayerNormOp<R> {
    fn execute(
        &self,
        handles: &mut HandleContainer<<FusionCubeRuntime<R> as FusionRuntime>::FusionHandle>,
    ) {
        let ([x, gamma, beta], [out]) = self.desc.as_fixed();
        let x = handles.get_float_tensor::<InnerB<R>>(x);
        let gamma = handles.get_float_tensor::<InnerB<R>>(gamma);
        let beta = self
            .has_beta
            .then(|| handles.get_float_tensor::<InnerB<R>>(beta));
        let out_t = launch_layernorm(x, gamma, beta, self.eps);
        handles.register_float_tensor::<InnerB<R>>(&out.id, out_t);
    }
}

#[derive(Clone, Debug)]
struct AddLayerNormOp<R: CubeRuntime> {
    desc: CustomOpIr,
    has_beta: bool,
    eps: f32,
    _r: PhantomData<R>,
}

impl<R: CubeRuntime> Operation<FusionCubeRuntime<R>> for AddLayerNormOp<R> {
    fn execute(
        &self,
        handles: &mut HandleContainer<<FusionCubeRuntime<R> as FusionRuntime>::FusionHandle>,
    ) {
        let ([x, residual, gamma, beta], [out_sum, out_norm]) = self.desc.as_fixed();
        let x = handles.get_float_tensor::<InnerB<R>>(x);
        let residual = handles.get_float_tensor::<InnerB<R>>(residual);
        let gamma = handles.get_float_tensor::<InnerB<R>>(gamma);
        let beta = self
            .has_beta
            .then(|| handles.get_float_tensor::<InnerB<R>>(beta));
        let (sum_t, norm_t) = launch_add_layernorm(x, residual, gamma, beta, self.eps);
        handles.register_float_tensor::<InnerB<R>>(&out_sum.id, sum_t);
        handles.register_float_tensor::<InnerB<R>>(&out_norm.id, norm_t);
    }
}

#[derive(Clone, Debug)]
struct BiasGeluOp<R: CubeRuntime> {
    desc: CustomOpIr,
    _r: PhantomData<R>,
}

impl<R: CubeRuntime> Operation<FusionCubeRuntime<R>> for BiasGeluOp<R> {
    fn execute(
        &self,
        handles: &mut HandleContainer<<FusionCubeRuntime<R> as FusionRuntime>::FusionHandle>,
    ) {
        let ([x, bias], [out]) = self.desc.as_fixed();
        let x = handles.get_float_tensor::<InnerB<R>>(x);
        let bias = handles.get_float_tensor::<InnerB<R>>(bias);
        let out_t = launch_bias_gelu(x, bias);
        handles.register_float_tensor::<InnerB<R>>(&out.id, out_t);
    }
}

/// `gelu_tanh(x + bias)` in a single kernel launch (NaFlex MLP fc1 epilogue).
pub(crate) fn gpu_bias_gelu<R: CubeRuntime>(
    x: &FusionTensor<FusionCubeRuntime<R>>,
    bias: &FusionTensor<FusionCubeRuntime<R>>,
) -> FusionTensor<FusionCubeRuntime<R>> {
    let client = x.client.clone();
    let streams = OperationStreams::with_inputs([x, bias]);
    let out_ir = TensorIr::uninit(client.create_empty_handle(), x.shape.clone(), x.dtype);
    let desc = CustomOpIr::new(
        "hydra_bias_gelu",
        &[x.clone().into_ir(), bias.clone().into_ir()],
        &[out_ir],
    );
    let op = BiasGeluOp::<R> {
        desc: desc.clone(),
        _r: PhantomData,
    };
    client
        .register(streams, OperationIr::Custom(desc), op)
        .output()
}

/// `layer_norm(x)` in a single kernel launch. All tensors must share the same
/// dtype (BF16 on Hydra GPU builds); stats accumulate in F32.
pub(crate) fn gpu_layernorm<R: CubeRuntime>(
    x: &FusionTensor<FusionCubeRuntime<R>>,
    gamma: &FusionTensor<FusionCubeRuntime<R>>,
    beta: Option<&FusionTensor<FusionCubeRuntime<R>>>,
    eps: f32,
) -> FusionTensor<FusionCubeRuntime<R>> {
    let client = x.client.clone();
    let beta_t = beta.unwrap_or(gamma);
    let streams = OperationStreams::with_inputs([x, gamma, beta_t]);
    let out_ir = TensorIr::uninit(client.create_empty_handle(), x.shape.clone(), x.dtype);
    let desc = CustomOpIr::new(
        "hydra_layernorm",
        &[x.clone().into_ir(), gamma.clone().into_ir(), beta_t.clone().into_ir()],
        &[out_ir],
    );
    let op = LayerNormOp::<R> {
        desc: desc.clone(),
        has_beta: beta.is_some(),
        eps,
        _r: PhantomData,
    };
    client
        .register(streams, OperationIr::Custom(desc), op)
        .output()
}

/// `(x + residual, layer_norm(x + residual))` in a single kernel launch.
pub(crate) fn gpu_add_layernorm<R: CubeRuntime>(
    x: &FusionTensor<FusionCubeRuntime<R>>,
    residual: &FusionTensor<FusionCubeRuntime<R>>,
    gamma: &FusionTensor<FusionCubeRuntime<R>>,
    beta: Option<&FusionTensor<FusionCubeRuntime<R>>>,
    eps: f32,
) -> (
    FusionTensor<FusionCubeRuntime<R>>,
    FusionTensor<FusionCubeRuntime<R>>,
) {
    let client = x.client.clone();
    let beta_t = beta.unwrap_or(gamma);
    let streams = OperationStreams::with_inputs([x, residual, gamma, beta_t]);
    let out_sum_ir = TensorIr::uninit(client.create_empty_handle(), x.shape.clone(), x.dtype);
    let out_norm_ir = TensorIr::uninit(client.create_empty_handle(), x.shape.clone(), x.dtype);
    let desc = CustomOpIr::new(
        "hydra_add_layernorm",
        &[
            x.clone().into_ir(),
            residual.clone().into_ir(),
            gamma.clone().into_ir(),
            beta_t.clone().into_ir(),
        ],
        &[out_sum_ir, out_norm_ir],
    );
    let op = AddLayerNormOp::<R> {
        desc: desc.clone(),
        has_beta: beta.is_some(),
        eps,
        _r: PhantomData,
    };
    let [sum, norm] = client
        .register(streams, OperationIr::Custom(desc), op)
        .try_into()
        .expect("hydra_add_layernorm produces two outputs");
    (sum, norm)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "burn-cuda"))]
mod tests {
    use super::*;
    use burn::tensor::{DType, Tensor, TensorData, TensorPrimitive};
    use cubecl::cuda::{CudaDevice, CudaRuntime};

    use crate::models::burn::init_cubecl_runtime;

    type Inner = CubeBackend<CudaRuntime, f32, i32, u8>;

    fn to_cube<const D: usize>(t: Tensor<Inner, D>) -> CubeTensor<CudaRuntime> {
        t.into_primitive().tensor()
    }

    fn from_cube<const D: usize>(t: CubeTensor<CudaRuntime>) -> Tensor<Inner, D> {
        Tensor::from_primitive(TensorPrimitive::Float(t))
    }

    fn pattern(len: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37 + seed * 11) % 211) as f32 - 105.0) * scale / 37.0)
            .collect()
    }

    /// Reference LayerNorm in F32 via burn's high-level ops.
    fn reference_ln(
        x: &Tensor<Inner, 2>,
        gamma: &Tensor<Inner, 1>,
        beta: Option<&Tensor<Inner, 1>>,
        eps: f64,
    ) -> Tensor<Inner, 2> {
        let hidden = x.dims()[1];
        let mean = x.clone().mean_dim(1);
        let centered = x.clone() - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(1);
        let inv_std = (var + eps).sqrt().recip();
        let normed = centered * inv_std;
        let scaled = normed * gamma.clone().reshape([1, hidden]);
        match beta {
            Some(b) => scaled + b.clone().reshape([1, hidden]),
            None => scaled,
        }
    }

    #[test]
    fn layernorm_kernel_matches_reference() {
        init_cubecl_runtime();
        let device = CudaDevice::new(0);
        let (rows, hidden) = (37usize, 1152usize);

        let x_f32 = Tensor::<Inner, 2>::from_data(
            TensorData::new(pattern(rows * hidden, 1, 4.0), [rows, hidden]),
            &device,
        );
        let g_f32 = Tensor::<Inner, 1>::from_data(
            TensorData::new(pattern(hidden, 2, 0.5), [hidden]),
            &device,
        );
        let b_f32 = Tensor::<Inner, 1>::from_data(
            TensorData::new(pattern(hidden, 3, 0.25), [hidden]),
            &device,
        );

        let expected = reference_ln(&x_f32, &g_f32, Some(&b_f32), LN_EPS as f64);

        for dtype in [DType::F32, DType::BF16] {
            let x = x_f32.clone().cast(dtype);
            let gamma = g_f32.clone().cast(dtype);
            let beta = b_f32.clone().cast(dtype);
            let out = from_cube::<2>(launch_layernorm(
                to_cube(x),
                to_cube(gamma),
                Some(to_cube(beta)),
                LN_EPS,
            ))
            .cast(DType::F32);

            let diff: f32 = (out - expected.clone()).abs().max().into_scalar();
            println!("layernorm kernel max abs diff ({dtype:?}): {diff}");
            assert!(diff < 0.05, "layernorm kernel {dtype:?} diff: {diff}");

            // Same accuracy check for burn's decomposed LayerNorm module on
            // the same backend/dtype, as an accuracy baseline.
            let mut ln = burn::nn::LayerNormConfig::new(hidden).init::<Inner>(&device);
            ln.gamma = burn::module::Param::from_tensor(g_f32.clone().cast(dtype));
            ln.beta = Some(burn::module::Param::from_tensor(b_f32.clone().cast(dtype)));
            let burn_out = ln
                .forward(x_f32.clone().cast(dtype))
                .cast(DType::F32);
            let burn_diff: f32 = (burn_out - expected.clone()).abs().max().into_scalar();
            println!("burn decomposed LN max abs diff ({dtype:?}): {burn_diff}");
        }
    }

    #[test]
    fn add_layernorm_kernel_matches_reference() {
        init_cubecl_runtime();
        let device = CudaDevice::new(0);
        let (rows, hidden) = (37usize, 1152usize);

        let x_f32 = Tensor::<Inner, 2>::from_data(
            TensorData::new(pattern(rows * hidden, 1, 4.0), [rows, hidden]),
            &device,
        );
        let r_f32 = Tensor::<Inner, 2>::from_data(
            TensorData::new(pattern(rows * hidden, 5, 2.0), [rows, hidden]),
            &device,
        );
        let g_f32 = Tensor::<Inner, 1>::from_data(
            TensorData::new(pattern(hidden, 2, 0.5), [hidden]),
            &device,
        );
        let b_f32 = Tensor::<Inner, 1>::from_data(
            TensorData::new(pattern(hidden, 3, 0.25), [hidden]),
            &device,
        );

        let sum_expected = x_f32.clone() + r_f32.clone();
        let norm_expected = reference_ln(&sum_expected, &g_f32, Some(&b_f32), LN_EPS as f64);

        for dtype in [DType::F32, DType::BF16] {
            let x = x_f32.clone().cast(dtype);
            let residual = r_f32.clone().cast(dtype);
            let gamma = g_f32.clone().cast(dtype);
            let beta = b_f32.clone().cast(dtype);
            let (sum_t, norm_t) = launch_add_layernorm(
                to_cube(x),
                to_cube(residual),
                to_cube(gamma),
                Some(to_cube(beta)),
                LN_EPS,
            );
            let sum = from_cube::<2>(sum_t).cast(DType::F32);
            let norm = from_cube::<2>(norm_t).cast(DType::F32);

            let sum_diff: f32 = (sum - sum_expected.clone()).abs().max().into_scalar();
            let norm_diff: f32 = (norm - norm_expected.clone()).abs().max().into_scalar();
            println!(
                "add_layernorm kernel max abs diff ({dtype:?}): sum={sum_diff} norm={norm_diff}"
            );
            // sum magnitudes reach ~16 where BF16 half-ulp is 0.0625.
            let sum_tol = if dtype == DType::BF16 { 0.07 } else { 0.05 };
            assert!(sum_diff < sum_tol, "add_layernorm {dtype:?} sum diff: {sum_diff}");
            assert!(
                norm_diff < 0.05,
                "add_layernorm {dtype:?} norm diff: {norm_diff}"
            );
        }
    }

    #[test]
    fn bias_gelu_kernel_matches_reference() {
        init_cubecl_runtime();
        let device = CudaDevice::new(0);
        let (rows, hidden) = (37usize, 4304usize);

        let x_f32 = Tensor::<Inner, 2>::from_data(
            TensorData::new(pattern(rows * hidden, 1, 3.0), [rows, hidden]),
            &device,
        );
        let b_f32 = Tensor::<Inner, 1>::from_data(
            TensorData::new(pattern(hidden, 3, 0.25), [hidden]),
            &device,
        );

        let gelu = |v: Tensor<Inner, 2>| {
            let inner = (v.clone().powf_scalar(3.0) * 0.044715 + v.clone()) * 0.7978845608028654;
            v * 0.5 * (inner.tanh() + 1.0)
        };
        let expected = gelu(x_f32.clone() + b_f32.clone().reshape([1, hidden]));

        for dtype in [DType::F32, DType::BF16] {
            let x = x_f32.clone().cast(dtype);
            let bias = b_f32.clone().cast(dtype);
            let out = from_cube::<2>(launch_bias_gelu(to_cube(x), to_cube(bias)))
                .cast(DType::F32);
            let diff: f32 = (out - expected.clone()).abs().max().into_scalar();
            println!("bias_gelu kernel max abs diff ({dtype:?}): {diff}");
            // x magnitudes reach ~9 where BF16 half-ulp is 0.03125.
            let tol = if dtype == DType::BF16 { 0.06 } else { 0.001 };
            assert!(diff < tol, "bias_gelu {dtype:?} diff: {diff}");
        }
    }
}
