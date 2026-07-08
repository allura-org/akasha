//! Load Hydra-3.5 weights from a `.safetensors` checkpoint into Burn modules.

use std::path::Path;

use anyhow::{Context, Result};
use burn::module::Param;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

use super::MODEL_DTYPE;
use super::fused_ops::{
    FastLinearBackend, FastRmsNormBackend, FusedAttentionBackend, FusedGluBackend,
    FusedHydraMidBlockBackend, FusedHydraPoolTailBackend, FusedMlpBackend, FusedNaFlexBlockBackend,
};
use super::modules::{
    Hydra, HydraEmbeds, HydraFeedForward, HydraMidBlock, HydraPool, HydraRmsNorm, LinearHead,
    NaFlexAttn, NaFlexBlock, NaFlexMlp,
};

/// Load Hydra-3.5 from a safetensors checkpoint.
pub fn load_hydra<
    B: FusedGluBackend
        + FusedMlpBackend
        + FusedAttentionBackend
        + FastLinearBackend
        + FastRmsNormBackend
        + FusedHydraMidBlockBackend
        + FusedHydraPoolTailBackend
        + FusedNaFlexBlockBackend,
>(
    path: &Path,
    device: &B::Device,
) -> Result<Hydra<B>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read model file: {}", path.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("failed to parse safetensors: {}", path.display()))?;

    let get_3d = |name: &str| -> Result<Tensor<B, 3>> {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("missing 3D tensor: {name}"))?;
        let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
        let data = tensor_data(&view)?;
        let tensor = Tensor::<B, 1>::from_data(data.as_slice(), (&device.clone(), MODEL_DTYPE))
            .reshape([shape[0], shape[1], shape[2]]);
        Ok(tensor)
    };

    let get = |name: &str| -> Result<Tensor<B, 2>> {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
        let data = tensor_data(&view)?;
        let tensor = Tensor::<B, 1>::from_data(data.as_slice(), (&device.clone(), MODEL_DTYPE))
            .reshape([shape[0], shape[1]]);
        Ok(tensor)
    };

    let get_1d = |name: &str| -> Result<Tensor<B, 1>> {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("missing 1D tensor: {name}"))?;
        let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
        let data = tensor_data(&view)?;
        let tensor = Tensor::<B, 1>::from_data(data.as_slice(), (&device.clone(), MODEL_DTYPE))
            .reshape([shape[0]]);
        Ok(tensor)
    };

    let mut blocks = Vec::with_capacity(27);
    for i in 0..27 {
        let (qkv, qkv_w_cache, qkv_b_cache) = load_linear_cached(
            &get(&format!("blocks.{i}.attn.qkv.weight"))?,
            Some(&get_1d(&format!("blocks.{i}.attn.qkv.bias"))?),
            device,
        );
        let (proj, proj_w_cache, proj_b_cache) = load_linear_cached(
            &get(&format!("blocks.{i}.attn.proj.weight"))?,
            Some(&get_1d(&format!("blocks.{i}.attn.proj.bias"))?),
            device,
        );
        blocks.push(NaFlexBlock {
            norm1: load_layer_norm(
                &get_1d(&format!("blocks.{i}.norm1.weight"))?,
                &get_1d(&format!("blocks.{i}.norm1.bias"))?,
                device,
            ),
            norm2: load_layer_norm(
                &get_1d(&format!("blocks.{i}.norm2.weight"))?,
                &get_1d(&format!("blocks.{i}.norm2.bias"))?,
                device,
            ),
            attn: NaFlexAttn {
                qkv,
                proj,
                qkv_w_cache,
                qkv_b_cache,
                proj_w_cache,
                proj_b_cache,
            },
            mlp: {
                let (fc1, fc1_w_cache, fc1_b_cache) = load_linear_cached(
                    &get(&format!("blocks.{i}.mlp.fc1.weight"))?,
                    Some(&get_1d(&format!("blocks.{i}.mlp.fc1.bias"))?),
                    device,
                );
                let (fc2, fc2_w_cache, fc2_b_cache) = load_linear_cached(
                    &get(&format!("blocks.{i}.mlp.fc2.weight"))?,
                    Some(&get_1d(&format!("blocks.{i}.mlp.fc2.bias"))?),
                    device,
                );
                NaFlexMlp {
                    fc1,
                    fc2,
                    fc1_w_cache,
                    fc1_b_cache,
                    fc2_w_cache,
                    fc2_b_cache,
                }
            },
        });
    }

    let pool_ff_glu_w = get("attn_pool.ff.proj_in.weight")?;
    let [pool_ff_glu_out2, pool_ff_glu_in] = pool_ff_glu_w.dims();
    let pool_ff_glu_w_cache = {
        let data = pool_ff_glu_w.to_data();
        let slice = data
            .as_slice::<f32>()
            .expect("pool ff glu weight is contiguous F32");
        transpose_row_major(slice, pool_ff_glu_out2, pool_ff_glu_in)
    };
    let (pool_ff_proj_out, pool_ff_proj_out_w_cache, pool_ff_proj_out_b_cache) =
        load_linear_cached(&get("attn_pool.ff.proj_out.weight")?, None, device);

    let (mid_q_proj, mid_q_proj_w_cache, mid_q_proj_b_cache) =
        load_linear_cached(&get("attn_pool.mid_blocks.0.q_proj.weight")?, None, device);
    let (mid_o_proj, mid_o_proj_w_cache, mid_o_proj_b_cache) =
        load_linear_cached(&get("attn_pool.mid_blocks.0.o_proj.weight")?, None, device);
    let mid_ff_glu_w = get("attn_pool.mid_blocks.0.ff.proj_in.weight")?;
    let [mid_ff_glu_out2, mid_ff_glu_in] = mid_ff_glu_w.dims();
    let mid_ff_glu_w_cache = {
        let data = mid_ff_glu_w.to_data();
        let slice = data
            .as_slice::<f32>()
            .expect("mid ff glu weight is contiguous F32");
        transpose_row_major(slice, mid_ff_glu_out2, mid_ff_glu_in)
    };
    let (mid_ff_proj_out, mid_ff_proj_out_w_cache, mid_ff_proj_out_b_cache) = load_linear_cached(
        &get("attn_pool.mid_blocks.0.ff.proj_out.weight")?,
        None,
        device,
    );

    let attn_pool = HydraPool {
        kv: load_linear(&get("attn_pool.kv.weight")?, None, device),
        q: Param::from_tensor(get_3d("attn_pool.q")?),
        qk_norm: HydraRmsNorm { eps: 1e-5 },
        ff: HydraFeedForward {
            norm: load_layer_norm_no_affine(2048, device),
            fused_glu_weight: Param::from_tensor(pool_ff_glu_w.transpose()),
            proj_out: pool_ff_proj_out,
            glu_w_cache: pool_ff_glu_w_cache,
            proj_out_w_cache: pool_ff_proj_out_w_cache,
            proj_out_b_cache: pool_ff_proj_out_b_cache,
        },
        mid_blocks: vec![HydraMidBlock {
            q_proj: mid_q_proj,
            q_norm: HydraRmsNorm { eps: 1e-5 },
            o_proj: mid_o_proj,
            ff: HydraFeedForward {
                norm: load_layer_norm_no_affine(2048, device),
                fused_glu_weight: Param::from_tensor(mid_ff_glu_w.transpose()),
                proj_out: mid_ff_proj_out,
                glu_w_cache: mid_ff_glu_w_cache,
                proj_out_w_cache: mid_ff_proj_out_w_cache,
                proj_out_b_cache: mid_ff_proj_out_b_cache,
            },
            q_proj_w_cache: mid_q_proj_w_cache,
            q_proj_b_cache: mid_q_proj_b_cache,
            o_proj_w_cache: mid_o_proj_w_cache,
            o_proj_b_cache: mid_o_proj_b_cache,
        }],
    };

    let model = Hydra {
        embeds: HydraEmbeds {
            proj: load_linear(
                &get("embeds.proj.weight")?,
                Some(&get_1d("embeds.proj.bias")?),
                device,
            ),
        },
        blocks,
        norm: load_layer_norm(&get_1d("norm.weight")?, &get_1d("norm.bias")?, device),
        attn_pool,
        head: LinearHead {
            weight: Param::from_tensor(get("head.weight")?),
        },
    };

    Ok(model)
}

fn load_linear<B: Backend>(
    weight: &Tensor<B, 2>,
    bias: Option<&Tensor<B, 1>>,
    device: &B::Device,
) -> Linear<B> {
    load_linear_cached(weight, bias, device).0
}

/// Load a `Linear` module plus contiguous F32 weight/bias caches for the fused
/// kernels. The cached weight is already transposed to Burn's row-major
/// `[in_features, out_features]` layout.
fn load_linear_cached<B: Backend>(
    weight: &Tensor<B, 2>,
    bias: Option<&Tensor<B, 1>>,
    device: &B::Device,
) -> (Linear<B>, Vec<f32>, Option<Vec<f32>>) {
    let [out_features, in_features] = weight.dims();
    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(bias.is_some())
        .init(device);

    // PyTorch weight is [out, in]; Burn Row layout expects [in, out].
    let weight_t = weight.clone().transpose();
    linear.weight = Param::from_tensor(weight_t);

    let weight_cache = {
        let data = weight.to_data();
        let slice = data.as_slice::<f32>().expect("weight is contiguous F32");
        transpose_row_major(slice, out_features, in_features)
    };

    let bias_cache = bias.map(|b| {
        let data = b.to_data();
        data.as_slice::<f32>()
            .expect("bias is contiguous F32")
            .to_vec()
    });

    if let Some(bias) = bias {
        linear.bias = Some(Param::from_tensor(bias.clone()));
    }

    (linear, weight_cache, bias_cache)
}

/// Transpose a row-major `[rows, cols]` matrix into `[cols, rows]` row-major.
fn transpose_row_major(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let src_base = r * cols;
        for c in 0..cols {
            out[c * rows + r] = data[src_base + c];
        }
    }
    out
}

fn load_layer_norm<B: Backend>(
    weight: &Tensor<B, 1>,
    bias: &Tensor<B, 1>,
    device: &B::Device,
) -> LayerNorm<B> {
    let [d] = weight.dims();
    let mut ln = LayerNormConfig::new(d).init(device);
    ln.gamma = Param::from_tensor(weight.clone().cast(super::MODEL_DTYPE));
    ln.beta = Some(Param::from_tensor(bias.clone().cast(super::MODEL_DTYPE)));
    ln
}

fn load_layer_norm_no_affine<B: Backend>(d: usize, device: &B::Device) -> LayerNorm<B> {
    // Burn's LayerNorm always has a gamma scale. To emulate PyTorch's
    // elementwise_affine=False we leave gamma as ones and omit beta, matching
    // the runtime dtype (F32).
    let mut ln = LayerNormConfig::new(d).with_bias(false).init(device);
    ln.gamma = Param::from_tensor(ln.gamma.val().cast(super::MODEL_DTYPE));
    ln
}

/// Decode a safetensors view into a flat Vec<f32>.
/// The returned vector is loaded into a Burn tensor with dtype F32.
fn tensor_data(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>> {
    match view.dtype() {
        safetensors::Dtype::BF16 => {
            let bytes = view.data();
            anyhow::ensure!(bytes.len() % 2 == 0, "invalid bf16 tensor data");
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let val = half::bf16::from_bits(bits);
                out.push(val.to_f32());
            }
            Ok(out)
        }
        safetensors::Dtype::F32 => {
            let bytes = view.data();
            anyhow::ensure!(bytes.len() % 4 == 0, "invalid f32 tensor data");
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        other => anyhow::bail!("unsupported safetensors dtype for Hydra: {other:?}"),
    }
}
