//! Load Hydra-3.5 weights from a `.safetensors` checkpoint into Burn modules.

use std::path::Path;

use anyhow::{Context, Result};
use burn::module::Param;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

use super::modules::{
    Hydra, HydraEmbeds, HydraFeedForward, HydraMidBlock, HydraPool, HydraRmsNorm, LinearHead,
    NaFlexAttn, NaFlexBlock, NaFlexMlp,
};
use super::MODEL_DTYPE;

/// Load Hydra-3.5 from a safetensors checkpoint.
pub fn load_hydra<B: Backend>(path: &Path, device: &B::Device) -> Result<Hydra<B>> {
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
        let tensor = Tensor::<B, 1>::from_data(
            data.as_slice(),
            (&device.clone(), MODEL_DTYPE),
        )
        .reshape([shape[0], shape[1], shape[2]]);
        Ok(tensor)
    };

    let get = |name: &str| -> Result<Tensor<B, 2>> {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
        let data = tensor_data(&view)?;
        let tensor = Tensor::<B, 1>::from_data(
            data.as_slice(),
            (&device.clone(), MODEL_DTYPE),
        )
        .reshape([shape[0], shape[1]]);
        Ok(tensor)
    };

    let get_1d = |name: &str| -> Result<Tensor<B, 1>> {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("missing 1D tensor: {name}"))?;
        let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
        let data = tensor_data(&view)?;
        let tensor = Tensor::<B, 1>::from_data(
            data.as_slice(),
            (&device.clone(), MODEL_DTYPE),
        )
        .reshape([shape[0]]);
        Ok(tensor)
    };

    let mut blocks = Vec::with_capacity(27);
    for i in 0..27 {
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
                qkv: load_linear(
                    &get(&format!("blocks.{i}.attn.qkv.weight"))?,
                    Some(&get_1d(&format!("blocks.{i}.attn.qkv.bias"))?),
                    device,
                ),
                proj: load_linear(
                    &get(&format!("blocks.{i}.attn.proj.weight"))?,
                    Some(&get_1d(&format!("blocks.{i}.attn.proj.bias"))?),
                    device,
                ),
            },
            mlp: NaFlexMlp {
                fc1: load_linear(
                    &get(&format!("blocks.{i}.mlp.fc1.weight"))?,
                    Some(&get_1d(&format!("blocks.{i}.mlp.fc1.bias"))?),
                    device,
                ),
                fc2: load_linear(
                    &get(&format!("blocks.{i}.mlp.fc2.weight"))?,
                    Some(&get_1d(&format!("blocks.{i}.mlp.fc2.bias"))?),
                    device,
                ),
            },
        });
    }

    let attn_pool = HydraPool {
        kv: load_linear(&get("attn_pool.kv.weight")?, None, device),
        q: Param::from_tensor(get_3d("attn_pool.q")?),
        qk_norm: HydraRmsNorm { eps: 1e-5 },
        ff: HydraFeedForward {
            norm: load_layer_norm_no_affine(2048, device),
            proj_in_weight: Param::from_tensor(get("attn_pool.ff.proj_in.weight")?),
            proj_out: load_linear(&get("attn_pool.ff.proj_out.weight")?, None, device),
        },
        mid_blocks: vec![HydraMidBlock {
            q_proj: load_linear(&get("attn_pool.mid_blocks.0.q_proj.weight")?, None, device),
            q_norm: HydraRmsNorm { eps: 1e-5 },
            o_proj: load_linear(&get("attn_pool.mid_blocks.0.o_proj.weight")?, None, device),
            ff: HydraFeedForward {
                norm: load_layer_norm_no_affine(2048, device),
                proj_in_weight: Param::from_tensor(get(
                    "attn_pool.mid_blocks.0.ff.proj_in.weight",
                )?),
                proj_out: load_linear(
                    &get("attn_pool.mid_blocks.0.ff.proj_out.weight")?,
                    None,
                    device,
                ),
            },
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
    let [out_features, in_features] = weight.dims();
    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(bias.is_some())
        .init(device);

    // PyTorch weight is [out, in]; Burn Row layout expects [in, out].
    let weight_t = weight.clone().transpose();
    linear.weight = Param::from_tensor(weight_t);

    if let Some(bias) = bias {
        linear.bias = Some(Param::from_tensor(bias.clone()));
    }

    linear
}

fn load_layer_norm<B: Backend>(
    weight: &Tensor<B, 1>,
    bias: &Tensor<B, 1>,
    device: &B::Device,
) -> LayerNorm<B> {
    let [d] = weight.dims();
    let mut ln = LayerNormConfig::new(d).init(device);
    ln.gamma = Param::from_tensor(weight.clone());
    ln.beta = Some(Param::from_tensor(bias.clone()));
    ln
}

fn load_layer_norm_no_affine<B: Backend>(d: usize, device: &B::Device) -> LayerNorm<B> {
    // Burn's LayerNorm always has a gamma scale. To emulate PyTorch's
    // elementwise_affine=False we leave gamma as ones and omit beta.
    LayerNormConfig::new(d).with_bias(false).init(device)
}

/// Decode a safetensors view into a flat Vec<f32>.
/// The returned vector is meant to be loaded into a Burn tensor with dtype BF16.
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
