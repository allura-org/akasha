//! GPU (CubeCL CUDA) matmul microbenchmark.
//!
//! Answers: are BF16 matmuls hitting the tensor cores (expect >100 TFLOPS on
//! an RTX 4090) or the SIMT fallback (expect <40)? Hydra's hot GEMM shapes
//! are included alongside a large square GEMM where any half-decent kernel
//! should saturate.
//!
//! Run with: `cargo test --release --features burn-cuda --test gpu_matmul_bench -- --nocapture`

#![cfg(feature = "burn-cuda")]

use std::time::Instant;

use burn::tensor::DType;
use burn::tensor::backend::Backend;
use burn::prelude::*;

type B = burn::backend::Cuda;

fn bench_shape(m: usize, k: usize, n: usize, dtype: DType) -> f64 {
    let device = burn::backend::cuda::CudaDevice::default();
    let a = Tensor::<B, 2>::random(
        [m, k],
        burn::tensor::Distribution::Default,
        &device,
    )
    .cast(dtype);
    let b = Tensor::<B, 2>::random(
        [k, n],
        burn::tensor::Distribution::Default,
        &device,
    )
    .cast(dtype);

    // Warmup (also absorbs one-time kernel compilation / autotune).
    for _ in 0..3 {
        let c = a.clone().matmul(b.clone());
        std::hint::black_box(&c);
    }
    B::sync(&device).expect("sync");

    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        let c = a.clone().matmul(b.clone());
        std::hint::black_box(&c);
    }
    B::sync(&device).expect("sync");
    let secs = start.elapsed().as_secs_f64() / iters as f64;
    2.0 * (m as f64) * (k as f64) * (n as f64) / secs / 1e12
}

#[test]
fn gpu_matmul_tflops() {
    let shapes: &[(usize, usize, usize, &str)] = &[
        (1024, 1152, 3456, "hydra qkv"),
        (1024, 1152, 1152, "hydra proj"),
        (1024, 1152, 4608, "hydra fc1/fc2"),
        (8886, 2048, 64, "hydra pool head-ish"),
        (4096, 4096, 4096, "large square"),
    ];
    for &(m, k, n, label) in shapes {
        for dtype in [DType::BF16, DType::F32] {
            let tf = bench_shape(m, k, n, dtype);
            eprintln!("{label:20} [{m:5}x{k:5}x{n:5}] {dtype:?}: {tf:7.2} TFLOPS");
        }
    }
}
