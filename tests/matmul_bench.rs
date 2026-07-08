use std::time::Instant;

use candle_core::{Device, Tensor};
use faer::linalg::matmul::matmul;
use faer::{Accum, Par, MatRef, MatMut};

fn rand_vec(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 123456789) % 1000) as f32 / 1000.0 - 0.5).collect()
}

fn bench_candle(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> f64 {
    let dev = Device::Cpu;
    let a_t = Tensor::from_vec(a.to_vec(), (m, k), &dev).unwrap();
    let b_t = Tensor::from_vec(b.to_vec(), (k, n), &dev).unwrap();
    // Warmup
    for _ in 0..2 {
        let _ = a_t.matmul(&b_t).unwrap();
    }
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        let _ = a_t.matmul(&b_t).unwrap();
    }
    start.elapsed().as_secs_f64() / runs as f64
}

fn bench_faer(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> f64 {
    let mut c = vec![0.0f32; m * n];
    let a_ref = MatRef::from_row_major_slice(a, m, k);
    let b_ref = MatRef::from_row_major_slice(b, k, n);
    let mut c_mut = MatMut::from_row_major_slice_mut(&mut c, m, n);
    // Warmup
    for _ in 0..2 {
        matmul(c_mut.as_mut(), Accum::Replace, a_ref, b_ref, 1.0f32, Par::rayon(0));
    }
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        matmul(c_mut.as_mut(), Accum::Replace, a_ref, b_ref, 1.0f32, Par::rayon(0));
    }
    start.elapsed().as_secs_f64() / runs as f64
}

fn bench_faer_seq(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> f64 {
    let mut c = vec![0.0f32; m * n];
    let a_ref = MatRef::from_row_major_slice(a, m, k);
    let b_ref = MatRef::from_row_major_slice(b, k, n);
    let mut c_mut = MatMut::from_row_major_slice_mut(&mut c, m, n);
    for _ in 0..2 {
        matmul(c_mut.as_mut(), Accum::Replace, a_ref, b_ref, 1.0f32, Par::Seq);
    }
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        matmul(c_mut.as_mut(), Accum::Replace, a_ref, b_ref, 1.0f32, Par::Seq);
    }
    start.elapsed().as_secs_f64() / runs as f64
}

fn bench_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> f64 {
    let mut c = vec![0.0f32; m * n];
    for _ in 0..2 {
        gemm_row_major(m, n, k, a, b, &mut c);
    }
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        gemm_row_major(m, n, k, a, b, &mut c);
    }
    start.elapsed().as_secs_f64() / runs as f64
}

fn gemm_row_major(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            1,
            n as isize,
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
}

fn bench_shape(name: &str, m: usize, k: usize, n: usize) {
    let a = rand_vec(m * k);
    let b = rand_vec(k * n);
    let t_candle = bench_candle(&a, &b, m, k, n);
    let t_faer = bench_faer(&a, &b, m, k, n);
    let t_faer_seq = bench_faer_seq(&a, &b, m, k, n);
    let t_gemm = bench_gemm(&a, &b, m, k, n);
    println!("{name:30} ({m:5}x{k:5}) @ ({k:5}x{n:5})  candle={t_candle:.3}s  faer_ray={t_faer:.3}s  faer_seq={t_faer_seq:.3}s  gemm={t_gemm:.3}s");
}

#[test]
fn matmul_bench() {
    bench_shape("qkv", 1024, 1152, 3456);
    bench_shape("mlp_fc1", 1024, 1152, 4608);
    bench_shape("mlp_fc2", 1024, 4608, 1152);
    bench_shape("pool_kv", 1024, 1152, 4096);
    bench_shape("pool_q_proj", 8886, 2048, 2048);
    bench_shape("pool_o_proj", 8886, 2048, 2048);
    bench_shape("pool_ff", 8886, 2048, 10240);
    bench_shape("mid_attn_qk", 8886, 64, 1024);
    bench_shape("mid_attn_sv", 8886, 1024, 64);
    bench_shape("naflex_attn_qk", 1024, 72, 1024);
    bench_shape("naflex_attn_sv", 1024, 1024, 72);
}
