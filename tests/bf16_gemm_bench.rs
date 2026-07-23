//! Prototype BF16 GEMM benchmark for the Hydra-3.5 hot shapes.
//!
//! Hypothesis: the 7950X has AVX512-BF16 (`VDPBF16PS`, 2x multiplier density
//! vs F32 FMA) and the Hydra weights are natively BF16, so a BF16 GEMM with
//! F32 accumulation should roughly double throughput over the `gemm`/`faer`
//! F32 path that currently dominates inference time.
//!
//! Design:
//! - B (weights) is pre-packed once into 32-column tiles of k/2 pair-rows,
//!   where each u32 lane holds `{B[2k][c], B[2k+1][c]}`. Packing happens once
//!   per model load, so it amortizes across all images — unlike the F32 path
//!   where `gemm`/`faer` re-pack the weight operand on every call.
//! - The microkernel computes MR=6 rows x NC=32 columns per tile with
//!   `VDPBF16PS` accumulating into 12 zmm F32 registers.
//! - Parallelism is over row tiles (rayon); each thread streams the packed B
//!   from L3 (42 MB packed for the pool_ff shape, L3-resident).
//!
//! Run with: `cargo test --release --test bf16_gemm_bench -- --nocapture`

#![allow(unsafe_op_in_unsafe_fn)]

use half::bf16;
use rayon::prelude::*;
use std::time::Instant;

/// Rows per microkernel tile. 6 rows x 2 zmm accumulators = 12 of 32 zmm regs.
const MR: usize = 6;
/// Columns per microkernel tile (2 x 16 F32 lanes).
const NC: usize = 32;

fn rand_vec(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 123456789) % 1000) as f32 / 1000.0 - 0.5)
        .collect()
}

fn to_bf16(src: &[f32]) -> Vec<bf16> {
    src.iter().map(|&v| bf16::from_f32(v)).collect()
}

/// Pack B `[k, n]` (row-major BF16) into 32-column tiles of k/2 pair-rows.
///
/// Per tile, per k-pair `kp`, lane `l` of the two zmm-wide halves holds
/// `{B[2kp][c0+l], B[2kp+1][c0+l]}` packed into a u32. This lets the kernel
/// broadcast one A pair and FMA against 32 columns per instruction pair.
fn pack_b(b: &[bf16], k: usize, n: usize) -> Vec<u32> {
    assert_eq!(k % 2, 0, "prototype requires even k");
    let n_tiles = n.div_ceil(NC);
    let mut packed = vec![0u32; n_tiles * (k / 2) * NC];
    for nt in 0..n_tiles {
        let c0 = nt * NC;
        for kp in 0..k / 2 {
            for l in 0..NC {
                if c0 + l < n {
                    let lo = b[2 * kp * n + c0 + l].to_bits() as u32;
                    let hi = b[(2 * kp + 1) * n + c0 + l].to_bits() as u32;
                    packed[(nt * (k / 2) + kp) * NC + l] = lo | (hi << 16);
                }
            }
        }
    }
    packed
}

/// `C[m, n] (f32) = A[m, k] (bf16) @ packed_B` with pre-packed B.
fn bf16_gemm_packed(a: &[bf16], packed_b: &[u32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16") {
        unsafe {
            return bf16_gemm_avx512(a, packed_b, c, m, n, k);
        }
    }
    bf16_gemm_scalar(a, packed_b, c, m, n, k);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn bf16_gemm_avx512(
    a: &[bf16],
    packed_b: &[u32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use std::arch::x86_64::*;

    let n_tiles = n.div_ceil(NC);
    let tile_stride = (k / 2) * NC;
    let kh = k / 2;

    // One task per MR-row tile; the last tile may be partial.
    c.par_chunks_mut(MR * n).enumerate().for_each(|(mt, c_chunk)| {
        let m0 = mt * MR;
        let rows = (m - m0).min(MR);
        let c_base = c_chunk.as_mut_ptr();
        for nt in 0..n_tiles {
            let b_tile = packed_b.as_ptr().add(nt * tile_stride);
            micro_6x32(a, b_tile, c_base, m0, rows, nt * NC, n, k, kh);
        }
    });
}

/// 6x32 (rows x cols) BF16 microkernel with F32 zmm accumulators.
///
/// `b_tile` points at the packed 32-column tile for this output tile,
/// `c_base` at row `m0` of C, `c_col` is the first output column.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_6x32(
    a: &[bf16],
    b_tile: *const u32,
    c_base: *mut f32,
    m0: usize,
    rows: usize,
    c_col: usize,
    n: usize,
    k: usize,
    kh: usize,
) {
    use std::arch::x86_64::*;
    let mut acc0 = [_mm512_setzero_ps(); MR];
    let mut acc1 = [_mm512_setzero_ps(); MR];
    for kp in 0..kh {
        let bp = b_tile.add(kp * NC);
        let b0: __m512bh = core::mem::transmute(core::ptr::read_unaligned(bp as *const __m512i));
        let b1: __m512bh =
            core::mem::transmute(core::ptr::read_unaligned(bp.add(16) as *const __m512i));
        for i in 0..rows {
            // A pair (A[m][2kp], A[m][2kp+1]) broadcast to all 16 lanes.
            let pair = (a.as_ptr().add((m0 + i) * k + 2 * kp) as *const u32).read_unaligned();
            let av: __m512bh = core::mem::transmute(_mm512_set1_epi32(pair as i32));
            acc0[i] = _mm512_dpbf16_ps(acc0[i], av, b0);
            acc1[i] = _mm512_dpbf16_ps(acc1[i], av, b1);
        }
    }
    let rem = n - c_col;
    if rem >= NC {
        for i in 0..rows {
            let dst = c_base.add(i * n + c_col);
            _mm512_storeu_ps(dst, acc0[i]);
            _mm512_storeu_ps(dst.add(16), acc1[i]);
        }
    } else {
        let lo = rem.min(16) as u16;
        let hi = rem.saturating_sub(16) as u16;
        let mask_lo: __mmask16 = if lo >= 16 { 0xFFFF } else { (1 << lo) - 1 };
        let mask_hi: __mmask16 = if hi >= 16 { 0xFFFF } else { (1 << hi) - 1 };
        for i in 0..rows {
            let dst = c_base.add(i * n + c_col);
            _mm512_mask_storeu_ps(dst, mask_lo, acc0[i]);
            _mm512_mask_storeu_ps(dst.add(16), mask_hi, acc1[i]);
        }
    }
}

/// Cache-blocked variant: tasks are (m-block, n-tile-group) pairs so the
/// packed B working set per task stays L2/L3-resident instead of streaming
/// the whole weight matrix per row tile. This matters for the big pool_ff
/// shape (40 MB packed B > 32 MB per-CCD L3).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn bf16_gemm_avx512_blocked(
    a: &[bf16],
    packed_b: &[u32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    /// Rows per m-block.
    const MC: usize = 512;
    /// 32-column tiles per n-group.
    const GT: usize = 32;

    let n_tiles = n.div_ceil(NC);
    let tile_stride = (k / 2) * NC;
    let kh = k / 2;
    let n_mb = m.div_ceil(MC);
    let n_ng = n_tiles.div_ceil(GT);
    let c_addr = c.as_mut_ptr() as usize;

    (0..n_mb * n_ng).into_par_iter().for_each(|task| {
        let mb = task / n_ng;
        let ng = task % n_ng;
        let m_end = m.min((mb + 1) * MC);
        let nt_end = n_tiles.min((ng + 1) * GT);
        let c_ptr = c_addr as *mut f32;
        for nt in ng * GT..nt_end {
            let b_tile = packed_b.as_ptr().add(nt * tile_stride);
            let mut m0 = mb * MC;
            while m0 < m_end {
                let rows = (m_end - m0).min(MR);
                micro_6x32(a, b_tile, c_ptr.add(m0 * n), m0, rows, nt * NC, n, k, kh);
                m0 += MR;
            }
        }
    });
}

/// Portability fallback: correct on any hardware, deliberately simple.
fn bf16_gemm_scalar(a: &[bf16], packed: &[u32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let kh = k / 2;
    for i in 0..m {
        for j in 0..n {
            let nt = j / NC;
            let l = j % NC;
            let mut acc = 0.0f32;
            for kp in 0..kh {
                let pair = packed[(nt * kh + kp) * NC + l];
                let b0 = bf16::from_bits((pair & 0xffff) as u16).to_f32();
                let b1 = bf16::from_bits((pair >> 16) as u16).to_f32();
                acc += a[i * k + 2 * kp].to_f32() * b0 + a[i * k + 2 * kp + 1].to_f32() * b1;
            }
            c[i * n + j] = acc;
        }
    }
}

fn bench_f32_faer(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> f64 {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, MatMut, MatRef, Par};
    let mut c = vec![0.0f32; m * n];
    let a_ref = MatRef::from_row_major_slice(a, m, k);
    let b_ref = MatRef::from_row_major_slice(b, k, n);
    let mut c_mut = MatMut::from_row_major_slice_mut(&mut c, m, n);
    for _ in 0..2 {
        matmul(c_mut.as_mut(), Accum::Replace, a_ref, b_ref, 1.0f32, Par::rayon(0));
    }
    let runs = 3;
    let start = Instant::now();
    for _ in 0..runs {
        matmul(c_mut.as_mut(), Accum::Replace, a_ref, b_ref, 1.0f32, Par::rayon(0));
    }
    start.elapsed().as_secs_f64() / runs as f64
}

fn bench_f32_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> f64 {
    let mut c = vec![0.0f32; m * n];
    let run = |c: &mut [f32]| unsafe {
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
    };
    for _ in 0..2 {
        run(&mut c);
    }
    let runs = 3;
    let start = Instant::now();
    for _ in 0..runs {
        run(&mut c);
    }
    start.elapsed().as_secs_f64() / runs as f64
}

fn bench_shape(name: &str, m: usize, k: usize, n: usize) {
    let a_f32 = rand_vec(m * k);
    let b_f32 = rand_vec(k * n);
    let a_bf = to_bf16(&a_f32);
    let b_bf = to_bf16(&b_f32);

    let t_pack_start = Instant::now();
    let packed = pack_b(&b_bf, k, n);
    let t_pack = t_pack_start.elapsed().as_secs_f64();

    // Correctness spot-check against the F32 faer result computed from the
    // same BF16-rounded inputs.
    let a_up: Vec<f32> = a_bf.iter().map(|v| v.to_f32()).collect();
    let b_up: Vec<f32> = b_bf.iter().map(|v| v.to_f32()).collect();
    let mut c_ref = vec![0.0f32; m * n];
    {
        use faer::linalg::matmul::matmul;
        use faer::{Accum, MatMut, MatRef, Par};
        matmul(
            MatMut::from_row_major_slice_mut(&mut c_ref, m, n),
            Accum::Replace,
            MatRef::from_row_major_slice(&a_up, m, k),
            MatRef::from_row_major_slice(&b_up, k, n),
            1.0f32,
            Par::rayon(0),
        );
    }
    let mut c_bf16 = vec![0.0f32; m * n];
    bf16_gemm_packed(&a_bf, &packed, &mut c_bf16, m, n, k);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (x, y) in c_bf16.iter().zip(c_ref.iter()) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / y.abs().max(1.0));
    }

    // Timing.
    let runs = 4;
    for _ in 0..2 {
        bf16_gemm_packed(&a_bf, &packed, &mut c_bf16, m, n, k);
    }
    let start = Instant::now();
    for _ in 0..runs {
        bf16_gemm_packed(&a_bf, &packed, &mut c_bf16, m, n, k);
    }
    let t_bf16 = start.elapsed().as_secs_f64() / runs as f64;

    // Cache-blocked variant: correctness spot-check + timing.
    #[cfg(target_arch = "x86_64")]
    let t_bf16_blocked = if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bf16")
    {
        let mut c_blk = vec![0.0f32; m * n];
        unsafe {
            bf16_gemm_avx512_blocked(&a_bf, &packed, &mut c_blk, m, n, k);
        }
        let mut blk_max_abs = 0.0f32;
        for (x, y) in c_blk.iter().zip(c_ref.iter()) {
            blk_max_abs = blk_max_abs.max((x - y).abs());
        }
        assert!(blk_max_abs < 1e-2, "blocked kernel mismatch: {blk_max_abs}");
        for _ in 0..2 {
            unsafe {
                bf16_gemm_avx512_blocked(&a_bf, &packed, &mut c_blk, m, n, k);
            }
        }
        let start = Instant::now();
        for _ in 0..runs {
            unsafe {
                bf16_gemm_avx512_blocked(&a_bf, &packed, &mut c_blk, m, n, k);
            }
        }
        start.elapsed().as_secs_f64() / runs as f64
    } else {
        f64::NAN
    };
    #[cfg(not(target_arch = "x86_64"))]
    let t_bf16_blocked = f64::NAN;

    let t_faer = bench_f32_faer(&a_f32, &b_f32, m, k, n);
    let t_gemm = bench_f32_gemm(&a_f32, &b_f32, m, k, n);

    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
    println!(
        "{name:12} ({m:5}x{k:5}) @ ({k:5}x{n:5})  bf16={t_bf16:.3}s ({:.2} TF/s)  \
         bf16_blocked={t_bf16_blocked:.3}s ({:.2} TF/s)  \
         faer_f32={t_faer:.3}s ({:.2} TF/s)  gemm_f32={t_gemm:.3}s  \
         speedup_blocked_vs_best_f32={:.2}x  pack={t_pack:.2}s  max_abs={max_abs:.4} max_rel={max_rel:.5}",
        gflop / t_bf16 / 1000.0,
        gflop / t_bf16_blocked / 1000.0,
        gflop / t_faer / 1000.0,
        t_faer.min(t_gemm) / t_bf16_blocked,
    );
}

#[test]
fn bf16_gemm_correctness() {
    // Odd m to exercise the partial row tile; n divisible by NC, even k.
    let (m, k, n) = (37, 64, 96);
    let a = to_bf16(&rand_vec(m * k));
    let b = to_bf16(&rand_vec(k * n));
    let packed = pack_b(&b, k, n);

    let mut c = vec![0.0f32; m * n];
    bf16_gemm_packed(&a, &packed, &mut c, m, n, k);

    let mut c_scalar = vec![0.0f32; m * n];
    bf16_gemm_scalar(&a, &packed, &mut c_scalar, m, n, k);

    let mut max_diff = 0.0f32;
    for (x, y) in c.iter().zip(c_scalar.iter()) {
        max_diff = max_diff.max((x - y).abs());
    }
    assert!(max_diff < 1e-3, "avx512 vs scalar mismatch: {max_diff}");
    println!("correctness ok (m={m}, k={k}, n={n}), max_diff={max_diff:.6}");
}

#[test]
fn bf16_pool_ff_bench() {
    println!(
        "rayon threads: {}",
        rayon::current_num_threads()
    );
    // Hydra-3.5 hot shapes: pool FF is the single biggest GEMM in the model;
    // mlp_fc1/mlp_fc2/qkv dominate the 27 NaFlex blocks. Note fc1's n = 4304
    // (not 4608 — the real MLP hidden dim), which exercises the padded-n
    // masked-store edge tile.
    bench_shape("pool_ff", 8886, 2048, 10240);
    bench_shape("pool_proj", 8886, 5120, 2048);
    bench_shape("mlp_fc1", 1024, 1152, 4304);
    bench_shape("mlp_fc2", 1024, 4304, 1152);
    bench_shape("qkv", 1024, 1152, 3456);
}
