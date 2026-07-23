#![allow(unsafe_op_in_unsafe_fn)]

//! BF16 GEMM with F32 accumulation for Hydra-3.5 CPU inference.
//!
//! The Hydra-3.5 checkpoint stores weights in BF16 and `weights.rs` upcasts
//! them to F32 losslessly at load, so converting the existing F32 weight caches
//! back to bf16 is exact — the only rounding happens on the activation inputs.
//! On CPUs with AVX512-BF16 (`VDPBF16PS`) this roughly doubles GEMM throughput
//! over the F32 `gemm`/`faer` path.
//!
//! Design (validated by `tests/bf16_gemm_bench.rs`):
//! - B (weights) is pre-packed once at load into 32-column tiles of k/2
//!   pair-rows, where each u32 lane holds `{B[2k][c], B[2k+1][c]}`.
//! - The microkernel computes MR=6 rows x NC=32 columns per tile with
//!   `VDPBF16PS`, accumulating into F32 zmm registers, and **accumulates into
//!   C** so callers keep the existing bias/residual epilogue pattern (they
//!   pre-fill C).
//! - Activations are converted F32 -> bf16 per GEMM call into a reusable
//!   scratch buffer.
//! - A cache-blocked driver is used when the packed weight exceeds
//!   `BLOCKED_MIN_BYTES` so the per-task working set stays L2/L3-resident.
//!
//! The whole path is gated at runtime by `AKASHA_USE_BF16_GEMM` (default on
//! when the CPU supports avx512f + avx512bf16); when disabled, weights are not
//! packed and the F32 path runs unchanged.

use half::bf16;
use rayon::prelude::*;

/// Rows per microkernel tile. 6 rows x 2 zmm accumulators = 12 of 32 zmm regs.
const MR: usize = 6;
/// Columns per microkernel tile (2 x 16 F32 lanes).
const NC: usize = 32;

/// Packed-B byte size above which the cache-blocked driver is used. Below this
/// the simple row-tile-parallel driver wins (the packed weight already fits in
/// L3); above it the per-task working set would thrash L3.
const BLOCKED_MIN_BYTES: usize = 24 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Runtime switch
// ---------------------------------------------------------------------------

/// Return `true` if the BF16 GEMM path should be used.
///
/// `AKASHA_USE_BF16_GEMM=0`/`false` forces it off; `1`/`true` forces it on
/// (on CPUs without avx512bf16 the GEMM itself still falls back to a portable
/// scalar implementation, so forcing it on is always safe). The default is on
/// when the CPU supports avx512f + avx512bf16.
pub fn use_bf16_gemm() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        if let Ok(v) = std::env::var("AKASHA_USE_BF16_GEMM") {
            if v == "0" || v.eq_ignore_ascii_case("false") {
                return false;
            }
            if v == "1" || v.eq_ignore_ascii_case("true") {
                return true;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

// ---------------------------------------------------------------------------
// Packed weights
// ---------------------------------------------------------------------------

/// A BF16 weight matrix pre-packed for the bf16 microkernel.
///
/// The original weight is row-major `[k, n]` (Burn's `[in, out]` layout); the
/// packed layout is `n/32` tiles, each holding `k/2` pair-rows of 32 u32 lanes
/// where each lane packs `{B[2kp][c], B[2kp+1][c]}`.
#[derive(Debug, Clone)]
pub struct PackedBf16Weight {
    packed: Vec<u32>,
    /// Reduction dimension (`in_features`).
    pub k: usize,
    /// Output dimension (`out_features`).
    pub n: usize,
}

impl PackedBf16Weight {
    /// Pack a row-major `[k, n]` F32 weight cache.
    ///
    /// Returns `None` when the shape is incompatible with the kernel (odd `k`
    /// or `n` not divisible by 32) so the caller can keep using the F32 path.
    /// The F32 -> bf16 conversion is exact for the Hydra checkpoint, whose
    /// weights are natively BF16.
    pub fn pack_f32(w: &[f32], k: usize, n: usize) -> Option<Self> {
        if k == 0 || n == 0 || w.len() != k * n || k % 2 != 0 || n % NC != 0 {
            return None;
        }
        let mut b = vec![bf16::ZERO; k * n];
        b.par_iter_mut().zip(w.par_iter()).for_each(|(d, &v)| {
            *d = bf16::from_f32(v);
        });
        let n_tiles = n / NC;
        let kh = k / 2;
        let mut packed = vec![0u32; n_tiles * kh * NC];
        packed
            .par_chunks_mut(kh * NC)
            .enumerate()
            .for_each(|(nt, tile)| {
                let c0 = nt * NC;
                for kp in 0..kh {
                    let r0 = 2 * kp * n + c0;
                    let r1 = (2 * kp + 1) * n + c0;
                    let dst = &mut tile[kp * NC..(kp + 1) * NC];
                    for l in 0..NC {
                        let lo = b[r0 + l].to_bits() as u32;
                        let hi = b[r1 + l].to_bits() as u32;
                        dst[l] = lo | (hi << 16);
                    }
                }
            });
        Some(Self { packed, k, n })
    }

    /// Size of the packed weight in bytes.
    fn byte_len(&self) -> usize {
        self.packed.len() * 4
    }
}

// ---------------------------------------------------------------------------
// Activation conversion
// ---------------------------------------------------------------------------

/// Convert `m` rows of `k` F32 values (row stride `src_stride`) into a
/// contiguous `[m, k]` bf16 buffer.
///
/// Round-to-nearest-even via integer bit manipulation, matching
/// `half::bf16::from_f32` for all finite values (NaN is not special-cased;
/// activations are finite by construction). The branchless form autovectorizes.
pub fn f32_to_bf16_strided(src: &[f32], src_stride: usize, dst: &mut [bf16], m: usize, k: usize) {
    debug_assert!(src_stride >= k);
    debug_assert!(src.len() >= (m - 1) * src_stride + k);
    debug_assert!(dst.len() >= m * k);
    dst.par_chunks_exact_mut(k)
        .take(m)
        .enumerate()
        .for_each(|(i, drow)| {
            let srow = &src[i * src_stride..i * src_stride + k];
            for (d, &v) in drow.iter_mut().zip(srow.iter()) {
                let bits = v.to_bits();
                let bias = 0x7fffu32 + ((bits >> 16) & 1);
                *d = bf16::from_bits((bits.wrapping_add(bias) >> 16) as u16);
            }
        });
}

// ---------------------------------------------------------------------------
// GEMM entry points
// ---------------------------------------------------------------------------

/// `C += A @ B` where `A` is `[m, k]` bf16 (contiguous), `B` is a packed bf16
/// weight, and `C` is `[m, n]` F32. Existing contents of `C` are accumulated
/// into, so callers pre-fill `C` with bias/residual epilogues as with the F32
/// accumulate GEMM.
pub fn gemm_accum(a: &[bf16], w: &PackedBf16Weight, c: &mut [f32], m: usize) {
    gemm(a, w, c, m, true);
}

/// `C = A @ B`, replacing the contents of `C`.
pub fn gemm_replace(a: &[bf16], w: &PackedBf16Weight, c: &mut [f32], m: usize) {
    gemm(a, w, c, m, false);
}

fn gemm(a: &[bf16], w: &PackedBf16Weight, c: &mut [f32], m: usize, accumulate: bool) {
    let (k, n) = (w.k, w.n);
    debug_assert_eq!(a.len(), m * k);
    debug_assert!(c.len() >= m * n);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16") {
        unsafe {
            if w.byte_len() > BLOCKED_MIN_BYTES {
                return gemm_avx512_blocked(a, &w.packed, c, m, n, k, accumulate);
            } else {
                return gemm_avx512(a, &w.packed, c, m, n, k, accumulate);
            }
        }
    }
    gemm_scalar(a, &w.packed, c, m, n, k, accumulate);
}

/// Convert `[m, k]` F32 activations (row stride `a_stride`) to bf16 and
/// accumulate `C += A @ W`.
///
/// Returns `false` without touching `c` or `scratch` if the buffers are
/// incompatible with the packed weight, so the caller can fall back to the F32
/// GEMM. `scratch` is a reusable workspace to avoid per-call allocations.
pub fn linear_accum_from_f32(
    a: &[f32],
    a_stride: usize,
    m: usize,
    w: &PackedBf16Weight,
    c: &mut [f32],
    scratch: &mut Vec<bf16>,
) -> bool {
    linear_from_f32(a, a_stride, m, w, c, scratch, true)
}

/// Like [`linear_accum_from_f32`], but writes `C = A @ W`, replacing the
/// existing contents of `c` (no pre-fill or zeroing required).
pub fn linear_replace_from_f32(
    a: &[f32],
    a_stride: usize,
    m: usize,
    w: &PackedBf16Weight,
    c: &mut [f32],
    scratch: &mut Vec<bf16>,
) -> bool {
    linear_from_f32(a, a_stride, m, w, c, scratch, false)
}

fn linear_from_f32(
    a: &[f32],
    a_stride: usize,
    m: usize,
    w: &PackedBf16Weight,
    c: &mut [f32],
    scratch: &mut Vec<bf16>,
    accumulate: bool,
) -> bool {
    let (k, n) = (w.k, w.n);
    if m == 0 || a_stride < k || a.len() < (m - 1) * a_stride + k || c.len() < m * n {
        return false;
    }
    if scratch.len() < m * k {
        scratch.resize(m * k, bf16::ZERO);
    }
    f32_to_bf16_strided(a, a_stride, &mut scratch[..m * k], m, k);
    gemm(&scratch[..m * k], w, c, m, accumulate);
    true
}

// ---------------------------------------------------------------------------
// AVX512-BF16 kernels
// ---------------------------------------------------------------------------

/// Row-tile-parallel driver: one task per MR-row tile; the last tile may be
/// partial. Best when the packed weight fits comfortably in L3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn gemm_avx512(
    a: &[bf16],
    packed_b: &[u32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) {
    let n_tiles = n / NC;
    let tile_stride = (k / 2) * NC;
    let kh = k / 2;

    // NOTE: the microkernel must be called from a `#[target_feature]` fn like
    // this one — a rayon closure does not inherit target features and perf
    // collapses if the kernel body sits inside it.
    c.par_chunks_mut(MR * n).enumerate().for_each(|(mt, c_chunk)| {
        let m0 = mt * MR;
        let rows = (m - m0).min(MR);
        let c_base = c_chunk.as_mut_ptr();
        for nt in 0..n_tiles {
            let b_tile = packed_b.as_ptr().add(nt * tile_stride);
            micro_6x32(a, b_tile, c_base, m0, rows, nt * NC, n, k, kh, accumulate);
        }
    });
}

/// Cache-blocked driver: tasks are (m-block, n-tile-group) pairs so the packed
/// B working set per task stays L2/L3-resident instead of streaming the whole
/// weight matrix per row tile. Wins for the largest shapes (e.g. pool FF).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn gemm_avx512_blocked(
    a: &[bf16],
    packed_b: &[u32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) {
    /// Rows per m-block.
    const MC: usize = 512;
    /// 32-column tiles per n-group.
    const GT: usize = 32;

    let n_tiles = n / NC;
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
                micro_6x32(a, b_tile, c_ptr.add(m0 * n), m0, rows, nt * NC, n, k, kh, accumulate);
                m0 += MR;
            }
        }
    });
}

/// 6x32 (rows x cols) BF16 microkernel with F32 zmm accumulators.
///
/// `b_tile` points at the packed 32-column tile for this output tile,
/// `c_base` at row `m0` of C, `c_col` is the first output column. When
/// `accumulate` is true the product is added to the existing contents of C,
/// otherwise it replaces them.
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
    accumulate: bool,
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
    for i in 0..rows {
        let dst = c_base.add(i * n + c_col);
        if accumulate {
            _mm512_storeu_ps(dst, _mm512_add_ps(_mm512_loadu_ps(dst), acc0[i]));
            _mm512_storeu_ps(dst.add(16), _mm512_add_ps(_mm512_loadu_ps(dst.add(16)), acc1[i]));
        } else {
            _mm512_storeu_ps(dst, acc0[i]);
            _mm512_storeu_ps(dst.add(16), acc1[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Portable fallback
// ---------------------------------------------------------------------------

/// Portability fallback: correct on any hardware, deliberately simple.
/// Accumulates into `c` when `accumulate` is true, replaces otherwise.
fn gemm_scalar(a: &[bf16], packed: &[u32], c: &mut [f32], m: usize, n: usize, k: usize, accumulate: bool) {
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
            if accumulate {
                c[i * n + j] += acc;
            } else {
                c[i * n + j] = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 123456789) % 1000) as f32 / 1000.0 - 0.5)
            .collect()
    }

    /// Reference `C += A @ B` over F32 slices using the `gemm` crate.
    fn gemm_f32_accum_ref(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                c.as_mut_ptr(),
                1,
                n as isize,
                true, // read_dst: accumulate into C
                a.as_ptr(),
                1,
                k as isize,
                b.as_ptr(),
                1,
                n as isize,
                1.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
    }

    fn upcast(v: &[bf16]) -> Vec<f32> {
        v.iter().map(|x| x.to_f32()).collect()
    }

    /// The bf16 kernel must match an F32 GEMM of the *same bf16-rounded
    /// inputs* almost exactly (only accumulation order differs), and match the
    /// full-F32 GEMM within bf16 activation-rounding error.
    fn check_shape(m: usize, k: usize, n: usize) {
        let a_f32 = rand_vec(m * k);
        let w_f32 = rand_vec(k * n);
        let bias = rand_vec(n);

        let packed = PackedBf16Weight::pack_f32(&w_f32, k, n).expect("Hydra-compatible shape");
        let mut a_bf = vec![bf16::ZERO; m * k];
        f32_to_bf16_strided(&a_f32, k, &mut a_bf, m, k);

        // Kernel result, pre-filled with bias like the fused epilogues do.
        let mut c_bf16 = vec![0.0f32; m * n];
        for row in c_bf16.chunks_exact_mut(n) {
            row.copy_from_slice(&bias);
        }
        gemm_accum(&a_bf, &packed, &mut c_bf16, m);

        // Reference: F32 GEMM of the bf16-rounded inputs.
        let a_up = upcast(&a_bf);
        let w_bf: Vec<bf16> = w_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let w_up = upcast(&w_bf);
        let mut c_ref_rounded = vec![0.0f32; m * n];
        for row in c_ref_rounded.chunks_exact_mut(n) {
            row.copy_from_slice(&bias);
        }
        gemm_f32_accum_ref(&a_up, &w_up, &mut c_ref_rounded, m, n, k);

        let mut max_abs = 0.0f32;
        for (x, y) in c_bf16.iter().zip(c_ref_rounded.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        assert!(
            max_abs < 1e-2,
            "bf16 kernel vs rounded-input F32 GEMM: max_abs={max_abs}"
        );

        // Reference: full-F32 GEMM (what the existing path computes).
        let mut c_ref_f32 = vec![0.0f32; m * n];
        for row in c_ref_f32.chunks_exact_mut(n) {
            row.copy_from_slice(&bias);
        }
        gemm_f32_accum_ref(&a_f32, &w_f32, &mut c_ref_f32, m, n, k);

        // The only error source is bf16 rounding of the GEMM inputs (~2^-9
        // relative per element, for both activations and weights). Judged
        // element-wise this blows up wherever the dot product nearly cancels,
        // so measure the error relative to the magnitude of a *typical*
        // output element, sqrt(k) * rms(a) * rms(w). The synthetic data here
        // shares a 1000-value cycle between A and W, which correlates the
        // rounding residuals and inflates the aggregate error well beyond the
        // independent-rounding estimate (~0.025 scaled); the end-to-end gate
        // for real data is the logits A/B in the ignored hydra test.
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        let out_scale = (k as f32).sqrt() * rms(&a_f32) * rms(&w_f32);
        let mut max_scaled = 0.0f32;
        for (x, y) in c_bf16.iter().zip(c_ref_f32.iter()) {
            max_scaled = max_scaled.max((x - y).abs() / out_scale);
        }
        assert!(
            max_scaled < 5e-2,
            "bf16 kernel vs full-F32 GEMM: max scaled error={max_scaled} (out_scale={out_scale})"
        );
    }

    #[test]
    fn bf16_gemm_matches_f32_on_hydra_shapes() {
        // NaFlex qkv / o_proj, fc1, fc2 and pool kv shapes with a small m.
        check_shape(37, 1152, 3456);
        check_shape(37, 1152, 1152);
        check_shape(37, 1152, 4608);
        check_shape(37, 4608, 1152);
        check_shape(37, 2048, 4096);
    }

    #[test]
    fn bf16_gemm_matches_f32_on_pool_ff_shapes() {
        // Pool/mid FF shapes (glu proj + proj_out); pool FF exercises the
        // cache-blocked driver (packed B > BLOCKED_MIN_BYTES).
        check_shape(37, 2048, 10240);
        check_shape(37, 5120, 2048);
        check_shape(37, 2048, 2048);
    }

    #[test]
    fn f32_to_bf16_matches_half() {
        let src = rand_vec(4096 + 3);
        let mut dst = vec![bf16::ZERO; src.len()];
        f32_to_bf16_strided(&src, src.len(), &mut dst, 1, src.len());
        for (&v, &d) in src.iter().zip(dst.iter()) {
            assert_eq!(bf16::from_f32(v).to_bits(), d.to_bits(), "value {v}");
        }
    }

    #[test]
    fn incompatible_shapes_fall_back() {
        // Odd k and n not divisible by NC must not pack.
        assert!(PackedBf16Weight::pack_f32(&rand_vec(7 * 32), 7, 32).is_none());
        assert!(PackedBf16Weight::pack_f32(&rand_vec(8 * 30), 8, 30).is_none());

        // Mismatched activation buffers must report failure without touching C.
        let packed = PackedBf16Weight::pack_f32(&rand_vec(64 * 96), 64, 96).unwrap();
        let mut scratch = Vec::new();
        let mut c = vec![1.0f32; 4 * 96];
        assert!(!linear_accum_from_f32(&rand_vec(4 * 32), 32, 4, &packed, &mut c, &mut scratch));
        assert!(c.iter().all(|&v| v == 1.0));
        assert!(!linear_accum_from_f32(&[], 64, 0, &packed, &mut c, &mut scratch));
    }

    #[test]
    fn scalar_fallback_matches_avx512() {
        let (m, k, n) = (37, 64, 96);
        let a_f32 = rand_vec(m * k);
        let w_f32 = rand_vec(k * n);
        let packed = PackedBf16Weight::pack_f32(&w_f32, k, n).unwrap();
        let mut a_bf = vec![bf16::ZERO; m * k];
        f32_to_bf16_strided(&a_f32, k, &mut a_bf, m, k);

        let mut c = vec![0.5f32; m * n];
        gemm_accum(&a_bf, &packed, &mut c, m);

        let mut c_scalar = vec![0.5f32; m * n];
        gemm_scalar(&a_bf, &packed.packed, &mut c_scalar, m, n, k, true);

        let mut max_diff = 0.0f32;
        for (x, y) in c.iter().zip(c_scalar.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff < 1e-3, "avx512 vs scalar mismatch: {max_diff}");

        // Replace mode must drop the pre-existing C contents entirely.
        let mut c_rep = vec![0.5f32; m * n];
        gemm_replace(&a_bf, &packed, &mut c_rep, m);
        let mut c_rep_scalar = vec![0.5f32; m * n];
        gemm_scalar(&a_bf, &packed.packed, &mut c_rep_scalar, m, n, k, false);
        let mut max_diff = 0.0f32;
        for (x, y) in c_rep.iter().zip(c_rep_scalar.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff < 1e-3, "replace-mode mismatch: {max_diff}");
        for (x, y) in c_rep.iter().zip(c_scalar.iter()) {
            // c_scalar = 0.5 + product, c_rep = product.
            max_diff = max_diff.max((x + 0.5 - y).abs());
        }
        assert!(max_diff < 1e-3, "replace vs accum mismatch: {max_diff}");
    }
}
