#!/usr/bin/env bash
# GPU run wrapper for Akasha.
#
# Fedora/RHEL systems often ship a 32-bit /usr/lib/libcuda.so that rustc
# finds before the 64-bit /usr/lib64/libcuda.so, causing:
#   /usr/lib/libcuda.so is incompatible with elf64-x86-64
#
# This script points builds at a shim directory that only contains symlinks
# to the 64-bit CUDA toolkit libraries plus the correct 64-bit driver lib.

set -euo pipefail

CUDA_TOOLKIT="${CUDA_TOOLKIT:-/usr/local/cuda-13.2}"
SHIM_DIR="${HOME}/.akasha_cuda"

# Recreate the shim so it tracks the selected toolkit.
rm -rf "${SHIM_DIR}"
mkdir -p "${SHIM_DIR}/lib64"
for f in "${CUDA_TOOLKIT}/lib64"/*; do
    ln -s "${f}" "${SHIM_DIR}/lib64/"
done
ln -sf /usr/lib64/libcuda.so "${SHIM_DIR}/lib64/libcuda.so"

export PATH="${CUDA_TOOLKIT}/bin:${PATH}"
export CUDA_HOME="${SHIM_DIR}"

cargo run --release --features "hevc simd-thumbnails candle remote onnx mistralrs cuda" "$@"
