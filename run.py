#!/usr/bin/env python3

import sys
import numpy as np
from cuda.core.experimental import (
    Device,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)

if np.__version__ < "2.1.0":
    print("This example requires NumPy 2.1.0 or later", file=sys.stderr)
    sys.exit(0)

# ======================================================================================
# Matrix dimensions and constants
# ======================================================================================
M = 16   # Rows in A and C
K = 16  # Columns in A, Rows in B
N = 8   # Columns in B and C
BLOCK_SIZE = 32  # single warp

# ======================================================================================
# CUDA kernel for FP16 matrix multiplication with FP32 accumulation
# ======================================================================================
code = f'''extern "C"
__global__ void matmul_fp16_fp32(const half* A, const half* B, float* C) {{
    const int WARP_SIZE = 32;
    const int tid = threadIdx.x;

    for (int i = tid; i < {M} * {N}; i += WARP_SIZE) {{
        int r = i / {N};
        int c = i % {N};
        float sum = 0.0f;
        for (int k = 0; k < {K}; ++k) {{
            half a = A[r * {K} + k];
            half b = B[k * {N} + c];
            sum += __half2float(a) * __half2float(b);
        }}
        C[r * {N} + c] = sum;
    }}
}}
'''

print(code)

# ======================================================================================
# CUDA setup
# ======================================================================================
dev = Device()
dev.set_current()
stream = dev.create_stream()

program_options = ProgramOptions(std="c++20", arch=f"sm_{dev.arch}", pre_include='/usr/local/cuda/include/cuda_fp16.h')
prog = Program(code, code_type="c++", options=program_options)
mod = prog.compile("cubin")
kernel = mod.get_kernel("matmul_fp16_fp32")

# Use pinned memory
pinned_mr = LegacyPinnedMemoryResource()

# Allocate matrices in pinned (CPU) memory
a_bytes = M * K * np.dtype(np.float16).itemsize
b_bytes = K * N * np.dtype(np.float16).itemsize
c_bytes = M * N * np.dtype(np.float32).itemsize

a_buf = pinned_mr.allocate(a_bytes, stream=stream)
b_buf = pinned_mr.allocate(b_bytes, stream=stream)
c_buf = pinned_mr.allocate(c_bytes, stream=stream)

# Convert to NumPy arrays using DLPack
a_host = np.from_dlpack(a_buf).view(np.float16).reshape(M, K)
b_host = np.from_dlpack(b_buf).view(np.float16).reshape(K, N)
c_host = np.from_dlpack(c_buf).view(np.float32).reshape(M, N)

# Initialize inputs
rng = np.random.default_rng()
a_host[:] = rng.random((M, K), dtype=np.float32)
b_host[:] = rng.random((K, N), dtype=np.float32)
c_host[:] = 0

# Keep reference for correctness check
ref = np.matmul(a_host.astype(np.float32), b_host.astype(np.float32))

stream.sync()

# ======================================================================================
# Kernel launch
# ======================================================================================
grid = 1
block = BLOCK_SIZE
config = LaunchConfig(grid=grid, block=block)

launch(stream, config, kernel, a_buf, b_buf, c_buf)
stream.sync()

# Verify results
assert np.allclose(c_host, ref, atol=1e-2), "Pinned memory matmul verification failed"

# Cleanup
a_buf.close(stream)
b_buf.close(stream)
c_buf.close(stream)
stream.close()

print("Pinned memory FP16×FP16→FP32 matrix multiplication example completed successfully!")

