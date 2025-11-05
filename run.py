#!/usr/bin/env python3

import sys
import numpy as np
np.set_printoptions(linewidth=9999999)
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
BLOCK_SIZE = 32  # single warp

# ======================================================================================
# Fragment definitions
# ======================================================================================

# TODO:

# ======================================================================================
# CUDA kernel for FP16 matrix multiplication with FP32 accumulation
# ======================================================================================
code_setup = f'''
    // Lane and group assignment within the warp
    unsigned int lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    unsigned int groupID = lane_id >> 2;
    unsigned int threadID_in_group = lane_id & 3;

    // Output accumulator fragments (FP32)
    float c_frag[4] = {{0.f, 0.f, 0.f, 0.f}};
    float d_frag[4];

    // MMA expects 2 half per 32b register (half2), so there are 4 half2 for A, 2 for B
    half2 a_frag[4];
    half2 b_frag[2];
'''

code_load = f'''
    // --- Fill A fragment (4 regs, 2 half each) ---
    for (int frag = 0; frag < 4; ++frag) {{
        // For each half in this half2
        half h0, h1;
        for (int subidx = 0; subidx < 2; ++subidx) {{
            int ai = frag * 2 + subidx;
            unsigned int row = ( (ai < 2) || (ai >= 4 && ai < 6) ) ? groupID : (groupID + 8);
            unsigned int col = (threadID_in_group * 2) + (ai & 1);
            if (ai >= 4) col += 8;
            int index = row * {K} + col;
            half value = __float2half(0.0f);
            if(row < {M} && col < {K})
                value = A[index];
            if(subidx == 0) h0 = value;
            else h1 = value;
        }}
        a_frag[frag] = __halves2half2(h0, h1);
    }}

    // --- Fill B fragment (2 regs, 2 half each) ---
    for (int frag = 0; frag < 2; ++frag) {{
        half h0, h1;
        for (int subidx = 0; subidx < 2; ++subidx) {{
            int bi = frag * 2 + subidx;
            unsigned int row = (threadID_in_group * 2) + (bi & 1);
            if (bi >= 2) row += 8;
            unsigned int col = groupID;
            int index = row * {N} + col;
            half value = __float2half(0.0f);
            if(row < {K} && col < {N})
                value = B[index];
            if(subidx == 0) h0 = value;
            else h1 = value;
        }}
        b_frag[frag] = __halves2half2(h0, h1);
    }}

    // Cast to unsigned for inline PTX
    unsigned int a_int[4], b_int[2];
    #pragma unroll
    for(int i = 0; i < 4; ++i)
        a_int[i] = reinterpret_cast<unsigned int &>(a_frag[i]);
    #pragma unroll
    for(int i = 0; i < 2; ++i)
        b_int[i] = reinterpret_cast<unsigned int &>(b_frag[i]);
'''

code_exec = f'''
    // --- MMA PTX ---
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0, %1, %2, %3}},{{%4, %5, %6, %7}},{{%8, %9}},{{%10, %11, %12, %13}};\\n"
        : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
        : "r"(a_int[0]), "r"(a_int[1]), "r"(a_int[2]), "r"(a_int[3]), "r"(b_int[0]), "r"(b_int[1]), "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3])
    );
'''

code_store = f'''
    // --- Write results by fragment-to-tile mapping ---
    for (int i = 0; i < 4; i++) {{
        unsigned int row = (i < 2) ? groupID : (groupID + 8);
        unsigned int col = (threadID_in_group * 2) + (i & 1);
        if (row < {M} && col < {N}) {{
            unsigned int idx = row * {N} + col;
            D[idx] = d_frag[i];
        }}
    }}
'''

code = f'''extern "C"
__global__ void matmul_fp16_fp32(const half* A, const half* B, float* C, float* D) {{
  {code_setup}
  {code_load}
  {code_exec}
  {code_store}
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
d_bytes = M * N * np.dtype(np.float32).itemsize

a_buf = pinned_mr.allocate(a_bytes, stream=stream)
b_buf = pinned_mr.allocate(b_bytes, stream=stream)
d_buf = pinned_mr.allocate(d_bytes, stream=stream)

# Convert to NumPy arrays using DLPack
a_host = np.from_dlpack(a_buf).view(np.float16).reshape(M, K)
b_host = np.from_dlpack(b_buf).view(np.float16).reshape(K, N)
d_host = np.from_dlpack(d_buf).view(np.float32).reshape(M, N)

# Initialize inputs
rng = np.random.default_rng()
a_host[:] = rng.random((M, K), dtype=np.float32)
b_host[:] = rng.random((K, N), dtype=np.float32)
d_host[:] = 0

# Keep reference for correctness check
ref = np.matmul(a_host.astype(np.float32), b_host.astype(np.float32))

stream.sync()

# ======================================================================================
# Kernel launch
# ======================================================================================
grid = 1
block = BLOCK_SIZE
config = LaunchConfig(grid=grid, block=block)

launch(stream, config, kernel, a_buf, b_buf, 0, d_buf)
stream.sync()

print(d_host)

# Verify results
assert np.allclose(d_host, ref, atol=1e-2), "Pinned memory matmul verification failed"

# Cleanup
a_buf.close(stream)
b_buf.close(stream)
d_buf.close(stream)
stream.close()

print("Pinned memory FP16×FP16→FP32 matrix multiplication example completed successfully!")

