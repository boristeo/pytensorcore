#!/usr/bin/env python3

import sys
import math
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
from functools import reduce
import operator

if np.__version__ < "2.1.0":
    print("This example requires NumPy 2.1.0 or later", file=sys.stderr)
    sys.exit(0)

# ======================================================================================
# Matrix dimensions and constants
# ======================================================================================
M = 4096   # Rows in A and C
K = 4096  # Columns in A, Rows in B
N = 4096   # Columns in B and C

Mmma = 16
Kmma = 16
Nmma = 8
BLOCK_SIZE = 32  # single warp
TILES_M = int(math.ceil(M / Mmma)) 
TILES_N = int(math.ceil(N / Nmma))
TILES_K = int(math.ceil(K / Kmma))
NBLOCKS = TILES_M * TILES_N * TILES_K

# ======================================================================================
# Fragment definitions
# ======================================================================================

class StridedShape:
  def __init__(self, shape, strides=None):
    self.shape = list(shape)
    if strides is None:
      strides = []
      acc = 1
      for dim in shape[::-1]:
        strides.insert(0, acc)
        acc *= dim
    self.strides = [0 if s == 1 else st for s, st in zip(shape, strides)]
    assert len(shape) == len(strides)
  def __repr__(self):
    return f"StridedShape(shape={self.shape}, strides={self.strides})"
  def reshape(self, newshape):
    old_shape, old_strides, new_shape = list(self.shape), list(self.strides), list(newshape)
    old_count = 1
    for s in old_shape: old_count *= s
    new_count = 1
    for s in new_shape: new_count *= s
    if old_count != new_count: raise ValueError("Incompatible reshape dimensions")
    if old_count == 0:
      elem_stride, newstrides, mul = 1, [], 1
      for s in reversed(new_shape):
        newstrides.insert(0, 0 if s == 1 else elem_stride * mul)
        if s != 1: mul *= s
      return StridedShape(new_shape, newstrides)
    old_non1 = [i for i, s in enumerate(old_shape) if s != 1]
    if not old_non1:
      newstrides, mul = [0 if s == 1 else 1 for s in new_shape], 1
      for i in range(len(new_shape)-1, -1, -1):
        if new_shape[i] != 1:
          newstrides[i] = mul
          mul *= new_shape[i]
      return StridedShape(new_shape, newstrides)
    last_old = old_non1[-1]
    base, mul = old_strides[last_old], 1
    for k in range(last_old, -1, -1):
      if old_shape[k] == 1: continue
      expected = base * mul
      if old_strides[k] != expected:
        raise ValueError("Reshape would require copying data")
      mul *= old_shape[k]
    newstrides, mul = [None]*len(new_shape), 1
    for i in range(len(new_shape)-1, -1, -1):
      s = new_shape[i]
      newstrides[i] = 0 if s == 1 else base * mul
      if s != 1: mul *= s
    return StridedShape(new_shape, newstrides)

class Fragments:
  def __init__(self, name, lshape, dtype, esize, nregs, locs):
    assert len(lshape.shape) == len(locs[0])
    self.name = name
    self.lshape = lshape
    self.dtype = dtype
    self.esize = esize
    self.nregs = nregs
    self.locs = ('+'.join(f'({loc})*{stride}' for loc, stride in zip(mdimloc, self.lshape.strides)) for mdimloc in locs)

# Multiplicand A:
#
# groupID           = %laneid >> 2
# threadID_in_group = %laneid % 4
# 
# row =      groupID            for ai where  0 <= i < 2 || 4 <= i < 6
#           groupID + 8         Otherwise
# 
# col =  (threadID_in_group * 2) + (i & 0x1)          for ai where i <  4
# (threadID_in_group * 2) + (i & 0x1) + 8      for ai where i >= 4

# TODO: Clean up syntax to specify these
a_locs = [('blockIdx.y', f'{(i//2%2)*Mmma//2}+(lane_id/4)', 'i', f'{(i%2)+(i//4%2)*Kmma//2}+(lane_id%4)*2') for i in range(8)]
a_frags = Fragments('A', StridedShape([TILES_M, Mmma, TILES_K, Kmma]), 'half', 2, 8, a_locs)

b_locs = [('i', f'{(i%2)+(i//2)*Kmma//2}+(lane_id%4)*2', 'blockIdx.x', 'lane_id/4') for i in range(4)]
b_frags = Fragments('B', StridedShape([TILES_K, Kmma, TILES_N, Nmma]), 'half', 2, 4, b_locs)

d_locs = [('blockIdx.y', f'{(i//2)*Mmma//2}+(lane_id/4)', 'blockIdx.x', f'{i%2}+(lane_id%4)*2') for i in range(4)]
d_frags = Fragments('D', StridedShape([TILES_M, Mmma, TILES_N, Nmma]), 'float', 4, 4, d_locs)


# ======================================================================================
# CUDA kernel for FP16 matrix multiplication with FP32 accumulation
# ======================================================================================
def code_decl(mat_frags, val=None):
  return f'''
    {mat_frags.dtype} {mat_frags.name.lower()}_frag[{mat_frags.nregs}]{f" = {val}" if val is not None else ""};
    unsigned int* {mat_frags.name.lower()}_int = reinterpret_cast<unsigned int *>({mat_frags.name.lower()}_frag);
  '''
def code_load_v2(mat_frags):
  return '      '.join((f'{mat_frags.name.lower()}_frag[{i}] = {mat_frags.name}[{loc}];\n'
      for i, loc in enumerate(mat_frags.locs)))

code_exec = f'''
    // --- MMA PTX ---
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}},{{%4,%5,%6,%7}},{{%8,%9}},{{%0,%1,%2,%3}};\\n"
        : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
        : "r"(a_int[0]), "r"(a_int[1]), "r"(a_int[2]), "r"(a_int[3]),
        "r"(b_int[0]), "r"(b_int[1])
    );
'''

def code_store_v2(mat_frags):
  return f'''
    {'\n    '.join((f'{mat_frags.name}[{loc}] = {mat_frags.name.lower()}_frag[{i}];' for i, loc in enumerate(mat_frags.locs)))}
  '''

code = f'''extern "C"
__global__ void matmul_fp16_fp32(const half* A, const half* B, float* C, float* D) {{
    unsigned int lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    {code_decl(a_frags)}
    {code_decl(b_frags)}
    {code_decl(d_frags, "{0.0f,0.0f,0.0f,0.0f}")}
    for (int i = 0; i < {TILES_K}; ++i) {{
      {code_load_v2(a_frags)}
      {code_load_v2(b_frags)}
      {code_exec}
    }}
    {code_store_v2(d_frags)}
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
grid = (TILES_N, TILES_M)
block = BLOCK_SIZE
config = LaunchConfig(grid=grid, block=block)

launch(stream, config, kernel, a_buf, b_buf, 0, d_buf)
stream.sync()

#print(d_host)

# Verify results
assert np.allclose(d_host, ref, atol=1e-1), "Pinned memory matmul verification failed"

# Cleanup
a_buf.close(stream)
b_buf.close(stream)
d_buf.close(stream)
stream.close()

print("Pinned memory FP16×FP16→FP32 matrix multiplication example completed successfully!")

