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
from functools import reduce
import operator

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


# Description of the access pattern as a strided multidim array. Format TBD

class StridedIndexMapper:
  def __init__(self, shape, strides):
      self.shape = shape
      self.strides = strides

  def logical_to_real(self, idx):
      coords = []
      for dim in reversed(self.shape):
          coords.append(idx % dim)
          idx //= dim
      coords.reverse()
      return sum(c * s for c, s in zip(coords, self.strides))

  def export_c_expression(self, idx_name="idx"):
    terms = []
    cumprod = 1
    for dim, stride in zip(reversed(self.shape), reversed(self.strides)):
        term = f"(({idx_name} / {cumprod}) % {dim}) * {stride}"
        terms.append(term)
        cumprod *= dim
    terms.reverse()
    return " + ".join(terms)

class Fragments:
  def __init__(self, name, lshape, dtype, esize, nregs, fshape, fstrides):
    self.name = name
    self.lshape = lshape
    self.dtype = dtype
    self.esize = esize
    self.nregs = nregs
    assert reduce(operator.mul, fshape, 1) == reduce(operator.mul, lshape, 1), 'Matrix view does not cover all elements'
    assert fshape[-1] >= 4 // esize, 'Misaligned accesses in view'
    # Merge < 4 byte loads into 4 byte loads
    #if 4 // esize > 1:
    #  assert fstrides[-1] == 1, 'Misaligned accesses in view'
    #  fshape[-1] //= (4 // esize)
    #  nregs //= (4 // esize)

    # Go from the right, find the dimension at which all registers are indexable
    elements = 1
    for i, dim in list(enumerate(fshape))[::-1]:
      elements *= dim
      if elements >= nregs:
        assert elements == nregs, 'Lanes should index starting at a new dimension'
        break
    self.lanes = StridedIndexMapper(fshape[:i], fstrides[:i])
    self.regs = StridedIndexMapper(fshape[i:], fstrides[i:])


  def linidx(self, lane, reg):
    return self.lanes.logical_to_real(lane) + self.regs.logical_to_real(reg)


a_locs = [(f'{(i//2%2)*M//2}+(lane_id/4)', f'{(i%2)+(i//4%2)*M//2}+(lane_id%4)*2') for i in range(8)]
a_shape = (M, K)
a_locs_flat = (f'({h})*{a_shape[-1]}+({c})' for h, c in a_locs)

b_locs = [(f'{(i%2)+(i//2)*K//2}+(lane_id%4)*2', f'lane_id/4') for i in range(4)]
b_shape = (K, N)
b_locs_flat = (f'({h})*{b_shape[-1]}+({c})' for h, c in b_locs)

d_locs = [(f'{(i//2)*(M//2)}+(lane_id/4)', f'{i%2}+(lane_id%4)*2') for i in range(4)]
d_shape = (M, N)
d_locs_flat = (f'({h})*{b_shape[-1]}+({c})' for h, c in d_locs)

a_frags = Fragments('A', (M, K), 'half', 2, 8, [M//2, K//4, 2, 2, 2], [M, 2, K//2, M*K//2, 1])
b_frags = Fragments('B', (K, N), 'half', 2, 4, [N, K//4, 2, 2], [1, 2*N, K*N//2, N])
d_frags = Fragments('D', (M, N), 'float', 4, 4, [8, 4, 2, 2], [N, 2, M*N//2, 1])


# ======================================================================================
# CUDA kernel for FP16 matrix multiplication with FP32 accumulation
# ======================================================================================
def code_load(mat_frags):
  return f'''
    unsigned int {mat_frags.name.lower()}_lane_offset = {mat_frags.lanes.export_c_expression("lane_id")};
    {mat_frags.dtype} {mat_frags.name.lower()}_frag[{mat_frags.nregs}];
    {'\n    '.join((
      f'{mat_frags.name.lower()}_frag[{i}] = {mat_frags.name}[{mat_frags.name.lower()}_lane_offset + {mat_frags.linidx(0, i)}];'
      for i in range(mat_frags.nregs)))}
    unsigned int* {mat_frags.name.lower()}_int = reinterpret_cast<unsigned int *>({mat_frags.name.lower()}_frag);
  '''

def code_load_v2(locs, mat_frags):
  return f'''
    {mat_frags.dtype} {mat_frags.name.lower()}_frag[{mat_frags.nregs}] {{
      {',\n      '.join((f'{mat_frags.name}[{loc}]' for i, loc in enumerate(locs)))}
    }};
    unsigned int* {mat_frags.name.lower()}_int = reinterpret_cast<unsigned int *>({mat_frags.name.lower()}_frag);
  '''

code_exec = f'''
    float c_frag[4] = {{0.f, 0.f, 0.f, 0.f}};
    float d_frag[4];
    // --- MMA PTX ---
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}},{{%4,%5,%6,%7}},{{%8,%9}},{{%10,%11,%12,%13}};\\n"
        : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
        : "r"(a_int[0]), "r"(a_int[1]), "r"(a_int[2]), "r"(a_int[3]),
        "r"(b_int[0]), "r"(b_int[1]),
        "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3])
    );
'''

def code_store(mat_frags):
  return f'''
    unsigned int {mat_frags.name.lower()}_lane_offset = {mat_frags.lanes.export_c_expression("lane_id")};
    {'\n    '.join((
      f'{mat_frags.name}[{mat_frags.name.lower()}_lane_offset + {mat_frags.linidx(0, i)}] = {mat_frags.name.lower()}_frag[{i}];'
      for i in range(mat_frags.nregs)))}
  '''

def code_store_v2(locs, mat_frags):
  return f'''
    unsigned int {mat_frags.name.lower()}_lane_offset = {mat_frags.lanes.export_c_expression("lane_id")};
    {'\n    '.join((f'{mat_frags.name}[{loc}] = {mat_frags.name.lower()}_frag[{i}];' for i, loc in enumerate(locs)))}
  '''

code = f'''extern "C"
__global__ void matmul_fp16_fp32(const half* A, const half* B, float* C, float* D) {{
    unsigned int lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    {code_load_v2(a_locs_flat, a_frags)}
    {code_load_v2(b_locs_flat, b_frags)}
    {code_exec}
    {code_store_v2(d_locs_flat, d_frags)}
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

