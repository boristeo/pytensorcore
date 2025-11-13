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
  def __getitem__(self, i):
    return self.shape[i]
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
  @property
  def frag_name(self):
    return f'{self.name.lower()}_frag'
  @property
  def int_name(self):
    return f'{self.name.lower()}_int'
  @property
  def nint32(self):
    return self.nregs // (4 // self.esize)

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

M, N, K = 4096, 4096, 4096
#M, N, K = 16, 8, 16

A_shape = StridedShape([M, K])
B_shape = StridedShape([K, N])
C_shape = StridedShape([M, N])

Mmma = 16
Kmma = 16
Nmma = 8

BLOCK_SIZE = 32  # single warp
TILES_M = int(math.ceil(M / Mmma)) 
TILES_N = int(math.ceil(N / Nmma))
TILES_K = int(math.ceil(K / Kmma))
NBLOCKS = TILES_M * TILES_N * TILES_K

A_sbuf_shape = StridedShape([Mmma, Kmma])
B_sbuf_shape = StridedShape([Kmma, Nmma])


# TODO: Clean up syntax to specify these
a_locs = [('by', f'{(i//2%2)*Mmma//2}+(tid/4)', 'i', f'{(i%2)+(i//4%2)*Kmma//2}+(tid%4)*2') for i in range(8)]
a_frags = Fragments('s_a', A_sbuf_shape.reshape([A_sbuf_shape[0]//Mmma, Mmma, A_sbuf_shape[1]//Kmma, Kmma]), 'half', 2, 8, a_locs)

b_locs = [('i', f'{(i%2)+(i//2)*Kmma//2}+(tid%4)*2', 'bx', 'tid/4') for i in range(4)]
b_frags = Fragments('s_b', B_sbuf_shape.reshape([B_sbuf_shape[0]//Kmma, Kmma, B_sbuf_shape[1]//Nmma, Nmma]), 'half', 2, 4, b_locs)

d_locs = [('by', f'{(i//2)*Mmma//2}+(tid/4)', 'bx', f'{i%2}+(tid%4)*2') for i in range(4)]
d_frags = Fragments('d', C_shape.reshape([TILES_M, Mmma, TILES_N, Nmma]), 'float', 4, 4, d_locs)


# ======================================================================================
# CUDA kernel for FP16 matrix multiplication with FP32 accumulation
# ======================================================================================
def code_decl(var, val=None):
  return f'{var.dtype} {var.frag_name}[{var.nregs}]{f" = {val}" if val is not None else ""};'
def code_load_v2(mat_frags):
  return '    '.join((f'{mat_frags.frag_name}[{i}] = {mat_frags.name}[{loc}];\n' for i, loc in enumerate(mat_frags.locs)))
def code_store_v2(mat_frags):
  return '\n  '.join((f'{mat_frags.name}[{loc}] = {mat_frags.frag_name}[{i}];' for i, loc in enumerate(mat_frags.locs)))
def code_exec(a, b, c_d):
  return f'''
    // --- MMA PTX ---
    unsigned int* {a.int_name} = reinterpret_cast<unsigned int *>({a.frag_name});
    unsigned int* {b.int_name} = reinterpret_cast<unsigned int *>({b.frag_name});
    unsigned int* {c_d.int_name} = reinterpret_cast<unsigned int *>({c_d.frag_name});
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}},{{%4,%5,%6,%7}},{{%8,%9}},{{%0,%1,%2,%3}};\\n"
      : {",".join(f"\"+r\"({c_d.int_name}[{i}])" for i in range(c_d.nint32))}
      : {",".join(f"\"r\"({a.int_name}[{i}])" for i in range(a.nint32))}
      , {",".join(f"\"r\"({b.int_name}[{i}])" for i in range(b.nint32))}
    );'''
def code_async_load_g2s(*, g, s, nbytes):
  return f'''
    asm volatile("cp.async.ca.shared.global [%0],[%1],{nbytes};\\n"
      :
      : "r"((int)({s}))
      , "l"({g})
    );'''
def code_commit():
    return 'asm volatile("cp.async.commit_group;\\n" ::);'
def code_sync():
  return f'''
    asm volatile("cp.async.wait_group 0;\\n" ::);
    __syncthreads();
  '''


code = f'''extern "C"
__global__ void matmul_fp16_fp32(
  const half* a,
  const half* b,
  float* c,
  float* d)
{{
  unsigned int tid = threadIdx.x;
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;
  unsigned int bz = blockIdx.z;
  {code_decl(a_frags)}
  {code_decl(b_frags)}
  {code_decl(d_frags, "{0.0f,0.0f,0.0f,0.0f}")}

  // TODO: Next step - double buf pipeline
  __shared__ half s_a[{Mmma*Kmma}];
  __shared__ half s_b[{Kmma*Nmma}];
  int s_a_addr = __cvta_generic_to_shared(s_a);
  int s_b_addr = __cvta_generic_to_shared(s_b);

  // Loop over TILES_K
  for (int i = 0; i < {TILES_K}; ++i) {{
    {code_async_load_g2s(s="s_a_addr+tid*8*sizeof(half)", g=f"&a[by*{Mmma*K}+i*{Kmma}+(tid/2)*{K}+(tid%2)*8]", nbytes=16)}
    {code_async_load_g2s(s="s_b_addr+tid*4*sizeof(half)", g=f"&b[i*{Kmma*N}+bx*{Nmma}+(tid/2)*{N}+(tid%2)*4]", nbytes=8)}
    {code_commit()}
    {code_sync()}
    // Done loading tile
    {code_load_v2(a_frags)}
    {code_load_v2(b_frags)}
    {code_exec(a_frags, b_frags, d_frags)}
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
block = (BLOCK_SIZE, )
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

