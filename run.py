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
from playground import *

if np.__version__ < "2.1.0":
    print("This example requires NumPy 2.1.0 or later", file=sys.stderr)
    sys.exit(0)


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


class Tensor:
  def __init__(self, name, shape, dtype):
    self.name = name
    self.shape = StridedShape(shape)
    self.dtype = dtype
  
# ======================================================================================
# Matrix dimensions and constants
# ======================================================================================

Mmma, Nmma, Kmma = 16, 8, 16

M, N, K = 4096, 4096, 4096
#M, N, K = 16, 8, 16

# ======================================================================================
# Variable definitions
# ======================================================================================


a_global = Tensor('g_a', [M, K], 'half')
b_global = Tensor('g_b', [K, N], 'half')
d_global = Tensor('g_d', [M, N], 'float')

a_sbuf = Tensor('s_a', [2, Mmma, Kmma], 'half')
b_sbuf = Tensor('s_b', [2, Kmma, Nmma], 'half')

a_reg = Tensor('r_a', [8], 'half')
b_reg = Tensor('r_b', [4], 'half')
d_reg = Tensor('r_d', [4], 'float')

# TODO: Clean up syntax to specify these
a_locs = [('i%2', 'by', f'{(i//2%2)*Mmma//2}+(tid/4)', 'i', f'{(i%2)+(i//4%2)*Kmma//2}+(tid%4)*2') for i in range(8)]
a_frags = Fragments('s_a', a_sbuf.shape.reshape([2, a_sbuf.shape[1]//Mmma, Mmma, a_sbuf.shape[2]//Kmma, Kmma]), 'half', 2, 8, a_locs)

b_locs = [('i%2', 'i', f'{(i%2)+(i//2)*Kmma//2}+(tid%4)*2', 'bx', 'tid/4') for i in range(4)]
b_frags = Fragments('s_b', b_sbuf.shape.reshape([2, b_sbuf.shape[1]//Kmma, Kmma, b_sbuf.shape[2]//Nmma, Nmma]), 'half', 2, 4, b_locs)

d_locs = [('by', f'{(i//2)*Mmma//2}+(tid/4)', 'bx', f'{i%2}+(tid%4)*2') for i in range(4)]
d_frags = Fragments('d', d_global.shape.reshape([M//Mmma, Mmma, N//Nmma, Nmma]), 'float', 4, 4, d_locs)


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
  __shared__ half s_a[2*{Mmma*Kmma}];
  __shared__ half s_b[2*{Kmma*Nmma}];
  int s_a_addr = __cvta_generic_to_shared(s_a);
  int s_b_addr = __cvta_generic_to_shared(s_b);

  {code_async_load_g2s(s=f"s_a_addr+tid*8*sizeof(half)", g=f"&a[by*{Mmma*K}+0*{Kmma}+(tid/2)*{K}+(tid%2)*8]", nbytes=16)}
  {code_async_load_g2s(s=f"s_b_addr+tid*4*sizeof(half)", g=f"&b[0*{Kmma*N}+bx*{Nmma}+(tid/2)*{N}+(tid%2)*4]", nbytes=8)}
  {code_commit()}

  // Loop over TILES_K
  for (int i = 0; i < {K//Kmma}; ++i) {{
    {code_sync()}
    if (i < {K//Kmma} - 1) {{
      int nexti = i+1;
      {code_async_load_g2s(s=f"s_a_addr+(nexti%2*{Mmma*Kmma}+tid*8)*sizeof(half)", g=f"&a[by*{Mmma*K}+nexti*{Kmma}+(tid/2)*{K}+(tid%2)*8]", nbytes=16)}
      {code_async_load_g2s(s=f"s_b_addr+(nexti%2*{Kmma*Nmma}+tid*4)*sizeof(half)", g=f"&b[nexti*{Kmma*N}+bx*{Nmma}+(tid/2)*{N}+(tid%2)*4]", nbytes=8)}
    }}
    {code_commit()}
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
grid = (N//Nmma, M//Mmma)
block = (32, )
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

