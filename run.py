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


class Tensor:
  def __init__(self, name, shape, dtype, *, init=None, indexer=None):
    self.name = name
    self.shape = StridedShape(shape)
    self.indexer = indexer or StridedIndexer(self.shape)
    self.dtype = dtype
    self.init = np.ones(shape)*init if isinstance(init, int) else init
  def reshape(self, *newshape):
    if len(newshape) == 1: newshape = newshape[0]
    return Tensor(self.name, self.shape.reshape(newshape), self.dtype)
  def permute(self, *permutation):
    if len(permutation) == 1: permutation = permutation[0]
    return Tensor(self.name, self.shape.permute(permutation), self.dtype)
  def flatten(self):
    return Tensor(self.name, [np.prod(self.shape.shape)], self.dtype, indexer=self.indexer.flat)
  def __getitem__(self, i):
    indexer = self.indexer[i]
    return Tensor(self.name, indexer.shape, self.dtype, indexer=indexer)
  def index(self):
    return self.indexer
  

sizeof = {
  'half': 2,
  'float': 4
}


# ======================================================================================
# Matrix dimensions and constants
# ======================================================================================

Mmma, Nmma, Kmma = 16, 8, 16

M, N, K = 4096, 4096, 4096
#M, N, K = 16, 8

# ======================================================================================
# Variable definitions
# ======================================================================================

l0 = SymbolicVar('tid')
l1 = SymbolicVar('l1')
l2 = SymbolicVar('l2')
g0 = SymbolicVar('bx')
g1 = SymbolicVar('by')
g2 = SymbolicVar('g2')

i = SymbolicVar('i')

a_global = Tensor('a', [M, K], 'half')
b_global = Tensor('b', [K, N], 'half')
d_global = Tensor('d', [M, N], 'float')

a_sbuf = Tensor('s_a', [2, Mmma, Kmma], 'half')
b_sbuf = Tensor('s_b', [2, Kmma, Nmma], 'half')

a_sbuf_frag = a_sbuf.reshape(2, 2, 8, 2, 4, 2).permute(0, 2, 4, 3, 1, 5)[i%2, l0//4, l0%4].flatten()
b_sbuf_frag = b_sbuf.reshape(2, 2, 4, 2, 8).permute(0, 4, 2, 1, 3)[i%2, l0//4, l0%4].flatten()

d_global_frag = d_global.reshape(M//Mmma, 2, 8, N//Nmma, 4, 2).permute(0, 3, 2, 4, 1, 5)[g1, g0, l0//4, l0%4].flatten()

a_reg = Tensor('s_a_frag', [8], 'half')  # r_a
b_reg = Tensor('s_b_frag', [4], 'half')  # r_b
d_reg = Tensor('d_frag', [4], 'float', init=0)  # r_d


# ======================================================================================
# CUDA kernel for FP16 matrix multiplication with FP32 accumulation
# ======================================================================================
def code_decl_v2(t):
  if t.init is not None:
    return f'{t.dtype} {t.name}{"".join(f"[{dim}]" for dim in t.shape.shape)} = {{{",".join(str(v) for v in t.init)}}};'
  else:
    return f'{t.dtype} {t.name}{"".join(f"[{dim}]" for dim in t.shape.shape)};'
def code_assign(l, r):
  init = []
  if r is not None:
    index = [0]*len(l.shape)
    while True:
      init.append(f'{l.name}[{expr(l.index()[*index])}] = {r.name}[{expr(r.index()[*index])}];')
      index[-1] += 1
      for i in reversed(range(1, len(index))):
        if index[i] >= l.shape.shape[i]:
          index[i] = 0
          index[i-1] += 1
        else:
          break
      if index[0] >= l.shape.shape[0]:
        break
  return '\n    '.join(init)

def code_load_v2(mat_frags):
  return '    '.join((f'{mat_frags.frag_name}[{i}] = {mat_frags.name}[{loc}];\n' for i, loc in enumerate(mat_frags.locs)))
def code_store_v2(mat_frags):
  return '\n  '.join((f'{mat_frags.name}[{loc}] = {mat_frags.frag_name}[{i}];' for i, loc in enumerate(mat_frags.locs)))
def code_exec(a, b, c_d):
  return f'''
    // --- MMA PTX ---
    unsigned int* {a.name}_int = reinterpret_cast<unsigned int *>({a.name});
    unsigned int* {b.name}_int = reinterpret_cast<unsigned int *>({b.name});
    unsigned int* {c_d.name}_int = reinterpret_cast<unsigned int *>({c_d.name});
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}},{{%4,%5,%6,%7}},{{%8,%9}},{{%0,%1,%2,%3}};\\n"
      : {",".join(f"\"+r\"({c_d.name}_int[{i}])" for i in range(c_d.shape.shape[0]//(4//sizeof[c_d.dtype])))}
      : {",".join(f"\"r\"({a.name}_int[{i}])" for i in range(a.shape.shape[0]//(4//sizeof[a.dtype])))}
      , {",".join(f"\"r\"({b.name}_int[{i}])" for i in range(b.shape.shape[0]//(4//sizeof[b.dtype])))}
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
  {code_decl_v2(a_reg)}
  {code_decl_v2(b_reg)}
  {code_decl_v2(d_reg)}

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
    {code_assign(a_reg, a_sbuf_frag)}
    {code_assign(b_reg, b_sbuf_frag)}
    {code_exec(a_reg, b_reg, d_reg)}
  }}
  {code_assign(d_global_frag, d_reg)}
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

