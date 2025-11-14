# Playground to test compiler ideas

"""
Synthetic indexing

The kernel is invoked with a 2x3D index:
<bx, by, bz>, <tx, ty, tz>

for uniformity this can be thought of as a 6D index:
<bz, by, bx, tz, ty, tx>

or

<i5, i4, i3, i2, i1, i0>

Each thread gets a unique index, thus the total number of threads
is the product of the size of all dimensions.

Unfortunately the choice of indexing scheme is not arbitrary,
there are performance considerations when trading off between
block and group sizes, since groups run within a SM and blocks,
across multiple. Also, it seems like the best thing to do with i0
is to reserve it to represent the lane id 0-31 within each warp.

This is annoying.

There needs to be a way to express dimensions in terms of these
indexes simply, without manual manipulation, and have automatic
tracking of the stride expression determining real offsets.

Layout of fragment registers for thread 0 in a 16x16 warp patch of A
0 1 _ _ _ _ _ _ 2 3 _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            ...
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
4 5 _ _ _ _ _ _ 6 7 _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            ...
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

Layout of fragment registers for thread 1 in a 16x16 warp patch of A
_ _ 0 1 _ _ _ _ _ _ 2 3 _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            ...
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ 4 5 _ _ _ _ _ _ 6 7 _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            ...
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
Layout of fragment registers for thread 4 in a 16x16 warp patch of A
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
0 1 _ _ _ _ _ _ 2 3 _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            ...
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
4 5 _ _ _ _ _ _ 6 7 _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            ...
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

Interpreting patch as shape
[2, 8, 2, 4, 2]
strides=[128, 16, 8, 2, 1]

Then permuting by
1 3 0 2 4

We get a shape
[8, 4, 2, 2, 2]
strides=[16, 2, 128, 8, 1]

Then, we can vectorize the tail - slice it off to get:
[8,4]

which maps to lane 0-31

"""

class StridedShape:
  def __init__(self, shape, strides=None):
    if isinstance(shape, StridedShape):
      self.shape = shape.shape
      self.strides = shape.strides
    else:
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
  def permute(self, permutation):
    assert len(permutation) == len(self.shape)
    return StridedShape([self.shape[i] for i in permutation], [self.strides[i] for i in permutation])
  def __len__(self):
    return len(self.shape)


class SymbolicVar:
  def __init__(self, expr):
    self.expr = expr.expr if isinstance(expr, SymbolicVar) else expr if isinstance(expr, str) else str(expr)
  def __add__(self, other):
    if other == 0: return SymbolicVar(self.expr)
    elif self == 0: return SymbolicVar(other)
    other_expr = other.expr if isinstance(other, SymbolicVar) else str(other)
    return SymbolicVar(f"({self.expr}+{other_expr})")
  def __sub__(self, other):
    if other == 0: return SymbolicVar(self.expr)
    other_expr = other.expr if isinstance(other, SymbolicVar) else str(other)
    return SymbolicVar(f"({self.expr}-{other_expr})")
  def __mul__(self, other):
    if self == 0 or other == 0: return SymbolicVar('0')
    other_expr = other.expr if isinstance(other, SymbolicVar) else str(other)
    return SymbolicVar(f"({self.expr}*{other_expr})")
  def __floordiv__(self, other):
    if self == 0: return SymbolicVar('0')
    if other == 0: raise ValueError('Divide by zero')
    other_expr = other.expr if isinstance(other, SymbolicVar) else str(other)
    return SymbolicVar(f"({self.expr}/{other_expr})")
  def __mod__(self, other):
    if self == 0: return SymbolicVar('0')
    if other == 0: raise ValueError('Divide by zero')
    other_expr = other.expr if isinstance(other, SymbolicVar) else str(other)
    return SymbolicVar(f"({self.expr}%{other_expr})")
  def __pow__(self, other):
    if other == 0: return SymbolicVar('1')
    elif self == 0: return SymbolicVar('0')
    other_expr = other.expr if isinstance(other, SymbolicVar) else str(other)
    return SymbolicVar(f"({self.expr}**{other_expr})")
  def __repr__(self):
    return f"SymbolicVar(expr='{self.expr}')"
  def __eq__(self, x):
    return self.expr == SymbolicVar(x).expr

class StridedIndexer:
  class FlatIndexer:
    def __init__(self, st):
      self.st = st
      self.shape = st.shape.shape
    def __getitem__(self, i):
      i = i[0] if getattr(i, '__getitem__', None) else i
      index = []
      for dim in self.st.shape.shape[::-1]:
        index.insert(0, i%dim)
        i //= dim
      return self.st[index]
  def __init__(self, shape:StridedShape, base_offset=0):
    assert isinstance(shape, StridedShape)
    self.shape = shape
    self.base_offset = base_offset
    self.flat = self.FlatIndexer(self)

  def _expand_index(self, idx):
    if not getattr(idx, '__getitem__', None): idx = (idx,)
    out, ell = [], False
    for i in idx:
      if i is Ellipsis:
        if ell: raise IndexError("Multiple ellipsis")
        ell = True
        n_missing = len(self.shape) - (len(idx) - 1)
        out += [slice(None)] * n_missing
      else:
        out.append(i)
    while len(out) < len(self.shape): out.append(slice(None))
    if len(out) > len(self.shape): raise IndexError("Too many indices")
    return tuple(out)

  def __getitem__(self, idx):
    idx = self._expand_index(idx)
    offset, new_shape, new_strides = self.base_offset, [], []
    for dim in range(len(self.shape)):
      i, dim_size, stride = idx[dim], self.shape.shape[dim], self.shape.strides[dim]
      if isinstance(i, SymbolicVar):
        if not isinstance(offset, SymbolicVar):
          offset = SymbolicVar(offset)
        offset += i * stride
      elif isinstance(i, slice):
        start, stop, step = i.start, i.stop, i.step
        start = 0 if start is None else (dim_size + start if start < 0 else start)
        stop = dim_size if stop is None else (dim_size + stop if stop < 0 else stop)
        step = 1 if step is None else step
        size = max(0, (stop - start + (step - 1)) // step)
        new_shape.append(size)
        new_strides.append(stride * step)
        offset += start * stride
      else:
        if i < 0: i += dim_size
        if not 0 <= i < dim_size: raise IndexError("Index out of bounds")
        offset += i * stride
    if not new_shape: return offset
    return StridedIndexer(StridedShape(new_shape, new_strides), offset)


if __name__ == '__main__':
  atile = StridedShape([16,16]).reshape([2, 8, 2, 4, 2]).permute([1, 3, 2, 0, 4])
  aindex = StridedIndexer(atile, 0)

  print(aindex[..., 1, 0, 1][0,0])
  print(aindex.flat[0])
  print(aindex.flat[1])
  print(aindex.flat[2])
  print(aindex.flat[3])
  print(aindex.flat[4])
  print(aindex.flat[5])
  print(aindex.flat[6])
  print(aindex.flat[7])
  print()
  print(aindex.flat[8+0])
  print(aindex.flat[8+1])
  print(aindex.flat[8+2])
  print(aindex.flat[8+3])
  print(aindex.flat[8+4])
  print(aindex.flat[8+5])
  print(aindex.flat[8+6])
  print(aindex.flat[8+7])
  print()


  tid = SymbolicVar('tid')

  print(aindex[tid//4, tid%4, 0, 0, 1])
  print(aindex[tid//4, tid%4, 0, 1, 0])
  print(aindex[tid//4, tid%4, 1, 0, 0])
  print()

  print(aindex[tid//4, tid%4].flat[0])
  print(aindex[tid//4, tid%4].flat[1])
  print(aindex[tid//4, tid%4].flat[2])


  btile = StridedShape([16,8]).reshape([2, 4, 2, 8]).permute([1, 3, 0, 2])
  bindex = StridedIndexer(btile, 0)

  print(bindex[tid%4,tid//4].flat[0])
  print(bindex[tid%4,tid//4].flat[1])
  print(bindex[tid%4,tid//4].flat[2])
  print(bindex[tid%4,tid//4].flat[3])
  print(bindex[tid%4,tid//4].flat[4])
