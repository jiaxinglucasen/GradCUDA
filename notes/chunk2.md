# Chunk 2: Autograd Engine

## What I built
- Added _children parameter to __init__, stored as self._prev
  (a set of parent Tensors that created this one)
- Added self._backward as a placeholder function (does nothing by
  default), later overridden by each operation with real logic
- Added backward() method: builds a topological sort of the
  computation graph, then walks it in reverse, calling each
  Tensor's _backward() to propagate gradients

## Backward rules implemented (one per operation)
- Add: gradient passes through unchanged to both inputs
  (self.grad += out.grad, other.grad += out.grad)
- Multiply: each input's gradient depends on the OTHER input's value
  (self.grad += other.data * out.grad)
- Matmul: uses matrix transpose (.T) — standard calculus result for
  matrix multiplication backward pass
  (self.grad += out.grad @ other.data.T)
- ReLU: gradient passes through only where input was positive,
  otherwise blocked (multiplied by 0)
  (self.grad += (self.data > 0) * out.grad)

## Key concept: topological sort
- Needed because gradients must be calculated in the right order —
  can't compute a Tensor's gradient before everything that depends
  on it has already been processed.
- build_topo() recursively visits all parent Tensors first, then
  appends the current Tensor — so by the end, topo is ordered from
  "earliest" to "latest" in the computation.
- backward() processes this list in REVERSE, since gradients flow
  backward from the final output toward the original inputs.

## Test results (all confirmed working)
- Add:    a.grad = [1.], b.grad = [1.]
- Mul:    a.grad = [3.], b.grad = [2.]  (each gets the OTHER's value)
- Matmul: ran without error, correct shapes
- ReLU:   a.grad = [0. 0. 0. 1. 1.]  (negatives blocked, positives pass)

## Milestone
- Chunk 2 complete: full autograd engine working across all four
  operations. This is the same core mechanism PyTorch's .backward()
  uses internally, built from scratch at a smaller scale.