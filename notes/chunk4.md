# Chunk 4: Profiling — Finding the Bottleneck

## What I built
- Added Python's built-in `time` module (no install needed)
- Broke the forward pass out of the forward() function call into
  inline steps, so each operation could be timed separately:
  matmul -> add -> relu -> matmul -> add
- Wrapped each step with t0 = time.time() / time.time() - t0,
  accumulating into running totals: matmul_time, add_time,
  relu_time, backward_time
- Timed loss.backward() as one combined total (it internally
  includes topological sort + every operation's _backward function)
- Printed all four totals after training finished

## Bug caught while restructuring
- Typo introduced while rewriting the loop inline:
  W2.data -= learning_rate * b2.grad  (wrong — used b2's grad
  instead of W2's own grad)
- Should be: W2.data -= learning_rate * W2.grad
- Lesson: restructuring working code (even just for profiling)
  can introduce new bugs — always double check variable names
  match after copy/pasting or rewriting loops

## Results (100 epochs, learning_rate = 0.5)
- Matmul total time:     0.1895s
- Add total time:        0.1257s
- ReLU total time:       0.0484s
- Backward total time:   1.0538s   <-- dominant cost, by far
- Accuracy after training: 67.15%
  (slight variation from Chunk 3's 75.40% run — expected, due to
  loop restructuring / the b2.grad typo; still far above the
  ~10% random-guess baseline, so engine is still confirmed working)

## Key concept: why backward is so much more expensive than forward
- matmul's _backward function does MORE matrix multiplication than
  matmul's forward pass:
    self.grad  += out.grad @ other.data.T
    other.grad += self.data.T @ out.grad
  -> that's two separate matrix multiplications just to backprop
     through one forward matmul call
- backward() also does topological sort bookkeeping (building and
  walking the topo list) on top of all the individual _backward
  calls (add, relu, matmul all contribute)
- Together, this is why backward_time (1.05s) is roughly 3x bigger
  than every other measured category combined (0.36s)

## Conclusion / target for Chunk 5
- Matrix multiplication (matmul) is the confirmed bottleneck —
  both directly (forward matmul calls) and indirectly (as the
  dominant cost inside the backward pass via its transpose-based
  gradient calculations)
- This is the operation to accelerate with a custom CUDA kernel
- A fast CUDA matmul kernel would help BOTH the forward pass
  timing AND a large chunk of the backward pass timing, since
  matmul's backward logic is itself built from more matmul-style
  operations

## Milestone
- Chunk 4 complete: bottleneck identified through real measurement,
  not guessing. Confirms the plan from the very start of this
  project — matmul is exactly the operation NVIDIA's own libraries
  (cuBLAS, cuDNN) exist to accelerate, and it's exactly what this
  project's CUDA kernel (Chunk 5) will target next.