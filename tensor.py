"""
tensor.py -- the core of GradCUDA: a from-scratch autograd engine.

A Tensor wraps a NumPy array (`data`) and remembers which Tensors it was
computed from. Calling `.backward()` on the final result (the loss) then walks
that history in reverse and fills in `.grad` -- the gradient of the loss with
respect to every Tensor that contributed to it. This is reverse-mode automatic
differentiation, the same idea PyTorch's autograd is built on.

Two matrix-multiply paths exist:
  * matmul      -- NumPy on the CPU, float64.
  * matmul_gpu  -- my custom CUDA kernel (matmul_cuda_ext.cu), float32.
Everything else in the engine (add, relu, backward, ...) runs on the CPU.
"""
import torch                 # Used only to move arrays onto the GPU. Must be imported BEFORE matmul_cuda_ext (Windows DLL search order).
import matmul_cuda_ext       # My compiled CUDA extension, built from matmul_cuda_ext.cu.

import numpy as np           # NumPy: stores every Tensor's values and does all CPU math.


def unbroadcast(grad, shape):
    """Shrink a gradient back to `shape` by summing over the dimensions NumPy broadcast.

    Example: in `x(2000, 64) + b(64,)`, NumPy silently stretches b to 2000 rows.
    The gradient coming back is (2000, 64), but b's gradient slot is (64,), so the
    2000 per-row gradients must be summed into one.
    """
    while grad.ndim > len(shape):           # The gradient has extra leading dimensions (e.g. the batch dimension)...
        grad = grad.sum(axis=0)             # ...so sum them away until the number of dimensions matches.
    for i, dim in enumerate(shape):         # Now check each remaining dimension against the target shape.
        if dim == 1 and grad.shape[i] != 1: # The original had size 1 here but the gradient is bigger, so it was stretched...
            grad = grad.sum(axis=i, keepdims=True)   # ...so sum along it, keeping the dimension as size 1.
    return grad                             # The gradient now has exactly the target shape.


class Tensor:
    """A NumPy array plus the bookkeeping needed for automatic differentiation."""

    def __init__(self, data, _children=()):
        # np.array copies the input and converts it to float64. By default it also KEEPS the input's
        # memory layout, so a column-major input stays column-major (this caused the GPU bug; see _to_cuda).
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)     # Gradient of the loss w.r.t. this Tensor; starts at zero, same shape as data.
        self._prev = set(_children)              # The Tensors this one was computed from (its parents in the computation graph).
        self._backward = lambda: None            # Function that pushes this Tensor's gradient to its parents; leaf Tensors do nothing.
        self.static = False    # True for constant inputs (like X_train) whose values never change, so the GPU copy can be cached.
        self._cuda = None      # Cached float32 copy of this Tensor on the GPU; only filled in when static is True.

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other))   # Forward: add the values; remember both inputs as parents.

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)    # d(a+b)/da = 1, so pass the gradient through (shrunk if a was broadcast).
            other.grad += unbroadcast(out.grad, other.data.shape)  # Same for b. "+=" because a Tensor can feed several operations.
        out._backward = _backward                # Attach the backward rule to the result.

        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other))   # Forward: element-wise multiply.

        def _backward():
            self.grad += other.data * out.grad   # d(a*b)/da = b, so a's gradient is b times the incoming gradient.
            other.grad += self.data * out.grad   # d(a*b)/db = a, so b's gradient is a times the incoming gradient.
        out._backward = _backward

        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other))   # Forward: matrix multiply on the CPU (NumPy, float64).

        def _backward():
            self.grad += out.grad @ other.data.T   # For C = A @ B: the gradient w.r.t. A is (incoming gradient) @ B transposed.
            other.grad += self.data.T @ out.grad   # And the gradient w.r.t. B is A transposed @ (incoming gradient).
        out._backward = _backward

        return out

    def _to_cuda(self):
        """Return this Tensor's values as a float32, row-major tensor on the GPU."""
        # The kernel reads raw memory as row-major float32, so the data must be converted to exactly that.
        # np.ascontiguousarray forces row-major layout; skipping it made the kernel read scrambled data
        # whenever the input was column-major (the main bug in this project).
        if self.static:                          # This Tensor never changes (e.g. the MNIST images)...
            if self._cuda is None:               # ...and has not been uploaded yet, so convert and upload it once.
                self._cuda = torch.from_numpy(np.ascontiguousarray(self.data, dtype=np.float32)).cuda()
            return self._cuda                    # Every later call reuses the cached GPU copy: no conversion, no upload.
        return torch.from_numpy(np.ascontiguousarray(self.data, dtype=np.float32)).cuda()   # Normal Tensors (weights, hidden values) change, so redo it every time.

    def matmul_gpu(self, other):
        A_torch = self._to_cuda()                # Left matrix on the GPU (float32, row-major).
        B_torch = other._to_cuda()               # Right matrix on the GPU.

        C_torch = matmul_cuda_ext.matmul_cuda(A_torch, B_torch)   # Run my CUDA kernel; the result stays on the GPU.

        C_numpy = C_torch.cpu().numpy().astype(np.float64)   # Copy the result back to the CPU, as a float64 NumPy array for the rest of the engine.
        out = Tensor(C_numpy, (self, other))     # Wrap it in a Tensor, remembering both inputs as parents.

        def _backward():
            self.grad += out.grad @ other.data.T   # Same backward math as the CPU matmul: it does not depend on where the forward pass ran.
            other.grad += self.data.T @ out.grad   # (The backward pass itself still runs on the CPU in NumPy.)
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))   # Forward: replace every negative value with 0.

        def _backward():
            self.grad += (self.data > 0) * out.grad   # Gradient passes through where the input was positive and is blocked (0) elsewhere.
        out._backward = _backward

        return out

    def backward(self):
        topo = []                                # Will hold every Tensor in the graph, parents before children.
        visited = set()                          # Tensors already placed in the list, so none is added twice.

        def build_topo(v):                       # Depth-first walk that lists each Tensor only after all of its parents.
            if v not in visited:
                visited.add(v)
                for child in v._prev:            # (_prev holds the Tensors v was computed from.)
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)      # Seed: the gradient of the final result with respect to itself is 1.
        for v in reversed(topo):                 # Go from the final result back toward the inputs, so each Tensor's gradient is complete
            v._backward()                        # before it is passed on to its own parents.

    def __sub__(self, other):
        out = Tensor(self.data - other.data, (self, other))   # Forward: element-wise subtraction.

        def _backward():
            self.grad += out.grad                # d(a-b)/da = +1.
            other.grad += -out.grad              # d(a-b)/db = -1.
        out._backward = _backward

        return out

    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,))   # Forward: raise every value to a constant power.

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad   # Power rule: d(x^p)/dx = p * x^(p-1), times the incoming gradient.
        out._backward = _backward

        return out

    def mean(self):
        out = Tensor(np.mean(self.data), (self,))   # Forward: average every value down to a single number.

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size   # Each value contributes equally: 1/N of the incoming gradient.
        out._backward = _backward

        return out


# Self-tests: run `python tensor.py` to check the engine on tiny cases with hand-checkable answers.
if __name__ == "__main__":
    # Test addition backward: both inputs should get gradient 1.
    a = Tensor([2.0])
    b = Tensor([3.0])
    c = a + b
    c.backward()
    print("Add -> a.grad:", a.grad, "b.grad:", b.grad)

    # Test multiplication backward: a's gradient should be b (3.0) and b's should be a (2.0).
    a2 = Tensor([2.0])
    b2 = Tensor([3.0])
    c2 = a2 * b2
    c2.backward()
    print("Mul -> a.grad:", a2.grad, "b.grad:", b2.grad)

    # Test matmul backward on a 2x2 example.
    a3 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b3 = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c3 = a3.matmul(b3)
    c3.backward()
    print("Matmul -> a.grad:\n", a3.grad, "\nb.grad:\n", b3.grad)

    # Test relu backward: gradient should be 0 for inputs <= 0 and 1 for inputs > 0.
    a4 = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    b4 = a4.relu()
    b4.backward()
    print("ReLU -> a.grad:", a4.grad)

# GPU-vs-CPU checks: the CUDA matmul must agree with the NumPy matmul.
if __name__ == "__main__":
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    c_cpu = a.matmul(b)          # NumPy result.
    c_gpu = a.matmul_gpu(b)      # CUDA kernel result.

    print("CPU result:", c_cpu.data)
    print("GPU result:", c_gpu.data)

    import numpy as np

    # Forward check at training scale (2000x784 @ 784x64): the two results should differ only by float32 rounding.
    # NOTE: random arrays are always row-major, so this test could NOT catch the memory-layout bug (see notes/bugs_fixed.md).
    np.random.seed(0)
    test_A = Tensor(np.random.randn(2000, 784).astype(np.float64))
    test_B = Tensor(np.random.randn(784, 64).astype(np.float64))

    result_cpu = test_A.matmul(test_B)
    result_gpu = test_A.matmul_gpu(test_B)

    diff = np.abs(result_cpu.data - result_gpu.data)
    print("Max difference between CPU and GPU result:", diff.max())
    print("Mean difference:", diff.mean())

    # Backward check: gradients through the CPU matmul vs. the GPU matmul, starting from identical data.
    # (Matmul's backward uses the original inputs, not the forward output, so this checks the backward wiring only.)
    test_A_cpu = Tensor(np.random.randn(2000, 784).astype(np.float64))
    test_B_cpu = Tensor(np.random.randn(784, 64).astype(np.float64))
    out_cpu = test_A_cpu.matmul(test_B_cpu)
    loss_cpu = out_cpu.mean()
    loss_cpu.backward()

    test_A_gpu = Tensor(test_A_cpu.data.copy())    # Same values, fresh Tensors, so the two runs cannot interfere.
    test_B_gpu = Tensor(test_B_cpu.data.copy())
    out_gpu = test_A_gpu.matmul_gpu(test_B_gpu)
    loss_gpu = out_gpu.mean()
    loss_gpu.backward()

    grad_diff_A = np.abs(test_A_cpu.grad - test_A_gpu.grad)
    grad_diff_B = np.abs(test_B_cpu.grad - test_B_gpu.grad)

    print("Max grad difference (A):", grad_diff_A.max())
    print("Max grad difference (B):", grad_diff_B.max())
