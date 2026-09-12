import torch
import matmul_cuda_ext

import numpy as np

def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data, _children=()):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def matmul_gpu(self, other):
        A_torch = torch.from_numpy(self.data.astype(np.float32)).cuda()
        B_torch = torch.from_numpy(other.data.astype(np.float32)).cuda()

        C_torch = matmul_cuda_ext.matmul_cuda(A_torch, B_torch)

        C_numpy = C_torch.cpu().numpy().astype(np.float64)
        out = Tensor(C_numpy, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __sub__(self, other):
        out = Tensor(self.data - other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += -out.grad
        out._backward = _backward

        return out

    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,))

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward

        return out

    def mean(self):
        out = Tensor(np.mean(self.data), (self,))

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size
        out._backward = _backward

        return out    


if __name__ == "__main__":
    # Test addition backward
    a = Tensor([2.0])
    b = Tensor([3.0])
    c = a + b
    c.backward()
    print("Add -> a.grad:", a.grad, "b.grad:", b.grad)

    # Test multiplication backward
    a2 = Tensor([2.0])
    b2 = Tensor([3.0])
    c2 = a2 * b2
    c2.backward()
    print("Mul -> a.grad:", a2.grad, "b.grad:", b2.grad)

    # Test matmul backward
    a3 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b3 = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c3 = a3.matmul(b3)
    c3.backward()
    print("Matmul -> a.grad:\n", a3.grad, "\nb.grad:\n", b3.grad)

    # Test relu backward
    a4 = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    b4 = a4.relu()
    b4.backward()
    print("ReLU -> a.grad:", a4.grad)

if __name__ == "__main__":
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    c_cpu = a.matmul(b)
    c_gpu = a.matmul_gpu(b)

    print("CPU result:", c_cpu.data)
    print("GPU result:", c_gpu.data)

import numpy as np

np.random.seed(0)
test_A = Tensor(np.random.randn(2000, 784).astype(np.float64))
test_B = Tensor(np.random.randn(784, 64).astype(np.float64))

result_cpu = test_A.matmul(test_B)
result_gpu = test_A.matmul_gpu(test_B)

diff = np.abs(result_cpu.data - result_gpu.data)
print("Max difference between CPU and GPU result:", diff.max())
print("Mean difference:", diff.mean())

test_A_cpu = Tensor(np.random.randn(2000, 784).astype(np.float64))
test_B_cpu = Tensor(np.random.randn(784, 64).astype(np.float64))
out_cpu = test_A_cpu.matmul(test_B_cpu)
loss_cpu = out_cpu.mean()
loss_cpu.backward()

test_A_gpu = Tensor(test_A_cpu.data.copy())
test_B_gpu = Tensor(test_B_cpu.data.copy())
out_gpu = test_A_gpu.matmul_gpu(test_B_gpu)
loss_gpu = out_gpu.mean()
loss_gpu.backward()

grad_diff_A = np.abs(test_A_cpu.grad - test_A_gpu.grad)
grad_diff_B = np.abs(test_B_cpu.grad - test_B_gpu.grad)

print("Max grad difference (A):", grad_diff_A.max())
print("Max grad difference (B):", grad_diff_B.max())