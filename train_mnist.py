import numpy as np
import torch
from tensor import Tensor
from sklearn.datasets import fetch_openml
import time

print("Loading MNIST... (this may take a minute the first time)")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Use a small subset for speed
X_train = X[:2000] / 255.0
y_train = y[:2000]

def one_hot(labels, num_classes=10):
    out = np.zeros((len(labels), num_classes))
    out[np.arange(len(labels)), labels] = 1
    return out

y_train_onehot = one_hot(y_train)

np.random.seed(42)

W1 = Tensor(np.random.randn(784, 64) * 0.01)
b1 = Tensor(np.zeros(64))
W2 = Tensor(np.random.randn(64, 10) * 0.01)
b2 = Tensor(np.zeros(10))

def forward(x):
    x = x.matmul(W1) + b1
    x = x.relu()
    x = x.matmul(W2) + b2
    return x

def forward_gpu(x):
    x = x.matmul_gpu(W1) + b1
    x = x.relu()
    x = x.matmul_gpu(W2) + b2
    return x

learning_rate = 0.5
epochs = 100

def accuracy(y_pred, y_true_labels):
    predicted_labels = np.argmax(y_pred.data, axis=1)
    correct = np.sum(predicted_labels == y_true_labels)
    return correct / len(y_true_labels)

initial_pred = forward(Tensor(X_train))
initial_acc = accuracy(initial_pred, y_train)
print(f"Accuracy before training: {initial_acc * 100:.2f}%")

matmul_time = 0.0
add_time = 0.0
relu_time = 0.0
backward_time = 0.0

for epoch in range(epochs):
    x = Tensor(X_train)
    y_true = Tensor(y_train_onehot)

    t0 = time.time()
    h = x.matmul(W1)
    matmul_time += time.time() - t0

    t0 = time.time()
    h = h + b1
    add_time += time.time() - t0

    t0 = time.time()
    h = h.relu()
    relu_time += time.time() - t0

    t0 = time.time()
    h2 = h.matmul(W2)
    matmul_time += time.time() - t0

    t0 = time.time()
    y_pred = h2 + b2
    add_time += time.time() - t0

    diff = y_pred - y_true
    loss = (diff * diff).mean()

    W1.grad = np.zeros_like(W1.data)
    b1.grad = np.zeros_like(b1.data)
    W2.grad = np.zeros_like(W2.data)
    b2.grad = np.zeros_like(b2.data)

    t0 = time.time()
    loss.backward()
    backward_time += time.time() - t0

    W1.data -= learning_rate * W1.grad
    b1.data -= learning_rate * b1.grad
    W2.data -= learning_rate * b2.grad
    b2.data -= learning_rate * b2.grad

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.data:.4f}")


print("\n--- Starting GPU-accelerated training ---")

np.random.seed(42)
W1_gpu = Tensor(np.random.randn(784, 64) * 0.01)
b1_gpu = Tensor(np.zeros(64))
W2_gpu = Tensor(np.random.randn(64, 10) * 0.01)
b2_gpu = Tensor(np.zeros(10))

gpu_matmul_time = 0.0
gpu_total_time_start = time.time()

for epoch in range(epochs):
    x = Tensor(X_train)
    y_true = Tensor(y_train_onehot)

    t0 = time.time()
    h = x.matmul_gpu(W1_gpu)
    gpu_matmul_time += time.time() - t0

    h = h + b1_gpu
    h = h.relu()

    t0 = time.time()
    h2 = h.matmul_gpu(W2_gpu)
    gpu_matmul_time += time.time() - t0

    y_pred = h2 + b2_gpu

    diff = y_pred - y_true
    loss = (diff * diff).mean()

    W1_gpu.grad = np.zeros_like(W1_gpu.data)
    b1_gpu.grad = np.zeros_like(b1_gpu.data)
    W2_gpu.grad = np.zeros_like(W2_gpu.data)
    b2_gpu.grad = np.zeros_like(b2_gpu.data)

    loss.backward()

    gpu_learning_rate = 0.1
    W1_gpu.data -= gpu_learning_rate * W1_gpu.grad
    b1_gpu.data -= gpu_learning_rate * b1_gpu.grad
    W2_gpu.data -= gpu_learning_rate * W2_gpu.grad
    b2_gpu.data -= gpu_learning_rate * b2_gpu.grad

    print(f"GPU Epoch {epoch+1}/{epochs}, Loss: {loss.data:.4f}")

gpu_total_time = time.time() - gpu_total_time_start

final_pred = forward(Tensor(X_train))
final_acc = accuracy(final_pred, y_train)
print(f"Accuracy after training: {final_acc * 100:.2f}%")

print("\n--- Profiling Results ---")
print(f"Matmul total time: {matmul_time:.4f}s")
print(f"Add total time: {add_time:.4f}s")
print(f"ReLU total time: {relu_time:.4f}s")
print(f"Backward total time: {backward_time:.4f}s")

print("\n--- Final Comparison ---")
print(f"CPU-only matmul time: {matmul_time:.4f}s")
print(f"GPU-accelerated matmul time: {gpu_matmul_time:.4f}s")
print(f"GPU speedup on matmul: {matmul_time / gpu_matmul_time:.2f}x")

final_pred_gpu = forward_gpu(Tensor(X_train))
final_acc_gpu = accuracy(final_pred_gpu, y_train)
print(f"GPU-trained accuracy: {final_acc_gpu * 100:.2f}%")

print("\n--- Starting PyTorch (reference) training ---")

torch.manual_seed(42)
W1_torch = (torch.randn(784, 64, dtype=torch.float64) * 0.01).clone().detach().requires_grad_(True)
b1_torch = torch.zeros(64, dtype=torch.float64, requires_grad=True)
W2_torch = (torch.randn(64, 10, dtype=torch.float64) * 0.01).clone().detach().requires_grad_(True)
b2_torch = torch.zeros(10, dtype=torch.float64, requires_grad=True)

X_train_torch = torch.tensor(X_train, dtype=torch.float64)
y_train_torch = torch.tensor(y_train_onehot, dtype=torch.float64)

torch_matmul_time = 0.0
torch_total_time_start = time.time()

for epoch in range(epochs):
    t0 = time.time()
    h = X_train_torch @ W1_torch
    torch_matmul_time += time.time() - t0

    h = h + b1_torch
    h = torch.relu(h)

    t0 = time.time()
    h2 = h @ W2_torch
    torch_matmul_time += time.time() - t0

    y_pred = h2 + b2_torch

    loss = ((y_pred - y_train_torch) ** 2).mean()

    for param in [W1_torch, b1_torch, W2_torch, b2_torch]:
        if param.grad is not None:
            param.grad.zero_()

    loss.backward()

    with torch.no_grad():
        W1_torch -= learning_rate * W1_torch.grad
        b1_torch -= learning_rate * b1_torch.grad
        W2_torch -= learning_rate * W2_torch.grad
        b2_torch -= learning_rate * b2_torch.grad

    print(f"PyTorch Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

torch_total_time = time.time() - torch_total_time_start

with torch.no_grad():
    h = torch.relu(X_train_torch @ W1_torch + b1_torch)
    final_pred_torch = h @ W2_torch + b2_torch
    predicted_labels_torch = torch.argmax(final_pred_torch, dim=1).numpy()
    torch_acc = np.mean(predicted_labels_torch == y_train)

print(f"\nPyTorch-trained accuracy: {torch_acc * 100:.2f}%")

print("\n--- Three-Way Final Comparison ---")
print(f"Your engine (CPU) matmul time:  {matmul_time:.4f}s")
print(f"Your engine (GPU) matmul time:  {gpu_matmul_time:.4f}s")
print(f"Real PyTorch matmul time:       {torch_matmul_time:.4f}s")
print()
print(f"Your engine (CPU) accuracy: {final_acc * 100:.2f}%")
print(f"Your engine (GPU) accuracy: {final_acc_gpu * 100:.2f}%")
print(f"Real PyTorch accuracy:      {torch_acc * 100:.2f}%")