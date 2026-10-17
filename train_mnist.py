"""
train_mnist.py -- trains the same 2-layer neural network on MNIST three different ways
and compares them:

  1. My engine, CPU   -- tensor.py with NumPy matmul (float64).
  2. My engine, GPU   -- tensor.py with my custom CUDA matmul kernel (float32).
  3. PyTorch          -- the professional library, as the reference answer.

Network: 784 pixels -> 64 hidden (ReLU) -> 10 outputs, trained with mean-squared-error
loss and plain full-batch gradient descent for 100 epochs.

What is compared: matmul time, total training time, and final accuracy. Only the forward
matmuls run on the GPU in path 2; add, relu and the whole backward pass still run on the CPU.
Timings on my laptop vary a lot from run to run, so treat single runs as indicative.
"""
import numpy as np                     # NumPy: arrays, random numbers, and all CPU math.
import torch                           # PyTorch: used only for path 3, the reference training run.
from tensor import Tensor              # My own hand-built autograd Tensor class (tensor.py).
from sklearn.datasets import fetch_openml   # Downloads the real MNIST dataset.
import time                            # Stopwatch functions, for timing each part of training.

print("Loading MNIST... (this may take a minute the first time)")   # Status message so it doesn't look frozen.
mnist = fetch_openml('mnist_784', version=1, as_frame=False)        # Downloads (or loads from cache) MNIST: 70,000 handwritten digits as 784 pixel values each.
X, y = mnist.data, mnist.target.astype(int)                         # X = pixel data for every image; y = the correct digit labels, converted from text to integers.

# Use a subset so a full training run finishes in seconds instead of minutes.
X_train = X[:2000] / 255.0             # First 2000 images; dividing by 255 rescales pixel brightness from 0-255 down to 0-1.
y_train = y[:2000]                     # The matching 2000 correct labels.

def one_hot(labels, num_classes=10):   # Converts a label like 7 into [0,0,0,0,0,0,0,1,0,0] so it can be compared with the network's 10 outputs.
    out = np.zeros((len(labels), num_classes))       # A grid of zeros: one row per image, one column per possible digit (0-9).
    out[np.arange(len(labels)), labels] = 1           # In each row, put a 1 in the column of the correct digit.
    return out                                        # Return the one-hot encoded labels.

y_train_onehot = one_hot(y_train)      # Convert all training labels to one-hot form (used only inside the loss).

np.random.seed(42)                     # Fix NumPy's randomness so every run starts from the same weights and results are reproducible.

W1 = Tensor(np.random.randn(784, 64) * 0.01)   # Layer-1 weights: 784 pixels -> 64 hidden values. Small random start (a tiny version of a bell curve).
b1 = Tensor(np.zeros(64))                       # Layer-1 bias: one number per hidden neuron, starts at zero.
W2 = Tensor(np.random.randn(64, 10) * 0.01)     # Layer-2 weights: 64 hidden values -> 10 digit scores.
b2 = Tensor(np.zeros(10))                       # Layer-2 bias: one number per digit, starts at zero.

def forward(x):                        # CPU forward pass: takes image data, returns the network's 10 scores per image. Uses the CPU weights W1, b1, W2, b2.
    x = x.matmul(W1) + b1              # Multiply the images by layer-1 weights (NumPy) and add the bias.
    x = x.relu()                       # ReLU: replace every negative value with 0.
    x = x.matmul(W2) + b2              # Multiply by layer-2 weights and add the bias; this is the final guess.
    return x                           # Return the guess.

def forward_gpu(x, w1, bb1, w2, bb2):  # Same network, but the matmuls use my CUDA kernel. The weights are passed in as arguments
    x = x.matmul_gpu(w1) + bb1         # so the caller must choose which weights to use. (An earlier version silently used the CPU-trained
    x = x.relu()                       # weights here, so the GPU run was graded on the wrong network -- see notes/bugs_fixed.md.)
    x = x.matmul_gpu(w2) + bb2         # Layer 2 on the GPU, plus bias.
    return x                           # Return the guess.

learning_rate = 0.5                    # Step size for gradient descent. Used by all three training loops so the comparison is fair.
epochs = 100                           # Number of full passes over the training data.

def accuracy(y_pred, y_true_labels):   # Fraction of images where the network's top-scoring digit equals the true digit.
    predicted_labels = np.argmax(y_pred.data, axis=1)   # For each image, the index (0-9) of the highest of the 10 scores is the predicted digit.
    correct = np.sum(predicted_labels == y_true_labels) # Count the predictions that match the true labels.
    return correct / len(y_true_labels)                 # Correct count divided by total = accuracy between 0 and 1.

# Diagnostic prints from the layout bug hunt: X_train came out column-major (F-contiguous), which the CUDA kernel misread.
print("X_train C-contiguous:", X_train.flags['C_CONTIGUOUS'], "| F-contiguous:", X_train.flags['F_CONTIGUOUS'])
print("Tensor(X_train).data C-contiguous:", Tensor(X_train).data.flags['C_CONTIGUOUS'])

initial_pred = forward(Tensor(X_train))         # Forward pass BEFORE any training, with the random starting weights.
initial_acc = accuracy(initial_pred, y_train)   # Should be near 10%, i.e. random guessing among 10 digits.
print(f"Accuracy before training: {initial_acc * 100:.2f}%")   # Print the baseline.

# ---------------------------------------------------------------------------------------------
# PATH 1: my engine on the CPU
# ---------------------------------------------------------------------------------------------
matmul_time = 0.0                      # Running total of time spent in matmuls (CPU path).
add_time = 0.0                         # Running total of time spent in additions (CPU path).
relu_time = 0.0                        # Running total of time spent in ReLU (CPU path).
backward_time = 0.0                    # Running total of time spent in loss.backward() (CPU path).

x = Tensor(X_train)                    # The 2,000 images as one Tensor, shape (2000, 784). Built ONCE, outside the loop, so the CPU
                                       # path isn't charged for rebuilding it every epoch (that unfairness once skewed the timings).
y_true = Tensor(y_train_onehot)        # The correct answers as one Tensor, shape (2000, 10). Only used inside the loss.
cpu_total_time_start = time.time()     # Start the stopwatch for the whole CPU training loop.

for epoch in range(epochs):            # CPU training loop: one iteration per epoch.

    t0 = time.time()                   # Start a stopwatch.
    h = x.matmul(W1)                   # Layer-1 multiplication: all images times W1 (CPU).
    matmul_time += time.time() - t0    # Add the elapsed time to the matmul total.

    t0 = time.time()                   # Start a stopwatch.
    h = h + b1                         # Add the layer-1 bias.
    add_time += time.time() - t0       # Add the elapsed time to the add total.

    t0 = time.time()                   # Start a stopwatch.
    h = h.relu()                       # Apply ReLU.
    relu_time += time.time() - t0      # Add the elapsed time to the ReLU total.

    t0 = time.time()                   # Start a stopwatch.
    h2 = h.matmul(W2)                  # Layer-2 multiplication: hidden values times W2 (CPU).
    matmul_time += time.time() - t0    # Add the elapsed time to the matmul total.

    t0 = time.time()                   # Start a stopwatch.
    y_pred = h2 + b2                   # Add the layer-2 bias; this is the network's guess for this epoch.
    add_time += time.time() - t0       # Add the elapsed time to the add total.

    diff = y_pred - y_true             # Error of each of the 10 outputs for each image.
    loss = (diff * diff).mean()        # Square the errors (no negatives, big errors count more), then average into one number: the loss.

    W1.grad = np.zeros_like(W1.data)   # Reset gradients to zero, because backward() adds into them with +=
    b1.grad = np.zeros_like(b1.data)   # and old gradients from the previous epoch would otherwise pile up.
    W2.grad = np.zeros_like(W2.data)   # (Same reset for W2...)
    b2.grad = np.zeros_like(b2.data)   # (...and b2.)

    t0 = time.time()                   # Start a stopwatch.
    loss.backward()                    # My hand-written backpropagation: fills in .grad for every weight.
    backward_time += time.time() - t0  # Add the elapsed time to the backward total.

    W1.data -= learning_rate * W1.grad # Gradient descent: move W1 a small step against its gradient (this is the "learning").
    b1.data -= learning_rate * b1.grad # Same update for b1.
    W2.data -= learning_rate * W2.grad # Same update for W2. (A typo here once used b2.grad, which capped CPU accuracy at 67%.)
    b2.data -= learning_rate * b2.grad # Same update for b2.

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.data:.4f}")   # Progress report.

cpu_total_time = time.time() - cpu_total_time_start   # Total wall-clock time of the CPU training loop.


# ---------------------------------------------------------------------------------------------
# PATH 2: my engine with the custom CUDA matmul
# ---------------------------------------------------------------------------------------------
print("\n--- Starting GPU-accelerated training ---")   # Marks the start of the GPU run in the output.

np.random.seed(42)                     # Same seed as before, so the GPU run starts from EXACTLY the same initial weights as the CPU run.
W1_gpu = Tensor(np.random.randn(784, 64) * 0.01)   # Fresh layer-1 weights for the GPU run.
b1_gpu = Tensor(np.zeros(64))                       # Fresh layer-1 bias.
W2_gpu = Tensor(np.random.randn(64, 10) * 0.01)     # Fresh layer-2 weights.
b2_gpu = Tensor(np.zeros(10))                       # Fresh layer-2 bias.

def one_step_check(use_gpu):           # Diagnostic: run ONE forward+backward pass on either CPU or GPU matmul, from identical weights, and return what came out.
    for p in [W1_gpu, b1_gpu, W2_gpu, b2_gpu]:     # Zero all gradients first so the two calls don't contaminate each other.
        p.grad = np.zeros_like(p.data)
    x = Tensor(X_train)                # Fresh input (local to this function; shadows the global x).
    y = Tensor(y_train_onehot)         # Fresh labels.
    mm = (lambda a, b: a.matmul_gpu(b)) if use_gpu else (lambda a, b: a.matmul(b))   # Pick which matmul to use.
    h = mm(x, W1_gpu) + b1_gpu         # Layer 1.
    h = h.relu()                       # ReLU.
    pred = mm(h, W2_gpu) + b2_gpu      # Layer 2.
    d = pred - y                       # Error.
    loss = (d * d).mean()              # Mean-squared-error loss.
    loss.backward()                    # Backward pass fills the gradients.
    return float(loss.data), pred.data.copy(), W1_gpu.grad.copy(), W2_gpu.grad.copy()   # Copies, so the next call can't overwrite them.

loss_c, pred_c, g1_c, g2_c = one_step_check(False)   # One step using the CPU matmul.
loss_g, pred_g, g1_g, g2_g = one_step_check(True)    # One step using the GPU matmul.

print("CHECK loss  CPU vs GPU:", loss_c, loss_g)     # The two losses should agree to many digits.
print("CHECK pred max diff:   ", np.abs(pred_c - pred_g).max())   # Largest difference in predictions; should be tiny (float32 rounding).
print("CHECK W1 grad max diff:", np.abs(g1_c - g1_g).max(), " (CPU grad size:", np.abs(g1_c).max(), ")")   # Gradient difference vs the gradient's own size.
print("CHECK W2 grad max diff:", np.abs(g2_c - g2_g).max(), " (CPU grad size:", np.abs(g2_c).max(), ")")   # Same for W2.

x_static = Tensor(X_train)             # The input images as a Tensor for the GPU loop.
x_static.static = True                 # Promise that these values never change, so tensor.py uploads them to the GPU once and reuses the copy.

_ = x_static.matmul_gpu(W1_gpu)        # Warm-up call: the first CUDA call is slow (driver start-up) and uploads X; do it before the timer starts.

y_true = Tensor(y_train_onehot)        # The correct answers again (rebuilt; same values as before).

gpu_matmul_time = 0.0                  # Running total of time spent in GPU matmul calls (includes conversion, upload, kernel, copy-back).
gpu_total_time_start = time.time()     # Start the stopwatch for the whole GPU training loop.

for epoch in range(epochs):            # GPU training loop: same structure as the CPU loop.
    x = x_static                        # Use the cached-on-GPU input tensor.

    t0 = time.time()                   # Start a stopwatch.
    h = x.matmul_gpu(W1_gpu)           # Layer 1 with my CUDA kernel (X is already on the GPU; only W1 is uploaded).
    gpu_matmul_time += time.time() - t0   # Add the elapsed time to the GPU matmul total.

    h = h + b1_gpu                     # Add layer-1 bias (CPU).
    h = h.relu()                       # ReLU (CPU).

    t0 = time.time()                   # Start a stopwatch.
    h2 = h.matmul_gpu(W2_gpu)          # Layer 2 with my CUDA kernel (the hidden values change each epoch, so they are uploaded each time).
    gpu_matmul_time += time.time() - t0   # Add the elapsed time to the GPU matmul total.

    y_pred = h2 + b2_gpu               # Add layer-2 bias; the network's guess.

    diff = y_pred - y_true             # Error against the correct answers.
    loss = (diff * diff).mean()        # Same loss formula as the CPU loop.

    W1_gpu.grad = np.zeros_like(W1_gpu.data)   # Reset gradients before the backward pass (same reason as in the CPU loop).
    b1_gpu.grad = np.zeros_like(b1_gpu.data)   # (b1)
    W2_gpu.grad = np.zeros_like(W2_gpu.data)   # (W2)
    b2_gpu.grad = np.zeros_like(b2_gpu.data)   # (b2)

    loss.backward()                    # Backward pass. It always runs on the CPU, even in this "GPU" loop; it is most of the epoch time.

    gpu_learning_rate = learning_rate            # Same learning rate as the CPU loop (an earlier version used 0.1 here, which made the comparison unfair).
    W1_gpu.data -= gpu_learning_rate * W1_gpu.grad   # Gradient-descent update for W1_gpu.
    b1_gpu.data -= gpu_learning_rate * b1_gpu.grad   # Update b1_gpu.
    W2_gpu.data -= gpu_learning_rate * W2_gpu.grad   # Update W2_gpu.
    b2_gpu.data -= gpu_learning_rate * b2_gpu.grad   # Update b2_gpu.

    print(f"GPU Epoch {epoch+1}/{epochs}, Loss: {loss.data:.4f}")   # Progress report.

gpu_total_time = time.time() - gpu_total_time_start   # Total wall-clock time of the GPU training loop.

final_pred = forward(Tensor(X_train))           # Forward pass with the CPU-TRAINED weights (W1, b1, W2, b2) to measure the CPU path's final accuracy.
final_acc = accuracy(final_pred, y_train)       # CPU path accuracy after 100 epochs.
print(f"Accuracy after training: {final_acc * 100:.2f}%")   # Print it.

print("\n--- Profiling Results ---")            # Header: where the CPU path spends its time.
print(f"Matmul total time: {matmul_time:.4f}s") # Total CPU matmul time over all epochs.
print(f"Add total time: {add_time:.4f}s")       # Total CPU add time.
print(f"ReLU total time: {relu_time:.4f}s")     # Total CPU ReLU time.
print(f"Backward total time: {backward_time:.4f}s")   # Total CPU backward time.

final_pred_gpu = forward_gpu(Tensor(X_train), W1_gpu, b1_gpu, W2_gpu, b2_gpu)   # NOTE: this line is repeated further below (line "Run a forward pass using the GPU-trained weights") and is redundant. Left as-is so the code's behaviour is unchanged.

print("\n--- Final Comparison ---")             # Header: CPU vs GPU matmul.
print(f"CPU-only matmul time: {matmul_time:.4f}s")            # CPU matmul total again, for side-by-side reading.
print(f"GPU-accelerated matmul time: {gpu_matmul_time:.4f}s") # GPU matmul total (includes conversion and transfer overhead).
print(f"GPU speedup on matmul: {matmul_time / gpu_matmul_time:.2f}x")   # CPU time / GPU time. Below 1.00x means the GPU path is SLOWER.

final_pred_gpu = forward_gpu(Tensor(X_train), W1_gpu, b1_gpu, W2_gpu, b2_gpu)   # Forward pass with the GPU-TRAINED weights, through the GPU forward function.
final_acc_gpu = accuracy(final_pred_gpu, y_train)   # GPU path accuracy after 100 epochs.
print(f"GPU-trained accuracy: {final_acc_gpu * 100:.2f}%")   # Print it.

# ---------------------------------------------------------------------------------------------
# PATH 3: plain PyTorch, the reference
# ---------------------------------------------------------------------------------------------
print("\n--- Starting PyTorch (reference) training ---")   # Marks the start of the PyTorch run.

torch.manual_seed(42)                  # Seed PyTorch's own random generator (it can't produce the same numbers as NumPy, so its starting weights differ slightly).
W1_torch = (torch.randn(784, 64, dtype=torch.float64) * 0.01).clone().detach().requires_grad_(True)   # Layer-1 weights. clone().detach() makes it a fresh leaf tensor, then requires_grad_ turns on gradient tracking.
b1_torch = torch.zeros(64, dtype=torch.float64, requires_grad=True)    # Layer-1 bias, tracked for gradients.
W2_torch = (torch.randn(64, 10, dtype=torch.float64) * 0.01).clone().detach().requires_grad_(True)     # Layer-2 weights (same leaf-tensor trick).
b2_torch = torch.zeros(10, dtype=torch.float64, requires_grad=True)    # Layer-2 bias, tracked for gradients.

X_train_torch = torch.tensor(X_train, dtype=torch.float64)         # The images as a PyTorch tensor (float64, to match my CPU engine).
y_train_torch = torch.tensor(y_train_onehot, dtype=torch.float64)  # The one-hot labels as a PyTorch tensor.

torch_matmul_time = 0.0                # Running total of PyTorch matmul time.
torch_total_time_start = time.time()   # Start the stopwatch for the whole PyTorch loop.

for epoch in range(epochs):            # PyTorch training loop: same 100 epochs, same data, same math.
    t0 = time.time()                   # Start a stopwatch.
    h = X_train_torch @ W1_torch       # Layer-1 multiplication with PyTorch's @ operator.
    torch_matmul_time += time.time() - t0   # Add the elapsed time.

    h = h + b1_torch                   # Add layer-1 bias.
    h = torch.relu(h)                  # PyTorch's own ReLU.

    t0 = time.time()                   # Start a stopwatch.
    h2 = h @ W2_torch                  # Layer-2 multiplication.
    torch_matmul_time += time.time() - t0   # Add the elapsed time.

    y_pred = h2 + b2_torch             # Add layer-2 bias; the guess.

    loss = ((y_pred - y_train_torch) ** 2).mean()   # Same mean-squared-error loss, written with PyTorch operations.

    for param in [W1_torch, b1_torch, W2_torch, b2_torch]:   # PyTorch also accumulates gradients, so reset them each epoch.
        if param.grad is not None:                            # On the first epoch there is no gradient yet.
            param.grad.zero_()                                # Reset in place to zero.

    loss.backward()                    # PyTorch's automatic backward pass: all gradients in one call.

    with torch.no_grad():              # Updating weights must not itself be recorded as part of the computation graph.
        W1_torch -= learning_rate * W1_torch.grad   # Gradient-descent update for W1.
        b1_torch -= learning_rate * b1_torch.grad   # Update b1.
        W2_torch -= learning_rate * W2_torch.grad   # Update W2.
        b2_torch -= learning_rate * b2_torch.grad   # Update b2.

    print(f"PyTorch Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")   # Progress; .item() turns a one-element tensor into a plain number.

torch_total_time = time.time() - torch_total_time_start   # Total wall-clock time of the PyTorch loop.

with torch.no_grad():                  # Evaluation only: no gradient tracking needed.
    h = torch.relu(X_train_torch @ W1_torch + b1_torch)         # Forward pass, layer 1 + ReLU, with the final trained weights.
    final_pred_torch = h @ W2_torch + b2_torch                  # Layer 2: the final 10 scores per image.
    predicted_labels_torch = torch.argmax(final_pred_torch, dim=1).numpy()   # Highest-scoring digit per image, converted to NumPy.
    torch_acc = np.mean(predicted_labels_torch == y_train)      # Fraction of predictions that match the true labels.

print(f"\nPyTorch-trained accuracy: {torch_acc * 100:.2f}%")   # Print PyTorch's final accuracy.

# ---------------------------------------------------------------------------------------------
# Summary of all three
# ---------------------------------------------------------------------------------------------
print("\n--- Three-Way Final Comparison ---")   # Header for the side-by-side summary.
print(f"Your engine (CPU) matmul time:  {matmul_time:.4f}s")   # CPU engine matmul time.
print(f"Your engine (GPU) matmul time:  {gpu_matmul_time:.4f}s")   # GPU engine matmul time.
print(f"Real PyTorch matmul time:       {torch_matmul_time:.4f}s")   # PyTorch matmul time.
print()                                # Blank line for readability.
print(f"Your engine (CPU) accuracy: {final_acc * 100:.2f}%")       # CPU engine accuracy.
print(f"Your engine (GPU) accuracy: {final_acc_gpu * 100:.2f}%")   # GPU engine accuracy.
print(f"Real PyTorch accuracy:      {torch_acc * 100:.2f}%")       # PyTorch accuracy.
print(f"CPU engine total training time: {cpu_total_time:.4f}s")    # Whole-loop time, CPU engine.
print(f"GPU engine total training time: {gpu_total_time:.4f}s")    # Whole-loop time, GPU engine.
print(f"PyTorch total training time: {torch_total_time:.4f}s")     # Whole-loop time, PyTorch.
