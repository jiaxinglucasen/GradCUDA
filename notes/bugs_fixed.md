## 1. The GPU run scored ~12% accuracy (memory layout bug) -- the big one

**Symptom.** The GPU version of the network trained to about 11-12% accuracy, which is chance level for
10 digits. The loss looked like it was going down, but the predictions were useless.

**Cause.** The CUDA kernel reads a matrix as one flat list of numbers in *row-major* order (row 0, then
row 1, ...). The MNIST array `X_train` was stored in memory *column-major* (column 0 first, then column
1, ...). NumPy doesn't care, because it tracks "strides" and finds the right element either way. My
kernel just reads raw memory, so it saw the pixels scrambled. `np.array(...)` keeps the input's layout,
so the column-major layout survived all the way to the GPU.

**How I found it.** I printed `flags['C_CONTIGUOUS']` and
`flags['F_CONTIGUOUS']` on `X_train`: C was False, F was True. My earlier correctness tests used
`np.random.randn` matrices, which are always row-major, so they could never have caught this.

**Fix.** In `tensor.py`, `_to_cuda` now calls `np.ascontiguousarray(self.data, dtype=np.float32)`, which
forces row-major layout. After that the GPU accuracy matched the CPU exactly (75.40%) and a one-step
CPU-vs-GPU check agreed to about 1e-8.

---

## 2. The GPU network was being graded with the wrong weights

**Symptom.** Accuracy numbers for the GPU path didn't match what training suggested.

**Cause.** `forward_gpu(x)` used the global `W1, W2` (the CPU-trained weights) instead of `W1_gpu, W2_gpu`.
So the "GPU accuracy" was really measuring the wrong network.

**Fix.** `forward_gpu(x, w1, bb1, w2, bb2)` now takes the weights as arguments, and the caller passes the
`_gpu` weights. (The TypeError I hit along the way was the call site still using the old signature.)

---

## 3. The GPU loop used a different learning rate (0.1 vs 0.5)

**Symptom / cause.** The GPU training loop had its own smaller learning rate, so the two runs weren't
doing the same experiment.

**Fix.** `gpu_learning_rate = learning_rate`, so both loops use 0.5.

---

## 4. A one-word typo capped the CPU accuracy at 67%

**Symptom.** CPU accuracy was 67.15% while PyTorch got about 75%.

**Cause.** The update line read `W2.data -= learning_rate * b2.grad` (b2's gradient applied to W2) instead of
`W2.grad`. NumPy broadcasting let the mismatched shapes through without an error, so the code ran, just wrongly.

**Fix.** Use `W2.grad`. CPU accuracy rose to 75.40%.

---

## 5. The timing comparison was unfair

**Symptom.** The GPU loop's *total* time looked about 33% faster than the CPU's even though its matmul
time was slower, which made no sense.

**Cause.** The CPU loop rebuilt `Tensor(X_train)` and `Tensor(y_train_onehot)` every epoch, while the GPU
loop reused one prepared tensor. The CPU was being charged for setup work the GPU wasn't.

**Fix.** Build `x` and `y_true` once before the CPU loop. Total times then agreed with matmul times.

---

## 6. The GPU matmul was slow because of repeated conversion and uploads (and start-up cost in the timer)

**Symptom.** The GPU matmul took over 1.5 s in a 2,000-image run, far slower than the CPU.

**Cause.** Every call converted the big input from float64 to float32 and uploaded it again, even though
the input images never change. Separately, the very first CUDA call includes driver start-up, which was
being counted in the timer.

**Fix.** A `static` flag plus a `_cuda` cache: a constant Tensor like the images is converted and uploaded
once, then reused. I also added a warm-up call before the timer starts. GPU matmul time dropped to about 0.45 s.

---

## 7. Broadcasting broke the backward pass

**Symptom.** `ValueError: non-broadcastable output operand` during `backward()`.

**Cause.** In the forward pass, adding a bias of shape (64,) to a (2000, 64) matrix works because NumPy
stretches the bias. The gradient coming back is (2000, 64), but the bias's gradient slot is (64,).

**Fix.** `unbroadcast(grad, shape)` sums the gradient over the stretched dimensions to get back to the
original shape. Real frameworks have to do the same thing.

---

## 8. PyTorch reference: "non-leaf tensor" gradients

**Symptom.** In the PyTorch reference run, the weight-update step crashed with a TypeError because `.grad` was `None`.

**Cause.** I had written `torch.randn(..., requires_grad=True) * 0.01`. The multiply makes the result a new,
non-"leaf" tensor, and PyTorch only stores `.grad` on leaf tensors.

**Fix.** `(...).clone().detach().requires_grad_(True)` makes a fresh leaf tensor that tracks gradients.

---

## 9. Getting the CUDA extension to build and import on Windows

Four separate problems, all environment related (details in `notes/chunk6.md`):

1. **`DISTUTILS_USE_SDK` warning** -- the build refused to continue inside the Visual Studio prompt until I set `DISTUTILS_USE_SDK=1`.
2. **CUDA / PyTorch version mismatch** -- PyTorch built for CUDA 12.4 vs a CUDA 13.x toolkit. Building an extension needs them to match, so I reinstalled PyTorch from the `cu130` index.
3. **MSVC preprocessor error** -- CUDA 13 headers need `/Zc:preprocessor`. Putting it under `cxx` did nothing for `.cu` files; it had to go under `nvcc` as `-Xcompiler /Zc:preprocessor`.
4. **`ImportError: DLL load failed`** -- PyTorch's DLLs are only findable after `import torch`, so `torch` must be imported before `matmul_cuda_ext`.

---

## 10. Smaller ones

- Indentation: new methods not lined up inside the `Tensor` class gave `unsupported operand type(s) for -`.
- Missing `pandas` for `fetch_openml`.
- Self-tests in `tensor.py` ran on every import; they now sit under `if __name__ == "__main__":`.
