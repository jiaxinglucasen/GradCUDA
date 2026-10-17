# GradCUDA

An autograd engine (in the spirit of Andrej Karpathy's micrograd) built from scratch in
Python using NumPy. Created CUDA kernel performing matrix multiplication to compare the accuracy and speed of calculation run on CPU and GPU. They are benchmarked head-to-head against PyTorch itself.

The goal was to understand the mathematic calculation underneath a deep learning framework by building it, measuring it, and finding out why my engines are slower than the PyTorch.

## Contents

- [What it does](#what-it-does)
- [Results](#results)
- [Why the GPU version is still slower than the CPU version](#why-the-gpu-version-is-still-slower-than-the-cpu-version)
- [Bugs I fixed](#bugs-i-fixed)
- [Project structure](#project-structure)
- [How to run](#how-to-run)
- [Future work](#future-work)

## What it does

- A `Tensor` class supporting `+`, `-`, `*`, `**`, `mean`, `matmul` and `relu`, each remembering how it was computed.
- Reverse-mode automatic differentiation: `backward()` does a topological sort over the computation graph and applies the chain rule, including un-broadcasting gradients.
- Trains a 2-layer MLP (784 -> 64 -> 10, ReLU, mean-squared-error loss, full-batch gradient descent, 100 epochs) on MNIST.
- A custom CUDA matmul kernel (one thread per output element), wrapped with pybind11 as a PyTorch extension, and plugged into the engine as `Tensor.matmul_gpu`.
- A caching mechanism that uploads constant inputs (the MNIST images) to the GPU once.
- Benchmark: my engine on CPU, my engine with the CUDA kernel, and PyTorch.

## Results

Same network, data, initial weights (for my two engine runs), learning rate (0.5) and 100 epochs in every column.
Timings are from one laptop run each and vary by 2x or more between runs (power state, heat, background GPU use),
so read them as indicative, not exact. Accuracy is measured on the same images each network trained on, not on unseen test images.

**10,000 training images**

| Metric | My engine (CPU) | My engine (GPU kernel) | PyTorch (CPU, float64) |
|---|---|---|---|
| Matmul time (100 epochs) | 0.72 s | 1.95 s | 0.24 s |
| Total training time | 5.56 s | 6.56 s | 0.60 s |
| Accuracy on the training images | 74.31% | 74.31% | 74.94% |

**2,000 training images**

| Metric | My engine (CPU) | My engine (GPU kernel) | PyTorch (CPU, float64) |
|---|---|---|---|
| Matmul time (100 epochs) | 0.17 s | 0.45 s | 0.06 s |
| Total training time | 1.30 s | 1.58 s | 0.20 s |
| Accuracy on the training images | 75.40% | 75.40% | 75.35% |

What this shows:

- **Correctness.** The GPU path reproduces the CPU path's accuracy exactly (74.31% / 75.40%), and a one-step check shows CPU and GPU losses, predictions and gradients agreeing to around 1e-8 or better. Both engine paths land within about a point of PyTorch; PyTorch uses a different random initialisation, so exact equality isn't expected.
- **Speed.** My GPU path is still slower than my CPU path, and PyTorch is faster than both. The next section explains why.

### Where the time goes for one layer-1 matmul (10000x784 @ 784x64)

From `bench_matmul_gpu.py`, one run, mean of 50 calls:

| Step | Time |
|---|---|
| NumPy matmul on CPU | 3.1 ms |
| Full GPU path (convert, upload, kernel, copy back, convert) | 42.2 ms |
| Full GPU path with the input cached on the GPU | 12.9 ms |
| &nbsp;&nbsp;cast float64 -> float32 (uncached only) | 10.7 ms |
| &nbsp;&nbsp;upload to GPU (uncached only) | 17.9 ms |
| &nbsp;&nbsp;**my kernel alone** | **10.2 ms** |
| &nbsp;&nbsp;copy result back | 1.5 ms |
| cuBLAS (PyTorch's matmul) alone | 2.1 ms |

## Why the GPU version is still slower than the CPU version

1. **My kernel is naive.** Every thread reads its inputs straight from slow global GPU memory, so the same numbers are fetched over and over. At this size it takes about 10 ms, versus about 2 ms for NVIDIA's cuBLAS.
2. **Even cuBLAS barely beats the CPU here.** The calculated matrices are small by GPU standards, so the best case (using PyTorch) is only around 1.5x faster than NumPy (2.1 ms vs 3.1 ms). A perfect kernel would not turn this into a big win by itself.
3. **Data still moves between CPU and GPU.** Caching the images removed the redundant conversion cost (see bug 6), but the hidden-layer values feeding layer 2 are produced on the CPU and must be uploaded every epoch, and every result is copied back.
4. **Everything else is still on the CPU.** In this project, only matmul function is performed in GPU and compared. Add, ReLU and the entire backward pass run in NumPy. In the CPU profile, `backward()` is roughly 73% of epoch time, and the GPU version runs backward function in CPU as well.

The takeaway: offloading one operation of a mostly-CPU pipeline doesn't pay off. Real frameworks keep tensors on the GPU from start to finish.

## Bugs I fixed

Full write-up with symptoms, causes and fixes in [`notes/bugs_fixed.md`](notes/bugs_fixed.md). The major ones:

1. **GPU accuracy stuck at ~12% (chance level).** The CUDA kernel assumes row-major memory, but the MNIST array was column-major. NumPy hides this with strides; a raw-pointer kernel doesn't. Fixed with `np.ascontiguousarray`. My earlier random-matrix tests couldn't catch it because random arrays are always row-major.
2. **GPU network graded with the wrong weights.** `forward_gpu` silently used the CPU-trained weights.
3. **Learning-rate mismatch** between the CPU and GPU loops (0.5 vs 0.1).
4. **One-word typo (`b2.grad` instead of `W2.grad`)** capped the CPU engine at 67% until fixed.
5. **Unfair timing.** The CPU loop rebuilt its input tensors every epoch while the GPU loop didn't.
6. **Repeated conversion and upload of constant data**, plus CUDA start-up counted in the timer. Fixed with a `static` flag, a GPU cache and a warm-up call.
7. **Broadcasting in the backward pass.** Bias gradients had to be summed back to the bias's shape (`unbroadcast`).
8. **PyTorch non-leaf tensor** in the reference run.
9. **Windows build chain:** `DISTUTILS_USE_SDK`, CUDA/PyTorch version match, MSVC preprocessor flag, and DLL import order.

## Project structure

- `tensor.py` -- the Tensor class and autograd engine (CPU matmul and `matmul_gpu`).
- `train_mnist.py` -- MNIST training for all three paths (CPU, GPU kernel, PyTorch), with profiling and comparison output.
- `matmul_cuda_ext.cu` -- the CUDA kernel wrapped as a PyTorch extension.
- `matmul_kernel.cu` -- the first standalone 2x2 CUDA test (early verification).
- `setup.py` -- build configuration for the extension.
- `bench_matmul_gpu.py` -- per-stage micro-benchmark of one GPU matmul.
- `notes/` -- the original build log (chunks 1-7), plus `bugs_fixed.md` and setup notes. A correction note at the top of `chunk7.md` gives the real cause of the accuracy bug (see bug 1).

## How to run

Developed on Windows 11 with an NVIDIA laptop GPU, Python 3.13, a CUDA 13.x toolkit, Visual Studio Build Tools (Desktop development with C++), and PyTorch built for the matching CUDA version (`cu130` index).

```bat
:: 1. In an "x64 Native Tools Command Prompt for VS", from the project folder, with the venv active:
set DISTUTILS_USE_SDK=1
python setup.py build_ext --inplace

:: 2. Make sure the CUDA bin folder is on PATH, then train and compare all three paths:
python train_mnist.py

:: 3. Optional: per-stage matmul micro-benchmark, and the Tensor self-tests
python bench_matmul_gpu.py
python tensor.py
```

`train_mnist.py` downloads MNIST through scikit-learn on first run (needs `scikit-learn` and `pandas`). `import torch` must come before `import matmul_cuda_ext`.

## Future work

- **Shared-memory tiling** for the matmul kernel, to stop re-reading global memory; compare against cuBLAS.
- **Keep tensors on the GPU** between operations so hidden activations never round-trip through the CPU.
- **GPU add, ReLU and backward kernels**, so the whole training step runs on the GPU. This is where the real speedup is, since backward dominates.
- **Skip unneeded gradients** (a `requires_grad`-style flag) so the input images don't get a gradient computed.
- **Error checking and input validation** in the extension (`TORCH_CHECK` for dtype/device, `.contiguous()` on inputs, `cudaGetLastError`), so a layout bug like bug 1 fails loudly instead of silently.
- **Switch to cross-entropy loss with softmax.** The network currently uses mean-squared-error loss, which works but learns slowly for classification. Cross-entropy with softmax is the standard choice for MNIST and should push accuracy well above the current ~75%.
- **Better benchmarking:** best-of-N timing, larger batches, and mini-batch training.

