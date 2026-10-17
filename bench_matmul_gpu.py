"""
bench_matmul_gpu.py -- micro-benchmark that breaks one GPU matmul call into its stages
(cast, upload, kernel, copy back, cast back) to show where the time goes, and compares
against NumPy on the CPU and cuBLAS (PyTorch's `@`) on the GPU.

Sizes match the two layers of the MNIST network at 10,000 images.
Each row is the mean of 50 calls after one warm-up call. Laptop timings are noisy (power,
heat, background GPU use), so compare rows within one run rather than across runs.
"""
import time                  # perf_counter: a high-resolution stopwatch.
import numpy as np           # CPU arrays and the CPU matmul baseline.
import torch                 # GPU tensors, synchronize(), and cuBLAS (via @). Must be imported before matmul_cuda_ext.
import matmul_cuda_ext       # My compiled CUDA kernel.

def sync():                  # GPU calls return before the GPU is finished; synchronize forces a wait so timings are real.
    torch.cuda.synchronize()

def timeit(fn, reps=50):     # Average time of fn() in milliseconds.
    fn(); sync()                      # warm-up call, not timed (first CUDA call is slow because of start-up)
    t0 = time.perf_counter()          # Start the clock.
    for _ in range(reps):             # Run the function `reps` times back to back...
        fn()
    sync()                            # ...and wait for the GPU to finish all of them before stopping the clock.
    return (time.perf_counter() - t0) / reps * 1000   # milliseconds per call

def full_path(A, B):         # Exactly what Tensor.matmul_gpu does for non-cached inputs, end to end.
    At = torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32)).cuda()   # Cast to contiguous float32 and upload A.
    Bt = torch.from_numpy(np.ascontiguousarray(B, dtype=np.float32)).cuda()   # Same for B.
    C = matmul_cuda_ext.matmul_cuda(At, Bt)                                   # Run my kernel.
    return C.cpu().numpy().astype(np.float64)                                 # Copy back to the CPU and cast to float64.

rng = np.random.default_rng(0)       # Seeded random generator so every run uses the same numbers.
for name, (M, K, N) in {"layer 1": (10000, 784, 64), "layer 2": (10000, 64, 10)}.items():   # Test both layer shapes of the network.
    A = rng.random((M, K))                      # float64 input, like Tensor.data.
    B = rng.standard_normal((K, N)) * 0.01      # float64 weights, small like the real initial weights.
    A32 = A.astype(np.float32)                  # Pre-cast copies, so the pure upload step can be timed on its own.
    B32 = B.astype(np.float32)
    At = torch.from_numpy(A32).cuda()           # A already on the GPU (used by the kernel-only and cached rows).
    Bt = torch.from_numpy(B32).cuda()           # B already on the GPU.
    C = matmul_cuda_ext.matmul_cuda(At, Bt); sync()   # One result on the GPU, used by the copy-back row.
    C_np = C.cpu().numpy()                      # The same result on the CPU, used by the cast-back row.

    def cached_path(B):                         # What the training loop does for layer 1 with a static input.
        # A (the big input) is already on the GPU; only B is converted and uploaded.
        Bt_ = torch.from_numpy(np.ascontiguousarray(B, dtype=np.float32)).cuda()   # Convert and upload only the weights.
        C_ = matmul_cuda_ext.matmul_cuda(At, Bt_)                                  # Kernel with the cached A.
        return C_.cpu().numpy().astype(np.float64)                                 # Copy back and cast.

    rows = [                                    # (label, function to time). The indented a-e rows are the stages of the FULL path.
        ("NumPy matmul on CPU (float64)",      lambda: A @ B),
        ("FULL GPU path (what matmul_gpu does)", lambda: full_path(A, B)),
        ("FULL GPU path, A cached on GPU",    lambda: cached_path(B)),
        ("  a. cast float64 -> float32",       lambda: np.ascontiguousarray(A, dtype=np.float32)),
        ("  b. copy A and B to GPU",           lambda: (torch.from_numpy(A32).cuda(), torch.from_numpy(B32).cuda())),
        ("  c. kernel only (data already on GPU)", lambda: matmul_cuda_ext.matmul_cuda(At, Bt)),
        ("  d. copy result back to CPU",       lambda: C.cpu().numpy()),
        ("  e. cast result float32 -> float64", lambda: C_np.astype(np.float64)),
        ("cuBLAS only (data already on GPU)",  lambda: At @ Bt),
    ]
    print(f"\n{name}: {M}x{K} @ {K}x{N}")      # Header for this layer's results.
    for label, fn in rows:                      # Time and print every row.
        print(f"  {label:<40} {timeit(fn):8.3f} ms")
