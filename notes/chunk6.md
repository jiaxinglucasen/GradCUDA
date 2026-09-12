# Chunk 6: Connecting the CUDA Kernel to PyTorch

## What I built
- matmul_cuda_ext.cu: same kernel logic as Chunk 5's standalone
  version, but wrapped as a PyTorch extension using torch::Tensor
  instead of manual cudaMalloc/cudaMemcpy
- torch::Tensor matmul_cuda(A, B): the bridge function PyTorch
  calls -- receives tensors already on the GPU, launches the kernel,
  returns a real PyTorch tensor as output
- PYBIND11_MODULE macro: exports matmul_cuda so Python can import
  and call it directly
- setup.py: build configuration using torch.utils.cpp_extension's
  CUDAExtension + BuildExtension, which handles invoking nvcc/cl.exe
  with all the correct PyTorch-specific compiler flags automatically

## Real debugging journey (significant, multi-stage)

### Issue 1: DISTUTILS_USE_SDK warning blocking the build
- PyTorch's build system detected the VC environment was already
  active (from the x64 Native Tools terminal) and refused to
  proceed without explicit confirmation
- Fix: set DISTUTILS_USE_SDK=1 in the terminal before building

### Issue 2: CUDA/PyTorch version mismatch
- Installed PyTorch was built for CUDA 12.4, but installed CUDA
  Toolkit was 13.4 -- building an extension (unlike just running
  PyTorch) requires exact version alignment
- Fix: uninstalled torch/torchvision/torchaudio, reinstalled using
  --index-url https://download.pytorch.org/whl/cu130 to match the
  actual installed CUDA Toolkit version

### Issue 3: MSVC traditional preprocessor error
- CUDA 13.4's headers refuse to compile unless cl.exe uses the
  newer "standard conforming preprocessor"
- First attempt: added extra_compile_args={'cxx': ['/Zc:preprocessor']}
  -- WRONG, this doesn't apply to .cu files compiled through nvcc
- Correct fix: extra_compile_args={'nvcc': ['-Xcompiler', '/Zc:preprocessor']}
  -- -Xcompiler is nvcc's own mechanism for passing flags through
  to the underlying C++ compiler, matching the pattern nvcc uses
  internally for all its other compiler flags

### Issue 4: ImportError - DLL load failed
- First guess (CUDA bin folder missing from PATH) was wrong
- Real cause: PyTorch's own DLLs (torch_python.dll, c10.dll, etc.)
  live in venv\Lib\site-packages\torch\lib, a folder only added to
  Windows' DLL search path when torch itself is imported first
- Fix: import torch BEFORE import matmul_cuda_ext, every time --
  this same order will need to be respected inside tensor.py once
  the extension is wired into the Tensor class

## Key concept: why "cxx" vs "nvcc" compiler args matter
- A CUDAExtension build has TWO compilers involved: nvcc (compiles
  .cu files, calling cl.exe internally via -Xcompiler flags) and
  cl.exe directly (for any plain .cpp files, if present)
- extra_compile_args={'cxx': [...], 'nvcc': [...]} lets you target
  each one separately -- flags meant for cl.exe still need to be
  wrapped as -Xcompiler when going through nvcc for a .cu file

## Verified result
- Import succeeded (with torch imported first)
- Full functional test through the real PyTorch extension pipeline:
  matmul_cuda_ext.matmul_cuda(a, b) on real CUDA tensors returned
  tensor([[19., 22.], [43., 50.]], device='cuda:0')
- Matches the hand-calculated and standalone-CUDA-verified result
  from earlier chunks -- correctness confirmed through three
  independent verification methods across the whole project

## Milestone
- Chunk 6 complete: successfully built, debugged, and verified a
  real PyTorch C++/CUDA extension from scratch, working through
  four distinct real-world build/environment issues along the way.
  This is genuinely representative of real CUDA extension
  development work, directly relevant to the "Deep Learning
  Frameworks and Libraries" role this whole project has been
  targeting from the start.

  ## Integration into tensor.py (final connection)

## What I added
- import torch and import matmul_cuda_ext at the top of tensor.py
  (torch imported first, per Chunk 6's DLL loading lesson)
- New method matmul_gpu(self, other) alongside the original
  NumPy-based matmul:
  - Converts self.data/other.data from float64 -> float32 (kernel
    requires float32), wraps as torch tensors, moves to GPU (.cuda())
  - Calls matmul_cuda_ext.matmul_cuda() -- the actual custom kernel
  - Converts result back: GPU -> CPU (.cpu()), torch -> numpy
    (.numpy()), float32 -> float64 (.astype)
  - Same _backward gradient logic as the original matmul (the
    backward MATH doesn't change based on which hardware ran the
    forward pass -- only the forward computation itself moved to GPU)

## Environment reminder for running (not just building)
- Running any script that imports matmul_cuda_ext still needs the
  CUDA bin folder in PATH for that terminal session, even though
  building is done -- this is because the compiled .pyd needs the
  CUDA runtime DLLs at import time, not just at compile time
- PowerShell syntax: $env:PATH = "C:\...\CUDA\v13.4\bin;" + $env:PATH
  (different from Command Prompt's `set PATH=...;%PATH%`)

## Verified result
- a.matmul(b) [CPU/NumPy]: [[19. 22.], [43. 50.]]
- a.matmul_gpu(b) [custom CUDA kernel via PyTorch extension]:
  [[19. 22.], [43. 50.]]
- Identical results confirmed -- the full pipeline (Tensor class ->
  CUDA kernel -> PyTorch extension bridge) is correct end to end

## Milestone
- The core technical goal of the entire project is now complete:
  a working, from-scratch autograd engine that can optionally
  execute its heaviest operation on the GPU using custom CUDA code,
  verified correct against the original CPU implementation.