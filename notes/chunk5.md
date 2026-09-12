# Chunk 5: CUDA Kernel for Matrix Multiplication

## What I built
- matmul_kernel.cu: a standalone CUDA program implementing matrix
  multiplication directly on the GPU
- __global__ matmulKernel: each GPU thread computes exactly ONE
  output element, using row/col calculated from threadIdx/blockIdx/
  blockDim (rather than one CPU core looping through everything)
- main(): host-side code that allocates GPU memory (cudaMalloc),
  copies data from CPU to GPU (cudaMemcpy HostToDevice), launches
  the kernel, copies the result back (cudaMemcpy DeviceToHost),
  and frees GPU memory (cudaFree)
- Added CUDA_CHECK macro to catch and print any CUDA error instead
  of silently continuing — critical for actually debugging this

## Environment setup issues (significant, real debugging practice)

### Issue 1: nvcc not found in PATH
- CUDA Toolkit folder didn't exist where expected — had to
  reinstall the CUDA Toolkit entirely
- After reinstalling, confirmed with: nvcc --version

### Issue 2: nvcc couldn't find cl.exe (Microsoft's C++ compiler)
- nvcc needs Microsoft's C++ compiler for part of the build process
- Regular terminals don't have this compiler set up in PATH
- Fix: must use "x64 Native Tools Command Prompt for VS" specifically
  — this special terminal runs vcvarsall.bat automatically, which
  sets up the correct environment for cl.exe to be found
- Must cd into the project folder each time a NEW terminal window
  is opened (this environment doesn't persist between sessions)

### Issue 3: Silent wrong output (0.0 0.0 / 0.0 0.0), no error shown
- Root cause: had no error-checking at all — CUDA calls can fail
  silently without any indication unless you explicitly check
- Fix: added CUDA_CHECK macro wrapping every CUDA API call, plus
  cudaGetLastError() and cudaDeviceSynchronize() after the kernel
  launch specifically -- these catch launch failures and execution
  failures that basic code would otherwise miss entirely

### Issue 4 (the real bug): "provided PTX was compiled with an
unsupported toolchain"
- Root cause: CUDA Toolkit version (13.4, quite new) was ahead of
  what my installed GPU driver supported -- toolkit and driver
  versions need to be compatible with each other
- Fix: updated NVIDIA GPU driver (GeForce Game Ready Driver, not
  Studio Driver -- Game Ready gets newer CUDA compatibility updates
  faster) via nvidia.com/drivers, then restarted the computer
- After driver update: cudaGetDeviceCount correctly reported 1
  device found, and the kernel executed correctly

## Key concept: why this bug was worth hitting
- Toolkit/driver version mismatches are a very common real-world
  CUDA development problem -- not a beginner mistake, a genuine
  environment issue every CUDA developer runs into
- Proper debugging process: don't guess -- add explicit error
  checking, read the SPECIFIC error message, understand what it
  actually means, then fix the root cause (driver update), not
  just the symptom

## Verified result
- Number of CUDA devices found: 1
- Kernel output: [[19.0, 22.0], [43.0, 50.0]]
- Matches the exact hand-verified result from Chunk 1's matmul
  test -- confirms the GPU kernel computes the mathematically
  correct answer, not just "runs without crashing"

## Milestone
- Chunk 5 complete: successfully wrote, compiled, and ran a real
  CUDA kernel performing matrix multiplication on the GPU, with
  correct, verified output. Solved a genuine environment/toolchain
  debugging problem along the way -- directly relevant experience
  for the target NVIDIA role, which explicitly values GPU
  programming and performance-oriented parallel programming skills.