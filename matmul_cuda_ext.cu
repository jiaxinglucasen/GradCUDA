// matmul_cuda_ext.cu -- my custom CUDA matrix-multiply kernel, wrapped as a PyTorch extension
// so Python (tensor.py) can call it as matmul_cuda_ext.matmul_cuda(A, B).
//
// This is a "naive" kernel: every thread computes ONE element of the output and reads
// everything it needs straight from slow global GPU memory. Correct and simple, but about
// 5x slower than NVIDIA's cuBLAS at 10000x784 @ 784x64. Shared-memory tiling is the planned fix.
//
// A and B are contiguous, row-major, float32 tensors on the GPU.
// tensor.py guarantees this with np.ascontiguousarray(..., dtype=np.float32).

#include <torch/extension.h>   // PyTorch's C++ API: torch::Tensor and the pybind11 bindings.
#include <cuda.h>              // CUDA driver API declarations.
#include <cuda_runtime.h>      // CUDA runtime API (dim3, threadIdx, blockIdx, ...).

// GPU kernel: each thread computes one element of the output matrix, all threads in parallel.
// Computes C (M x N) = A (M x K) @ B (K x N). Matrices are stored row-major in flat 1-D arrays.
__global__ void matmulKernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // Which output row this thread owns: its block's row offset plus its position inside the block.
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Which output column this thread owns, computed the same way.

    if (row < M && col < N) {                          // The grid is rounded UP to whole 16x16 blocks, so some threads fall outside the matrix; they must do nothing.
        float sum = 0.0f;                              // Running dot product, kept in a fast register.
        for (int i = 0; i < K; i++) {                  // Walk along row `row` of A and down column `col` of B.
            sum += A[row * K + i] * B[i * N + col];    // Row-major indexing: element (r, c) of an R x C matrix lives at r*C + c.
        }
        C[row * N + col] = sum;                        // Write the finished dot product into the output.
    }
}

// Host-side wrapper: works out the sizes, allocates the output, and launches matmulKernel on the GPU.
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);                                 // Rows of A (and of the output).
    int K = A.size(1);                                 // Columns of A = rows of B: the length of each dot product.
    int N = B.size(1);                                 // Columns of B (and of the output).

    auto C = torch::zeros({M, N}, A.options());        // Output matrix on the same GPU with the same dtype as A, filled with zeros.

    dim3 threadsPerBlock(16, 16);                      // Each block is a 16 x 16 square of threads (256 threads).
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);      // Enough blocks to cover the whole output; "+15 then /16" rounds up.

    matmulKernel<<<numBlocks, threadsPerBlock>>>(      // Launch the kernel on the GPU (asynchronous: returns before it finishes).
        A.data_ptr<float>(),                           // Raw pointer to A's float data in GPU memory.
        B.data_ptr<float>(),                           // Raw pointer to B's data.
        C.data_ptr<float>(),                           // Raw pointer to the output buffer.
        M, N, K                                        // The dimensions.
    );

    return C;                                          // Hand the result back to Python as a real PyTorch tensor (still on the GPU).
}

// Registers matmul_cuda so it's callable from Python as matmul_cuda_ext.matmul_cuda(...).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {             // TORCH_EXTENSION_NAME is filled in by the build (setup.py) with "matmul_cuda_ext".
    m.def("matmul_cuda", &matmul_cuda, "Matrix multiplication (CUDA)");   // Python name, C++ function, docstring.
}
