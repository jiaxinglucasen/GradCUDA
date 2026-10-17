// matmul_kernel.cu -- the very first CUDA step of the project: a standalone program (no Python, no PyTorch)
// that multiplies two 2x2 matrices on the GPU and prints the result, to prove the kernel logic works.
// Expected output: [[19, 22], [43, 50]]. Build with: nvcc matmul_kernel.cu -o matmul_kernel
// The kernel was later copied into matmul_cuda_ext.cu, where it is wrapped for PyTorch.

#include <stdio.h>                                     // printf.

// Wraps any CUDA call: if it returns an error code, print the file, line and a readable message.
// Without this, CUDA failures are silent and you just get wrong numbers.
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    }

// GPU kernel: each thread computes one element of C (M x N) = A (M x K) @ B (K x N), all row-major.
__global__ void matmulKernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // Output row owned by this thread.
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Output column owned by this thread.

    if (row < M && col < N) {                          // Skip threads that fall outside the matrix.
        float sum = 0.0f;                              // Running dot product.
        for (int i = 0; i < K; i++) {                  // Walk along A's row and down B's column.
            sum += A[row * K + i] * B[i * N + col];    // Row-major indexing: (r, c) of an R x C matrix is at r*C + c.
        }
        C[row * N + col] = sum;                        // Store the result.
    }
}

int main() {
    int deviceCount = 0;                               // Will hold how many CUDA GPUs the program can see.
    cudaGetDeviceCount(&deviceCount);                  // Ask the CUDA runtime.
    printf("Number of CUDA devices found: %d\n", deviceCount);   // Sanity check: should be at least 1.

    int M = 2, N = 2, K = 2;                           // All three matrices are 2 x 2.
    size_t size = M * N * sizeof(float);               // Bytes per matrix: 4 floats. (Valid for all three only because they are all 2x2.)

    float h_A[4] = {1, 2, 3, 4};                       // Host (CPU) copy of A = [[1,2],[3,4]], flattened row-major.
    float h_B[4] = {5, 6, 7, 8};                       // Host copy of B = [[5,6],[7,8]].
    float h_C[4];                                      // Host buffer that will receive the result.

    float *d_A, *d_B, *d_C;                            // Device (GPU) pointers; "d_" marks memory that lives on the GPU.
    CUDA_CHECK(cudaMalloc(&d_A, size));                // Allocate GPU memory for A.
    CUDA_CHECK(cudaMalloc(&d_B, size));                // Allocate GPU memory for B.
    CUDA_CHECK(cudaMalloc(&d_C, size));                // Allocate GPU memory for the result.

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));   // Copy A from CPU to GPU.
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));   // Copy B from CPU to GPU.

    dim3 threadsPerBlock(2, 2);                        // One block of 2 x 2 threads: one thread per output element.
    dim3 numBlocks(1, 1);                              // A single block is enough for a 2 x 2 output.
    matmulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);   // Launch the kernel.

    CUDA_CHECK(cudaGetLastError());                    // Did the launch itself fail (bad configuration, etc.)?
    CUDA_CHECK(cudaDeviceSynchronize());               // Wait for the GPU to finish, and catch errors that happened while it ran.

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));   // Copy the result from GPU back to CPU.

    printf("Result:\n");                               // Print the result matrix row by row.
    for (int i = 0; i < M; i++) {                      // For each row...
        for (int j = 0; j < N; j++) {                  // ...and each column in it...
            printf("%.1f ", h_C[i * N + j]);           // ...print the element (row-major index).
        }
        printf("\n");                                  // New line at the end of each row.
    }

    cudaFree(d_A);                                     // Release the GPU memory.
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;                                          // Success.
}
