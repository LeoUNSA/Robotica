#include "cuda_kernels.h"
#include <stdio.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Matrix multiplication kernel: C = A * B
// A: m x k, B: k x n, C: m x n
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Optimized matrix multiplication with shared memory
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C,
                                     int m, int n, int k) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (k + 31) / 32; tile++) {
        // Load tiles into shared memory
        if (row < m && (tile * 32 + threadIdx.x) < k)
            As[threadIdx.y][threadIdx.x] = A[row * k + tile * 32 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < n && (tile * 32 + threadIdx.y) < k)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        for (int i = 0; i < 32; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// Element-wise addition: C = A + B
__global__ void add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// Add bias to each row: out[i,j] = in[i,j] + bias[j]
__global__ void add_bias_kernel(const float* input, const float* bias, float* output,
                                int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] = input[idx] + bias[col];
    }
}

// ReLU activation: out = max(0, in)
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU derivative: out = (in > 0) ? 1 : 0
__global__ void relu_derivative_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Element-wise multiplication: C = A * B
__global__ void elementwise_mul_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

// Scalar multiplication: out = scalar * in
__global__ void scalar_mul_kernel(const float* input, float scalar, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * scalar;
    }
}

// Xavier initialization for weights
__global__ void xavier_init_kernel(float* data, int size, float scale, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * scale;
    }
}

// Initialize bias to zero
__global__ void zero_init_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

// Copy network parameters
__global__ void copy_kernel(const float* src, float* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// Soft update: target = tau * source + (1 - tau) * target
__global__ void soft_update_kernel(const float* source, float* target, float tau, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        target[idx] = tau * source[idx] + (1.0f - tau) * target[idx];
    }
}

// MSE loss and gradient: loss = 0.5 * (pred - target)^2, grad = (pred - target)
__global__ void mse_loss_kernel(const float* pred, const float* target, 
                                float* loss, float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        if (loss) loss[idx] = 0.5f * diff * diff;
        if (grad) grad[idx] = diff;
    }
}

// Transpose matrix: B = A^T
// A: m x n, B: n x m
__global__ void transpose_kernel(const float* A, float* B, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        B[col * m + row] = A[row * n + col];
    }
}

// Sum reduction along rows for bias gradient
__global__ void sum_rows_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

// Gradient clipping
__global__ void clip_gradient_kernel(float* grad, float max_norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = grad[idx];
        if (val > max_norm) grad[idx] = max_norm;
        else if (val < -max_norm) grad[idx] = -max_norm;
    }
}

// Adam optimizer update
__global__ void adam_update_kernel(float* param, float* grad, float* m, float* v,
                                   float lr, float beta1, float beta2, float epsilon,
                                   int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];
        
        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        
        // Update parameters
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// Host functions
void cuda_matmul(const float* A, const float* B, float* C, int m, int n, int k, bool use_shared) {
    if (use_shared) {
        dim3 block(32, 32);
        dim3 grid((n + 31) / 32, (m + 31) / 32);
        matmul_shared_kernel<<<grid, block>>>(A, B, C, m, n, k);
    } else {
        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (m + 15) / 16);
        matmul_kernel<<<grid, block>>>(A, B, C, m, n, k);
    }
    CUDA_CHECK(cudaGetLastError());
}

void cuda_add(const float* A, const float* B, float* C, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<<<blocks, BLOCK_SIZE>>>(A, B, C, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_add_bias(const float* input, const float* bias, float* output, int rows, int cols) {
    int size = rows * cols;
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bias_kernel<<<blocks, BLOCK_SIZE>>>(input, bias, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_relu(const float* input, float* output, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_relu_derivative(const float* input, float* output, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_derivative_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_elementwise_mul(const float* A, const float* B, float* C, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elementwise_mul_kernel<<<blocks, BLOCK_SIZE>>>(A, B, C, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_scalar_mul(const float* input, float scalar, float* output, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scalar_mul_kernel<<<blocks, BLOCK_SIZE>>>(input, scalar, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_xavier_init(float* data, int size, float scale, unsigned long long seed) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    xavier_init_kernel<<<blocks, BLOCK_SIZE>>>(data, size, scale, seed);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_zero_init(float* data, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zero_init_kernel<<<blocks, BLOCK_SIZE>>>(data, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_copy(const float* src, float* dst, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    copy_kernel<<<blocks, BLOCK_SIZE>>>(src, dst, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_soft_update(const float* source, float* target, float tau, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    soft_update_kernel<<<blocks, BLOCK_SIZE>>>(source, target, tau, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_mse_loss(const float* pred, const float* target, float* loss, float* grad, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mse_loss_kernel<<<blocks, BLOCK_SIZE>>>(pred, target, loss, grad, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_transpose(const float* A, float* B, int m, int n) {
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    transpose_kernel<<<grid, block>>>(A, B, m, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_sum_rows(const float* input, float* output, int rows, int cols) {
    int blocks = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sum_rows_kernel<<<blocks, BLOCK_SIZE>>>(input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_clip_gradient(float* grad, float max_norm, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_gradient_kernel<<<blocks, BLOCK_SIZE>>>(grad, max_norm, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_adam_update(float* param, float* grad, float* m, float* v,
                      float lr, float beta1, float beta2, float epsilon,
                      int t, int size) {
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adam_update_kernel<<<blocks, BLOCK_SIZE>>>(param, grad, m, v, lr, beta1, beta2, epsilon, t, size);
    CUDA_CHECK(cudaGetLastError());
}
