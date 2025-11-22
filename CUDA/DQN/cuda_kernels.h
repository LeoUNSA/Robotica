#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Matrix operations
void cuda_matmul(const float* A, const float* B, float* C, int m, int n, int k, bool use_shared);
void cuda_add(const float* A, const float* B, float* C, int size);
void cuda_add_bias(const float* input, const float* bias, float* output, int rows, int cols);
void cuda_transpose(const float* A, float* B, int m, int n);

// Activation functions
void cuda_relu(const float* input, float* output, int size);
void cuda_relu_derivative(const float* input, float* output, int size);

// Element-wise operations
void cuda_elementwise_mul(const float* A, const float* B, float* C, int size);
void cuda_scalar_mul(const float* input, float scalar, float* output, int size);

// Initialization
void cuda_xavier_init(float* data, int size, float scale, unsigned long long seed);
void cuda_zero_init(float* data, int size);

// Utility functions
void cuda_copy(const float* src, float* dst, int size);
void cuda_soft_update(const float* source, float* target, float tau, int size);

// Loss functions
void cuda_mse_loss(const float* pred, const float* target, float* loss, float* grad, int size);

// Gradient operations
void cuda_sum_rows(const float* input, float* output, int rows, int cols);
void cuda_clip_gradient(float* grad, float max_norm, int size);

// Optimizer
void cuda_adam_update(float* param, float* grad, float* m, float* v,
                      float lr, float beta1, float beta2, float epsilon,
                      int t, int size);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H
