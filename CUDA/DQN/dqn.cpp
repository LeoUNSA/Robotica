#include "dqn.h"
#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ==================== Layer Implementation ====================

Layer::Layer(int in_size, int out_size) 
    : input_size(in_size), output_size(out_size), cached_batch_size(0) {
    
    // Allocate device memory for weights and biases
    CUDA_CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, output_size * sizeof(float)));
    
    // Allocate memory for gradients
    CUDA_CHECK(cudaMalloc(&d_weight_grad, input_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_grad, output_size * sizeof(float)));
    
    // Allocate memory for Adam optimizer states
    CUDA_CHECK(cudaMalloc(&d_weight_m, input_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight_v, input_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_m, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_v, output_size * sizeof(float)));
    
    // Initialize optimizer states to zero
    cuda_zero_init(d_weight_m, input_size * output_size);
    cuda_zero_init(d_weight_v, input_size * output_size);
    cuda_zero_init(d_bias_m, output_size);
    cuda_zero_init(d_bias_v, output_size);
    
    d_output = nullptr;
    d_input_cache = nullptr;
}

Layer::~Layer() {
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_weight_grad);
    cudaFree(d_bias_grad);
    cudaFree(d_weight_m);
    cudaFree(d_weight_v);
    cudaFree(d_bias_m);
    cudaFree(d_bias_v);
    if (d_output) cudaFree(d_output);
    if (d_input_cache) cudaFree(d_input_cache);
}

void Layer::init_weights(unsigned long long seed) {
    // Xavier initialization
    float scale = sqrtf(6.0f / (input_size + output_size));
    cuda_xavier_init(d_weights, input_size * output_size, scale, seed);
    cuda_zero_init(d_bias, output_size);
}

void Layer::forward(const float* d_input, int batch_size) {
    // Reallocate output buffer if batch size changed
    if (!d_output || cached_batch_size != batch_size) {
        // Synchronize before freeing
        cudaDeviceSynchronize();
        
        if (d_output) {
            cudaFree(d_output);
            d_output = nullptr;
        }
        if (d_input_cache) {
            cudaFree(d_input_cache);
            d_input_cache = nullptr;
        }
        
        CUDA_CHECK(cudaMalloc(&d_output, batch_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_input_cache, batch_size * input_size * sizeof(float)));
        cached_batch_size = batch_size;
    }
    
    // Cache input for backward pass
    CUDA_CHECK(cudaMemcpy(d_input_cache, d_input, 
                          batch_size * input_size * sizeof(float), 
                          cudaMemcpyDeviceToDevice));
    
    // Matrix multiplication: output = input * weights^T
    // input: batch_size x input_size
    // weights: output_size x input_size (stored as input_size x output_size, need transpose)
    float* d_weights_T;
    CUDA_CHECK(cudaMalloc(&d_weights_T, input_size * output_size * sizeof(float)));
    cuda_transpose(d_weights, d_weights_T, input_size, output_size);
    
    cuda_matmul(d_input, d_weights_T, d_output, batch_size, output_size, input_size, true);
    cudaFree(d_weights_T);
    
    // Add bias
    cuda_add_bias(d_output, d_bias, d_output, batch_size, output_size);
}

void Layer::backward(const float* d_grad_output, float* d_grad_input, int batch_size) {
    // Compute gradient w.r.t. input: grad_input = grad_output * weights
    cuda_matmul(d_grad_output, d_weights, d_grad_input, batch_size, input_size, output_size, true);
    
    // Compute gradient w.r.t. weights: weight_grad = input^T * grad_output
    float* d_input_T;
    CUDA_CHECK(cudaMalloc(&d_input_T, batch_size * input_size * sizeof(float)));
    cuda_transpose(d_input_cache, d_input_T, batch_size, input_size);
    
    cuda_matmul(d_input_T, d_grad_output, d_weight_grad, input_size, output_size, batch_size, true);
    cudaFree(d_input_T);
    
    // Compute gradient w.r.t. bias: bias_grad = sum(grad_output, axis=0)
    cuda_sum_rows(d_grad_output, d_bias_grad, batch_size, output_size);
}

void Layer::update_weights(float lr, float beta1, float beta2, float epsilon, int t) {
    // Update weights using Adam optimizer
    cuda_adam_update(d_weights, d_weight_grad, d_weight_m, d_weight_v,
                     lr, beta1, beta2, epsilon, t, input_size * output_size);
    
    cuda_adam_update(d_bias, d_bias_grad, d_bias_m, d_bias_v,
                     lr, beta1, beta2, epsilon, t, output_size);
}

// ==================== DQN Implementation ====================

DQN::DQN(int state_dim, int action_dim, const std::vector<int>& hidden_dims) 
    : state_size(state_dim), action_size(action_dim) {
    
    // Build network architecture
    int prev_size = state_dim;
    unsigned long long seed = 12345;
    
    for (int hidden_size : hidden_dims) {
        Layer* layer = new Layer(prev_size, hidden_size);
        layer->init_weights(seed++);
        layers.push_back(layer);
        prev_size = hidden_size;
    }
    
    // Output layer
    Layer* output_layer = new Layer(prev_size, action_dim);
    output_layer->init_weights(seed);
    layers.push_back(output_layer);
    
    // Initialize temporary buffers to nullptr
    for (int i = 0; i < 10; i++) {
        d_layer_outputs[i] = nullptr;
        d_relu_masks[i] = nullptr;
    }
}

DQN::~DQN() {
    for (Layer* layer : layers) {
        delete layer;
    }
    
    for (int i = 0; i < 10; i++) {
        if (d_layer_outputs[i]) cudaFree(d_layer_outputs[i]);
        if (d_relu_masks[i]) cudaFree(d_relu_masks[i]);
    }
}

void DQN::forward(const float* d_state, float* d_q_values, int batch_size) {
    const float* current_input = d_state;
    
    // Forward through all layers except the last
    for (size_t i = 0; i < layers.size() - 1; i++) {
        layers[i]->forward(current_input, batch_size);
        
        // Allocate buffer for ReLU output
        int output_size = layers[i]->output_size;
        if (!d_layer_outputs[i]) {
            CUDA_CHECK(cudaMalloc(&d_layer_outputs[i], batch_size * output_size * sizeof(float)));
        }
        if (!d_relu_masks[i]) {
            CUDA_CHECK(cudaMalloc(&d_relu_masks[i], batch_size * output_size * sizeof(float)));
        }
        
        // Apply ReLU activation
        cuda_relu(layers[i]->d_output, d_layer_outputs[i], batch_size * output_size);
        
        // Store ReLU mask for backward pass
        cuda_relu_derivative(layers[i]->d_output, d_relu_masks[i], batch_size * output_size);
        
        current_input = d_layer_outputs[i];
    }
    
    // Forward through output layer (no activation)
    layers.back()->forward(current_input, batch_size);
    
    // Copy output to q_values
    CUDA_CHECK(cudaMemcpy(d_q_values, layers.back()->d_output,
                          batch_size * action_size * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

void DQN::backward(const float* d_grad, int batch_size) {
    // Allocate temporary buffer for gradients
    float* d_grad_current;
    float* d_grad_next;
    
    int max_size = 0;
    for (Layer* layer : layers) {
        max_size = std::max(max_size, layer->input_size * batch_size);
    }
    
    CUDA_CHECK(cudaMalloc(&d_grad_current, max_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_next, max_size * sizeof(float)));
    
    // Start with gradient from loss
    CUDA_CHECK(cudaMemcpy(d_grad_current, d_grad,
                          batch_size * action_size * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    
    // Backward through layers in reverse order
    for (int i = layers.size() - 1; i >= 0; i--) {
        // If not the last layer, apply ReLU derivative
        if (i < (int)layers.size() - 1) {
            cuda_elementwise_mul(d_grad_current, d_relu_masks[i],
                               d_grad_current, batch_size * layers[i]->output_size);
        }
        
        // Backward through layer
        layers[i]->backward(d_grad_current, d_grad_next, batch_size);
        
        // Swap buffers
        float* temp = d_grad_current;
        d_grad_current = d_grad_next;
        d_grad_next = temp;
    }
    
    cudaFree(d_grad_current);
    cudaFree(d_grad_next);
}

void DQN::update(float lr, float beta1, float beta2, float epsilon, int t) {
    for (Layer* layer : layers) {
        layer->update_weights(lr, beta1, beta2, epsilon, t);
    }
}

void DQN::copy_weights_from(DQN* source) {
    for (size_t i = 0; i < layers.size(); i++) {
        Layer* src_layer = source->layers[i];
        Layer* dst_layer = layers[i];
        
        cuda_copy(src_layer->d_weights, dst_layer->d_weights,
                 src_layer->input_size * src_layer->output_size);
        cuda_copy(src_layer->d_bias, dst_layer->d_bias, src_layer->output_size);
    }
}

void DQN::soft_update_from(DQN* source, float tau) {
    for (size_t i = 0; i < layers.size(); i++) {
        Layer* src_layer = source->layers[i];
        Layer* dst_layer = layers[i];
        
        cuda_soft_update(src_layer->d_weights, dst_layer->d_weights, tau,
                        src_layer->input_size * src_layer->output_size);
        cuda_soft_update(src_layer->d_bias, dst_layer->d_bias, tau,
                        src_layer->output_size);
    }
}

void DQN::save_weights(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    for (Layer* layer : layers) {
        int w_size = layer->input_size * layer->output_size;
        int b_size = layer->output_size;
        
        float* h_weights = new float[w_size];
        float* h_bias = new float[b_size];
        
        CUDA_CHECK(cudaMemcpy(h_weights, layer->d_weights, w_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_bias, layer->d_bias, b_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        file.write(reinterpret_cast<char*>(h_weights), w_size * sizeof(float));
        file.write(reinterpret_cast<char*>(h_bias), b_size * sizeof(float));
        
        delete[] h_weights;
        delete[] h_bias;
    }
    
    file.close();
}

void DQN::load_weights(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return;
    }
    
    for (Layer* layer : layers) {
        int w_size = layer->input_size * layer->output_size;
        int b_size = layer->output_size;
        
        float* h_weights = new float[w_size];
        float* h_bias = new float[b_size];
        
        file.read(reinterpret_cast<char*>(h_weights), w_size * sizeof(float));
        file.read(reinterpret_cast<char*>(h_bias), b_size * sizeof(float));
        
        CUDA_CHECK(cudaMemcpy(layer->d_weights, h_weights, w_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layer->d_bias, h_bias, b_size * sizeof(float), cudaMemcpyHostToDevice));
        
        delete[] h_weights;
        delete[] h_bias;
    }
    
    file.close();
}

// ==================== Replay Buffer Implementation ====================

ReplayBuffer::ReplayBuffer(int cap) : capacity(cap), position(0) {
    std::random_device rd;
    rng.seed(rd());
}

void ReplayBuffer::add(const std::vector<float>& state, int action, float reward,
                       const std::vector<float>& next_state, bool done) {
    Experience exp = {state, action, reward, next_state, done};
    
    if (buffer.size() < (size_t)capacity) {
        buffer.push_back(exp);
    } else {
        buffer[position] = exp;
    }
    
    position = (position + 1) % capacity;
}

std::vector<Experience> ReplayBuffer::sample(int batch_size) {
    std::vector<Experience> samples;
    std::uniform_int_distribution<int> dist(0, buffer.size() - 1);
    
    for (int i = 0; i < batch_size; i++) {
        int idx = dist(rng);
        samples.push_back(buffer[idx]);
    }
    
    return samples;
}

// ==================== DQN Agent Implementation ====================

DQNAgent::DQNAgent(int state_dim, int action_dim, const std::vector<int>& hidden_dims,
                   int buffer_capacity, float gamma_val, float epsilon_start,
                   float epsilon_end, float epsilon_decay_val, float tau_val)
    : state_size(state_dim), action_size(action_dim), gamma(gamma_val),
      epsilon(epsilon_start), epsilon_min(epsilon_end), epsilon_decay(epsilon_decay_val),
      tau(tau_val), update_step(0), max_batch_size(256) {
    
    // Create networks
    policy_net = new DQN(state_dim, action_dim, hidden_dims);
    target_net = new DQN(state_dim, action_dim, hidden_dims);
    target_net->copy_weights_from(policy_net);
    
    // Create replay buffer
    replay_buffer = new ReplayBuffer(buffer_capacity);
    
    // Initialize random number generator
    std::random_device rd;
    rng.seed(rd());
    
    // Allocate device memory for batch processing
    CUDA_CHECK(cudaMalloc(&d_states, max_batch_size * state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_states, max_batch_size * state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_values, max_batch_size * action_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_q_values, max_batch_size * action_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target_q_values, max_batch_size * action_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, max_batch_size * action_size * sizeof(float)));
}

DQNAgent::~DQNAgent() {
    delete policy_net;
    delete target_net;
    delete replay_buffer;
    
    cudaFree(d_states);
    cudaFree(d_next_states);
    cudaFree(d_q_values);
    cudaFree(d_next_q_values);
    cudaFree(d_target_q_values);
    cudaFree(d_grad);
}

int DQNAgent::select_action(const std::vector<float>& state, bool training) {
    if (training) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng) < epsilon) {
            // Random action
            std::uniform_int_distribution<int> action_dist(0, action_size - 1);
            return action_dist(rng);
        }
    }
    
    // Greedy action
    float* d_state;
    float* d_q_vals;
    CUDA_CHECK(cudaMalloc(&d_state, state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_vals, action_size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_state, state.data(), state_size * sizeof(float), cudaMemcpyHostToDevice));
    
    policy_net->forward(d_state, d_q_vals, 1);
    
    std::vector<float> q_values(action_size);
    CUDA_CHECK(cudaMemcpy(q_values.data(), d_q_vals, action_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_state);
    cudaFree(d_q_vals);
    
    return std::max_element(q_values.begin(), q_values.end()) - q_values.begin();
}

void DQNAgent::store_experience(const std::vector<float>& state, int action, float reward,
                                const std::vector<float>& next_state, bool done) {
    replay_buffer->add(state, action, reward, next_state, done);
}

float DQNAgent::train_step(int batch_size, float lr, float beta1, float beta2, float epsilon_adam) {
    if (!replay_buffer->can_sample(batch_size)) {
        return 0.0f;
    }
    
    update_step++;
    
    // Sample from replay buffer
    auto experiences = replay_buffer->sample(batch_size);
    
    // Prepare batch data
    std::vector<float> h_states(batch_size * state_size);
    std::vector<float> h_next_states(batch_size * state_size);
    std::vector<int> actions(batch_size);
    std::vector<float> rewards(batch_size);
    std::vector<bool> dones(batch_size);
    
    for (int i = 0; i < batch_size; i++) {
        std::copy(experiences[i].state.begin(), experiences[i].state.end(),
                 h_states.begin() + i * state_size);
        std::copy(experiences[i].next_state.begin(), experiences[i].next_state.end(),
                 h_next_states.begin() + i * state_size);
        actions[i] = experiences[i].action;
        rewards[i] = experiences[i].reward;
        dones[i] = experiences[i].done;
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_states, h_states.data(), batch_size * state_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next_states, h_next_states.data(), batch_size * state_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Forward pass through policy network
    policy_net->forward(d_states, d_q_values, batch_size);
    
    // Forward pass through target network
    target_net->forward(d_next_states, d_next_q_values, batch_size);
    
    // Compute target Q-values on host (for simplicity)
    std::vector<float> h_q_values(batch_size * action_size);
    std::vector<float> h_next_q_values(batch_size * action_size);
    std::vector<float> h_target_q_values(batch_size * action_size);
    
    CUDA_CHECK(cudaMemcpy(h_q_values.data(), d_q_values, batch_size * action_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_next_q_values.data(), d_next_q_values, batch_size * action_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_loss = 0.0f;
    
    for (int i = 0; i < batch_size; i++) {
        // Copy current Q-values to target
        std::copy(h_q_values.begin() + i * action_size,
                 h_q_values.begin() + (i + 1) * action_size,
                 h_target_q_values.begin() + i * action_size);
        
        // Compute target for the action taken
        float target;
        if (dones[i]) {
            target = rewards[i];
        } else {
            float max_next_q = *std::max_element(
                h_next_q_values.begin() + i * action_size,
                h_next_q_values.begin() + (i + 1) * action_size);
            target = rewards[i] + gamma * max_next_q;
        }
        
        // Update only the Q-value for the action taken
        int action = actions[i];
        float old_q = h_target_q_values[i * action_size + action];
        h_target_q_values[i * action_size + action] = target;
        
        total_loss += (target - old_q) * (target - old_q);
    }
    
    // Copy target Q-values to device
    CUDA_CHECK(cudaMemcpy(d_target_q_values, h_target_q_values.data(),
                          batch_size * action_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compute gradient (MSE loss gradient)
    cuda_mse_loss(d_q_values, d_target_q_values, nullptr, d_grad, batch_size * action_size);
    
    // Clip gradients for stability
    cuda_clip_gradient(d_grad, 1.0f, batch_size * action_size);
    
    // Backward pass
    policy_net->backward(d_grad, batch_size);
    
    // Clip gradients in each layer for extra stability
    for (Layer* layer : policy_net->layers) {
        cuda_clip_gradient(layer->d_weight_grad, 1.0f, layer->input_size * layer->output_size);
        cuda_clip_gradient(layer->d_bias_grad, 1.0f, layer->output_size);
    }
    
    // Update weights
    policy_net->update(lr, beta1, beta2, epsilon_adam, update_step);
    
    return total_loss / batch_size;
}

void DQNAgent::update_target_network() {
    target_net->soft_update_from(policy_net, tau);
}

void DQNAgent::decay_epsilon() {
    epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
}

void DQNAgent::save(const char* filename) {
    policy_net->save_weights(filename);
}

void DQNAgent::load(const char* filename) {
    policy_net->load_weights(filename);
    target_net->copy_weights_from(policy_net);
}
