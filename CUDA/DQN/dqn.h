#ifndef DQN_H
#define DQN_H

#include <vector>
#include <random>

// Neural Network Layer
class Layer {
public:
    int input_size;
    int output_size;
    
    // Device pointers
    float* d_weights;
    float* d_bias;
    float* d_output;
    float* d_input_cache;  // Cache for backward pass
    
    // Gradient pointers
    float* d_weight_grad;
    float* d_bias_grad;
    
    // Adam optimizer states
    float* d_weight_m;
    float* d_weight_v;
    float* d_bias_m;
    float* d_bias_v;
    
    int cached_batch_size;
    
    Layer(int in_size, int out_size);
    ~Layer();
    
    void forward(const float* d_input, int batch_size);
    void backward(const float* d_grad_output, float* d_grad_input, int batch_size);
    void init_weights(unsigned long long seed);
    void update_weights(float lr, float beta1, float beta2, float epsilon, int t);
};

// Deep Q-Network
class DQN {
public:
    std::vector<Layer*> layers;
    int state_size;
    int action_size;
    
    // Temporary buffers
    float* d_layer_outputs[10];  // Max 10 layers
    float* d_relu_masks[10];     // For ReLU backward pass
    
    DQN(int state_dim, int action_dim, const std::vector<int>& hidden_dims);
    ~DQN();
    
    void forward(const float* d_state, float* d_q_values, int batch_size);
    void backward(const float* d_grad, int batch_size);
    void update(float lr, float beta1, float beta2, float epsilon, int t);
    void copy_weights_from(DQN* source);
    void soft_update_from(DQN* source, float tau);
    
    void save_weights(const char* filename);
    void load_weights(const char* filename);
};

// Experience Replay Buffer
struct Experience {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
};

class ReplayBuffer {
private:
    std::vector<Experience> buffer;
    int capacity;
    int position;
    std::mt19937 rng;
    
public:
    ReplayBuffer(int capacity);
    
    void add(const std::vector<float>& state, int action, float reward,
             const std::vector<float>& next_state, bool done);
    
    std::vector<Experience> sample(int batch_size);
    
    int size() const { return buffer.size(); }
    bool can_sample(int batch_size) const { return buffer.size() >= batch_size; }
};

// DQN Agent
class DQNAgent {
private:
    DQN* policy_net;
    DQN* target_net;
    ReplayBuffer* replay_buffer;
    
    int state_size;
    int action_size;
    float gamma;
    float epsilon;
    float epsilon_min;
    float epsilon_decay;
    float tau;
    
    int update_step;
    std::mt19937 rng;
    
    // Device memory for batch processing
    float* d_states;
    float* d_next_states;
    float* d_q_values;
    float* d_next_q_values;
    float* d_target_q_values;
    float* d_grad;
    
    int max_batch_size;
    
public:
    DQNAgent(int state_dim, int action_dim, const std::vector<int>& hidden_dims,
             int buffer_capacity = 100000, float gamma = 0.99f,
             float epsilon_start = 1.0f, float epsilon_end = 0.01f,
             float epsilon_decay = 0.995f, float tau = 0.001f);
    
    ~DQNAgent();
    
    int select_action(const std::vector<float>& state, bool training = true);
    
    void store_experience(const std::vector<float>& state, int action, float reward,
                         const std::vector<float>& next_state, bool done);
    
    float train_step(int batch_size, float lr, float beta1 = 0.9f, 
                     float beta2 = 0.999f, float epsilon_adam = 1e-8f);
    
    void update_target_network();
    
    void decay_epsilon();
    
    void save(const char* filename);
    void load(const char* filename);
    
    float get_epsilon() const { return epsilon; }
};

#endif // DQN_H
