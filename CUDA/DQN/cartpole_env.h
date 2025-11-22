#ifndef CARTPOLE_ENV_H
#define CARTPOLE_ENV_H

#include <vector>
#include <random>

// Simple CartPole environment for testing DQN
class CartPoleEnv {
private:
    // State variables
    float cart_position;
    float cart_velocity;
    float pole_angle;
    float pole_angular_velocity;
    
    // Environment parameters
    const float gravity = 9.8f;
    const float mass_cart = 1.0f;
    const float mass_pole = 0.1f;
    const float total_mass = mass_cart + mass_pole;
    const float pole_length = 0.5f;
    const float pole_mass_length = mass_pole * pole_length;
    const float force_magnitude = 10.0f;
    const float dt = 0.02f;  // Time step
    
    // Thresholds for failure
    const float theta_threshold = 12.0f * M_PI / 180.0f;  // ~12 degrees
    const float x_threshold = 2.4f;
    
    // Random number generator
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    
    int step_count;
    const int max_steps = 500;
    
public:
    CartPoleEnv();
    
    std::vector<float> reset();
    
    struct StepResult {
        std::vector<float> next_state;
        float reward;
        bool done;
    };
    
    StepResult step(int action);
    
    std::vector<float> get_state() const;
    
    int get_state_size() const { return 4; }
    int get_action_size() const { return 2; }  // Left or Right
    
    bool is_done() const;
};

#endif // CARTPOLE_ENV_H
