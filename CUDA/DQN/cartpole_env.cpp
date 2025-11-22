#include "cartpole_env.h"
#include <cmath>

CartPoleEnv::CartPoleEnv() : dist(-0.05f, 0.05f) {
    std::random_device rd;
    rng.seed(rd());
    step_count = 0;
}

std::vector<float> CartPoleEnv::reset() {
    // Initialize state with small random values
    cart_position = dist(rng);
    cart_velocity = dist(rng);
    pole_angle = dist(rng);
    pole_angular_velocity = dist(rng);
    step_count = 0;
    
    return get_state();
}

CartPoleEnv::StepResult CartPoleEnv::step(int action) {
    step_count++;
    
    // Apply force based on action (0 = left, 1 = right)
    float force = (action == 1) ? force_magnitude : -force_magnitude;
    
    // Physics simulation (using simplified CartPole dynamics)
    float cos_theta = std::cos(pole_angle);
    float sin_theta = std::sin(pole_angle);
    
    float temp = (force + pole_mass_length * pole_angular_velocity * pole_angular_velocity * sin_theta) / total_mass;
    float theta_acc = (gravity * sin_theta - cos_theta * temp) / 
                     (pole_length * (4.0f/3.0f - mass_pole * cos_theta * cos_theta / total_mass));
    float x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;
    
    // Update state using Euler integration
    cart_position += dt * cart_velocity;
    cart_velocity += dt * x_acc;
    pole_angle += dt * pole_angular_velocity;
    pole_angular_velocity += dt * theta_acc;
    
    // Check if episode is done
    bool done = is_done() || step_count >= max_steps;
    
    // Reward is 1 for every step the pole stays up
    float reward = done ? 0.0f : 1.0f;
    
    StepResult result;
    result.next_state = get_state();
    result.reward = reward;
    result.done = done;
    
    return result;
}

std::vector<float> CartPoleEnv::get_state() const {
    return {cart_position, cart_velocity, pole_angle, pole_angular_velocity};
}

bool CartPoleEnv::is_done() const {
    return std::abs(cart_position) > x_threshold ||
           std::abs(pole_angle) > theta_threshold;
}
