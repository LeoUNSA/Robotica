#include "dqn.h"
#include "cartpole_env.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

void print_progress_bar(int current, int total, float reward, float loss, float epsilon) {
    const int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);
    
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% "
              << "Episode: " << current << "/" << total
              << " | Reward: " << std::fixed << std::setprecision(2) << reward
              << " | Loss: " << std::setprecision(4) << loss
              << " | ε: " << std::setprecision(3) << epsilon;
    std::cout.flush();
}

int main(int argc, char** argv) {
    std::cout << "=== Deep Q-Network (DQN) Training on CartPole ===" << std::endl;
    std::cout << "Implemented in CUDA" << std::endl << std::endl;
    
    // Hyperparameters
    const int num_episodes = 500;
    const int batch_size = 32;
    const float learning_rate = 0.0001f;  // Reduced for stability
    const float gamma = 0.99f;
    const float epsilon_start = 1.0f;
    const float epsilon_end = 0.01f;
    const float epsilon_decay = 0.995f;
    const float tau = 0.005f;  // Increased for faster target network updates
    const int update_frequency = 1;  // Update every step
    const int target_update_frequency = 4;  // More frequent updates
    
    // Network architecture
    std::vector<int> hidden_dims = {128, 128};  // Larger network
    
    // Create environment
    CartPoleEnv env;
    int state_size = env.get_state_size();
    int action_size = env.get_action_size();
    
    std::cout << "Environment: CartPole" << std::endl;
    std::cout << "State size: " << state_size << std::endl;
    std::cout << "Action size: " << action_size << std::endl;
    std::cout << "Network architecture: " << state_size;
    for (int dim : hidden_dims) {
        std::cout << " -> " << dim;
    }
    std::cout << " -> " << action_size << std::endl << std::endl;
    
    // Create DQN agent
    DQNAgent agent(state_size, action_size, hidden_dims, 10000, gamma,
                   epsilon_start, epsilon_end, epsilon_decay, tau);
    
    std::cout << "Hyperparameters:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Gamma: " << gamma << std::endl;
    std::cout << "  Epsilon decay: " << epsilon_decay << std::endl;
    std::cout << "  Tau: " << tau << std::endl;
    std::cout << "  Update frequency: " << update_frequency << std::endl;
    std::cout << "  Target update frequency: " << target_update_frequency << std::endl;
    std::cout << std::endl;
    
    // Training statistics
    std::vector<float> episode_rewards;
    std::vector<float> episode_losses;
    float running_reward = 0.0f;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Training loop
    for (int episode = 0; episode < num_episodes; episode++) {
        auto state = env.reset();
        float episode_reward = 0.0f;
        float episode_loss = 0.0f;
        int steps = 0;
        
        bool done = false;
        while (!done) {
            // Select action
            int action = agent.select_action(state, true);
            
            // Take action in environment
            auto result = env.step(action);
            
            // Store experience
            agent.store_experience(state, action, result.reward, result.next_state, result.done);
            
            // Update state
            state = result.next_state;
            episode_reward += result.reward;
            done = result.done;
            steps++;
            
            // Train agent
            if (steps % update_frequency == 0) {
                float loss = agent.train_step(batch_size, learning_rate);
                episode_loss += loss;
            }
            
            // Update target network
            if (steps % target_update_frequency == 0) {
                agent.update_target_network();
            }
        }
        
        // Decay epsilon
        agent.decay_epsilon();
        
        // Update statistics
        episode_rewards.push_back(episode_reward);
        episode_losses.push_back(episode_loss / steps);
        running_reward = 0.05f * episode_reward + 0.95f * running_reward;
        
        // Print progress
        print_progress_bar(episode + 1, num_episodes, episode_reward,
                          episode_loss / steps, agent.get_epsilon());
        
        // Print detailed stats every 10 episodes
        if ((episode + 1) % 10 == 0) {
            std::cout << std::endl;
            std::cout << "Episode " << episode + 1 << " | "
                      << "Reward: " << episode_reward << " | "
                      << "Running avg: " << running_reward << " | "
                      << "Loss: " << episode_loss / steps << std::endl;
        }
        
        // Save model every 100 episodes
        if ((episode + 1) % 100 == 0) {
            std::string filename = "dqn_model_episode_" + std::to_string(episode + 1) + ".bin";
            agent.save(filename.c_str());
            std::cout << "Model saved to " << filename << std::endl;
        }
        
        // Early stopping if solved
        if (running_reward > 475.0f) {
            std::cout << std::endl << std::endl;
            std::cout << "Environment solved in " << episode + 1 << " episodes!" << std::endl;
            std::cout << "Running average reward: " << running_reward << std::endl;
            break;
        }
    }
    
    std::cout << std::endl << std::endl;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "Training completed in " << duration.count() << " seconds" << std::endl;
    
    // Save final model
    agent.save("dqn_model_final.bin");
    std::cout << "Final model saved to dqn_model_final.bin" << std::endl;
    
    // Print final statistics
    std::cout << std::endl << "=== Training Statistics ===" << std::endl;
    
    float avg_reward = 0.0f;
    for (float r : episode_rewards) {
        avg_reward += r;
    }
    avg_reward /= episode_rewards.size();
    
    std::cout << "Average episode reward: " << avg_reward << std::endl;
    std::cout << "Final running average: " << running_reward << std::endl;
    
    // Test the trained agent
    std::cout << std::endl << "=== Testing Trained Agent ===" << std::endl;
    const int test_episodes = 10;
    float total_test_reward = 0.0f;
    
    for (int episode = 0; episode < test_episodes; episode++) {
        auto state = env.reset();
        float episode_reward = 0.0f;
        bool done = false;
        
        while (!done) {
            int action = agent.select_action(state, false);  // Greedy action
            auto result = env.step(action);
            state = result.next_state;
            episode_reward += result.reward;
            done = result.done;
        }
        
        total_test_reward += episode_reward;
        std::cout << "Test episode " << episode + 1 << ": Reward = " << episode_reward << std::endl;
    }
    
    std::cout << "Average test reward: " << total_test_reward / test_episodes << std::endl;
    
    return 0;
}
