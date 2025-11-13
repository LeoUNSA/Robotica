import numpy as np
import random
import os
import sys
from collections import deque
from controller import Robot

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    print("ERROR: PyTorch not found. Please install: pip install torch")
    sys.exit(1)

# ---------- CONFIGURATION ----------
TIME_STEP = 64
EPISODES = 2000
MAX_STEPS_PER_EPISODE = 500
TRAINING_MODE = True  # Set to False for greedy policy execution

SENSOR_NAMES = ['ds_left', 'ds_right']
STATE_SIZE = len(SENSOR_NAMES)
ACTIONS = ['forward', 'left', 'right', 'back']
N_ACTIONS = len(ACTIONS)

# Velocities
FORWARD_V = 5.0
TURN_V = 3.5
BACK_V = -4.0

# Rewards
REWARD_COLLISION = -10.0
REWARD_STEP = -0.05
REWARD_SAFE_FORWARD = 1.0
REWARD_TURN_NEAR_WALL = 0.5  # Reward turning when near obstacles
COLLISION_THRESHOLD = 0.20
SAFE_FORWARD_THRESHOLD = 0.6
NEAR_WALL_THRESHOLD = 0.35  # Threshold for "near wall" detection

# DQN Hyperparameters
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 5  # Update target network every N episodes
MIN_REPLAY_SIZE = 1000  # Start training after this many experiences

MODEL_FILE = 'dqn_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- NEURAL NETWORK ----------
class DQN(nn.Module):
    """Deep Q-Network with 3 hidden layers."""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# ---------- REPLAY BUFFER ----------
class ReplayBuffer:
    """Experience replay buffer for storing transitions."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


# ---------- DQN AGENT ----------
class DQNAgent:
    """DQN Agent with experience replay and target network."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Main network and target network
        self.policy_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        """Perform one training step on a batch of experiences."""
        if len(self.memory) < MIN_REPLAY_SIZE:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def save(self, filepath):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', EPSILON_MIN)


# ---------- WEBOTS INITIALIZATION ----------
robot = Robot()
timestep = int(robot.getBasicTimeStep()) if robot.getBasicTimeStep() else TIME_STEP

# Motors
motor_names = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
motors = {}

for name in motor_names:
    dev = robot.getDevice(name)
    if dev:
        motors[name] = dev
    else:
        print(f"WARNING: motor {name} not found")

if len(motors) < 4:
    print("ERROR: missing one or more motors 'wheel1'..'wheel4'.")
    sys.exit(1)

# Group motors by side
right_motors = [motors['wheel1'], motors['wheel2']]
left_motors = [motors['wheel3'], motors['wheel4']]

for m in left_motors + right_motors:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# Distance sensors
distance_sensors = []
for name in SENSOR_NAMES:
    ds = robot.getDevice(name)
    if ds is None:
        print(f"ERROR: sensor {name} not found.")
        sys.exit(1)
    ds.enable(timestep)
    distance_sensors.append(ds)

# Initialize DQN agent
agent = DQNAgent(STATE_SIZE, N_ACTIONS)

# Load existing model if available
if os.path.exists(MODEL_FILE):
    try:
        agent.load(MODEL_FILE)
        print(f"Loaded model from {MODEL_FILE}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Starting with fresh model")


# ---------- UTILITY FUNCTIONS ----------
def read_normalized_sensors():
    """Read and normalize sensor values (1=free, 0=collision)."""
    vals = []
    for ds in distance_sensors:
        max_val = ds.getMaxValue() if ds.getMaxValue() > 0 else 4096.0
        v = ds.getValue() / max_val
        vals.append(1.0 - max(0.0, min(1.0, v)))  # Invert scale
    return np.array(vals, dtype=np.float32)


def set_motors_for_action(action):
    """Control motors based on action."""
    if action is None:
        v_left = v_right = 0.0
    elif ACTIONS[action] == 'forward':
        v_left = FORWARD_V
        v_right = FORWARD_V
    elif ACTIONS[action] == 'left':
        v_left = -TURN_V
        v_right = TURN_V
    elif ACTIONS[action] == 'right':
        v_left = TURN_V
        v_right = -TURN_V
    elif ACTIONS[action] == 'back':
        v_left = v_right = BACK_V
    else:
        v_left = v_right = 0.0

    for m in left_motors:
        m.setVelocity(v_left)
    for m in right_motors:
        m.setVelocity(v_right)


def compute_reward(sensor_vals, action, prev_min_dist=None):
    """Compute reward based on state and action."""
    min_dist = np.min(sensor_vals)
    avg_dist = np.mean(sensor_vals)
    left_dist = sensor_vals[0]
    right_dist = sensor_vals[1]
    
    # Base penalty for time
    reward = REWARD_STEP
    done = False
    
    # Collision penalty
    if min_dist < COLLISION_THRESHOLD:
        reward += REWARD_COLLISION
        done = True
        return reward, done
    
    # Reward turning when near walls
    if min_dist < NEAR_WALL_THRESHOLD:
        if ACTIONS[action] in ['left', 'right']:
            reward += REWARD_TURN_NEAR_WALL
            # Extra reward for turning away from closer obstacle
            if ACTIONS[action] == 'left' and right_dist < left_dist:
                reward += 0.3
            elif ACTIONS[action] == 'right' and left_dist < right_dist:
                reward += 0.3
    
    # Reward forward movement in open space
    if ACTIONS[action] == 'forward' and avg_dist > SAFE_FORWARD_THRESHOLD:
        reward += REWARD_SAFE_FORWARD
    
    # Penalty for backward movement (discourage excessive backing up)
    if ACTIONS[action] == 'back':
        reward -= 0.2
    
    # Reward for maintaining distance from walls
    reward += 0.3 * avg_dist
    
    # Penalty for getting closer to obstacles
    if prev_min_dist is not None and min_dist < prev_min_dist - 0.05:
        reward -= 0.3
    
    return reward, done


# ---------- TRAINING ----------
if TRAINING_MODE:
    print(f"Starting DQN training on {DEVICE}")
    
    for ep in range(1, EPISODES + 1):
        set_motors_for_action(None)
        robot.step(timestep)

        state = read_normalized_sensors()
        total_reward = 0.0
        losses = []
        prev_min_dist = None

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state, training=True)
            set_motors_for_action(action)

            if robot.step(timestep) == -1:
                break

            next_state = read_normalized_sensors()
            reward, done = compute_reward(next_state, action, prev_min_dist)

            # Store experience
            agent.memory.push(state, action, reward, next_state, float(done))

            # Train on batch
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            prev_min_dist = np.min(next_state)
            total_reward += reward

            if done:
                set_motors_for_action(None)
                robot.step(timestep)
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Update target network periodically
        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Logging
        if ep % 50 == 0:
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Episode {ep}/{EPISODES} | Reward: {total_reward:.3f} | "
                  f"Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f} | "
                  f"Memory: {len(agent.memory)}")
        
        # Save model periodically
        if ep % 200 == 0:
            agent.save(MODEL_FILE)
            print(f"Model saved at episode {ep}")

    agent.save(MODEL_FILE)
    print("Training completed. Model saved.")

# ---------- GREEDY POLICY EXECUTION ----------
print("Executing greedy policy (Ctrl+C to stop).")
agent.epsilon = 0.0  # No exploration

while robot.step(timestep) != -1:
    state = read_normalized_sensors()
    action = agent.select_action(state, training=False)
    set_motors_for_action(action)