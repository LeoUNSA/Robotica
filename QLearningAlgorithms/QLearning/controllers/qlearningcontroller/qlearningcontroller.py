#!/usr/bin/env python3
# qlearningcontroller.py
# Controlador mejorado con Q-Learning para FourWheelsRobot
# Compatible con motores: wheel1..wheel4
# Sensores de distancia: ds_left, ds_right

import numpy as np
import random
import os
import sys
from controller import Robot

# ---------- CONFIGURACIÓN ----------
TIME_STEP = 64
EPISODES = 2000
MAX_STEPS_PER_EPISODE = 500
TRAINING_MODE = True  # Cambia a False para ejecutar la política greedy

SENSOR_NAMES = ['ds_left', 'ds_right']
BINS = 3
ACTIONS = ['forward', 'left', 'right', 'back']
N_ACTIONS = len(ACTIONS)

# Velocidades
FORWARD_V = 5.0
TURN_V = 3.5
BACK_V = -4.0

# Recompensas
REWARD_COLLISION = -5.0
REWARD_STEP = -0.01
REWARD_SAFE_FORWARD = 0.3
COLLISION_THRESHOLD = 0.15
SAFE_FORWARD_THRESHOLD = 0.5

# Parámetros Q-Learning
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9997

QTABLE_FILE = 'qtable.npy'


# ---------- FUNCIONES DE UTILIDAD ----------
def make_bins(n_bins):
    # 0.0 (colisión) → 1.0 (libre)
    return np.linspace(0.0, 1.0, n_bins + 1)[1:-1]


def discretize_state(sensor_values, bins):
    return tuple(int(np.digitize(v, bins)) for v in sensor_values)


def state_to_index(state_idxs, base):
    idx = 0
    for s in state_idxs:
        idx = idx * base + s
    return idx


# ---------- INICIALIZACIÓN WEBOTS ----------
robot = Robot()
timestep = int(robot.getBasicTimeStep()) if robot.getBasicTimeStep() else TIME_STEP

# Motores
motor_names = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
motors = {}

for name in motor_names:
    dev = robot.getDevice(name)
    if dev:
        motors[name] = dev
    else:
        print(f"ADVERTENCIA: motor {name} no encontrado")

if len(motors) < 4:
    print("ERROR: faltan uno o más motores 'wheel1'..'wheel4'.")
    sys.exit(1)

# Agrupar por lado (según PROTO)
right_motors = [motors['wheel1'], motors['wheel2']]
left_motors = [motors['wheel3'], motors['wheel4']]

for m in left_motors + right_motors:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# Sensores de distancia
distance_sensors = []
for name in SENSOR_NAMES:
    ds = robot.getDevice(name)
    if ds is None:
        print(f"ERROR: sensor {name} no encontrado.")
        sys.exit(1)
    ds.enable(timestep)
    distance_sensors.append(ds)

bins = make_bins(BINS)
n_states = BINS ** len(SENSOR_NAMES)

# Cargar o inicializar Q-table
if os.path.exists(QTABLE_FILE):
    qtable = np.load(QTABLE_FILE)
    if qtable.shape != (n_states, N_ACTIONS):
        print("Tamaño de Q-table incompatible, reiniciando...")
        qtable = np.zeros((n_states, N_ACTIONS))
else:
    qtable = np.zeros((n_states, N_ACTIONS))


# ---------- FUNCIONES DE CONTROL ----------
def read_normalized_sensors():
    """Lee y normaliza los valores de los sensores (invertidos: 1=libre, 0=colisión)."""
    vals = []
    for ds in distance_sensors:
        v = ds.getValue() / ds.getMaxValue() if ds.getMaxValue() > 0 else ds.getValue() / 4096.0
        vals.append(1.0 - max(0.0, min(1.0, v)))  # invertir escala
    return vals


def set_motors_for_action(action):
    """Controla los motores según la acción."""
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


def choose_action(state_idx, epsilon):
    """Elige una acción (ε-greedy)."""
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    return int(np.argmax(qtable[state_idx]))


def compute_reward(sensor_vals, action):
    """Define la recompensa en base al estado y acción."""
    min_dist = min(sensor_vals)
    avg_dist = sum(sensor_vals) / len(sensor_vals)
    reward = REWARD_STEP + 0.5 * avg_dist  # incentivo continuo

    if min_dist < COLLISION_THRESHOLD:
        reward += REWARD_COLLISION
        done = True
    elif ACTIONS[action] == 'forward' and avg_dist > SAFE_FORWARD_THRESHOLD:
        reward += REWARD_SAFE_FORWARD
        done = False
    else:
        done = False

    return reward, done


# ---------- ENTRENAMIENTO ----------
if TRAINING_MODE:
    epsilon = EPSILON_START

    for ep in range(1, EPISODES + 1):
        set_motors_for_action(None)
        robot.step(timestep)

        sens_vals = read_normalized_sensors()
        state_idxs = discretize_state(sens_vals, bins)
        state_idx = state_to_index(state_idxs, BINS)

        total_reward = 0.0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = choose_action(state_idx, epsilon)
            set_motors_for_action(action)

            if robot.step(timestep) == -1:
                break

            new_sens_vals = read_normalized_sensors()
            new_state_idxs = discretize_state(new_sens_vals, bins)
            new_state_idx = state_to_index(new_state_idxs, BINS)

            reward, done = compute_reward(new_sens_vals, action)

            old_q = qtable[state_idx, action]
            future_q = 0 if done else np.max(qtable[new_state_idx])
            qtable[state_idx, action] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)

            state_idx = new_state_idx
            total_reward += reward

            if done:
                set_motors_for_action(None)
                robot.step(timestep)
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if ep % 50 == 0:
            print(f"Episode {ep}/{EPISODES} total_reward={total_reward:.3f} epsilon={epsilon:.3f}")
        if ep % 200 == 0:
            np.save(QTABLE_FILE, qtable)

    np.save(QTABLE_FILE, qtable)
    print("Entrenamiento finalizado. Q-table guardada.")

# ---------- EJECUCIÓN POLÍTICA GREEDY ----------
print("Ejecutando política greedy (Ctrl+C para detener).")

while robot.step(timestep) != -1:
    sens_vals = read_normalized_sensors()
    state_idxs = discretize_state(sens_vals, bins)
    state_idx = state_to_index(state_idxs, BINS)
    action = int(np.argmax(qtable[state_idx]))
    set_motors_for_action(action)
