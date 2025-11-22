# Deep Q-Network (DQN) en CUDA

Implementación completa de Deep Q-Network usando CUDA para aceleración GPU. Este proyecto incluye una red neuronal profunda implementada desde cero en CUDA, un agente DQN con experience replay, y un entorno CartPole para entrenamiento.

## Características

- **Red Neuronal en CUDA**: Implementación completa de propagación hacia adelante y hacia atrás con kernels CUDA optimizados
- **Kernels CUDA Optimizados**: 
  - Multiplicación de matrices con memoria compartida
  - Funciones de activación ReLU
  - Operaciones element-wise
  - Optimizador Adam
- **Algoritmo DQN**:
  - Experience replay buffer
  - Target network con soft update
  - Epsilon-greedy exploration
  - MSE loss para Q-learning
- **Entorno CartPole**: Simulación física del problema clásico de control

## Estructura del Proyecto

```
DQN/
├── cuda_kernels.cu      # Kernels CUDA para operaciones de red neuronal
├── cuda_kernels.h       # Headers de kernels CUDA
├── dqn.h               # Definiciones de clases DQN, Layer, ReplayBuffer, Agent
├── dqn.cpp             # Implementación de DQN y componentes
├── cartpole_env.h      # Header del entorno CartPole
├── cartpole_env.cpp    # Implementación del entorno CartPole
├── main.cpp            # Loop de entrenamiento principal
├── Makefile            # Build con Make
└── CMakeLists.txt      # Build con CMake
```

## Requisitos

- NVIDIA GPU con compute capability 6.0 o superior
- CUDA Toolkit 10.0 o superior
- GCC/G++ 7.0 o superior
- CMake 3.10 o superior (opcional, para build con CMake)

## Compilación

### Opción 1: Usando Makefile

```bash
# Verificar instalación de CUDA
make check-cuda

# Compilar el proyecto
make

# Compilar y ejecutar
make run
```

### Opción 2: Usando CMake

```bash
# Crear directorio de build
mkdir build
cd build

# Configurar y compilar
cmake ..
make

# Ejecutar
./dqn_train
```

## Uso

### Entrenamiento

Ejecutar el entrenamiento con configuración por defecto:

```bash
./dqn_train
```

El programa:
1. Crea un agente DQN con arquitectura 4 → 64 → 64 → 2
2. Entrena por 500 episodios (o hasta resolver el entorno)
3. Guarda checkpoints cada 100 episodios
4. Guarda el modelo final como `dqn_model_final.bin`
5. Evalúa el agente entrenado por 10 episodios de prueba

### Hiperparámetros

Los hiperparámetros principales se pueden modificar en `main.cpp`:

```cpp
const int num_episodes = 500;        // Número de episodios de entrenamiento
const int batch_size = 64;           // Tamaño del batch
const float learning_rate = 0.001f;  // Tasa de aprendizaje
const float gamma = 0.99f;           // Factor de descuento
const float epsilon_start = 1.0f;    // Epsilon inicial (exploración)
const float epsilon_end = 0.01f;     // Epsilon final
const float epsilon_decay = 0.995f;  // Tasa de decay de epsilon
const float tau = 0.001f;            // Tasa de soft update
```

## Arquitectura DQN

### Componentes Principales

1. **Layer**: Capa de red neuronal con:
   - Forward pass: multiplicación de matrices + bias + activación
   - Backward pass: cálculo de gradientes
   - Optimizador Adam integrado

2. **DQN**: Red neuronal profunda que:
   - Aproxima la función Q(s,a)
   - Soporta arquitecturas de múltiples capas
   - Incluye save/load de pesos

3. **ReplayBuffer**: Buffer de experiencias con:
   - Almacenamiento circular
   - Muestreo aleatorio uniforme
   - Capacidad configurable

4. **DQNAgent**: Agente de aprendizaje que:
   - Mantiene policy network y target network
   - Implementa epsilon-greedy exploration
   - Ejecuta pasos de entrenamiento
   - Actualiza target network con soft update

### Kernels CUDA Implementados

- `matmul_kernel`: Multiplicación de matrices básica
- `matmul_shared_kernel`: Multiplicación optimizada con memoria compartida
- `relu_kernel`: Activación ReLU
- `add_bias_kernel`: Suma de bias a matriz
- `adam_update_kernel`: Actualización de parámetros con Adam
- `mse_loss_kernel`: Cálculo de pérdida MSE
- `transpose_kernel`: Transposición de matrices
- Más kernels para operaciones auxiliares

## Ejemplo de Salida

```
=== Deep Q-Network (DQN) Training on CartPole ===
Implemented in CUDA

Environment: CartPole
State size: 4
Action size: 2
Network architecture: 4 -> 64 -> 64 -> 2

Hyperparameters:
  Batch size: 64
  Learning rate: 0.001
  Gamma: 0.99
  ...

[==================================================] 100% Episode: 500/500 | Reward: 500.00 | Loss: 0.0023 | ε: 0.010

Episode 10 | Reward: 23.00 | Running avg: 21.85 | Loss: 0.1234
Episode 20 | Reward: 45.00 | Running avg: 38.72 | Loss: 0.0876
...
Environment solved in 245 episodes!
Running average reward: 475.3

Training completed in 123 seconds
Final model saved to dqn_model_final.bin

=== Training Statistics ===
Average episode reward: 234.56
Final running average: 475.30

=== Testing Trained Agent ===
Test episode 1: Reward = 500.00
Test episode 2: Reward = 500.00
...
Average test reward: 497.50
```

## Detalles de Implementación

### Algoritmo DQN

El agente implementa el algoritmo DQN estándar:

1. Observa estado s
2. Selecciona acción a usando ε-greedy
3. Ejecuta acción, observa recompensa r y siguiente estado s'
4. Almacena transición (s, a, r, s', done) en replay buffer
5. Muestrea batch de transiciones del buffer
6. Calcula target: y = r + γ * max_a' Q_target(s', a')
7. Actualiza Q_policy minimizando (Q_policy(s,a) - y)²
8. Actualiza Q_target con soft update

### Optimizaciones CUDA

- **Memoria compartida**: Usado en multiplicación de matrices para reducir accesos a memoria global
- **Coalesced memory access**: Patrones de acceso optimizados para máximo ancho de banda
- **Kernel fusion**: Operaciones combinadas donde es posible
- **Streams**: Preparado para procesamiento asíncrono

## Limitaciones Conocidas

- Batch size máximo: 256 (configurable en `DQNAgent`)
- Número máximo de capas: 10 (configurable en `DQN`)
- Solo soporta capas densas (fully connected)
- Activación ReLU solamente (se puede extender)

## Posibles Extensiones

- [ ] Soporte para más funciones de activación (tanh, sigmoid, etc.)
- [ ] Double DQN
- [ ] Dueling DQN
- [ ] Prioritized Experience Replay
- [ ] Más entornos de prueba
- [ ] Visualización en tiempo real
- [ ] Convolutional layers para estados de imagen

## Referencias

- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
- Van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.

## Licencia

Este proyecto es código educativo y está disponible para uso académico.

## Autor

Implementado para el curso de Robótica - UNSA

## Notas

- Ajustar `arch=sm_XX` en Makefile según tu GPU
- Para debugging, reducir optimización flags (-O0) y agregar -g
- Usar `nvidia-smi` para monitorear uso de GPU durante entrenamiento
