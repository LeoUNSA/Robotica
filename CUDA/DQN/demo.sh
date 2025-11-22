#!/bin/bash

echo "=========================================="
echo "  DQN en CUDA - Demostración Completa"
echo "=========================================="
echo ""

# Mostrar información de GPU
echo "📊 Información de GPU:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

# Mostrar arquitectura compilada
echo "🏗️  Arquitectura del proyecto:"
echo "   - CUDA Kernels: cuda_kernels.cu/h"
echo "   - DQN Agent: dqn.cpp/h"
echo "   - Environment: cartpole_env.cpp/h"
echo "   - Main: main.cpp"
echo ""

# Contar líneas de código
echo "📝 Líneas de código:"
echo "   CUDA kernels: $(wc -l < cuda_kernels.cu) líneas"
echo "   DQN implementation: $(wc -l < dqn.cpp) líneas"
echo "   Headers: $(cat *.h | wc -l) líneas"
echo "   Total: $(cat *.cu *.cpp *.h | wc -l) líneas"
echo ""

# Mostrar modelos guardados
echo "💾 Modelos guardados:"
ls -lh *.bin 2>/dev/null | awk '{print "   - " $9 " (" $5 ")"}'
echo ""

echo "🚀 Opciones:"
echo "   1. Ver README completo: cat README.md"
echo "   2. Ver resultados del test: cat TEST_RESULTS.md"
echo "   3. Ver log de entrenamiento: cat training_output.log"
echo "   4. Ejecutar nuevo entrenamiento: ./dqn_train"
echo "   5. Compilar desde cero: make clean && make"
echo ""

echo "✅ Implementación completa y funcional!"
echo ""
