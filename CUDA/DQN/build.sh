#!/bin/bash

# Script de compilación y ejecución para DQN en CUDA

echo "================================================"
echo "  DQN (Deep Q-Network) en CUDA - Build Script"
echo "================================================"
echo ""

# Verificar CUDA
echo "Verificando instalación de CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc no encontrado. Por favor instala CUDA Toolkit."
    exit 1
fi

echo "✓ CUDA Toolkit encontrado:"
nvcc --version | grep "release"

# Verificar drivers NVIDIA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA drivers instalados"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n 1
else
    echo "⚠ nvidia-smi no encontrado. Asegúrate de tener los drivers instalados."
fi

echo ""
echo "Compilando proyecto..."
echo ""

# Limpiar build anterior
make clean

# Compilar
if make; then
    echo ""
    echo "================================================"
    echo "✓ Compilación exitosa"
    echo "================================================"
    echo ""
    echo "Para ejecutar el entrenamiento:"
    echo "  ./dqn_train"
    echo ""
    echo "O ejecutar directamente con:"
    echo "  make run"
    echo ""
else
    echo ""
    echo "================================================"
    echo "✗ Error en la compilación"
    echo "================================================"
    exit 1
fi
