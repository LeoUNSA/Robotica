# Compiler settings
CUDA_PATH = /opt/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# Compiler flags
NVCC_FLAGS = -std=c++11 -O3 -arch=sm_75 --compiler-options -fPIC
CXX_FLAGS = -std=c++11 -O3 -Wall -fPIC

# CUDA flags
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcurand
INCLUDES = -I$(CUDA_PATH)/include

# Target executable
TARGET = dqn_train

# Source files
CUDA_SOURCES = cuda_kernels.cu
CPP_SOURCES = dqn.cpp cartpole_env.cpp main.cpp

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
OBJECTS = $(CUDA_OBJECTS) $(CPP_OBJECTS)

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_LIBS)
	@echo "Build complete: $(TARGET)"

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET) *.bin
	@echo "Cleaned build artifacts"

# Run the training
run: $(TARGET)
	./$(TARGET)

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "NVCC not found. Please install CUDA toolkit."
	@nvidia-smi || echo "nvidia-smi not found. Please install NVIDIA drivers."

# Help message
help:
	@echo "Available targets:"
	@echo "  all        - Build the project (default)"
	@echo "  clean      - Remove build artifacts"
	@echo "  run        - Build and run the training"
	@echo "  check-cuda - Check CUDA installation"
	@echo "  help       - Show this help message"

.PHONY: all clean run check-cuda help
