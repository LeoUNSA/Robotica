cmake_minimum_required(VERSION 3.10)
project(DQN_CUDA LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(CUDA_SOURCES
    cuda_kernels.cu
)

set(CPP_SOURCES
    dqn.cpp
    cartpole_env.cpp
    main.cpp
)

# Create executable
add_executable(dqn_train
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

# Set CUDA properties
set_target_properties(dqn_train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(dqn_train
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
)

# Compiler flags
target_compile_options(dqn_train PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
)

# Print configuration
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
