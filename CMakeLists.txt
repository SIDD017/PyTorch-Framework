cmake_minimum_required(VERSION 3.12)
project(MyTensorProject LANGUAGES CXX CUDA)

# Set Python paths (adjust to match your Python installation)
set(PYTHON_INCLUDE_DIR "C:/Program Files/Python/python310/include")
set(PYTHON_LIB_DIR "C:/Program Files/Python/python310/libs")
set(PYTHON_LIB "python310")

# Find Pybind11
find_package(pybind11 REQUIRED)

# Set CUDA architecture and include paths
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O2;-std=c++17;--compiler-options;-fPIC")

# Include directories
include_directories(${PYTHON_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CUDA_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/tensor/tensor.cpp
    src/tensor/tensor_kernels.cu
    src/loss/mse_loss.cpp
    src/loss/cross_entropy_loss.cpp
    src/loss/loss_pybind.cpp
)

# Add library
add_library(my_tensor_module SHARED ${SOURCES})
set_target_properties(my_tensor_module PROPERTIES PREFIX "" SUFFIX ".pyd")

# Link libraries
target_link_libraries(my_tensor_module PRIVATE ${PYTHON_LIB} pybind11::module ${CUDA_LIBRARIES})

# Output the shared library (Python module)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python_module)
