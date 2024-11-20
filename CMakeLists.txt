cmake_minimum_required(VERSION 3.12)
project(custom_torch LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -forward-unknown-to-host-compiler -fopenmp -arch=sm_61 -O3 -Wall -shared -std=c++11 -fPIC")

find_package(OpenMP)
find_package(pybind11 REQUIRED)
# Just add all new cc or cu files you create. In time use a gloc recurse for traversing the project hierechy / structure
# pybind11_add_module(custom_torch MODULE tensor_pybind.cc tensor.cu)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/src/tensor ${PROJECT_SOURCE_DIR}/src/layers ${PROJECT_SOURCE_DIR}/src/loss ${PROJECT_SOURCE_DIR}/src/optimizers)

# Add subdirectories for each component
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

# Collect all source and Pybind files
# file(GLOB_RECURSE TENSOR_HDR "${SRC_DIR}/tensor/*.h")
file(GLOB_RECURSE TENSOR_SRC "${SRC_DIR}/tensor/*.cc" "${SRC_DIR}/tensor/*.cu")
file(GLOB_RECURSE TENSOR_PYBIND "${SRC_DIR}/tensor/*_pybind.cc")

# file(GLOB_RECURSE LAYER_HDR "${SRC_DIR}/layers/*.h")
file(GLOB_RECURSE LAYER_SRC "${SRC_DIR}/layers/*.cc" "${SRC_DIR}/layers/*.cu")
file(GLOB_RECURSE LAYER_PYBIND "${SRC_DIR}/layers/*_pybind.cc")

# file(GLOB_RECURSE LOSS_HDR "${SRC_DIR}/loss/*.h")
file(GLOB_RECURSE LOSS_SRC "${SRC_DIR}/loss/*.cc" "${SRC_DIR}/loss/*.cu")
file(GLOB_RECURSE LOSS_PYBIND "${SRC_DIR}/loss/*_pybind.cc")

# file(GLOB_RECURSE OPTIMIZER_HDR "${SRC_DIR}/optimizers/*.h")
file(GLOB_RECURSE OPTIMIZER_SRC "${SRC_DIR}/optimizers/*.cc" "${SRC_DIR}/optimizers/*.cu")
file(GLOB_RECURSE OPTIMIZER_PYBIND "${SRC_DIR}/optimizers/*_pybind.cc")

# Combine all sources
set(SOURCES 
    ${TENSOR_SRC}
    ${LAYER_SRC}
    ${LOSS_SRC}
    ${OPTIMIZER_SRC}
)

set(PYBINDS 
    ${TENSOR_PYBIND}
    ${LAYER_PYBIND}
    ${LOSS_PYBIND}
    ${OPTIMIZER_PYBIND}
)

# Add the Pybind11 module
pybind11_add_module(custom_torch MODULE ${SOURCES} ${PYBINDS})


if (OpenMP_CXX_FOUND)
  include_directories(OpenMP_CXX_INCLUDE_DIRS)
  target_link_libraries(custom_torch PUBLIC OpenMP::OpenMP_CXX)
endif()

