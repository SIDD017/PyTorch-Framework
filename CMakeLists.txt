cmake_minimum_required(VERSION 3.12)
project(hw3 LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Release)
##set(CMAKE_CUDA_ARCH_BIN "61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -forward-unknown-to-host-compiler -fopenmp -arch=sm_61 -O3 -Wall -shared -std=c++11 -fPIC")

##add_subdirectory(pybind11)

find_package(OpenMP)
find_package(pybind11 REQUIRED)
pybind11_add_module(hw3tensor MODULE hw3tensor.cu)

if (OpenMP_CXX_FOUND)
  include_directories(OpenMP_CXX_INCLUDE_DIRS)
  target_link_libraries(hw3tensor PUBLIC OpenMP::OpenMP_CXX)
endif()

