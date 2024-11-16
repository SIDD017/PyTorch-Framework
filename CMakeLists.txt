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
pybind11_add_module(custom_torch MODULE tensor_pybind.cc tensor.cu)

if (OpenMP_CXX_FOUND)
  include_directories(OpenMP_CXX_INCLUDE_DIRS)
  target_link_libraries(custom_torch PUBLIC OpenMP::OpenMP_CXX)
endif()

