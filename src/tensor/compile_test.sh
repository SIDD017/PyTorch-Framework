nvcc -forward-unknown-to-host-compiler -fopenmp -arch=sm_61 -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) tendor_pybind.cc -o hw3tensor$(python3-config --extension-suffix)
