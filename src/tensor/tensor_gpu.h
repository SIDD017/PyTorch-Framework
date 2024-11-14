#ifndef TENSOR_GPU_H
#define TENSOR_GPU_H

#include "tensor.h"
#include <string>
#include <omp.h>
#include<iostream>

#define NUM_BLOCKS 64
#define NUM_THREADS 32

class TensorGPU : public Tensor {
public:
  double *d_data;
  int num_blocks = NUM_BLOCKS;
  int num_threads = NUM_THREADS;

  void copyToDevice();
  void copyToHost();
  
  TensorGPU(std::vector<size_t> dims);
  TensorGPU(std::vector<size_t> dims,std::vector<std::vector<size_t>> idx,std::vector<double> val);
  static TensorGPU onesGPU(std::vector<size_t> dims);
  size_t indexGPU(std::vector<size_t> x);
  TensorGPU reshapeGPU(std::vector<size_t> new_dims);
  TensorGPU transposeGPU();
  TensorGPU negGPU();
  TensorGPU reciprocalGPU();
  TensorGPU addGPU(TensorGPU x);
  TensorGPU subtractGPU(TensorGPU x);
  TensorGPU multGPU(double x);
  TensorGPU elementwise_multGPU(TensorGPU x);
  TensorGPU powGPU(double x);
  TensorGPU reluGPU();
  TensorGPU binarilizeGPU();
  TensorGPU expGPU();
  TensorGPU matmulGPU(TensorGPU x);
};

#endif