#include "tensor.h"

#define CHECK_CUDA_ERRORS(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
         std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";

         cudaDeviceReset();
         exit(99);
    }   
}

void Tensor::copyToDevice() const {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_data, data.data(), data.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void Tensor::copyToHost() {
    CHECK_CUDA_ERRORS(cudaMemcpy(data.data(), d_data, data.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

/** KERNELS */
__global__ static void d_ones(double *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    data[i] = 1.0;
  }
}

__global__ static void d_zeros(double *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    data[i] = 0.0;
  }
}

__global__ void d_reshape(double *out, const double *in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = in[i];
  }
}

__global__ void d_neg(double *out, const double *in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = -in[i];
  }
}

__global__ void d_reciprocal(double *out, const double *in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = 1.0 / in[i];
  }
}

__global__ void d_reciprocal_grad(double *out_grad, const double *ret_grad, 
                                  const double *input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out_grad[i] = -ret_grad[i] / (input[i] * input[i]);
  }
}

__global__ void d_add(double *out, const double *in1, const double *in2, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = in1[i] + in2[i];
  }
}

__global__ void d_add_grad(double* grad_out, const double* grad_in, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_out[idx] = grad_in[idx];
    }
}

__global__ void d_subtract(double *out, const double *in1, const double *in2, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = in1[i] - in2[i];
  }
}

__global__ void d_scalar_mult(double *out, const double *in, double scalar, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
      out[i] = in[i] * scalar;
  }
}

__global__ void d_elementwise_mult(double *out, const double *in1, const double *in2, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = in1[i] * in2[i];
  }
}

__global__ void d_elementwise_mult_grad(
  double *out_grad1, double *out_grad2, 
  const double *ret_grad, 
  const double *input1, const double *input2, 
  int size
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out_grad1[i] = input2[i] * ret_grad[i];
    out_grad2[i] = input1[i] * ret_grad[i];
  }
}

__global__ void d_pow(double *out, const double *in, double x, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = pow(in[i], x);
  }
}

__global__ void d_pow_grad(double* grad_out, const double* grad_in, const double* input, 
                          double power, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_out[idx] = grad_in[idx] * power * pow(input[idx], power - 1);
    }
}

__global__ void d_relu(double *out, const double *in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = in[i] > 0 ? in[i] : 0;
  }
}

__global__ void d_relu_grad(
  double *out_grad, 
  const double *ret_grad, 
  const double *input, 
  int size
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out_grad[i] = input[i] > 0 ? ret_grad[i] : 0;
  }
}

__global__ void d_binarilize(double *out, const double *in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = in[i] > 0 ? 1.0 : 0.0;
  }
}

__global__ void d_exp(double *out, const double *in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = exp(in[i]);
  }
}

__global__ void d_exp_grad(
  double *out_grad, 
  const double *ret_grad, 
  const double *output, 
  int size
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out_grad[i] = output[i] * ret_grad[i];
  }
}

__global__ void d_matmul_2d(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
      }
      C[row * N + col] = sum;
    }
}

__global__ void d_matmul_3d(const double* A, const double* B, double* C, int batch_size, int M, int N, int K) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch < batch_size && row < M && col < N) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A[batch * M * K + row * K + k] * B[k * N + col];
      }
      C[batch * M * N + row * N + col] = sum;
    }
}

// Gradient computation kernels
__global__ void d_matmul_grad_left_2d(const double* dC, const double* B, double* dA,
                                     int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        double sum = 0.0;
        for (int n = 0; n < N; ++n) {
            sum += dC[row * N + n] * B[col * N + n];
        }
        dA[row * K + col] = sum;
    }
}

__global__ void d_matmul_grad_right_2d(const double* A, const double* dC, double* dB,
                                      int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < K && col < N) {
        double sum = 0.0;
        for (int m = 0; m < M; ++m) {
            sum += A[m * K + row] * dC[m * N + col];
        }
        dB[row * N + col] = sum;
    }
}

__global__ void d_matmul_grad_left_3d(const double* dC, const double* B, double* dA,
                                     int batch_size, int M, int K, int N) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < batch_size && row < M && col < K) {
        double sum = 0.0;
        for (int n = 0; n < N; ++n) {
            sum += dC[batch * M * N + row * N + n] * B[col * N + n];
        }
        dA[batch * M * K + row * K + col] = sum;
    }
}

__global__ void d_matmul_grad_right_3d(const double* A, const double* dC, double* dB,
                                      int batch_size, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < K && col < N) {
        double sum = 0.0;
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int m = 0; m < M; ++m) {
                sum += A[batch * M * K + m * K + row] * dC[batch * M * N + m * N + col];
            }
        }
        dB[row * N + col] = sum;
    }
}

__global__ void d_transpose_2d(const double* in, double* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
      out[col * rows + row] = in[row * cols + col];
    }
}

__global__ void d_transpose_3d(const double* in, double* out, int batch_size, int rows, int cols) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch < batch_size && row < rows && col < cols) {
      out[batch * cols * rows + col * rows + row] = in[batch * rows * cols + row * cols + col];
    }
}

__global__ void d_reshape_copy(double* out, const double* in, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
      out[idx] = in[idx];
    }
}

__global__ void d_log(double *out, const double *in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = log(in[i]);
  }
}

// CUDA kernel for backward pass gradient computation
__global__ void d_log_grad(
  double *out_grad, 
  const double *ret_grad, 
  const double *input, 
  int size
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out_grad[i] = ret_grad[i] / input[i];
  }
}

Tensor::Tensor(std::vector<size_t> dims, std::string dev = "cpu", bool requires_grad = false) : dims(dims), requires_grad(requires_grad) {
  device = dev;
  size_t len = 1;
  for(auto d : dims)
    len *= d;
  data.resize(len);
  if (requires_grad) {
    grad = std::make_shared<std::vector<double>>(len, 0.0);
  }
  if(dev == "cuda") {
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
    copyToDevice();
  }
  else {
    d_data = nullptr;
  }
}

Tensor::Tensor(std::vector<double> data1, std::string dev = "cpu", bool requires_grad = false) : requires_grad(requires_grad) {
  device = dev;
  dims.push_back(data1.size());
  size_t len = 1;
  for(auto d : dims)
    len *= d;
  // data.resize(len);
  data.insert(data.end(), data1.begin(), data1.end()); 
  if (requires_grad) {
    grad = std::make_shared<std::vector<double>>(len, 0.0);
  }
  if(dev == "cuda") {
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
    copyToDevice();
  }
  else {
    d_data = nullptr;
  }
}

Tensor::Tensor(std::vector<std::vector<double>> data1, std::string dev = "cpu", bool requires_grad = false) : requires_grad(requires_grad) {
  device = dev;
  dims.push_back(data1.size());
  dims.push_back(data1[0].size());
  for(size_t i = 0; i < data1.size(); ++i) {
    data.insert(data.end(), data1[i].begin(), data1[i].end()); 
  }
  if (requires_grad) {
    grad = std::make_shared<std::vector<double>>(data.size(), 0.0);
  }
  if(dev == "cuda") {
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
    copyToDevice();
  }
  else {
    d_data = nullptr;
  }
}

Tensor::Tensor(std::vector<std::vector<std::vector<double>>> data1, std::string dev = "cpu", bool requires_grad = false) : requires_grad(requires_grad) {
  device = dev;
  dims.push_back(data1.size());
  dims.push_back(data1[0].size());
  dims.push_back(data1[0][0].size()); 
  for(size_t i = 0; i < data1.size(); ++i) {
    for(size_t j = 0; j < data1[0].size(); ++j) {
      data.insert(data.end(), data1[i][j].begin(), data1[i][j].end()); 
    }
  }
  if (requires_grad) {
    grad = std::make_shared<std::vector<double>>(data.size(), 0.0);
  }
  if(dev == "cuda") {
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
    copyToDevice();
  }
  else {
    d_data = nullptr;
  }
}

void Tensor::to(std::string dev) {
  if(dev != "cpu" && dev != "cuda") {
    throw std::runtime_error("Incorrect deviec name specified");
  }
  if(dev == device) {
    return;
  }
  device = dev;
  if(dev == "cuda") {
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
    copyToDevice();
  }
  else {
    d_data = nullptr;
  }
}

Tensor Tensor::ones(std::vector<size_t> dims, std::string dev, bool requires_grad = false) {
  Tensor ret(dims, dev, requires_grad);
  if(dev == "cpu") {
    for(size_t i = 0;i < ret.data.size();++i)
      ret.data[i] = 1;
  }
  else {
    ret.copyToDevice();
    d_ones<<<NUM_BLOCKS, NUM_THREADS>>>(ret.d_data, ret.data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
  }
  return ret;
}

Tensor Tensor::zeros(std::vector<size_t> dims, std::string dev, bool requires_grad = false) {
  Tensor ret(dims, dev, requires_grad);
  if(dev == "cpu") {
    for(size_t i = 0;i < ret.data.size();++i)
      ret.data[i] = 1;
  }
  else {
    ret.copyToDevice();
    d_ones<<<NUM_BLOCKS, NUM_THREADS>>>(ret.d_data, ret.data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
  }
  return ret;
}

size_t Tensor::index(std::vector<size_t> x) const {
  if(x.size() != dims.size())
    throw std::runtime_error("Mismatched dims in index");
  size_t ret = 0;
  size_t prod = 1;
  for(int i = dims.size() - 1;i >= 0;--i){
    if(x[i] >= dims[i])
      throw std::runtime_error("Index out of bound");
    ret += x[i] * prod;
    prod *= dims[i];
  } 
  return ret;
}

Tensor Tensor::reshape(std::vector<size_t> new_dims) {
  size_t len = 1;
  for(auto d : new_dims)
    len *= d;
  if(len != data.size())
    throw std::runtime_error("Mismatched dims in reshape");
  
  Tensor ret(new_dims, device, requires_grad);
  ret.data = data;

  if (device == "cpu") {
    // Backward pass for CPU
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      ret.backward_fn = [self, ret, new_dims]() mutable {
        if (self->requires_grad) {
          // Ensure gradient has same shape as original tensor
          if (!self->grad) {
            self->grad = std::make_shared<std::vector<double>>(self->data.size(), 0.0);
          }
          
          // Copy gradients back to original shape, maintaining the same values
          for (size_t i = 0; i < self->data.size(); ++i) {
            self->grad->at(i) += ret.grad->at(i);
          }
        }
      };
    }
  } else {
    // Backward pass for GPU
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      int temp1 = num_blocks, temp2 = num_threads;
      ret.backward_fn = [self, ret, temp1, temp2]() mutable {
        if (self->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);

          // Copy gradients to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          self_grad_gpu.copyToDevice();
          grad_gpu.copyToDevice();

          // Kernel to copy gradients back to original shape
          d_reshape<<<temp1, temp2>>>(self_grad_gpu.d_data, grad_gpu.d_data, self->data.size());
          CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

          self_grad_gpu.copyToHost();

          // Accumulate gradients
          for (size_t i = 0; i < self->grad->size(); ++i) {
            self->grad->at(i) += self_grad_gpu.data[i];
          }
        }
      };
    }
  }

  return ret;
}

Tensor Tensor::transpose() const {
  if (dims.size() != 2 && dims.size() != 3) {
    throw std::runtime_error("The tensor must be 2D or batched 2D tensors");
  }

  if(device == "cpu") {
    if(dims.size() == 2) {
      std::vector<size_t> temp = {dims[1], dims[0]};
      Tensor ret(temp, device, requires_grad);
      
      // Forward pass
      for(size_t i = 0; i < dims[0]; ++i) {
        for(size_t j = 0; j < dims[1]; ++j) {
          ret.data[ret.index({j,i})] = data[index({i,j})];
        }
      }

      if (requires_grad) {
        auto self = std::make_shared<Tensor>(*this);
        ret.parents = {self};
        ret.backward_fn = [self, ret]() mutable {
          // Backward pass for 2D
          if (self->requires_grad) {
            for(size_t i = 0; i < self->dims[0]; ++i) {
              for(size_t j = 0; j < self->dims[1]; ++j) {
                self->grad->at(self->index({i,j})) += ret.grad->at(ret.index({j,i}));
              }
            }
          }
        };
      }
      return ret;

    } else {  // dims.size() == 3
      std::vector<size_t> temp = {dims[0], dims[2], dims[1]};
      Tensor ret(temp, device, requires_grad);
      
      // Forward pass
      for(size_t b = 0; b < dims[0]; ++b) {
        for(size_t i = 0; i < dims[1]; ++i) {
          for(size_t j = 0; j < dims[2]; ++j) {
            ret.data[ret.index({b,j,i})] = data[index({b,i,j})];
          }
        }
      }

      if (requires_grad) {
        auto self = std::make_shared<Tensor>(*this);
        ret.parents = {self};
        ret.backward_fn = [self, ret]() mutable {
          // Backward pass for 3D
          if (self->requires_grad) {
            for(size_t b = 0; b < self->dims[0]; ++b) {
              for(size_t i = 0; i < self->dims[1]; ++i) {
                for(size_t j = 0; j < self->dims[2]; ++j) {
                  self->grad->at(self->index({b,i,j})) += ret.grad->at(ret.index({b,j,i}));
                }
              }
            }
          }
        };
      }
      return ret;
    }
  }
  else {
    std::vector<size_t> ret_dims = dims.size() == 2 ? 
      std::vector<size_t>{dims[1], dims[0]} : 
      std::vector<size_t>{dims[0], dims[2], dims[1]};
    
    Tensor ret(ret_dims, device, requires_grad);
    
    // Forward pass
    copyToDevice();
    ret.copyToDevice();
    dim3 threadsPerBlock(NUM_THREADS, NUM_THREADS);
    
    if (dims.size() == 2) {
      dim3 numBlocks((dims[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (dims[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);
      d_transpose_2d<<<numBlocks, threadsPerBlock>>>(d_data, ret.d_data, dims[0], dims[1]);
    } else {
      dim3 numBlocks((dims[2] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (dims[1] + threadsPerBlock.y - 1) / threadsPerBlock.y,
                     dims[0]);
      d_transpose_3d<<<numBlocks, threadsPerBlock>>>(d_data, ret.d_data, dims[0], dims[1], dims[2]);
    }
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      ret.backward_fn = [self, ret, threadsPerBlock]() mutable {
        if (self->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);
          
          // Copy gradients to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          self_grad_gpu.copyToDevice();
          grad_gpu.copyToDevice();
          
          // Backward pass
          if (self->dims.size() == 2) {
            dim3 numBlocks((self->dims[0] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                          (self->dims[1] + threadsPerBlock.y - 1) / threadsPerBlock.y);
            d_transpose_2d<<<numBlocks, threadsPerBlock>>>(
              grad_gpu.d_data, self_grad_gpu.d_data, ret.dims[0], ret.dims[1]
            );
          } else {
            dim3 numBlocks((self->dims[2] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                          (self->dims[1] + threadsPerBlock.y - 1) / threadsPerBlock.y,
                          self->dims[0]);
            d_transpose_3d<<<numBlocks, threadsPerBlock>>>(
              grad_gpu.d_data, self_grad_gpu.d_data, ret.dims[0], ret.dims[2], ret.dims[1]
            );
          }
          CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
          
          self_grad_gpu.copyToHost();
          // Accumulate gradients
          for (size_t i = 0; i < self->grad->size(); ++i) {
            self->grad->at(i) += self_grad_gpu.data[i];
          }
        }
      };
    }
    return ret;
  }
}

Tensor Tensor::neg() const {
  Tensor ret(dims, device, requires_grad);

  if (device == "cpu") {
    // Forward pass
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = -data[i];

    // Backward pass
    if (requires_grad) {  
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      ret.backward_fn = [self, ret]() mutable {
        if (self->requires_grad) {
          for (size_t i = 0; i < self->data.size(); ++i) {
            // Reusing negation logic for backward pass
            self->grad->at(i) += -ret.grad->at(i);
          }
        }
      };
    }
  } else {
    // Forward pass on GPU
    copyToDevice();
    ret.copyToDevice();
    d_neg<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      int temp1 = num_blocks, temp2 = num_threads;
      ret.backward_fn = [self, ret, temp1, temp2]() mutable {
        if (self->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);

          // Copy gradients and input to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          self_grad_gpu.copyToDevice();
          grad_gpu.copyToDevice();

          // Reuse the d_neg kernel for backward pass
          d_neg<<<temp1, temp2>>>(self_grad_gpu.d_data, grad_gpu.d_data, self->data.size());
          CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

          self_grad_gpu.copyToHost();

          // Accumulate gradients
          for (size_t i = 0; i < self->grad->size(); ++i) {
            self->grad->at(i) += self_grad_gpu.data[i];
          }
        }
      };
    }
  }
  return ret;
}

Tensor Tensor::reciprocal() {
  Tensor ret(dims, device, requires_grad);

  if (device == "cpu") {
    // Forward pass
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = 1.0 / data[i];

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      ret.backward_fn = [self, ret]() mutable {
        if (self->requires_grad) {
          for (size_t i = 0; i < self->data.size(); ++i) {
            // Gradient: -1 / (x^2)
            self->grad->at(i) += -ret.grad->at(i) / (self->data[i] * self->data[i]);
          }
        }
      };
    }
  } else {
    // Forward pass on GPU
    copyToDevice();
    ret.copyToDevice();
    d_reciprocal<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      int temp1 = num_blocks, temp2 = num_threads;
      ret.backward_fn = [self, ret, temp1, temp2]() mutable {
        if (self->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);

          // Copy gradients and input to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          self_grad_gpu.copyToDevice();
          grad_gpu.copyToDevice();

          // Kernel for backward pass gradient computation
          d_reciprocal_grad<<<temp1, temp2>>>(self_grad_gpu.d_data, grad_gpu.d_data, 
                                              self->d_data, self->data.size());
          CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

          self_grad_gpu.copyToHost();

          // Accumulate gradients
          for (size_t i = 0; i < self->grad->size(); ++i) {
            self->grad->at(i) += self_grad_gpu.data[i];
          }
        }
      };
    }
  }
  return ret;
}

Tensor Tensor::add(const Tensor &x) const {
  if(dims != x.dims)
    throw std::runtime_error("Mismatched shape in add");
  if(device != x.device) {
    throw std::runtime_error("Expected all tensors to be on the same device.");
  }

  bool needs_grad = requires_grad || x.requires_grad;
  Tensor ret(dims, device, needs_grad);

  if(device == "cpu") {
    // CPU forward pass
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] + x.data[i];

    if (needs_grad) {
      auto self = std::make_shared<Tensor>(*this);
      auto other = std::make_shared<Tensor>(x);
      ret.parents = {self, other};
      ret.backward_fn = [self, other, ret]() mutable {
        // CPU backward pass
        if (self->requires_grad) {
          for (size_t i = 0; i < ret.data.size(); ++i) {
            self->grad->at(i) += ret.grad->at(i);
          }
        }
        if (other->requires_grad) {
          for (size_t i = 0; i < ret.data.size(); ++i) {
            other->grad->at(i) += ret.grad->at(i);
          }
        }
      };
    }
  }
  else {
    // GPU forward pass
    copyToDevice();
    x.copyToDevice();
    ret.copyToDevice();
    d_add<<<num_blocks, num_threads>>>(ret.d_data, d_data, x.d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    if (needs_grad) {
      auto self = std::make_shared<Tensor>(*this);
      auto other = std::make_shared<Tensor>(x);
      ret.parents = {self, other};
      int temp1 = num_blocks, temp2 = num_threads, temp3 = data.size();
      ret.backward_fn = [self, other, ret, temp1, temp2, temp3]() mutable {
        if (self->requires_grad || other->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          
          // Copy gradients to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          grad_gpu.copyToDevice();

          if (self->requires_grad) {
            Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);
            self_grad_gpu.copyToDevice();
            
            // GPU backward pass for self
            d_add_grad<<<temp1, temp2>>>(
              self_grad_gpu.d_data, grad_gpu.d_data, temp3
            );
            CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
            
            self_grad_gpu.copyToHost();
            // Accumulate gradients
            for (size_t i = 0; i < self->grad->size(); ++i) {
              self->grad->at(i) += self_grad_gpu.data[i];
            }
          }

          if (other->requires_grad) {
            Tensor other_grad_gpu = Tensor(other->dims, "cuda", false);
            other_grad_gpu.copyToDevice();
            
            // GPU backward pass for other
            d_add_grad<<<temp1, temp2>>>(
              other_grad_gpu.d_data, grad_gpu.d_data, temp3
            );
            CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
            
            other_grad_gpu.copyToHost();
            // Accumulate gradients
            for (size_t i = 0; i < other->grad->size(); ++i) {
              other->grad->at(i) += other_grad_gpu.data[i];
            }
          }
        }
      };
    }
  }
  return ret;
}
  
Tensor Tensor::subtract(const Tensor &x) const {
  if(dims != x.dims)
    throw std::runtime_error("Mismatched shape in subtract");
  if(device != x.device) {
    throw std::runtime_error("Expected all tensors to be on the same device.");
  }
  return add(x.neg());
}

Tensor Tensor::mult(double x) const {
  Tensor ret(dims, device, requires_grad);

  if (device == "cpu") {
    // Forward pass
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] * x;

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      auto scalar = x; // Capture scalar value for backward pass
      ret.backward_fn = [self, scalar, ret]() mutable {
        if (self->requires_grad) {
          for (size_t i = 0; i < self->data.size(); ++i) {
            // Gradient of multiplication with a scalar: d/dx(x * c) = c
            self->grad->at(i) += ret.grad->at(i) * scalar;
          }
        }
      };
    }
  } else {
    // Forward pass on GPU
    copyToDevice();
    ret.copyToDevice();
    d_scalar_mult<<<num_blocks, num_threads>>>(ret.d_data, d_data, x, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      auto scalar = x;
      int temp1 = num_blocks, temp2 = num_threads;
      ret.backward_fn = [self, scalar, ret, temp1, temp2]() mutable {
        if (self->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);

          // Copy gradients and input to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          self_grad_gpu.copyToDevice();
          grad_gpu.copyToDevice();

          // Compute gradients on GPU
          d_scalar_mult<<<temp1, temp2>>>(
              self_grad_gpu.d_data, grad_gpu.d_data, scalar, self->data.size());
          CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

          self_grad_gpu.copyToHost();

          // Accumulate gradients
          for (size_t i = 0; i < self->grad->size(); ++i) {
            self->grad->at(i) += self_grad_gpu.data[i];
          }
        }
      };
    }
  }
  return ret;
}

Tensor Tensor::elementwise_mult(const Tensor &x) const {
  if(dims != x.dims)
    throw std::runtime_error("Mismatched shape in elementwise_mult");
  if(device != x.device) {
    throw std::runtime_error("Expected all tensors to be on the same device.");
  }
  
  Tensor ret(dims, device, requires_grad || x.requires_grad);
  
  if(device == "cpu") {
    // Forward pass
    for(size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] * x.data[i];

    // Backward pass
    if (requires_grad || x.requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      auto other = std::make_shared<Tensor>(x);
      ret.parents = {self, other};
      
      ret.backward_fn = [self, other, ret]() mutable {
        if (self->requires_grad) {
          for (size_t i = 0; i < self->data.size(); ++i) {
            // Gradient: y * dL/dz
            self->grad->at(i) += other->data[i] * ret.grad->at(i);
          }
        }
        
        if (other->requires_grad) {
          for (size_t i = 0; i < other->data.size(); ++i) {
            // Gradient: x * dL/dz
            other->grad->at(i) += self->data[i] * ret.grad->at(i);
          }
        }
      };
    }
  } else {
    // Forward pass on GPU
    copyToDevice();
    x.copyToDevice();
    ret.copyToDevice();
    
    d_elementwise_mult<<<num_blocks, num_threads>>>(ret.d_data, d_data, x.d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    // Backward pass
    if (requires_grad || x.requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      auto other = std::make_shared<Tensor>(x);
      ret.parents = {self, other};
      int temp1 = num_blocks, temp2 = num_threads;
      
      ret.backward_fn = [self, other, ret, temp1, temp2]() mutable {
        if (self->requires_grad || other->requires_grad) {
          // Gradient computation tensors
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);
          Tensor other_grad_gpu = Tensor(other->dims, "cuda", false);
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_input_gpu = Tensor(self->dims, "cuda", false);
          Tensor other_input_gpu = Tensor(other->dims, "cuda", false);

          // Copy data to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          std::copy(self->data.begin(), self->data.end(), self_input_gpu.data.begin());
          std::copy(other->data.begin(), other->data.end(), other_input_gpu.data.begin());
          
          self_grad_gpu.copyToDevice();
          other_grad_gpu.copyToDevice();
          grad_gpu.copyToDevice();
          self_input_gpu.copyToDevice();
          other_input_gpu.copyToDevice();

          // Kernel for gradient computation
          d_elementwise_mult_grad<<<temp1, temp2>>>(
            self_grad_gpu.d_data, other_grad_gpu.d_data, 
            grad_gpu.d_data, 
            self_input_gpu.d_data, other_input_gpu.d_data, 
            self->data.size()
          );
          
          CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

          self_grad_gpu.copyToHost();
          other_grad_gpu.copyToHost();

          // Accumulate gradients
          if (self->requires_grad) {
            for (size_t i = 0; i < self->grad->size(); ++i) {
              self->grad->at(i) += self_grad_gpu.data[i];
            }
          }
          
          if (other->requires_grad) {
            for (size_t i = 0; i < other->grad->size(); ++i) {
              other->grad->at(i) += other_grad_gpu.data[i];
            }
          }
        }
      };
    }
  }
  return ret;
}

Tensor Tensor::pow(double x) const {
  Tensor ret(dims, device, requires_grad);
  
  if(device == "cpu") {
    // Forward pass
    for(size_t i = 0; i < data.size(); ++i) {
      ret.data[i] = std::pow(data[i], x);
    }

    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      auto power = x; // Capture the power value
      ret.backward_fn = [self, power, ret]() mutable {
        // Backward pass using power rule: d/dx(x^n) = n * x^(n-1)
        if (self->requires_grad) {
          for (size_t i = 0; i < self->data.size(); ++i) {
            self->grad->at(i) += ret.grad->at(i) * power * std::pow(self->data[i], power - 1);
          }
        }
      };
    }
  }
  else {
    // Forward pass on GPU
    copyToDevice();
    ret.copyToDevice();
    d_pow<<<num_blocks, num_threads>>>(ret.d_data, d_data, x, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      auto power = x;
      int temp1 = num_blocks, temp2 = num_threads;
      ret.backward_fn = [self, power, ret, temp1, temp2]() mutable {
        if (self->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);
          
          // Copy input tensor and gradients to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          self_grad_gpu.copyToDevice();
          grad_gpu.copyToDevice();
          self->copyToDevice();
          
          // Compute gradients on GPU
          d_pow_grad<<<temp1, temp2>>>(
            self_grad_gpu.d_data, grad_gpu.d_data, self->d_data, power, self->data.size()
          );
          CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
          
          self_grad_gpu.copyToHost();
          // Accumulate gradients
          for (size_t i = 0; i < self->grad->size(); ++i) {
            self->grad->at(i) += self_grad_gpu.data[i];
          }
        }
      };
    }
  }
  return ret;
}

Tensor Tensor::relu() {
  Tensor ret(dims, device, requires_grad);
  
  if (device == "cpu") {
    // Forward pass
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] > 0 ? data[i] : 0;

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      ret.backward_fn = [self, ret]() mutable {
        if (self->requires_grad) {
          for (size_t i = 0; i < self->data.size(); ++i) {
            // Gradient: 1 if x > 0, 0 otherwise
            self->grad->at(i) += (self->data[i] > 0 ? ret.grad->at(i) : 0);
          }
        }
      };
    }
  } else {
    // Forward pass on GPU
    copyToDevice();
    ret.copyToDevice();
    d_relu<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      int temp1 = num_blocks, temp2 = num_threads;
      
      ret.backward_fn = [self, ret, temp1, temp2]() mutable {
        if (self->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);
          Tensor self_input_gpu = Tensor(self->dims, "cuda", false);

          // Copy data to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          std::copy(self->data.begin(), self->data.end(), self_input_gpu.data.begin());
          
          self_grad_gpu.copyToDevice();
          grad_gpu.copyToDevice();
          self_input_gpu.copyToDevice();

          // Kernel for backward pass gradient computation
          d_relu_grad<<<temp1, temp2>>>(
            self_grad_gpu.d_data, 
            grad_gpu.d_data, 
            self_input_gpu.d_data, 
            self->data.size()
          );
          
          CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

          self_grad_gpu.copyToHost();

          // Accumulate gradients
          for (size_t i = 0; i < self->grad->size(); ++i) {
            self->grad->at(i) += self_grad_gpu.data[i];
          }
        }
      };
    }
  }
  return ret;
}

Tensor Tensor::binarilize() {
  Tensor ret(dims, device, requires_grad);
  
  if (device == "cpu") {
    // Forward pass
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] > 0 ? 1 : 0;

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      ret.backward_fn = [self, ret]() mutable {
        if (self->requires_grad) {
          for (size_t i = 0; i < self->data.size(); ++i) {
            // Gradient is 0 since binarilize is not differentiable
            self->grad->at(i) += 0;
          }
        }
      };
    }
  } else {
    // Forward pass on GPU
    copyToDevice();
    ret.copyToDevice();
    d_binarilize<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      int temp1 = num_blocks, temp2 = num_threads;
      
      ret.backward_fn = [self, ret, temp1, temp2]() mutable {
        if (self->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);

          // Zero out gradients since binarilize is not differentiable
          std::fill(self_grad_gpu.data.begin(), self_grad_gpu.data.end(), 0);
          
          self_grad_gpu.copyToDevice();

          // Accumulate gradients (which will be zero)
          for (size_t i = 0; i < self->grad->size(); ++i) {
            self->grad->at(i) += 0;
          }
        }
      };
    }
  }
  return ret;
}

Tensor Tensor::exp() {
  Tensor ret(dims, device, requires_grad);
  
  if (device == "cpu") {
    // Forward pass
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = std::exp(data[i]);

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      ret.backward_fn = [self, ret]() mutable {
        if (self->requires_grad) {
          for (size_t i = 0; i < self->data.size(); ++i) {
            // Gradient: exp(x) * dL/dz
            self->grad->at(i) += ret.data[i] * ret.grad->at(i);
          }
        }
      };
    }
  } else {
    // Forward pass on GPU
    copyToDevice();
    ret.copyToDevice();
    d_exp<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      int temp1 = num_blocks, temp2 = num_threads;
      
      ret.backward_fn = [self, ret, temp1, temp2]() mutable {
        if (self->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);
          Tensor self_input_gpu = Tensor(self->dims, "cuda", false);

          // Copy data to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          std::copy(ret.data.begin(), ret.data.end(), self_input_gpu.data.begin());
          
          self_grad_gpu.copyToDevice();
          grad_gpu.copyToDevice();
          self_input_gpu.copyToDevice();

          // Kernel for backward pass gradient computation
          d_exp_grad<<<temp1, temp2>>>(
            self_grad_gpu.d_data, 
            grad_gpu.d_data, 
            self_input_gpu.d_data, 
            self->data.size()
          );
          
          CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

          self_grad_gpu.copyToHost();

          // Accumulate gradients
          for (size_t i = 0; i < self->grad->size(); ++i) {
            self->grad->at(i) += self_grad_gpu.data[i];
          }
        }
      };
    }
  }
  return ret;
}

Tensor Tensor::matmul(Tensor x) const {
  if(device != x.device) {
    throw std::runtime_error("Expected all tensors to be on the same device.");
  }
  if(x.dims.size() != 2){
    throw std::runtime_error("The right operand of matmul must be 2D tensors");
  }
  if(dims.size() != 2 && dims.size() != 3){
    throw std::runtime_error("The left operand of matmul must be 2D tensors or batched 2D tensors");
  }
  if(dims[dims.size() - 1] != x.dims[0]){
    throw std::runtime_error("Mismatched matmul matrix dimensions");
  }

  bool needs_grad = requires_grad || x.requires_grad;
  
  if(device == "cpu") {
    if(dims.size() == 2){
      std::vector<size_t> temp = {dims[0], x.dims[1]};
      Tensor ret(temp, device, needs_grad);
      
      // Forward pass for 2D
      for(size_t i = 0; i < dims[0]; ++i){
        for(size_t j = 0; j < x.dims[1]; ++j){
          for(size_t k = 0; k < dims[1]; ++k){
            ret.data[ret.index({i,j})] += data[index({i,k})] * x.data[x.index({k,j})];
          }
        }
      }

      if (needs_grad) {
        auto self = std::make_shared<Tensor>(*this);
        auto other = std::make_shared<Tensor>(x);
        ret.parents = {self, other};
        ret.backward_fn = [self, other, ret]() mutable {
          // For left matrix A: dL/dA = dL/dC * B^T
          if (self->requires_grad) {
            for(size_t i = 0; i < self->dims[0]; ++i) {
              for(size_t k = 0; k < self->dims[1]; ++k) {
                double grad_sum = 0;
                for(size_t j = 0; j < other->dims[1]; ++j) {
                  grad_sum += ret.grad->at(ret.index({i,j})) * other->data[other->index({k,j})];
                }
                self->grad->at(self->index({i,k})) += grad_sum;
              }
            }
          }
          
          // For right matrix B: dL/dB = A^T * dL/dC
          if (other->requires_grad) {
            for(size_t k = 0; k < other->dims[0]; ++k) {
              for(size_t j = 0; j < other->dims[1]; ++j) {
                double grad_sum = 0;
                for(size_t i = 0; i < self->dims[0]; ++i) {
                  grad_sum += self->data[self->index({i,k})] * ret.grad->at(ret.index({i,j}));
                }
                other->grad->at(other->index({k,j})) += grad_sum;
              }
            }
          }
        };
      }
      return ret;
    } else {
      // 3D (batched) case
      std::vector<size_t> temp = {dims[0], dims[1], x.dims[1]};
      Tensor ret(temp, device, needs_grad);
      
      // Forward pass for 3D
      for(size_t b = 0; b < dims[0]; ++b){
        for(size_t i = 0; i < dims[1]; ++i){
          for(size_t j = 0; j < x.dims[1]; ++j){
            for(size_t k = 0; k < dims[2]; ++k){
              ret.data[ret.index({b,i,j})] += data[index({b,i,k})] * x.data[x.index({k,j})];
            }
          }
        }
      }

      if (needs_grad) {
        auto self = std::make_shared<Tensor>(*this);
        auto other = std::make_shared<Tensor>(x);
        ret.parents = {self, other};
        ret.backward_fn = [self, other, ret]() mutable {
          // For left matrix A: dL/dA = dL/dC * B^T (for each batch)
          if (self->requires_grad) {
            for(size_t b = 0; b < self->dims[0]; ++b) {
              for(size_t i = 0; i < self->dims[1]; ++i) {
                for(size_t k = 0; k < self->dims[2]; ++k) {
                  double grad_sum = 0;
                  for(size_t j = 0; j < other->dims[1]; ++j) {
                    grad_sum += ret.grad->at(ret.index({b,i,j})) * other->data[other->index({k,j})];
                  }
                  self->grad->at(self->index({b,i,k})) += grad_sum;
                }
              }
            }
          }
          
          // For right matrix B: sum(A_b^T * dL/dC_b) over all batches
          if (other->requires_grad) {
            for(size_t k = 0; k < other->dims[0]; ++k) {
              for(size_t j = 0; j < other->dims[1]; ++j) {
                double grad_sum = 0;
                for(size_t b = 0; b < self->dims[0]; ++b) {
                  for(size_t i = 0; i < self->dims[1]; ++i) {
                    grad_sum += self->data[self->index({b,i,k})] * ret.grad->at(ret.index({b,i,j}));
                  }
                }
                other->grad->at(other->index({k,j})) += grad_sum;
              }
            }
          }
        };
      }
      return ret;
    }
  } else {
    std::vector<size_t> temp;
    if (dims.size() == 2) {
      temp = {dims[0], x.dims[1]};
    } else {
      temp = {dims[0], dims[1], x.dims[1]};
    }
    Tensor ret(temp, device, needs_grad);

    copyToDevice();
    x.copyToDevice();
    ret.copyToDevice();
    
    if (dims.size() == 2) {
      dim3 threadsPerBlock(16, 16);
      dim3 numBlocks((x.dims[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (dims[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);
      
      d_matmul_2d<<<numBlocks, threadsPerBlock>>>(d_data, x.d_data, ret.d_data, 
                                                 dims[0], x.dims[1], dims[1]);
    } else {
      dim3 threadsPerBlock(16, 16);
      dim3 numBlocksBatched((x.dims[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (dims[1] + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           dims[0]);
      
      d_matmul_3d<<<numBlocksBatched, threadsPerBlock>>>(d_data, x.d_data, ret.d_data, 
                                                        dims[0], dims[1], x.dims[1], dims[2]);
    }

    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    if (needs_grad) {
      auto self = std::make_shared<Tensor>(*this);
      auto other = std::make_shared<Tensor>(x);
      ret.parents = {self, other};
      ret.backward_fn = [self, other, ret]() mutable {
        if (self->requires_grad || other->requires_grad) {
          // Allocate temporary tensors for gradients on GPU
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);
          Tensor other_grad_gpu = Tensor(other->dims, "cuda", false);
          Tensor ret_grad_gpu = Tensor(ret.dims, "cuda", false);

          // Copy gradients to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), ret_grad_gpu.data.begin());
          ret_grad_gpu.copyToDevice();

          if (self->requires_grad) {
            self_grad_gpu.copyToDevice();
            if (self->dims.size() == 2) {
              dim3 threadsPerBlock(16, 16);
              dim3 numBlocks((self->dims[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                            (self->dims[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);
              
              d_matmul_grad_left_2d<<<numBlocks, threadsPerBlock>>>(
                ret_grad_gpu.d_data, other->d_data, self_grad_gpu.d_data,
                self->dims[0], self->dims[1], other->dims[1]
              );
            } else {
              dim3 threadsPerBlock(16, 16);
              dim3 numBlocks((self->dims[2] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                            (self->dims[1] + threadsPerBlock.y - 1) / threadsPerBlock.y,
                            self->dims[0]);
              
              d_matmul_grad_left_3d<<<numBlocks, threadsPerBlock>>>(
                ret_grad_gpu.d_data, other->d_data, self_grad_gpu.d_data,
                self->dims[0], self->dims[1], self->dims[2], other->dims[1]
              );
            }
            CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
            self_grad_gpu.copyToHost();
            
            // Accumulate gradients
            for (size_t i = 0; i < self->grad->size(); ++i) {
              self->grad->at(i) += self_grad_gpu.data[i];
            }
          }

          if (other->requires_grad) {
            other_grad_gpu.copyToDevice();
            if (self->dims.size() == 2) {
              dim3 threadsPerBlock(16, 16);
              dim3 numBlocks((other->dims[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                            (other->dims[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);
              
              d_matmul_grad_right_2d<<<numBlocks, threadsPerBlock>>>(
                self->d_data, ret_grad_gpu.d_data, other_grad_gpu.d_data,
                self->dims[0], other->dims[0], other->dims[1]
              );
            } else {
              dim3 threadsPerBlock(16, 16);
              dim3 numBlocks((other->dims[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                            (other->dims[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);
              
              d_matmul_grad_right_3d<<<numBlocks, threadsPerBlock>>>(
                self->d_data, ret_grad_gpu.d_data, other_grad_gpu.d_data,
                self->dims[0], self->dims[1], other->dims[0], other->dims[1]
              );
            }
            CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
            other_grad_gpu.copyToHost();
            
            // Accumulate gradients
            for (size_t i = 0; i < other->grad->size(); ++i) {
              other->grad->at(i) += other_grad_gpu.data[i];
            }
          }
        }
      };
    }
    return ret;
  }
}


void Tensor::print(){
  if(dims.size() == 1) {
    printf("[");
    for(auto x : data)
      printf("%s, ",std::to_string(x).c_str());
    printf("]\n");
  }
  else if(dims.size() == 2) {
    size_t index = 0;
    printf("[");
    for(auto x : data) {
      if(index % dims[1] == 0) {
        printf("[");
      }
      printf("%s, ",std::to_string(x).c_str());
      if(index % dims[1] == dims[1] - 1) {
        printf("],\n");
      }
      index++;
    }
    printf("]\n");
  }
  else if(dims.size() == 3) {
    size_t index = 0;
    printf("[\n");
    for(size_t i = 0; i < dims[0]; i++) {
        printf("  [\n");
        for(size_t j = 0; j < dims[1]; j++) {
            printf("    [");
            for(size_t k = 0; k < dims[2]; k++) {
                printf("%s", std::to_string(data[index]).c_str());
                if(k < dims[2] - 1) {
                    printf(", ");
                }
                index++;
            }
            printf("],\n");
        }
        printf("  ],\n");
    }
    printf("]\n");
  }
  else{
    printf("Invalid dimensions\n");
  }
}

std::vector<double> Tensor::get_data()  const{
  return data;
}

std::vector<size_t> Tensor::get_dims() const{
  return dims;
}

// Overloaded operators for tensor-tensor operations
// Tensor Tensor::operator+(const Tensor& other) const {
//   if (this->device != other.device) {
//     throw std::runtime_error("Tensors must be on the same device for addition.");
//   }
//   // Convert (1) or (1,1) dimensional vectors to scalars and then broadcast them to required dimesions of this
//   if((other.dims.size() == 1 && other.dims[0] == 1) || (other.dims.size() == 2 && other.dims[0] == 1  && other.dims[1] == 1)) {
//     std::vector<double> temp(this->data.size(), other.data[0]);
//     Tensor t(this->dims, this->device);
//     t.data = temp;
//     return this->add(t);
//   } 
//   if (this->dims != other.dims) {
//     throw std::invalid_argument("Tensors must have the same dimensions for addition.");
//   }
//   return this->add(other); 
// }

// Kernel for GPU broadcasting
__global__ void broadcast_kernel(double* output, const double* input,
                               const size_t* old_dims, const size_t* new_dims,
                               size_t old_rank, size_t new_rank,
                               const size_t* old_strides, size_t total_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    size_t old_idx = 0;
    size_t temp = idx;

    for (int dim = 0; dim < new_rank; ++dim) {
        size_t coord = temp;
        for (int j = dim + 1; j < new_rank; ++j) {
            coord /= new_dims[j];
        }
        coord %= new_dims[dim];
        
        if (dim >= new_rank - old_rank) {
            size_t old_dim = dim - (new_rank - old_rank);
            if (old_dims[old_dim] > 1) {
                old_idx += (coord % old_dims[old_dim]) * old_strides[old_dim];
            }
        }
    }
    output[idx] = input[old_idx];
}

// Kernel for GPU gradient accumulation
__global__ void broadcast_backward_kernel(double* grad_out, const double* grad_in,
                                        const size_t* old_dims, const size_t* new_dims,
                                        size_t old_rank, size_t new_rank,
                                        const size_t* old_strides, size_t total_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    size_t old_idx = 0;
    size_t temp = idx;

    for (int dim = 0; dim < new_rank; ++dim) {
        size_t coord = temp;
        for (int j = dim + 1; j < new_rank; ++j) {
            coord /= new_dims[j];
        }
        coord %= new_dims[dim];
        
        if (dim >= new_rank - old_rank) {
            size_t old_dim = dim - (new_rank - old_rank);
            if (old_dims[old_dim] > 1) {
                old_idx += (coord % old_dims[old_dim]) * old_strides[old_dim];
            }
        }
    }
    atomicAdd(&grad_out[old_idx], grad_in[idx]);
}

std::vector<size_t> calculate_broadcast_shape(const std::vector<size_t>& a_dims, 
                                            const std::vector<size_t>& b_dims) {
    size_t max_dims = std::max(a_dims.size(), b_dims.size());
    std::vector<size_t> broadcast_dims(max_dims);
    
    for (int i = 0; i < max_dims; i++) {
        size_t a_dim = i < a_dims.size() ? 
            a_dims[a_dims.size() - 1 - i] : 1;
        size_t b_dim = i < b_dims.size() ? 
            b_dims[b_dims.size() - 1 - i] : 1;
        
        if (a_dim == b_dim) {
            broadcast_dims[max_dims - 1 - i] = a_dim;
        } else if (a_dim == 1) {
            broadcast_dims[max_dims - 1 - i] = b_dim;
        } else if (b_dim == 1) {
            broadcast_dims[max_dims - 1 - i] = a_dim;
        } else {
            throw std::runtime_error("Incompatible shapes for broadcasting");
        }
    }
    return broadcast_dims;
}

// Function to broadcast tensor to new shape
Tensor broadcast_to(const Tensor& t, const std::vector<size_t>& new_shape) {
    if (t.dims == new_shape) {
        return t;
    }

    Tensor result(new_shape, t.device, t.requires_grad);
    size_t total_size = std::accumulate(new_shape.begin(), new_shape.end(), 
                                      1UL, std::multiplies<size_t>());

    // Calculate strides for the original tensor
    std::vector<size_t> old_strides(t.dims.size(), 1);
    for (int i = t.dims.size() - 2; i >= 0; --i) {
        old_strides[i] = old_strides[i + 1] * t.dims[i + 1];
    }

    if (t.device == "cpu") {
        // CPU broadcasting
        for (size_t i = 0; i < total_size; ++i) {
            size_t old_idx = 0;
            size_t temp = i;
            
            // Calculate source index
            for (int dim = 0; dim < new_shape.size(); ++dim) {
                size_t coord = temp / std::accumulate(new_shape.begin() + dim + 1, new_shape.end(), 
                                                    1UL, std::multiplies<size_t>());
                temp %= std::accumulate(new_shape.begin() + dim + 1, new_shape.end(), 
                                      1UL, std::multiplies<size_t>());
                
                if (dim >= new_shape.size() - t.dims.size()) {
                    size_t old_dim = dim - (new_shape.size() - t.dims.size());
                    if (t.dims[old_dim] > 1) {
                        old_idx += (coord % t.dims[old_dim]) * old_strides[old_dim];
                    }
                }
            }
            result.data[i] = t.data[old_idx];
        }
    } else {
        // GPU broadcasting - first copy to device
        t.copyToDevice();
        result.copyToDevice();

        size_t* d_old_dims;
        size_t* d_new_dims;
        size_t* d_old_strides;
        
        CHECK_CUDA_ERRORS(cudaMalloc(&d_old_dims, t.dims.size() * sizeof(size_t)));
        CHECK_CUDA_ERRORS(cudaMalloc(&d_new_dims, new_shape.size() * sizeof(size_t)));
        CHECK_CUDA_ERRORS(cudaMalloc(&d_old_strides, old_strides.size() * sizeof(size_t)));
        
        CHECK_CUDA_ERRORS(cudaMemcpy(d_old_dims, t.dims.data(), 
                                    t.dims.size() * sizeof(size_t), 
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_new_dims, new_shape.data(), 
                                    new_shape.size() * sizeof(size_t), 
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_old_strides, old_strides.data(), 
                                    old_strides.size() * sizeof(size_t), 
                                    cudaMemcpyHostToDevice));
        
        // Launch kernel for broadcasting
        broadcast_kernel<<<(total_size + 255) / 256, 256>>>(
            result.d_data, t.d_data,
            d_old_dims, d_new_dims,
            t.dims.size(), new_shape.size(),
            d_old_strides, total_size
        );
        CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
        
        // Clean up GPU memory
        CHECK_CUDA_ERRORS(cudaFree(d_old_dims));
        CHECK_CUDA_ERRORS(cudaFree(d_new_dims));
        CHECK_CUDA_ERRORS(cudaFree(d_old_strides));
        
        result.copyToHost();
    }

    // If gradient is required, set up backward function
    if (t.requires_grad) {
        auto self = std::make_shared<Tensor>(t);
        result.parents = {self};
        std::vector<size_t> old_strides_capture = old_strides;
        std::vector<size_t> new_shape_capture = new_shape;
        
        result.backward_fn = [self, new_shape_capture, old_strides_capture, result]() mutable {
            if (!self->grad) {
                self->grad = std::make_shared<std::vector<double>>(self->data.size(), 0.0f);
            }
            
            size_t total_size = std::accumulate(new_shape_capture.begin(), 
                                              new_shape_capture.end(), 
                                              1UL, std::multiplies<size_t>());
            
            if (self->device == "cpu") {
                for (size_t i = 0; i < total_size; ++i) {
                    size_t old_idx = 0;
                    size_t temp = i;
                    
                    for (int dim = 0; dim < new_shape_capture.size(); ++dim) {
                        size_t coord = temp / std::accumulate(new_shape_capture.begin() + dim + 1, 
                                                            new_shape_capture.end(), 
                                                            1UL, std::multiplies<size_t>());
                        temp %= std::accumulate(new_shape_capture.begin() + dim + 1, 
                                              new_shape_capture.end(), 
                                              1UL, std::multiplies<size_t>());
                        
                        if (dim >= new_shape_capture.size() - self->dims.size()) {
                            size_t old_dim = dim - (new_shape_capture.size() - self->dims.size());
                            if (self->dims[old_dim] > 1) {
                                old_idx += (coord % self->dims[old_dim]) * old_strides_capture[old_dim];
                            }
                        }
                    }
                    self->grad->at(old_idx) += result.grad->at(i);
                }
            } else {
                // GPU gradient accumulation with device memory
                Tensor grad_gpu = Tensor(self->dims, "cuda", false);
                grad_gpu.copyToDevice();
                
                // Allocate and copy arrays to GPU
                size_t* d_old_dims;
                size_t* d_new_dims;
                size_t* d_old_strides;
                double* d_result_grad_data;
                
                CHECK_CUDA_ERRORS(cudaMalloc(&d_old_dims, 
                                           self->dims.size() * sizeof(size_t)));
                CHECK_CUDA_ERRORS(cudaMalloc(&d_new_dims, 
                                           new_shape_capture.size() * sizeof(size_t)));
                CHECK_CUDA_ERRORS(cudaMalloc(&d_old_strides, 
                                           old_strides_capture.size() * sizeof(size_t)));
                CHECK_CUDA_ERRORS(cudaMalloc(&d_result_grad_data, 
                                           result.grad->size() * sizeof(double)));
                
                CHECK_CUDA_ERRORS(cudaMemcpy(d_old_dims, self->dims.data(), 
                                           self->dims.size() * sizeof(size_t), 
                                           cudaMemcpyHostToDevice));
                CHECK_CUDA_ERRORS(cudaMemcpy(d_new_dims, new_shape_capture.data(), 
                                           new_shape_capture.size() * sizeof(size_t), 
                                           cudaMemcpyHostToDevice));
                CHECK_CUDA_ERRORS(cudaMemcpy(d_old_strides, old_strides_capture.data(), 
                                           old_strides_capture.size() * sizeof(size_t), 
                                           cudaMemcpyHostToDevice));
                CHECK_CUDA_ERRORS(cudaMemcpy(d_result_grad_data, result.grad->data(), 
                                           result.grad->size() * sizeof(double), 
                                           cudaMemcpyHostToDevice));
                
                broadcast_backward_kernel<<<(total_size + 255) / 256, 256>>>(
                    grad_gpu.d_data, d_result_grad_data,
                    d_old_dims, d_new_dims,
                    self->dims.size(), new_shape_capture.size(),
                    d_old_strides, total_size
                );
                CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
                
                // Clean up GPU memory
                CHECK_CUDA_ERRORS(cudaFree(d_old_dims));
                CHECK_CUDA_ERRORS(cudaFree(d_new_dims));
                CHECK_CUDA_ERRORS(cudaFree(d_old_strides));
                CHECK_CUDA_ERRORS(cudaFree(d_result_grad_data));
                
                grad_gpu.copyToHost();
                for (size_t i = 0; i < self->grad->size(); ++i) {
                    self->grad->at(i) += grad_gpu.data[i];
                }
            }
        };
    }
    
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    // If shapes are identical, use original add directly
    if (this->dims == other.dims) {
        return this->add(other);
    }
    
    // Calculate broadcast shape
    std::vector<size_t> broadcast_shape = calculate_broadcast_shape(this->dims, other.dims);
    
    // Broadcast both tensors to the new shape
    Tensor broadcast_a = broadcast_to((*this), broadcast_shape);
    Tensor broadcast_b = broadcast_to(other, broadcast_shape);
    
    // Use original add implementation with broadcasted tensors
    return broadcast_a.add(broadcast_b);
}

Tensor Tensor::operator-(const Tensor& other) const {
  if (this->device != other.device) {
    throw std::runtime_error("Tensors must be on the same device for addition.");
  }
  // Convert (1) or (1,1) dimensional vectors to scalars and then broadcast them to required dimesions of this
  if((other.dims.size() == 1 && other.dims[0] == 1) || (other.dims.size() == 2 && other.dims[0] == 1  && other.dims[1] == 1)) {
    std::vector<double> temp(this->data.size(), other.data[0]);
    Tensor t(this->dims, this->device);
    t.data = temp;
    return this->subtract(t);
  } 
  if (this->dims != other.dims) {
    throw std::invalid_argument("Tensors must have the same dimensions for addition.");
  }
  return this->subtract(other);
}

Tensor Tensor::operator*(const Tensor& other) const {
  if (this->dims != other.dims) {
      throw std::invalid_argument("Tensors must have the same dimensions for element-wise multiplication.");
  }

  if (this->device == other.device) {
    return this->elementwise_mult(other);
  }  
  throw std::runtime_error("Tensors must be on the same device for addition.");
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (this->dims != other.dims) {
        throw std::invalid_argument("Tensors must have the same dimensions for element-wise division.");
    }

    std::vector<double> result_data(this->data.size());
    for (size_t i = 0; i < this->data.size(); ++i) {
        if (other.data[i] == 0) {
            throw std::invalid_argument("Division by zero.");
        }
        result_data[i] = this->data[i] / other.data[i];
    }
    return Tensor(result_data, this->device);
}

// Scalar operations
Tensor Tensor::operator+(double scalar) const {
  std::vector<double> temp(this->data.size(), scalar);
  Tensor t(this->dims, this->device);
  t.data = temp;
  return this->add(t);
}

Tensor Tensor::operator-(double scalar) const {
  std::vector<double> temp(this->data.size(), scalar);
  Tensor t(this->dims, this->device);
  t.data = temp;
  return this->subtract(t);
}

Tensor Tensor::operator*(double scalar) const {
  return this->mult(scalar);
}

Tensor Tensor::operator/(double scalar) const {
    if (scalar == 0) {
        throw std::invalid_argument("Division by zero.");
    }

    std::vector<double> result_data(this->data.size());
    for (size_t i = 0; i < this->data.size(); ++i) {
        result_data[i] = this->data[i] / scalar;
    }
    return Tensor(result_data, this->device);
}

// // In-place operations
// Tensor& Tensor::operator+=(const Tensor& other) {
//     if (this->dims != other.dims) {
//         throw std::invalid_argument("Tensors must have the same dimensions for addition.");
//     }

//     for (size_t i = 0; i < this->data.size(); ++i) {
//         this->data[i] += other.data[i];
//     }
//     return *this;
// }

// Tensor& Tensor::operator-=(const Tensor& other) {
//     if (this->dims != other.dims) {
//         throw std::invalid_argument("Tensors must have the same dimensions for subtraction.");
//     }

//     for (size_t i = 0; i < this->data.size(); ++i) {
//         this->data[i] -= other.data[i];
//     }
//     return *this;
// }

// Tensor& Tensor::operator*=(const Tensor& other) {
//     if (this->dims != other.dims) {
//         throw std::invalid_argument("Tensors must have the same dimensions for element-wise multiplication.");
//     }

//     for (size_t i = 0; i < this->data.size(); ++i) {
//         this->data[i] *= other.data[i];
//     }
//     return *this;
// }

// Tensor& Tensor::operator/=(const Tensor& other) {
//     if (this->dims != other.dims) {
//         throw std::invalid_argument("Tensors must have the same dimensions for element-wise division.");
//     }

//     for (size_t i = 0; i < this->data.size(); ++i) {
//         if (other.data[i] == 0) {
//             throw std::invalid_argument("Division by zero.");
//         }
//         this->data[i] /= other.data[i];
//     }
//     return *this;
// }

// // Scalar in-place operations
// Tensor& Tensor::operator+=(double scalar) {
//     for (size_t i = 0; i < this->data.size(); ++i) {
//         this->data[i] += scalar;
//     }
//     return *this;
// }

// Tensor& Tensor::operator-=(double scalar) {
//     for (size_t i = 0; i < this->data.size(); ++i) {
//         this->data[i] -= scalar;
//     }
//     return *this;
// }

// Tensor& Tensor::operator*=(double scalar) {
//     for (size_t i = 0; i < this->data.size(); ++i) {
//         this->data[i] *= scalar;
//     }
//     return *this;
// }

// Tensor& Tensor::operator/=(double scalar) {
//     if (scalar == 0) {
//         throw std::invalid_argument("Division by zero.");
//     }

//     for (size_t i = 0; i < this->data.size(); ++i) {
//         this->data[i] /= scalar;
//     }
//     return *this;
// }

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(";
    os << "device=" << tensor.device << ", ";
    os << "dims=[";
    for (size_t i = 0; i < tensor.dims.size(); ++i) {
        os << tensor.dims[i];
        if (i < tensor.dims.size() - 1) os << ", ";
    }
    os << "], data=[";
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        os << tensor.data[i];
        if (i < tensor.data.size() - 1) os << ", ";
    }
    os << "])";
    return os;
}

std::string Tensor::toString() const {
  std::ostringstream oss;
  if(dims.size() == 1) {
    oss << "[";
    for(auto x : data)
      oss << std::to_string(x) << ", ";
    oss << "]\n";
  }
  else if(dims.size() == 2) {
    size_t index = 0;
    oss << "[";
    for(auto x : data) {
      if(index % dims[1] == 0) {
        oss << "[";
      }
      oss << std::to_string(x) << ", ";
      if(index % dims[1] == dims[1] - 1) {
        oss << "],\n";
      }
      index++;
    }
    oss << "]\n";
  }
  else if(dims.size() == 3) {
    size_t index = 0;
    oss << "[\n";
    for(size_t i = 0; i < dims[0]; i++) {
        oss << "  [\n";
        for(size_t j = 0; j < dims[1]; j++) {
            oss << "    [";
            for(size_t k = 0; k < dims[2]; k++) {
                oss << std::to_string(data[index]) << ", ";
                if(k < dims[2] - 1) {
                    oss << ", ";
                }
                index++;
            }
            oss << "],\n";
        }
        oss << "  ],\n";
    }
    oss << "]\n";
  }
  return oss.str();
}

Tensor Tensor::rand(std::vector<size_t> dims, std::string dev, bool requires_grad = false) {
    size_t total_elements = 1;
    for (size_t dim : dims) total_elements *= dim;

    std::vector<double> random_data(total_elements);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (double& val : random_data) {
        val = dis(gen);
    }

    return Tensor(random_data, dev,requires_grad).reshape(dims);
}

Tensor Tensor::randn(std::vector<size_t> dims, std::string dev, bool requires_grad = false) {
    size_t total_elements = 1;
    for (size_t dim : dims) total_elements *= dim;

    std::vector<double> random_data(total_elements);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0); // Mean = 0, Stddev = 1

    for (double& val : random_data) {
        val = dis(gen);
    }

    return Tensor(random_data, dev, requires_grad).reshape(dims);
}

Tensor Tensor::log() const {
  Tensor ret(dims, device, requires_grad);
  
  if (device == "cpu") {
    // Forward pass
    for (size_t i = 0; i < data.size(); ++i) {
      if (data[i] <= 0) {
        throw std::domain_error("Logarithm undefined for non-positive values");
      }
      ret.data[i] = std::log(data[i]);
    }

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      ret.backward_fn = [self, ret]() mutable {
        if (self->requires_grad) {
          for (size_t i = 0; i < self->data.size(); ++i) {
            // Gradient: 1/x * dL/dz
            self->grad->at(i) += ret.grad->at(i) / self->data[i];
          }
        }
      };
    }
  } else {
    // Forward pass on GPU
    copyToDevice();
    ret.copyToDevice();
    d_log<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();

    // Backward pass
    if (requires_grad) {
      auto self = std::make_shared<Tensor>(*this);
      ret.parents = {self};
      int temp1 = num_blocks, temp2 = num_threads;
      
      ret.backward_fn = [self, ret, temp1, temp2]() mutable {
        if (self->requires_grad) {
          // Create temporary GPU tensors for gradients
          Tensor grad_gpu = Tensor(ret.dims, "cuda", false);
          Tensor self_grad_gpu = Tensor(self->dims, "cuda", false);
          Tensor self_input_gpu = Tensor(self->dims, "cuda", false);

          // Copy data to GPU
          std::copy(ret.grad->begin(), ret.grad->end(), grad_gpu.data.begin());
          std::copy(self->data.begin(), self->data.end(), self_input_gpu.data.begin());
          
          self_grad_gpu.copyToDevice();
          grad_gpu.copyToDevice();
          self_input_gpu.copyToDevice();

          // Kernel for backward pass gradient computation
          d_log_grad<<<temp1, temp2>>>(
            self_grad_gpu.d_data, 
            grad_gpu.d_data, 
            self_input_gpu.d_data, 
            self->data.size()
          );
          
          CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

          self_grad_gpu.copyToHost();

          // Accumulate gradients
          for (size_t i = 0; i < self->grad->size(); ++i) {
            self->grad->at(i) += self_grad_gpu.data[i];
          }
        }
      };
    }
  }
  return ret;
}

Tensor Tensor::sum(size_t dim) const {
    if (dim > dims.size()) {
        // Sum all elements
        double total = std::accumulate(data.begin(), data.end(), 0.0);
        std::vector<double> temp = {total};
        Tensor result(temp, device, requires_grad);

        if (requires_grad) {
            // Store parent tensor for backward pass
            auto this_ptr = std::make_shared<Tensor>(*this);
            result.parents.push_back(std::weak_ptr<Tensor>(this_ptr));

            // Define backward function
            result.backward_fn = [this_ptr, result]() {
                if (!result.grad) {
                    throw std::runtime_error("Called backward on tensor with no grad");
                }
                
                // Initialize gradient if not already done
                if (!this_ptr->grad) {
                    this_ptr->grad = std::make_shared<std::vector<double>>(this_ptr->data.size(), 0.0);
                }

                // Gradient of sum with respect to each input is 1.0
                // Broadcast the gradient to all elements
                double grad_value = (*result.grad)[0];
                std::fill(this_ptr->grad->begin(), this_ptr->grad->end(), grad_value);
            };
        }

        return result;
    } else if (dim < dims.size()) {
        // Sum along a specific dimension
        std::vector<size_t> new_dims = dims;
        new_dims[dim] = 1; // Reduce the specified dimension

        std::vector<double> result_data(new_dims[0], 0.0);
        size_t stride = dims[dim];
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i / stride] += data[i];
        }

        Tensor result(result_data, device, requires_grad);

        if (requires_grad) {
            // Store parent tensor for backward pass
            auto this_ptr = std::make_shared<Tensor>(*this);
            result.parents.push_back(std::weak_ptr<Tensor>(this_ptr));

            // Define backward function
            result.backward_fn = [this_ptr, result, stride, dim]() {
                if (!result.grad) {
                    throw std::runtime_error("Called backward on tensor with no grad");
                }

                // Initialize gradient if not already done
                if (!this_ptr->grad) {
                    this_ptr->grad = std::make_shared<std::vector<double>>(this_ptr->data.size(), 0.0);
                }

                // Distribute gradients along the summed dimension
                for (size_t i = 0; i < this_ptr->data.size(); i++) {
                    (*this_ptr->grad)[i] += (*result.grad)[i / stride];
                }
            };
        }

        return result;
    } else {
        throw std::invalid_argument("Invalid dimension for sum operation.");
    }
}

void Tensor::zero_grad() {
    if (requires_grad && grad) {
        std::fill(grad->begin(), grad->end(), 0.0);
    }
}

void Tensor::backward() {
  if (!requires_grad) {
    throw std::runtime_error("Cannot call backward on a tensor that does not require gradients.");
  }

  // Initialize the gradient of this tensor to 1 (assuming it's the output scalar)
  if (!grad) {
    grad = std::make_shared<std::vector<double>>(data.size(), 0.0);
  }
  std::fill(grad->begin(), grad->end(), 1.0);

  // A queue for propagating the computation graph
  std::vector<std::shared_ptr<Tensor>> to_visit = {std::make_shared<Tensor>(*this)};
  std::unordered_map<Tensor*, bool> visited;

  while (!to_visit.empty()) {
    auto current = to_visit.back();
    to_visit.pop_back();

    // Skip if already visited
    if (visited[current.get()]) {
      continue;
    }
    visited[current.get()] = true;

    // Perform the gradient computation for this tensor
    if (current->backward_fn) {
      current->backward_fn();
    }

    // Add parents to the visit queue
    for (const auto& weak_parent : current->parents) {
      if (auto parent = weak_parent.lock()) {
        to_visit.push_back(parent);
      }
    }
  }
}

