#include<vector>
#include<stdexcept>
#include <string>
#include <omp.h>
#include<iostream>

#define NUM_BLOCKS 64
#define NUM_THREADS 32

#define CHECK_CUDA_ERRORS(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
         std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";

         cudaDeviceReset();
         exit(99);
    }   
}

__global__ static void d_ones(double *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    data[i] = 1.0;
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

__global__ void d_add(double *out, const double *in1, const double *in2, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = in1[i] + in2[i];
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

__global__ void d_pow(double *out, const double *in, double x, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = pow(in[i], x);
  }
}

__global__ void d_relu(double *out, const double *in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += step) {
    out[i] = in[i] > 0 ? in[i] : 0;
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

class Tensor{
public:
  std::vector<double> data;
  std::vector<size_t> dims;
  double *d_data;

  int num_blocks = NUM_BLOCKS;
  int num_threads = NUM_THREADS;
  
  Tensor(std::vector<size_t> dims) : dims(dims){
    size_t len = 1;
    for(auto d : dims)
      len *= d;
    data.resize(len);
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
  }

  Tensor(std::vector<size_t> dims,std::vector<std::vector<size_t>> idx,std::vector<double> val) : dims(dims){
    size_t len = 1;
    for(auto d : dims)
      len *= d;
    data.resize(len);
    if(idx.size() != val.size())
      throw std::runtime_error("Mismatched idx and val size");
    for(size_t i = 0;i < idx.size();++i){
      data[index(idx[i])] = val[i];
    }
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
    copyToDevice();
  }

  void copyToDevice() {
      CHECK_CUDA_ERRORS(cudaMemcpy(d_data, data.data(), data.size() * sizeof(double), cudaMemcpyHostToDevice));
  }

  void copyToHost() {
      CHECK_CUDA_ERRORS(cudaMemcpy(data.data(), d_data, data.size() * sizeof(double), cudaMemcpyDeviceToHost));
  }

  static Tensor ones(std::vector<size_t> dims){
    Tensor ret(dims);
    ret.copyToDevice();
    d_ones<<<NUM_BLOCKS, NUM_THREADS>>>(ret.d_data, ret.data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }

  size_t index(std::vector<size_t> x){
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

  Tensor reshape(std::vector<size_t> new_dims){
    size_t len = 1;
    for(auto d : new_dims)
      len *= d;
    if(len != data.size())
      throw std::runtime_error("Mismatched dims in reshape");
    Tensor ret(new_dims);
    ret.data = data;
    return ret;
  }

  Tensor transpose(){
    if(dims.size() == 2){
      Tensor ret({dims[1],dims[0]});
      for(size_t i = 0;i < dims[0];++i){
        for(size_t j = 0;j < dims[1];++j){
          ret.data[ret.index({j,i})] = data[index({i,j})];
        }
      }
      return ret;
    }else if(dims.size() == 3){
      Tensor ret({dims[0],dims[2],dims[1]});
      for(size_t b = 0;b < dims[0];++b){
        for(size_t i = 0;i < dims[1];++i){
          for(size_t j = 0;j < dims[2];++j){
            ret.data[ret.index({b,j,i})] = data[index({b,i,j})];
          }
        }
      }
      return ret;
    }else{
      throw std::runtime_error("The tensor must be 2D or batched 2D tensors");
    }
  }

  Tensor neg(){
    Tensor ret(dims);
    copyToDevice();
    ret.copyToDevice();
    d_neg<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }
  
  Tensor reciprocal(){
    Tensor ret(dims);
    copyToDevice();
    ret.copyToDevice();
    d_reciprocal<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }

  Tensor add(Tensor x){
    if(dims != x.dims)
      throw std::runtime_error("Mismatched shape in add");
    Tensor ret(dims);
    copyToDevice();
    x.copyToDevice();
    ret.copyToDevice();
    d_add<<<num_blocks, num_threads>>>(ret.d_data, d_data, x.d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }
  
  Tensor subtract(Tensor x){
    if(dims != x.dims)
      throw std::runtime_error("Mismatched shape in subtract");
    return add(x.neg());
  }

  Tensor mult(double x){
    Tensor ret(dims);
    copyToDevice();
    ret.copyToDevice();
    d_scalar_mult<<<num_blocks, num_threads>>>(ret.d_data, d_data, x, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }
  
  Tensor elementwise_mult(Tensor x){
    if(dims != x.dims)
      throw std::runtime_error("Mismatched shape in elementwise_mult");
    Tensor ret(dims);
    copyToDevice();
    x.copyToDevice();
    ret.copyToDevice();
    d_elementwise_mult<<<num_blocks, num_threads>>>(ret.d_data, d_data, x.d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }
  
  Tensor pow(double x){
    Tensor ret(dims);
    copyToDevice();
    ret.copyToDevice();
    d_pow<<<num_blocks, num_threads>>>(ret.d_data, d_data, x, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }


  Tensor relu(){
    Tensor ret(dims);
    copyToDevice();
    ret.copyToDevice();
    d_relu<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }

  Tensor binarilize(){
    Tensor ret(dims);
    copyToDevice();
    ret.copyToDevice();
    d_binarilize<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }

  Tensor exp(){
    Tensor ret(dims);
    copyToDevice();
    ret.copyToDevice();
    d_exp<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }

  Tensor matmul(Tensor x) {
    if (x.dims.size() != 2) {
        throw std::runtime_error("The right operand of matmul must be 2D tensors");
    }
    if (dims.size() != 2 && dims.size() != 3) {
        throw std::runtime_error("The left operand of matmul must be 2D tensors or batched 2D tensors");
    }
    if (dims[dims.size() - 1] != x.dims[0]) {
        throw std::runtime_error("Mismatched matmul matrix dimensions");
    }

    Tensor ret({dims[0], x.dims[1]});

    copyToDevice();
    x.copyToDevice();
    ret.copyToDevice();
    
    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((x.dims[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (dims[0] + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    if (dims.size() == 2) {
        d_matmul_2d<<<numBlocks, threadsPerBlock>>>(d_data, x.d_data, ret.d_data, dims[0], x.dims[1], dims[1]);
    } else {
        // For batched matrix multiplication
        dim3 numBlocksBatched((x.dims[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (dims[1] + threadsPerBlock.y - 1) / threadsPerBlock.y,
                              dims[0]);
        d_matmul_3d<<<numBlocksBatched, threadsPerBlock>>>(d_data, x.d_data, ret.d_data, dims[0], dims[1], x.dims[1], dims[2]);
    }

    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
    return ret;
  }

  void print(){
    for(auto x : data)
      printf("%s\n",std::to_string(x).c_str());
  }

  std::vector<double> get_data(){
    return data;
  }

  std::vector<size_t> get_dims(){
    return dims;
  }
  
};
