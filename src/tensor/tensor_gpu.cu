#include "tensor_gpu.h"

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
  
TensorGPU::TensorGPU(std::vector<size_t> dims) : Tensor(dims) {
  CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
}

TensorGPU::TensorGPU(std::vector<size_t> dims,std::vector<std::vector<size_t>> idx,std::vector<double> val) : Tensor(dims, idx, val){
  CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
  copyToDevice();
}

void TensorGPU::copyToDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_data, data.data(), data.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void TensorGPU::copyToHost() {
    CHECK_CUDA_ERRORS(cudaMemcpy(data.data(), d_data, data.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

static TensorGPU onesGPU(std::vector<size_t> dims){
  TensorGPU ret(dims);
  ret.copyToDevice();
  d_ones<<<NUM_BLOCKS, NUM_THREADS>>>(ret.d_data, ret.data.size());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  ret.copyToHost();
  return ret;
}

size_t TensorGPU::indexGPU(std::vector<size_t> x){
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

TensorGPU TensorGPU::reshapeGPU(std::vector<size_t> new_dims){
  size_t len = 1;
  for(auto d : new_dims)
    len *= d;
  if(len != data.size())
    throw std::runtime_error("Mismatched dims in reshape");
  TensorGPU ret(new_dims);
  ret.data = data;
  return ret;
}

TensorGPU TensorGPU::TensorGPU::transposeGPU(){
  if(dims.size() == 2){
    TensorGPU ret({dims[1],dims[0]});
    for(size_t i = 0;i < dims[0];++i){
      for(size_t j = 0;j < dims[1];++j){
        ret.data[ret.index({j,i})] = data[index({i,j})];
      }
    }
    return ret;
  }else if(dims.size() == 3){
    TensorGPU ret({dims[0],dims[2],dims[1]});
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

TensorGPU TensorGPU::negGPU(){
  TensorGPU ret(dims);
  copyToDevice();
  ret.copyToDevice();
  d_neg<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  ret.copyToHost();
  return ret;
}
  
TensorGPU TensorGPU::reciprocalGPU(){
  TensorGPU ret(dims);
  copyToDevice();
  ret.copyToDevice();
  d_reciprocal<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  ret.copyToHost();
  return ret;
}

TensorGPU TensorGPU::addGPU(TensorGPU x){
  if(dims != x.dims)
    throw std::runtime_error("Mismatched shape in add");
  TensorGPU ret(dims);
  copyToDevice();
  x.copyToDevice();
  ret.copyToDevice();
  d_add<<<num_blocks, num_threads>>>(ret.d_data, d_data, x.d_data, data.size());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  ret.copyToHost();
  return ret;
}
  
TensorGPU TensorGPU::subtractGPU(TensorGPU x){
  if(dims != x.dims)
    throw std::runtime_error("Mismatched shape in subtract");
  return addGPU(x.negGPU());
}

TensorGPU TensorGPU::multGPU(double x){
  TensorGPU ret(dims);
  copyToDevice();
  ret.copyToDevice();
  d_scalar_mult<<<num_blocks, num_threads>>>(ret.d_data, d_data, x, data.size());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  ret.copyToHost();
  return ret;
}
  
TensorGPU TensorGPU::elementwise_multGPU(TensorGPU x){
  if(dims != x.dims)
    throw std::runtime_error("Mismatched shape in elementwise_mult");
  TensorGPU ret(dims);
  copyToDevice();
  x.copyToDevice();
  ret.copyToDevice();
  d_elementwise_mult<<<num_blocks, num_threads>>>(ret.d_data, d_data, x.d_data, data.size());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  ret.copyToHost();
  return ret;
}
  
TensorGPU TensorGPU::powGPU(double x){
  TensorGPU ret(dims);
  copyToDevice();
  ret.copyToDevice();
  d_pow<<<num_blocks, num_threads>>>(ret.d_data, d_data, x, data.size());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  ret.copyToHost();
  return ret;
}


TensorGPU TensorGPU::reluGPU(){
  TensorGPU ret(dims);
  copyToDevice();
  ret.copyToDevice();
  d_relu<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  ret.copyToHost();
  return ret;
}

TensorGPU TensorGPU::binarilizeGPU(){
  TensorGPU ret(dims);
  copyToDevice();
  ret.copyToDevice();
  d_binarilize<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  ret.copyToHost();
  return ret;
}

TensorGPU TensorGPU::expGPU(){
  TensorGPU ret(dims);
  copyToDevice();
  ret.copyToDevice();
  d_exp<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
  ret.copyToHost();
  return ret;
}

TensorGPU TensorGPU::matmulGPU(TensorGPU x) {
  if (x.dims.size() != 2) {
      throw std::runtime_error("The right operand of matmul must be 2D tensors");
  }
  if (dims.size() != 2 && dims.size() != 3) {
      throw std::runtime_error("The left operand of matmul must be 2D tensors or batched 2D tensors");
  }
  if (dims[dims.size() - 1] != x.dims[0]) {
      throw std::runtime_error("Mismatched matmul matrix dimensions");
  }

  TensorGPU ret({dims[0], x.dims[1]});

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
