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

Tensor::Tensor(std::vector<size_t> dims, std::string dev = "cpu") : dims(dims) {
  device = dev;
  size_t len = 1;
  for(auto d : dims)
    len *= d;
  data.resize(len);
  if(dev == "cuda") {
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
    copyToDevice();
  }
  else {
    d_data = nullptr;
  }
}

Tensor::Tensor(std::vector<double> data1, std::string dev = "cpu") {
  device = dev;
  dims.push_back(data1.size());
  size_t len = 1;
  for(auto d : dims)
    len *= d;
  // data.resize(len);
  data.insert(data.end(), data1.begin(), data1.end()); 
  if(dev == "cuda") {
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
    copyToDevice();
  }
  else {
    d_data = nullptr;
  }
}

Tensor::Tensor(std::vector<std::vector<double>> data1, std::string dev = "cpu") {
  device = dev;
  dims.push_back(data1.size());
  dims.push_back(data1[0].size());
  for(size_t i = 0; i < data1.size(); ++i) {
    data.insert(data.end(), data1[i].begin(), data1[i].end()); 
  }
  if(dev == "cuda") {
    CHECK_CUDA_ERRORS(cudaMalloc(&d_data, sizeof(double) * data.size()));
    copyToDevice();
  }
  else {
    d_data = nullptr;
  }
}

Tensor::Tensor(std::vector<std::vector<std::vector<double>>> data1, std::string dev = "cpu") {
  device = dev;
  dims.push_back(data1.size());
  dims.push_back(data1[0].size());
  dims.push_back(data1[0][0].size()); 
  for(size_t i = 0; i < data1.size(); ++i) {
    for(size_t j = 0; j < data1[0].size(); ++j) {
      data.insert(data.end(), data1[i][j].begin(), data1[i][j].end()); 
    }
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

Tensor Tensor::ones(std::vector<size_t> dims, std::string dev){
  Tensor ret(dims, dev);
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

Tensor Tensor::reshape(std::vector<size_t> new_dims){
  size_t len = 1;
  for(auto d : new_dims)
    len *= d;
  if(len != data.size())
    throw std::runtime_error("Mismatched dims in reshape");
  Tensor ret(new_dims, device);
  ret.data = data;
  if(ret.device == "cuda") {
    ret.copyToDevice();
  }
  return ret;
}

Tensor Tensor::transpose() const {
  if (dims.size() != 2 && dims.size() != 3) {
    throw std::runtime_error("The tensor must be 2D or batched 2D tensors");
  }
  if(device == "cpu") {
    if(dims.size() == 2){
      std::vector<size_t> temp = {dims[1],dims[0]};
      Tensor ret(temp, device);
      for(size_t i = 0;i < dims[0];++i){
        for(size_t j = 0;j < dims[1];++j){
          ret.data[ret.index({j,i})] = data[index({i,j})];
        }
      }
      return ret;
    }else if(dims.size() == 3){
      std::vector<size_t> temp = {dims[0],dims[2],dims[1]};
      Tensor ret(temp, device);
      for(size_t b = 0;b < dims[0];++b){
        for(size_t i = 0;i < dims[1];++i){
          for(size_t j = 0;j < dims[2];++j){
            ret.data[ret.index({b,j,i})] = data[index({b,i,j})];
          }
        }
      }
      return ret;
    }
  }
  else {
    Tensor ret(dims.size() == 2 ? std::vector<size_t>{dims[1], dims[0]} : std::vector<size_t>{dims[0], dims[2], dims[1]}, device);
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
    return ret;
  }
}

Tensor Tensor::neg() const {
  Tensor ret(dims, device);
  if(device == "cpu") {
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = -data[i];
  }
  else {
    copyToDevice();
    ret.copyToDevice();
    d_neg<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
  }
  return ret;
}
  
Tensor Tensor::reciprocal(){
  Tensor ret(dims, device);
  if(device == "cpu") {
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = 1.0 / data[i];
  }
  else {
    copyToDevice();
    ret.copyToDevice();
    d_reciprocal<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
  }
  return ret;
}

Tensor Tensor::add(const Tensor &x) const {
  if(dims != x.dims)
    throw std::runtime_error("Mismatched shape in add");
  if(device != x.device) {
    throw std::runtime_error("Expected all tensors to be on the same device.");
  }
  Tensor ret(dims, device);
  if(device == "cpu") {
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] + x.data[i];
  }
  else {
    copyToDevice();
    x.copyToDevice();
    ret.copyToDevice();
    d_add<<<num_blocks, num_threads>>>(ret.d_data, d_data, x.d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
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
  Tensor ret(dims, device);
  if(device == "cpu") {
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] * x;
  }
  else {
    copyToDevice();
    ret.copyToDevice();
    d_scalar_mult<<<num_blocks, num_threads>>>(ret.d_data, d_data, x, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
  }
  return ret;
}
  
Tensor Tensor::elementwise_mult(const Tensor &x) const {
  if(dims != x.dims)
    throw std::runtime_error("Mismatched shape in elementwise_mult");
  if(device != x.device) {
    throw std::runtime_error("Expected all tensors to be on the same device.");
  }
  Tensor ret(dims, device);
  if(device ==  "cpu") {
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] * x.data[i];
  }
  else {
    copyToDevice();
    x.copyToDevice();
    ret.copyToDevice();
    d_elementwise_mult<<<num_blocks, num_threads>>>(ret.d_data, d_data, x.d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
  }
  return ret;
}
  
Tensor Tensor::pow(double x) const {
  Tensor ret(dims, device);
  if(device == "cpu") {
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = std::pow(data[i],x);
  }
  else {
    copyToDevice();
    ret.copyToDevice();
    d_pow<<<num_blocks, num_threads>>>(ret.d_data, d_data, x, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
  }
  return ret;
}
  
Tensor Tensor::relu(){
  Tensor ret(dims, device);
  if(device == "cpu") {
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] > 0 ? data[i] : 0;
  }
  else {
    copyToDevice();
    ret.copyToDevice();
    d_relu<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
  }
  return ret;
}

Tensor Tensor::binarilize(){
  Tensor ret(dims, device);
  if(device == "cpu") {
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] > 0 ? 1 : 0;
  }
  else {
    copyToDevice();
    ret.copyToDevice();
    d_binarilize<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
  }
  return ret;
}

Tensor Tensor::exp(){
  Tensor ret(dims, device);
  if(device == "cpu") {
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = std::exp(data[i]);
  }
  else {
    copyToDevice();
    ret.copyToDevice();
    d_exp<<<num_blocks, num_threads>>>(ret.d_data, d_data, data.size());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    ret.copyToHost();
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
    throw std::runtime_error("Mismatched matmul matrix dimentions");
  }
  if(device == "cpu") {
    if(dims.size() == 2){
      std::vector<size_t> temp = {dims[0],x.dims[1]};
      Tensor ret(temp, device);
      for(size_t i = 0;i < dims[0];++i){
        for(size_t j = 0;j < x.dims[1];++j){
          for(size_t k = 0;k < dims[1];++k){
            ret.data[ret.index({i,j})] += data[index({i,k})] * x.data[x.index({k,j})];
          }
        }
      }
      return ret;
    }else{
      std::vector<size_t> temp = {dims[0],dims[1],x.dims[1]};
      Tensor ret(temp, device);
      for(size_t b = 0;b < dims[0];++b){
        for(size_t i = 0;i < dims[1];++i){
          for(size_t j = 0;j < x.dims[1];++j){
            for(size_t k = 0;k < dims[2];++k){
              ret.data[ret.index({b,i,j})] += data[index({b,i,k})] * x.data[x.index({k,j})];
            }
          }
        }
      }
      return ret;
    }
  }
  else {
    std::vector<size_t> temp = {dims[0], x.dims[1]};
    Tensor ret(temp, device);

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
Tensor Tensor::operator+(const Tensor& other) const {
  if (this->dims != other.dims) {
    throw std::invalid_argument("Tensors must have the same dimensions for addition.");
  }

  if (this->device == other.device) {
    return this->add(other);
  }  
  throw std::runtime_error("Tensors must be on the same device for addition.");
}

Tensor Tensor::operator-(const Tensor& other) const {
  if (this->dims != other.dims) {
    throw std::invalid_argument("Tensors must have the same dimensions for subtraction.");
  }

  if (this->device == other.device) {
    this->subtract(other);
  }  
  throw std::runtime_error("Tensors must be on the same device for addition.");
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
  Tensor t(temp, this->device);
  return this->add(t);
}

Tensor Tensor::operator-(double scalar) const {
  std::vector<double> temp(this->data.size(), scalar);
  Tensor t(temp, this->device);
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

Tensor Tensor::rand(std::vector<size_t> dims, std::string dev) {
    size_t total_elements = 1;
    for (size_t dim : dims) total_elements *= dim;

    std::vector<double> random_data(total_elements);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (double& val : random_data) {
        val = dis(gen);
    }

    return Tensor(random_data, dev).reshape(dims);
}

Tensor Tensor::randn(std::vector<size_t> dims, std::string dev) {
    size_t total_elements = 1;
    for (size_t dim : dims) total_elements *= dim;

    std::vector<double> random_data(total_elements);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0); // Mean = 0, Stddev = 1

    for (double& val : random_data) {
        val = dis(gen);
    }

    return Tensor(random_data, dev).reshape(dims);
}

Tensor Tensor::log() const {
    Tensor result = *this; // Create a copy of the tensor
    for (double& val : result.data) {
        if (val <= 0) {
            throw std::domain_error("Logarithm undefined for non-positive values");
        }
        val = std::log(val);
    }
    return result;
}

Tensor Tensor::sum(size_t dim) const {
    if (dim == SIZE_MAX) {
        // Sum all elements
        double total = std::accumulate(data.begin(), data.end(), 0.0);
        std::vector<double> temp = {total};
        return Tensor(temp, device);
    } else if (dim < dims.size()) {
        // Sum along a specific dimension
        std::vector<size_t> new_dims = dims;
        new_dims[dim] = 1; // Reduce the specified dimension

        std::vector<double> result(new_dims[0], 0.0);
        size_t stride = dims[dim];
        for (size_t i = 0; i < data.size(); i++) {
            result[i / stride] += data[i];
        }

        return Tensor(result, device);
    } else {
        throw std::invalid_argument("Invalid dimension for sum operation.");
    }
}
