#ifndef TENSOR_H
#define TENSOR_H

#include<vector>
#include<stdexcept>

class Tensor{
public:
  std::vector<double> data;
  std::vector<size_t> dims;
  
  Tensor(std::vector<size_t> dims);
  Tensor(std::vector<size_t> dims,std::vector<std::vector<size_t>> idx,std::vector<double> val);
  static Tensor ones(std::vector<size_t> dims);
  size_t index(std::vector<size_t> x);
  Tensor reshape(std::vector<size_t> new_dims);
  Tensor transpose();
  Tensor neg();
  Tensor reciprocal();
  Tensor add(Tensor x);
  Tensor subtract(Tensor x);
  Tensor mult(double x);
  Tensor elementwise_mult(Tensor x);
  Tensor pow(double x);
  Tensor relu();
  Tensor binarilize();
  Tensor exp();
  Tensor matmul(Tensor x);
  void print();
  std::vector<double> get_data();
  std::vector<size_t> get_dims();
};

#endif TENSOR_H