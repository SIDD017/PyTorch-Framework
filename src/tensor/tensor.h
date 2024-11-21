#ifndef TENSOR_H
#define TENSOR_H

#include<vector>
#include<stdexcept>
#include <string>
#include <omp.h>
#include<iostream>
#include <ostream>
#include <sstream>
#include <random>
#include <cmath>
#include <numeric>

#define NUM_BLOCKS 64
#define NUM_THREADS 32

class Tensor{
public:
  std::vector<double> data;
  std::vector<size_t> dims;
  std::string device;
  double *d_data;

  int num_blocks = NUM_BLOCKS;
  int num_threads = NUM_THREADS;
  
  Tensor() {}
  Tensor(std::vector<size_t> dims, std::string dev);
  Tensor(std::vector<double> data1, std::string dev);
  Tensor(std::vector<std::vector<double>> data1, std::string dev);
  Tensor(std::vector<std::vector<std::vector<double>>> data1, std::string dev);
  void to(std::string dev);
  size_t index(std::vector<size_t> x) const;
  void print();
  std::vector<double> get_data() const;
  std::vector<size_t> get_dims() const;

  void copyToDevice() const;
  void copyToHost();
  static Tensor ones(std::vector<size_t> dims, std::string dev);
  /* HACK: Do this on CPU only */
  Tensor reshape(std::vector<size_t> new_dims);
  Tensor transpose() const;
  Tensor neg() const;
  Tensor reciprocal();
  Tensor add(const Tensor &x) const;
  Tensor subtract(const Tensor &x) const;
  Tensor mult(double x) const;
  Tensor elementwise_mult(const Tensor &x) const;
  Tensor pow(double x) const;
  Tensor relu();
  Tensor binarilize();
  Tensor exp();
  Tensor matmul(Tensor x) const;
  Tensor sum(size_t dim = SIZE_MAX) const;

  // Overloaded operators
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  // Scalar operations
  Tensor operator+(double scalar) const;
  Tensor operator-(double scalar) const;
  Tensor operator*(double scalar) const;
  Tensor operator/(double scalar) const;

  // // In-place operations
  // Tensor& operator+=(const Tensor& other);
  // Tensor& operator-=(const Tensor& other);
  // Tensor& operator*=(const Tensor& other);
  // Tensor& operator/=(const Tensor& other);

  // Tensor& operator+=(double scalar);
  // Tensor& operator-=(double scalar);
  // Tensor& operator*=(double scalar);
  // Tensor& operator/=(double scalar);

  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
  std::string toString() const;

  static Tensor rand(std::vector<size_t> dims, std::string dev);
  static Tensor randn(std::vector<size_t> dims, std::string dev);
  Tensor log() const;
};

#endif
