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

#define NUM_BLOCKS 64
#define NUM_THREADS 32

class Tensor{
public:
  std::vector<double> data;
  std::vector<size_t> dims;
  char *device;
  double *d_data;

  int num_blocks = NUM_BLOCKS;
  int num_threads = NUM_THREADS;
  
  Tensor(std::vector<size_t> dims, char *dev);
  Tensor(std::vector<double> data1, char *dev);
  Tensor(std::vector<std::vector<double>> data1, char *dev);
  Tensor(std::vector<std::vector<std::vector<double>>> data1, char *dev);
  void to(char* dev);
  size_t index(std::vector<size_t> x);
  void print();
  std::vector<double> get_data();
  std::vector<size_t> get_dims();

  void copyToDevice();
  void copyToHost();
  static Tensor ones(std::vector<size_t> dims, char* dev);
  /* HACK: Do this on CPU only */
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

  // In-place operations
  Tensor& operator+=(const Tensor& other);
  Tensor& operator-=(const Tensor& other);
  Tensor& operator*=(const Tensor& other);
  Tensor& operator/=(const Tensor& other);

  Tensor& operator+=(double scalar);
  Tensor& operator-=(double scalar);
  Tensor& operator*=(double scalar);
  Tensor& operator/=(double scalar);

  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
  std::string toString() const;

  static Tensor rand(std::vector<size_t> dims, char* dev);
  static Tensor randn(std::vector<size_t> dims, char* dev);
  Tensor log() const;
};

#endif
