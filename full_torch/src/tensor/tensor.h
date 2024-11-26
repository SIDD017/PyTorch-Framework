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
  #include <functional>
  #include <algorithm>
  #include <memory>
  #include <unordered_map>

  #define NUM_BLOCKS 64
  #define NUM_THREADS 32

  class Tensor{
  public:
    int num_blocks = NUM_BLOCKS;
    int num_threads = NUM_THREADS;

    mutable std::vector<double> data;
    std::vector<size_t> dims;
    std::string device;
    mutable double *d_data;

    bool requires_grad;
    // Function pointer to a lambda for calculating this tensor's gradient in computation graph
    std::function<void()> backward_fn;
    // Pointers to this tensor's parent nodes in the computation graph
    std::vector<std::weak_ptr<Tensor>> parents;
    // Stores this tensor's gradients when backwards() is called
    std::shared_ptr<std::vector<double>> grad;

    // Constructors

    Tensor() {}
    Tensor(std::vector<size_t> dims, std::string dev, bool requires_grad);
    Tensor(std::vector<double> data1, std::string dev, bool requires_grad);
    Tensor(std::vector<std::vector<double>> data1, std::string dev, bool requires_grad);
    Tensor(std::vector<std::vector<std::vector<double>>> data1, std::string dev, bool requires_grad);

    // Tensor Initializers

    static Tensor ones(std::vector<size_t> dims, std::string dev, bool requires_grad);
    static Tensor zeros(std::vector<size_t> dims, std::string dev, bool requires_grad);
    static Tensor rand(std::vector<size_t> dims, std::string dev, bool requires_grad);
    static Tensor randn(std::vector<size_t> dims, std::string dev, bool requires_grad);

    // Tensor memory management

    void copyToDevice() const;
    void copyToHost() const;
    void freeDeviceMemory();
    void to(std::string dev);

    // Tensor operations

    size_t index(std::vector<size_t> x) const;
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
    Tensor clamp(double min_value, double max_value) const;
    Tensor max(size_t dim) const;
    Tensor log() const;

    // Gradient calculation
    
    void backward();
    void zero_grad();

    // Overloaded operators

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(double scalar) const;
    Tensor operator-(double scalar) const;
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    // Utility functions

    void print();
    std::string toString() const;
    std::vector<double> get_data() const;
    std::vector<size_t> get_dims() const;
    std::vector<double> get_data_1d() const;
    std::vector<std::vector<double>> get_data_2d() const;
    std::vector<std::vector<std::vector<double>>> get_data_3d() const;
  };

  #endif
