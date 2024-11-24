#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class Linear {
public:
    Linear(size_t input_dim, size_t output_dim, std::string device, bool requires_grad)
        : weights(Tensor::rand({input_dim, output_dim}, device, requires_grad)),
        bias(Tensor::zeros({1, output_dim}, device, requires_grad)) {}

    Tensor forward(const Tensor& x) {
        return x.matmul(weights) + bias;
    }

    Tensor weights;
    Tensor bias;
};

#endif