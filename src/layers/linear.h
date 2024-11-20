#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"
#include <random>

class Linear {
public:
    Tensor weights;
    Tensor bias;
    char *device;

    Linear(size_t in_features, size_t out_features, char *device = "cpu")
        : device(device) {
        weights = Tensor::randn({in_features, out_features}, device);
        bias = Tensor::randn({out_features}, device);
    }

    Tensor forward(const Tensor& input) const {
        return input.matmul(weights).add(bias);
    }

    Tensor operator()(const Tensor& input) const {
        return forward(input);
    }
};

#endif
