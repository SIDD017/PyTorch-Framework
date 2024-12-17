#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class Linear {
public:
    Tensor weights;
    Tensor bias;

    Linear(size_t input_dim, size_t output_dim, std::string device, bool requires_grad)
        : weights(kaiming_uniform({input_dim, output_dim}, input_dim, device, requires_grad)),
        bias(Tensor::zeros({1, output_dim}, device, requires_grad)) {}

    Tensor forward(const Tensor& x) {
        return x.matmul(weights) + bias;
    }

private:
    Tensor kaiming_uniform(const std::vector<size_t>& shape, size_t fan_in, const std::string& device, bool requires_grad) {
        float limit = std::sqrt(6.0f / fan_in);
        Tensor normal_tensor = Tensor::randn(shape, device, requires_grad);
        return normal_tensor * (limit / std::sqrt(3.0f));
    }
};

#endif