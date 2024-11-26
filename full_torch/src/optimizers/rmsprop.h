#ifndef RMSPROP_H
#define RMSPROP_H

#include "tensor.h"
#include <unordered_map>

class RMSprop {
public:
    RMSprop(std::vector<Tensor*> parameters, double lr = 0.01, double alpha = 0.99, double epsilon = 1e-8) 
        : parameters(parameters), lr(lr), alpha(alpha), epsilon(epsilon) {
        for (auto& param : parameters) {
            size_t size = param->data.size();
            squared_grads[param] = std::vector<double>(size, 0.0);
        }
    }
    
    void step() {
        for (auto& param : parameters) {
            if(!param->grad) {
                continue;
            }
            auto& sq_grad_avg = squared_grads[param];
            const auto& grad = *(param->grad);
            for (size_t i = 0; i < param->data.size(); ++i) {
                sq_grad_avg[i] = alpha * sq_grad_avg[i] + (1 - alpha) * grad[i] * grad[i];
                param->data[i] -= lr * grad[i] / (std::sqrt(sq_grad_avg[i] + epsilon));
            }
        }
    }
    
    void zero_grad() {
        for (auto& param : parameters) {
            param->zero_grad();
        }
    }

private:
    std::vector<Tensor*> parameters;
    double lr;
    double alpha;
    double epsilon;
    std::unordered_map<Tensor*, std::vector<double>> squared_grads;
};

#endif