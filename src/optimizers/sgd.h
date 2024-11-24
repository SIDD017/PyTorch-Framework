#ifndef SGD_H
#define SGD_H

#include "tensor.h"

class SGD {
public:
    SGD(std::vector<Tensor*> parameters, double lr) : parameters(parameters), lr(lr) {}
    
    void step() {
        for (auto& param : parameters) {
            if (param->grad) {
                auto grad = *(param->grad);
                for (size_t i = 0; i < param->data.size(); ++i) {
                    param->data[i] -= lr * grad[i];
                }
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
};

#endif