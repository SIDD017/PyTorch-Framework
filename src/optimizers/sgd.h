#ifndef SGD_H
#define SGD_H

#include "tensor.h"
#include <vector>

class SGD {
public:
    double lr;

    SGD(double lr) : lr(lr) {}

    void step(std::vector<Tensor*>& params, const std::vector<Tensor>& grads) {
        for (size_t i = 0; i < params.size(); ++i) {
            *params[i] = params[i]->subtract(grads[i].mult(lr));
        }
    }
};

#endif
