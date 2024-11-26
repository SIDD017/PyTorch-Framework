#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "tensor.h"

class MSELoss {
public:
    Tensor operator()(const Tensor& prediction, const Tensor& target) {
        auto diff = prediction - target;
        auto squared = diff.elementwise_mult(diff);
        return squared.sum() * (1.0 / prediction.get_data().size());
    }
};

#endif