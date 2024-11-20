#ifndef MSELOSS_H
#define MSELOSS_H

#include "tensor.h"

class MSELoss {
public:
    MSELoss() {}

    Tensor forward(const Tensor& prediction, const Tensor& target) const {
        Tensor diff = prediction.subtract(target);
        return diff.pow(2).sum() * (1.0 / target.get_data().size());
    }

    Tensor operator()(const Tensor& prediction, const Tensor& target) const {
        return forward(prediction, target);
    }
};

#endif
