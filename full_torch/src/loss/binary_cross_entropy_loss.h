#ifndef BINARY_CROSS_ENTROPY_LOSS_H
#define BINARY_CROSS_ENTROPY_LOSS_H

#include "tensor.h"
#include <cmath>

class BinaryCrossEntropyLoss {
public:
    Tensor operator()(const Tensor& prediction, const Tensor& target) {
        // Ensure predictions are between 0 and 1
        auto epsilon = 1e-7;
        auto clipped_pred = prediction.clamp(epsilon, 1.0 - epsilon);
        
        // Binary cross-entropy loss calculation
        // Loss = -[y * log(p) + (1-y) * log(1-p)]
        auto log_pred = clipped_pred.log();
        auto log_inv_pred = (Tensor::ones(clipped_pred.get_dims(), clipped_pred.device, false) - clipped_pred).log();
        
        auto loss = 
            (target.elementwise_mult(log_pred) + 
              (Tensor::ones(target.get_dims(), target.device, false) - target).elementwise_mult(log_inv_pred)).neg();
        
        // Return mean loss
        return loss.sum() * (1.0 / prediction.get_data().size());
    }
};

#endif