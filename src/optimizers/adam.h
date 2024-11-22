#ifndef ADAM_H
#define ADAM_H

#include "tensor.h"
#include <vector>
#include <unordered_map>
#include <string>

namespace adam_ns {
    class Adam {
    private:
        double learning_rate;  // Learning rate for the optimizer.
        double beta1;          // Decay rate for the first moment (momentum).
        double beta2;          // Decay rate for the second moment (RMSProp).
        double epsilon;        // Small constant to prevent division by zero.
        size_t time_step;      // Keeps track of the number of steps (t).

        std::unordered_map<std::string, Tensor> m; ///< First moment vector for each parameter.
        std::unordered_map<std::string, Tensor> v; ///< Second moment vector for each parameter.

    public:
        /**
         * @brief Constructor for Adam optimizer.
         *
         * @param lr Learning rate (default: 0.001).
         * @param beta1 Decay rate for the first moment (default: 0.9).
         * @param beta2 Decay rate for the second moment (default: 0.999).
         * @param epsilon Small constant for numerical stability (default: 1e-8).
         */
        Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

        /**
         * @brief Update parameters using the provided gradients.
         *
         * @param gradients Gradients for each parameter (same order as parameters).
         * @param parameters Parameters to be updated (e.g., weights, biases).
         * @param param_names Names of the parameters (used for moment vectors).
         */
        void step(const std::vector<Tensor>& gradients, std::vector<Tensor>& parameters, const std::vector<std::string>& param_names);

        /**
         * @brief Get the learning rate.
         *
         * @return Current learning rate.
         */
        double get_learning_rate() const;
    };

}

#endif
