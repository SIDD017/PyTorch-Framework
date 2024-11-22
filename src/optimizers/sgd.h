#ifndef SGD_H
#define SGD_H

#include "tensor.h"
#include "linear.h"
#include <vector>

namespace sgd_ns {
    class SGD {
    private:
        double learning_rate; // Learning rate for the optimizer.

    public:
        /**
         * @brief Constructor for SGD optimizer.
         *
         * @param lr Learning rate for the optimizer.
         */
        explicit SGD(double lr);

        /**
         * @brief Update parameters using the provided gradients.
         *
         * @param gradients Gradients for each parameter (same order as parameters).
         * @param parameters Parameters to be updated (e.g., weights, biases).
         */
        void step(const std::vector<Tensor>& gradients, std::vector<Tensor>& parameters);

        /**
         * @brief Get the learning rate.
         *
         * @return Current learning rate.
         */
        double get_learning_rate() const;
    };

}

#endif
