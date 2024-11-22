#include "sgd.h"
#include <stdexcept> // For exceptions

namespace sgd_ns {

    SGD::SGD(double lr) : learning_rate(lr) {
        if (lr <= 0.0) {
            throw std::invalid_argument("Learning rate must be positive.");
        }
    }

    void SGD::step(const std::vector<Tensor>& gradients, std::vector<Tensor>& parameters) {
        if (gradients.size() != parameters.size()) {
            throw std::invalid_argument("Number of gradients and parameters must match.");
        }

        // Update each parameter using its corresponding gradient
        for (size_t i = 0; i < parameters.size(); ++i) {
            // Compute the update: param = param - learning_rate * gradient
            Tensor scaled_gradient = gradients[i].mult(learning_rate); // Scale gradient by learning rate
            parameters[i] = parameters[i].subtract(scaled_gradient);   // Update parameter
        }
    }

    double SGD::get_learning_rate() const {
        return learning_rate;
    }

} // namespace sgd_ns
