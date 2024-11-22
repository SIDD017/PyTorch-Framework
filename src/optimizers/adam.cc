#include "adam.h"
#include <stdexcept> // For exceptions

namespace adam_ns {

    Adam::Adam(double lr, double beta1, double beta2, double epsilon)
        : learning_rate(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), time_step(0) {
        if (lr <= 0.0) {
            throw std::invalid_argument("Learning rate must be positive.");
        }
        if (beta1 <= 0.0 || beta1 >= 1.0) {
            throw std::invalid_argument("Beta1 must be in the range (0, 1).");
        }
        if (beta2 <= 0.0 || beta2 >= 1.0) {
            throw std::invalid_argument("Beta2 must be in the range (0, 1).");
        }
        if (epsilon <= 0.0) {
            throw std::invalid_argument("Epsilon must be positive.");
        }
    }

    void Adam::step(const std::vector<Tensor>& gradients, std::vector<Tensor>& parameters, const std::vector<std::string>& param_names) {
        if (gradients.size() != parameters.size() || parameters.size() != param_names.size()) {
            throw std::invalid_argument("Gradients, parameters, and parameter names must have the same size.");
        }

        ++time_step; // Increment the time step (t)

        for (size_t i = 0; i < parameters.size(); ++i) {
            const std::string& name = param_names[i];
            const Tensor& grad = gradients[i];
            Tensor& param = parameters[i];

            // Initialize m and v for this parameter if not already done
            if (m.find(name) == m.end()) {
                m[name] = Tensor(param.get_dims(), param.get_device()).zeros(); // First moment
                v[name] = Tensor(param.get_dims(), param.get_device()).zeros(); // Second moment
            }

            // Update biased first and second moment estimates
            m[name] = m[name].mult(beta1).add(grad.mult(1 - beta1)); // m_t = beta1 * m_t-1 + (1 - beta1) * g_t
            v[name] = v[name].mult(beta2).add(grad.pow(2).mult(1 - beta2)); // v_t = beta2 * v_t-1 + (1 - beta2) * g_t^2

            // Bias correction
            Tensor m_hat = m[name].mult(1.0 / (1.0 - std::pow(beta1, time_step))); // m_hat = m / (1 - beta1^t)
            Tensor v_hat = v[name].mult(1.0 / (1.0 - std::pow(beta2, time_step))); // v_hat = v / (1 - beta2^t)

            // Parameter update
            Tensor update = m_hat.div(v_hat.sqrt().add(epsilon)).mult(learning_rate); // Update = lr * m_hat / (sqrt(v_hat) + epsilon)
            param = param.subtract(update); // param = param - update
        }
    }

    double Adam::get_learning_rate() const {
        return learning_rate;
    }

}
