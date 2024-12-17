#ifndef ADAM_H
#define ADAM_H

#include "tensor.h"
#include <unordered_map>

class Adam {
public:
    Adam(std::vector<Tensor*> parameters, double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) 
        : parameters(parameters), lr(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {
        for (auto& param : parameters) {
            size_t size = param->data.size();
            m[param] = std::vector<double>(size, 0.0);
            v[param] = std::vector<double>(size, 0.0);
        }
    }
    
    void step() {
        t++;
        for (auto& param : parameters) {
            if (!param->grad) {
                continue;
            }
            auto& m_t = m[param];
            auto& v_t = v[param];
            const auto& grad = *(param->grad);
            for (size_t i = 0; i < param->data.size(); ++i) {
                m_t[i] = beta1 * m_t[i] + (1 - beta1) * grad[i];
                v_t[i] = beta2 * v_t[i] + (1 - beta2) * grad[i] * grad[i];
                double m_hat = m_t[i] / (1 - std::pow(beta1, t));
                double v_hat = v_t[i] / (1 - std::pow(beta2, t));
                param->data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
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
    double beta1;
    double beta2;
    double epsilon;
    size_t t;
    std::unordered_map<Tensor*, std::vector<double>> m;
    std::unordered_map<Tensor*, std::vector<double>> v;
};

#endif