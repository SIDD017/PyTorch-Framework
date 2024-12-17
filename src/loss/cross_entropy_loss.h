#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "tensor.h"
#include <cmath>
#include <limits>

class CrossEntropyLoss {
public:
    Tensor operator()(const Tensor& prediction, const Tensor& target) { 
        auto softmax_output = compute_softmax(prediction);
        auto log_softmax = softmax_output.log();
        std::vector<double> loss_values;
        auto softmax_data = softmax_output.get_data_2d();
        auto target_data = target.get_data_2d();
        double total_loss = 0.0;
        size_t batch_size = prediction.get_dims()[0];
        size_t num_classes = prediction.get_dims()[1];
        for (size_t i = 0; i < batch_size; ++i) {
            int target_class = -1;
            for (size_t j = 0; j < num_classes; ++j) {
                if (std::abs(target_data[i][j] - 1.0) < 1e-6) {
                    target_class = j;
                    break;
                }
            }
            if (target_class == -1) {
                throw std::runtime_error("Invalid one-hot encoded target");
            }
            double class_loss = -std::log(std::max(softmax_data[i][target_class], 1e-10));
            total_loss += class_loss;
        }
        std::vector<double> temp = {total_loss / batch_size};
        return Tensor(temp, prediction.device, true);
    }

private:
    Tensor compute_softmax(const Tensor& input) {
        auto input_max = input.max(1);
        std::vector<std::vector<double>> temp;
        temp.push_back(input_max.get_data());
        Tensor foo(temp, input.device, input.requires_grad);
        auto shifted_input = input - foo.transpose();
        auto exp_input = shifted_input.exp();
        auto sum_exp = exp_input.sum(1);
        std::vector<std::vector<double>> temp2;
        temp2.push_back(sum_exp.get_data());
        Tensor foo2(temp2, input.device, input.requires_grad);
        std::vector<std::vector<double>> softmax_values;
        auto exp_data = exp_input.get_data_2d();
        auto sum_data = foo2.transpose().get_data_1d();
        for (size_t i = 0; i < exp_data.size(); ++i) {
            std::vector<double> row_softmax;
            for (double exp_val : exp_data[i]) {
                row_softmax.push_back(exp_val / sum_data[i]);
            }
            softmax_values.push_back(row_softmax);
        }
        return Tensor(softmax_values, input.device, true);
    }
};

#endif