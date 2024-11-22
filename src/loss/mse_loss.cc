#include "mse_loss.h"
#include <stdexcept>

namespace mse_loss_ns {

    double MSELoss::forward(const Tensor& prediction, const Tensor& target) {
        // Ensure predictions and targets have the same dimensions
        if (prediction.get_dims() != target.get_dims()) {
            throw std::invalid_argument("Predictions and targets must have the same dimensions");
        }

        // Calculate the element-wise difference and square it
        Tensor difference = prediction.subtract(target); // Element-wise subtraction
        Tensor squared = difference.pow(2.0);            // Element-wise squaring

        // Sum up the squared elements
        const std::vector<double>& squared_data = squared.get_data();
        double sum = 0.0;
        for (double value : squared_data) {
            sum += value;
        }

        // Calculate the mean by dividing by the total number of elements
        size_t total_elements = 1;
        for (size_t dim : prediction.get_dims()) {
            total_elements *= dim;
        }

        return sum / total_elements; // Mean Squared Error
    }

} // namespace mse_loss_ns
