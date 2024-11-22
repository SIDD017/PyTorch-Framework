#include "cross_entropy_loss.h"
#include <stdexcept>


namespace cross_entropy_loss_ns {

    double CrossEntropyLoss::forward(const Tensor& prediction, const Tensor& target) {
        // Ensure predictions and targets have the same dimensions
        if (prediction.get_dims() != target.get_dims()) {
            throw std::invalid_argument("Prediction and target tensors must have the same dimensions.");
        }

        // Initialize loss
        double loss = 0.0;
        const auto& pred_data = prediction.get_data(); // Access prediction data
        const auto& target_data = target.get_data();   // Access target data
        size_t total_elements = pred_data.size();
        size_t num_samples = prediction.get_dims()[0]; // Number of rows (samples)

        // Compute the Cross-Entropy Loss
        for (size_t i = 0; i < total_elements; ++i) {
            if (pred_data[i] <= 0.0 || pred_data[i] > 1.0) {
                throw std::invalid_argument("Predicted probabilities must be in the range (0, 1].");
            }

            // Cross-Entropy Formula: -y * log(p)
            loss += -target_data[i] * std::log(pred_data[i]);
        }

        // Normalize by the number of samples
        return loss / num_samples;
    }
}