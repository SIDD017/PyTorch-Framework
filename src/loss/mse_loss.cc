#include "mse_loss.h"
#include <stdexcept>

namespace mse_loss_ns {

	double MSELoss::forward(const tensor& prediction, const tensor& target) {
		if (predictions.get_dims() != targets.get_dims()) {
			throw std::invalid_argument("Predictions and targets must have the same dimensions");
		}

		double sum = 0.0;
		size_t total_elements = predictions.get_data().size();

		for (size_t i = 0; i < total_elements; ++i) {
			double diff = predictions.get_data()[i] - targets.get_data()[i];
			sum += diff * diff;
		}

		return sum / total_elements;
	}

}