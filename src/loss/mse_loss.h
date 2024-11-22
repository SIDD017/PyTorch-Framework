#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "tensor.h"

namespace mse_loss_ns {

	class MSELoss {
	public:
		double forward(const Tensor& prediction, const Tensor& target);
	};
}

#endif