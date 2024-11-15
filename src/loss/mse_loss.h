#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "tensor.h"

namespace mse_loss_ns {

	class MSELoss {
	public:
		double forward(const tensor& prediction, const tensor& target);
	};
}

#endif