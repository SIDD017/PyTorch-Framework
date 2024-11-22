#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "tensor.h"

namespace cross_entropy_loss_ns {

	class CrossEntropyLoss {
	public:
		double forward(const Tensor& prediction, const Tensor& target);
	};
}

#endif
