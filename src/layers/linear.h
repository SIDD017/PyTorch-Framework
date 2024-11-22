#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"
#include <vector>
#include <string>

namespace linear {

	class Linear {
	private:
		Tensor weights; //Weight Tensor
		Tensor biases; //Bias Tensor
		std::string device;

	public:
		Linear(size_t input_size, size_t output_size, const std::string& device = "cpu");

		Tensor forward(const Tensor& input);

		Tensor get_weights() const;
		Tensor get_biases() const;
	};
}

#endif