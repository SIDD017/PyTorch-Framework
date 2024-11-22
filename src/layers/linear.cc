#include "linear.h"
#include <stdexcept>

namespace linear {
	Linear::Linear(size_t input_size, size_t output_size, const std::string& device)
		:device(device) {
		weights = Tensor({ input_size, output_size }, device).randomize();
		biases = Tensor({ output_size }, device).randomize();
	}

	Tensor Linear::forward(const Tensor& input) {
		if (input.get_dims()[1] != weights.get_dims()[0]) {
			throw std::invalid_argument("Input tensor's last dimension must match the Linear layer's input_size.");
		}

		Tensor matmul_output = input.matmul(weights);
		Tensor output = matmul_output.add(biases);

		return output;
	}

	Tensor Linear::get_weights() const {
		return weights;
	}

	Tensor Linear::get_biases() const {
		return biases;
	}
}