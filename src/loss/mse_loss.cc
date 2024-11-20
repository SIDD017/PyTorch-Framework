// #include "mse_loss.h"

// double MSELoss::forward(const Tensor& predictions, const Tensor& targets) {
//     Tensor diff = predictions.subtract(targets);
//     return diff.pow(2).sum() / predictions.get_data().size();
// }

// Tensor MSELoss::backward(const Tensor& predictions, const Tensor& targets) {
//     return predictions.subtract(targets).mult(2.0 / predictions.get_data().size());
// }
