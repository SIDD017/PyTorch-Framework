#include <pybind11/pybind11.h>
#include "mse_loss.h"

namespace py = pybind11;

PYBIND11_MODULE(my_tensor_module, m) {
    py::class_<mse_loss_ns::MSELoss>(m, "MSELoss")
        .def(py::init<>())  // Expose the default constructor
        .def("forward", &mse_loss_ns::MSELoss::forward,
            "Computes the Mean Squared Error between prediction and target tensors",
            py::arg("prediction"), py::arg("target"));
}