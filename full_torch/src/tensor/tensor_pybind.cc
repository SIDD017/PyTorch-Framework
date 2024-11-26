#include<stdio.h>
#include<string.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include"tensor.h"
#include "sgd.h"
#include "mse_loss.h"
#include "linear.h"
#include "rmsprop.h"
#include "adam.h"
#include "binary_cross_entropy_loss.h"
#include "cross_entropy_loss.h"

namespace py = pybind11;

PYBIND11_MODULE(custom_torch,m){
  py::class_<Tensor>(m,"Tensor")
    .def(py::init<std::vector<double>,std::string,bool>())
    .def(py::init<std::vector<std::vector<double>>,std::string,bool>())
    .def(py::init<std::vector<std::vector<std::vector<double>>>,std::string,bool>())
    .def_static("ones",&Tensor::ones)
    .def_static("zeros", &Tensor::zeros)
    .def_static("rand", &Tensor::rand)
    .def_static("randn", &Tensor::randn)
    .def("print",&Tensor::print)
    .def("__repr__", &Tensor::toString)
    .def("get_data",&Tensor::get_data)
    .def("get_dims",&Tensor::get_dims)
    .def("get_data_1d",&Tensor::get_data_1d)
    .def("get_data_2d",&Tensor::get_data_2d)
    .def("get_data_3d",&Tensor::get_data_3d)
    .def("to",&Tensor::to)
    .def("freeDeviceMemory",&Tensor::freeDeviceMemory)
    .def("index",&Tensor::index)
    .def("reshape",&Tensor::reshape)
    .def("transpose",&Tensor::transpose)
    .def("neg",&Tensor::neg)
    .def("reciprocal",&Tensor::reciprocal)
    .def("add",&Tensor::add)
    .def("subtract",&Tensor::subtract)
    .def("mult",&Tensor::mult)
    .def("elementwise_mult",&Tensor::elementwise_mult)
    .def("pow",&Tensor::pow)
    .def("relu",&Tensor::relu)
    .def("binarilize",&Tensor::binarilize)
    .def("exp",&Tensor::exp)
    .def("matmul",&Tensor::matmul)
    .def("sum",&Tensor::sum)
    .def("max",&Tensor::max)
    .def("log", &Tensor::log)
    .def("backward",&Tensor::backward)
    .def("zero_grad",&Tensor::zero_grad)
    .def("__add__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator+))
    .def("__sub__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
    .def("__mul__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator*))
    .def("__truediv__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator/))
    .def("__add__", static_cast<Tensor (Tensor::*)(double) const>(&Tensor::operator+))
    .def("__sub__", static_cast<Tensor (Tensor::*)(double) const>(&Tensor::operator-))
    .def("__mul__", static_cast<Tensor (Tensor::*)(double) const>(&Tensor::operator*))
    .def("__truediv__", static_cast<Tensor (Tensor::*)(double) const>(&Tensor::operator/))
    .def_property_readonly("grad", [](const Tensor& t) -> py::list {
        if (t.grad) {
            return py::cast(*t.grad);
        } 
        else {
            return py::none();
        }
    }
    );

  py::class_<Linear>(m, "Linear")
    .def(py::init<size_t, size_t, std::string, bool>())
    .def("forward", &Linear::forward)
    .def_readonly("weights", &Linear::weights)
    .def_readonly("bias", &Linear::bias);

  py::class_<SGD>(m, "SGD")
    .def(py::init<std::vector<Tensor*>, double>())
    .def("step", &SGD::step)
    .def("zero_grad", &SGD::zero_grad);

  py::class_<MSELoss>(m, "MSELoss")
    .def(py::init<>())
    .def("__call__", &MSELoss::operator());

  py::class_<BinaryCrossEntropyLoss>(m, "BinaryCrossEntropyLoss")
    .def(py::init<>())
    .def("__call__", &BinaryCrossEntropyLoss::operator());

  py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
    .def(py::init<>())
    .def("__call__", &CrossEntropyLoss::operator());

  py::class_<Adam>(m, "Adam")
    .def(py::init<std::vector<Tensor*>, double, double, double, double>(),
        py::arg("parameters"),
        py::arg("lr") = 0.001,
        py::arg("beta1") = 0.9,
        py::arg("beta2") = 0.999,
        py::arg("epsilon") = 1e-8)
    .def("step", &Adam::step)
    .def("zero_grad", &Adam::zero_grad);

  py::class_<RMSprop>(m, "RMSprop")
    .def(py::init<std::vector<Tensor*>, double, double, double>(),
         py::arg("parameters"),
         py::arg("lr") = 0.01,
         py::arg("alpha") = 0.99,
         py::arg("epsilon") = 1e-8)
    .def("step", &RMSprop::step)
    .def("zero_grad", &RMSprop::zero_grad);

}
