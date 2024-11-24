#include<stdio.h>
#include<string.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include"tensor.h"
#include "sgd.h"
#include "mse_loss.h"
#include "linear.h"

namespace py = pybind11;

PYBIND11_MODULE(custom_torch,m){
  py::class_<Tensor>(m,"Tensor")
    // .def(py::init<std::vector<size_t>,char*>())
    .def(py::init<std::vector<double>,std::string,bool>())
    .def(py::init<std::vector<std::vector<double>>,std::string,bool>())
    .def(py::init<std::vector<std::vector<std::vector<double>>>,std::string,bool>())
    .def("to",&Tensor::to)
    .def_static("ones",&Tensor::ones)
    .def_static("zeros", &Tensor::zeros)
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
    .def("print",&Tensor::print)
    .def("get_data",&Tensor::get_data)
    .def("get_dims",&Tensor::get_dims)

    .def("__add__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator+))
    .def("__sub__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
    .def("__mul__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator*))
    .def("__truediv__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator/))

    // Overloaded binary operators (Tensor op double)
    .def("__add__", static_cast<Tensor (Tensor::*)(double) const>(&Tensor::operator+))
    .def("__sub__", static_cast<Tensor (Tensor::*)(double) const>(&Tensor::operator-))
    .def("__mul__", static_cast<Tensor (Tensor::*)(double) const>(&Tensor::operator*))
    .def("__truediv__", static_cast<Tensor (Tensor::*)(double) const>(&Tensor::operator/))

    // // In-place operators (Tensor += Tensor)
    // .def("__iadd__", static_cast<Tensor& (Tensor::*)(const Tensor&)>(&Tensor::operator+=))
    // .def("__isub__", static_cast<Tensor& (Tensor::*)(const Tensor&)>(&Tensor::operator-=))
    // .def("__imul__", static_cast<Tensor& (Tensor::*)(const Tensor&)>(&Tensor::operator*=))
    // .def("__itruediv__", static_cast<Tensor& (Tensor::*)(const Tensor&)>(&Tensor::operator/=))

    // // In-place operators (Tensor += double)
    // .def("__iadd__", static_cast<Tensor& (Tensor::*)(double)>(&Tensor::operator+=))
    // .def("__isub__", static_cast<Tensor& (Tensor::*)(double)>(&Tensor::operator-=))
    // .def("__imul__", static_cast<Tensor& (Tensor::*)(double)>(&Tensor::operator*=))
    // .def("__itruediv__", static_cast<Tensor& (Tensor::*)(double)>(&Tensor::operator/=))

    .def("__repr__", &Tensor::toString)

    .def_static("rand", &Tensor::rand)
    .def_static("randn", &Tensor::randn)
    .def("log", &Tensor::log)

    .def("backward",&Tensor::backward)
    .def("zero_grad",&Tensor::zero_grad)

    .def("sum",&Tensor::sum)

    .def_property_readonly("grad",
                      [](const Tensor& t) -> py::list {
                          if (t.grad) {
                              return py::cast(*t.grad);
                          } else {
                              return py::none();
                          }
                      });

    py::class_<Linear>(m, "Linear")
        .def(py::init<size_t, size_t, std::string, bool>())
        .def("forward", &Linear::forward)
        .def_readonly("weights", &Linear::weights)
        .def_readonly("bias", &Linear::bias);

    // SGD Optimizer
    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<Tensor*>, double>())
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad);

    // MSE Loss
    py::class_<MSELoss>(m, "MSELoss")
        .def(py::init<>())
        .def("__call__", &MSELoss::operator());

}
