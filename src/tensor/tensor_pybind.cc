#include<stdio.h>
#include<string.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include"tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(custom_torch,m){
  py::class_<Tensor>(m,"Tensor")
    .def(py::init<std::vector<double>,char*>())
    .def(py::init<std::vector<std::vector<double>>,char*>())
    .def(py::init<std::vector<std::vector<std::vector<double>>>,char*>())
    .def("index",&Tensor::index)
    .def("print",&Tensor::print)
    .def("get_data",&Tensor::get_data)
    .def("get_dims",&Tensor::get_dims);
}
