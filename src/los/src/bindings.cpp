#include "core/scene2d.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace los2d;

PYBIND11_MODULE(los2d_py, m) {
  m.doc() = "New C++ 2D LOS core";

  py::class_<Point2D>(m, "Point2D")
      .def(py::init<float, float>())
      .def_readwrite("x", &Point2D::x)
      .def_readwrite("y", &Point2D::y);

  py::class_<Scene2D::TXResults>(m, "TXResults")
      .def_property_readonly("adjacency_matrix", [](const Scene2D::TXResults &r) {
        return py::array_t<int8_t>({r.num_tx, r.m + 1, r.m + 1}, r.adjacency_matrix.data());
      })
      .def_property_readonly("angle_matrix", [](const Scene2D::TXResults &r) {
        return py::array_t<float>({r.num_tx, r.m, (size_t)2}, r.angle_matrix.data());
      })
      .def_property_readonly("projection_matrix", [](const Scene2D::TXResults &r) {
        return py::array_t<float>({r.num_tx, r.m, (size_t)3}, r.projection_matrix.data());
      })
      .def_property_readonly("dist_matrix", [](const Scene2D::TXResults &r) {
        return py::array_t<float>({r.num_tx, r.m}, r.dist_matrix.data());
      })
      .def_property_readonly("source_to_proj_dist", [](const Scene2D::TXResults &r) {
        return py::array_t<float>({r.num_tx, r.m}, r.source_to_proj_dist.data());
      });

  py::class_<Scene2D::SelectedTXResults>(m, "SelectedTXResults")
      .def_readwrite("selected_indices", &Scene2D::SelectedTXResults::selected_indices)
      .def_property_readonly("street_mask", [](const Scene2D::SelectedTXResults &r) {
        return py::array_t<int8_t>({r.selected_indices.size(), (size_t)257, (size_t)257}, r.street_mask.data());
      })
      .def_property_readonly("vertex_mask", [](const Scene2D::SelectedTXResults &r) {
        return py::array_t<int8_t>({r.selected_indices.size(), (size_t)257, (size_t)257}, r.vertex_mask.data());
      });

  py::class_<Scene2D>(m, "Scene2D")
      .def(py::init<>())
      .def("load_polygons", &Scene2D::load_polygons)
      .def("compute_tx_perspective", &Scene2D::compute_tx_perspective)
      .def("compute_selected_tx", &Scene2D::compute_selected_tx)
      .def("compute_vertex_perspective", &Scene2D::compute_vertex_perspective)
      .def("generate_street_mask", [](Scene2D &scene, float grid_step, float device_height) {
        auto points = scene.generate_street_mask(grid_step, device_height);
        py::array_t<float> arr({points.size(), (size_t)2});
        auto a = arr.mutable_unchecked<2>();
        for (size_t i = 0; i < points.size(); ++i) {
          a(i, 0) = points[i].x;
          a(i, 1) = points[i].y;
        }
        return arr;
      }, py::arg("grid_step") = 1.0f, py::arg("device_height") = 1.5f);
}
