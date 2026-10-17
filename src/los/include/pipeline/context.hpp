#pragma once
#include "core/scene2d.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace los2d {
namespace pipeline {

struct RunContext {
  Scene2D scene;
  std::vector<std::vector<std::vector<float>>> polygons;
  std::vector<Point2D> street_points;
  std::vector<Point2D> source_points;
  std::vector<int> selected_indices;
  std::vector<Point2D> selected_source_points;
  Scene2D::TXResults tx_results;
  Scene2D::SelectedTXResults selected_results;

  std::vector<int8_t> source_visibility_vertex_mask;
  std::vector<float> selected_angle_matrix;
  std::vector<float> selected_visible_vertex_points;

  std::vector<int8_t> raw_vertex_visibility_mask;
  std::vector<int8_t> vertex_visibility_mask;
  std::vector<int8_t> street_visibility_mask;
  std::vector<float> vertex_distance_mask;
  std::vector<float> vertex_angle_mask;
  std::vector<float> street_distance_mask;
  std::vector<float> street_angle_mask;
};

} // namespace pipeline
} // namespace los2d
