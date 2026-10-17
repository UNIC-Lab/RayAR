#pragma once
#include "core/point2d.hpp"
#include <vector>

namespace los2d {

struct Building {
  std::vector<Point2D> points;
  std::vector<int> global_vertex_indices;
  int id{-1};
};

} // namespace los2d
