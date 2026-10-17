#pragma once
#include "core/point2d.hpp"

namespace los2d {

struct Intersection {
  bool hit{false};
  float dist{1e9f};
  Point2D pt;
  int type{-1}; // 0 boundary, 1 edge, 2 vertex
};

} // namespace los2d
