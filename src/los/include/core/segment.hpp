#pragma once
#include "core/point2d.hpp"

namespace los2d {

struct Segment {
  Point2D p1;
  Point2D p2;
  int building_id{-1};
  int v1_idx{-1};
  int v2_idx{-1};
};

} // namespace los2d
