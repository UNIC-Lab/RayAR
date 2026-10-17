#include "core/scene2d.hpp"
#include <cmath>

namespace los2d {

namespace {

bool point_on_segment(const Point2D &pt, const Point2D &a, const Point2D &b, float tolerance) {
  float cross = (pt.y - a.y) * (b.x - a.x) - (pt.x - a.x) * (b.y - a.y);
  if (std::abs(cross) > tolerance) {
    return false;
  }

  float dot = (pt.x - a.x) * (b.x - a.x) + (pt.y - a.y) * (b.y - a.y);
  float len_sq = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
  if (len_sq < 1e-10f) {
    return pt.distance_to(a) < tolerance;
  }

  float param = dot / len_sq;
  return param >= 0.0f && param <= 1.0f;
}

} // namespace

bool Scene2D::point_on_obstacle_boundary(const Point2D &pt, float tolerance) const {
  for (const auto &seg : all_segments) {
    if (point_on_segment(pt, seg.p1, seg.p2, tolerance)) {
      return true;
    }
  }
  return false;
}

} // namespace los2d
