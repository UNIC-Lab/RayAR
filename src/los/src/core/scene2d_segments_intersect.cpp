#include "core/scene2d.hpp"
#include <cmath>

namespace los2d {

bool Scene2D::segments_intersect(const Point2D &a, const Point2D &b, const Point2D &c, const Point2D &d) const {
  float dx1 = b.x - a.x;
  float dy1 = b.y - a.y;
  float dx2 = d.x - c.x;
  float dy2 = d.y - c.y;
  float dx3 = c.x - a.x;
  float dy3 = c.y - a.y;

  float denom = dx1 * dy2 - dy1 * dx2;
  if (std::abs(denom) < kLineParallelEps) {
    return false;
  }

  float t = (dx3 * dy2 - dy3 * dx2) / denom;
  float s = (dx3 * dy1 - dy3 * dx1) / denom;

  const float eps = kRayForwardEps;
  return (t > eps && t < 1.0f - eps && s > eps && s < 1.0f - eps);
}

} // namespace los2d
