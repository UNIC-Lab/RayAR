#include "core/scene2d.hpp"

namespace los2d {

bool Scene2D::point_in_polygon(const Point2D &pt, const std::vector<Point2D> &polygon) const {
  int n = static_cast<int>(polygon.size());
  if (n < 3) {
    return false;
  }

  int count = 0;
  for (int i = 0; i < n; ++i) {
    const Point2D &p1 = polygon[i];
    const Point2D &p2 = polygon[(i + 1) % n];

    if ((p1.y <= pt.y && pt.y < p2.y) || (p2.y <= pt.y && pt.y < p1.y)) {
      float x_intersect = p1.x + (pt.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
      if (pt.x < x_intersect) {
        ++count;
      }
    }
  }

  return (count % 2) == 1;
}

} // namespace los2d
