#include "core/scene2d.hpp"
#include <algorithm>
#include <cmath>

namespace los2d {

std::vector<Point2D> Scene2D::calculate_visibility_polygon(const Point2D &observer) const {
  std::vector<float> angles = get_visibility_angles(observer);
  std::vector<std::pair<float, Point2D>> hits;
  hits.reserve(angles.size());

  for (float angle : angles) {
    Intersection inter = cast_ray_angle(observer, angle, -1);
    if (inter.hit) {
      hits.emplace_back(angle, inter.pt);
    }
  }

  std::sort(hits.begin(), hits.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

  std::vector<Point2D> polygon;
  polygon.reserve(hits.size());
  for (const auto &kv : hits) {
    if (!polygon.empty() && polygon.back().distance_to(kv.second) < 1e-6f) {
      continue;
    }
    polygon.push_back(kv.second);
  }
  return polygon;
}

} // namespace los2d
