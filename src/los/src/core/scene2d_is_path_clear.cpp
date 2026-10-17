#include "core/scene2d.hpp"

namespace los2d {

bool Scene2D::is_path_clear(const Point2D &start, const Point2D &end) const {
  const int num_samples = 20;
  for (int i = 1; i < num_samples - 1; ++i) {
    float t = static_cast<float>(i) / static_cast<float>(num_samples - 1);
    Point2D sample(start.x + t * (end.x - start.x), start.y + t * (end.y - start.y));
    for (const auto &bld : original_buildings) {
      if (point_in_polygon(sample, bld.points)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace los2d
