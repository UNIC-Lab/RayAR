#include "core/scene2d.hpp"
#include <cmath>

namespace los2d {

Intersection Scene2D::cast_ray_angle(const Point2D &origin, float angle, int ignore_building_id) const {
  Point2D dir(std::cos(angle), std::sin(angle));
  return find_intersection(origin, dir, ignore_building_id);
}

} // namespace los2d
