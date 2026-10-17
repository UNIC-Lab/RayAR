#pragma once
#include <cmath>

namespace los2d {

struct Point2D {
  float x{0.0f};
  float y{0.0f};

  Point2D() = default;
  Point2D(float x_, float y_) : x(x_), y(y_) {}

  float distance_to(const Point2D &other) const {
    float dx = x - other.x;
    float dy = y - other.y;
    return std::sqrt(dx * dx + dy * dy);
  }
};

} // namespace los2d
