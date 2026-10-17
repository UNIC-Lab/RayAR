#include "core/scene2d.hpp"
#include <algorithm>

namespace los2d {

std::vector<float> Scene2D::get_visibility_angles(const Point2D &observer) const {
  constexpr float kPi = 3.14159265358979323846f;
  std::vector<float> events;
  events.reserve(8 + all_segments.size() * 2);

  std::vector<Point2D> boundary = {
      Point2D(0.0f, 0.0f),
      Point2D(map_size, 0.0f),
      Point2D(map_size, map_size),
      Point2D(0.0f, map_size),
  };

  for (const auto &p : boundary) {
    if (observer.distance_to(p) > kRayForwardEps) {
      events.push_back(std::atan2(p.y - observer.y, p.x - observer.x));
    }
  }

  for (const auto &seg : all_segments) {
    if (observer.distance_to(seg.p1) > kRayForwardEps) {
      events.push_back(std::atan2(seg.p1.y - observer.y, seg.p1.x - observer.x));
    }
    if (observer.distance_to(seg.p2) > kRayForwardEps) {
      events.push_back(std::atan2(seg.p2.y - observer.y, seg.p2.x - observer.x));
    }
  }

  std::sort(events.begin(), events.end());

  std::vector<float> out;
  out.reserve(events.size() * 3);
  for (float a : events) {
    out.push_back(a - kAngleEps);
    out.push_back(a);
    out.push_back(a + kAngleEps);
  }

  for (float &a : out) {
    while (a > kPi) a -= 2.0f * kPi;
    while (a < -kPi) a += 2.0f * kPi;
  }
  return out;
}

} // namespace los2d
