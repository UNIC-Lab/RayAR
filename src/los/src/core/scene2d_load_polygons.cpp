#include "core/scene2d.hpp"

namespace los2d {

bool Scene2D::load_polygons(
    const std::vector<std::vector<std::vector<float>>> &poly_data) {
  buildings.clear();
  constexpr float kDedupEps = 1e-6f;

  for (size_t i = 0; i < poly_data.size(); ++i) {
    Building bld;
    bld.id = static_cast<int>(i);
    for (const auto &pt_data : poly_data[i]) {
      if (pt_data.size() >= 2) {
        bld.points.emplace_back(pt_data[0], pt_data[1]);
      }
    }

    if (bld.points.size() >= 2 &&
        bld.points.front().distance_to(bld.points.back()) < kDedupEps) {
      bld.points.pop_back();
    }

    std::vector<Point2D> deduped;
    deduped.reserve(bld.points.size());
    for (const auto &pt : bld.points) {
      if (deduped.empty() || deduped.back().distance_to(pt) >= kDedupEps) {
        deduped.push_back(pt);
      }
    }
    if (deduped.size() >= 2 &&
        deduped.front().distance_to(deduped.back()) < kDedupEps) {
      deduped.pop_back();
    }
    bld.points.swap(deduped);

    buildings.push_back(bld);
  }

  preprocess_geometry();
  return true;
}

} // namespace los2d
