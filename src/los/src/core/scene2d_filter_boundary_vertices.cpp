#include "core/scene2d.hpp"
#include <cmath>

namespace los2d {

void Scene2D::filter_boundary_vertices() {
  const float eps = kBoundaryEps;
  const float limit = map_size;

  auto on_any_boundary = [&](const Point2D &p) {
    return (std::abs(p.x) < eps || std::abs(p.x - limit) < eps || std::abs(p.y) < eps || std::abs(p.y - limit) < eps);
  };

  for (auto &bld : buildings) {
    size_t n = bld.points.size();
    if (n < 3) {
      continue;
    }

    std::vector<bool> to_remove(n, false);
    for (size_t i = 0; i < n; ++i) {
      if (!on_any_boundary(bld.points[i])) {
        continue;
      }
      const auto &prev = bld.points[(i + n - 1) % n];
      const auto &next = bld.points[(i + 1) % n];
      bool has_adj = on_any_boundary(prev) || on_any_boundary(next);
      if (has_adj) {
        to_remove[i] = true;
        if (on_any_boundary(prev)) {
          to_remove[(i + n - 1) % n] = true;
        }
        if (on_any_boundary(next)) {
          to_remove[(i + 1) % n] = true;
        }
      }
    }

    std::vector<Point2D> filtered;
    filtered.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      if (!to_remove[i]) {
        filtered.push_back(bld.points[i]);
      }
    }
    bld.points.swap(filtered);
  }
}

} // namespace los2d
