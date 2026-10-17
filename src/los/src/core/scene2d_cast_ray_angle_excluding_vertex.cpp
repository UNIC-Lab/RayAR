#include "core/scene2d.hpp"
#include <algorithm>
#include <cmath>

namespace los2d {

Intersection
Scene2D::cast_ray_angle_excluding_vertex(const Point2D &origin, float angle,
                                         const Point2D &exclude_vertex,
                                         int ignore_building_id) const {
  const float vertex_match_threshold = kPointOnEdgeEps;
  Point2D dir(std::cos(angle), std::sin(angle));

  Intersection closest;
  closest.hit = false;
  closest.dist = 1e9f;

  // Boundary candidate (same style as find_intersection), but do not accept
  // the excluded vertex itself.
  float t_map = 1e9f;
  auto accept_boundary_t = [&](float t) {
    if (t <= kRayForwardEps || t >= t_map) {
      return;
    }
    float xh = origin.x + dir.x * t;
    float yh = origin.y + dir.y * t;
    if (xh < -kBoundaryEps || xh > map_size + kBoundaryEps ||
        yh < -kBoundaryEps || yh > map_size + kBoundaryEps) {
      return;
    }
    Point2D hit_pt(xh, yh);
    if (hit_pt.distance_to(exclude_vertex) < vertex_match_threshold) {
      return;
    }
    t_map = t;
  };

  if (std::abs(dir.x) > kBoundaryEps) {
    accept_boundary_t(-origin.x / dir.x);
    accept_boundary_t((map_size - origin.x) / dir.x);
  }
  if (std::abs(dir.y) > kBoundaryEps) {
    accept_boundary_t(-origin.y / dir.y);
    accept_boundary_t((map_size - origin.y) / dir.y);
  }

  if (t_map < 1e8f) {
    closest.hit = true;
    closest.dist = t_map;
    closest.pt = Point2D(origin.x + dir.x * t_map, origin.y + dir.y * t_map);
    closest.type = 0;
  }

  float x = origin.x;
  float y = origin.y;
  int step_x = (dir.x > 0) ? 1 : -1;
  int step_y = (dir.y > 0) ? 1 : -1;

  float cell_size = spatial_grid.cell_size;
  float t_max_x, t_max_y, t_delta_x, t_delta_y;
  if (std::abs(dir.x) < kLineParallelEps) {
    t_max_x = t_delta_x = 1e30f;
  } else {
    t_max_x = (dir.x > 0)
                  ? (std::floor(x / cell_size + 1.0f) * cell_size - x) / dir.x
                  : (x - std::floor(x / cell_size) * cell_size) / std::abs(dir.x);
    t_delta_x = cell_size / std::abs(dir.x);
  }
  if (std::abs(dir.y) < kLineParallelEps) {
    t_max_y = t_delta_y = 1e30f;
  } else {
    t_max_y = (dir.y > 0)
                  ? (std::floor(y / cell_size + 1.0f) * cell_size - y) / dir.y
                  : (y - std::floor(y / cell_size) * cell_size) / std::abs(dir.y);
    t_delta_y = cell_size / std::abs(dir.y);
  }

  int cur_c =
      std::max(0, std::min(spatial_grid.cols - 1, static_cast<int>(x / cell_size)));
  int cur_r =
      std::max(0, std::min(spatial_grid.rows - 1, static_cast<int>(y / cell_size)));

  float t_curr = 0.0f;
  float max_dist = std::min(closest.dist, 3000.0f);

  while (t_curr < max_dist) {
    const auto &cell_segs = spatial_grid.cells[cur_r * spatial_grid.cols + cur_c];
    for (int seg_idx : cell_segs) {
      const auto &seg = all_segments[seg_idx];
      if (seg.building_id == ignore_building_id) {
        continue;
      }
      if (seg.p1.distance_to(exclude_vertex) < vertex_match_threshold ||
          seg.p2.distance_to(exclude_vertex) < vertex_match_threshold) {
        continue;
      }

      Point2D s1 = seg.p1;
      Point2D s2 = seg.p2;
      float den = (s1.x - s2.x) * dir.y - (s1.y - s2.y) * dir.x;
      if (std::abs(den) < kLineParallelEps) {
        continue;
      }

      float t = ((s1.x - origin.x) * dir.y - (s1.y - origin.y) * dir.x) / den;
      float u = -((s1.x - s2.x) * (s1.y - origin.y) -
                  (s1.y - s2.y) * (s1.x - origin.x)) /
                den;

      if (t >= 0.0f && t <= 1.0f && u > kRayForwardEps && u < closest.dist) {
        closest.hit = true;
        closest.dist = u;
        closest.pt = Point2D(origin.x + dir.x * u, origin.y + dir.y * u);
        float d1 = closest.pt.distance_to(s1);
        float d2 = closest.pt.distance_to(s2);
        closest.type =
            (d1 < kProjectionDistanceThreshold || d2 < kProjectionDistanceThreshold)
                ? 2
                : 1;
        max_dist = closest.dist;
      }
    }

    if (t_max_x < t_max_y) {
      t_curr = t_max_x;
      t_max_x += t_delta_x;
      cur_c += step_x;
    } else {
      t_curr = t_max_y;
      t_max_y += t_delta_y;
      cur_r += step_y;
    }

    if (cur_c < 0 || cur_c >= spatial_grid.cols || cur_r < 0 ||
        cur_r >= spatial_grid.rows) {
      break;
    }
  }

  return closest;
}

} // namespace los2d
