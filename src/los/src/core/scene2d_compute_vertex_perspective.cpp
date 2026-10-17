#include "core/scene2d.hpp"
#include <cmath>
#include <omp.h>

namespace los2d {

Scene2D::VertexResults Scene2D::compute_vertex_perspective(float freq_ghz) {
  (void)freq_ghz;
  size_t num_v = all_vertices.size();

  VertexResults res;
  res.m = num_v;
  res.feature_points.reserve(num_v * 2);
  for (const auto &v : all_vertices) {
    res.feature_points.push_back(v.x);
    res.feature_points.push_back(v.y);
  }

  res.vertex_visibility_grid.assign(num_v * 257 * 257, 0);
  res.vertex_adjacency.assign(num_v * num_v, 0);
  res.vertex_dist_grid.assign(num_v * 257 * 257, 0.0f);
  res.vertex_angles_grid.assign(num_v * 257 * 257 * 2, 0.0f);

#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < static_cast<int>(num_v); ++i) {
    Point2D v = all_vertices[i];
    int b_id = vertex_to_building[i];

    for (int j = 0; j < static_cast<int>(num_v); ++j) {
      if (i == j) continue;
      if (is_visible(v, all_vertices[j], b_id, i, j)) {
        res.vertex_adjacency[static_cast<size_t>(i) * num_v + j] = 1;
      }
    }

    for (int r = 0; r < 257; ++r) {
      for (int c = 0; c < 257; ++c) {
        Point2D street_pt(static_cast<float>(c), static_cast<float>(r));
        if (is_visible(v, street_pt, b_id)) {
          size_t idx = static_cast<size_t>(i) * 257 * 257 + static_cast<size_t>(r) * 257 + c;
          res.vertex_visibility_grid[idx] = 1;
          float dx = street_pt.x - v.x;
          float dy = street_pt.y - v.y;
          float d2d = std::sqrt(dx * dx + dy * dy);
          float dz = device_height - building_height;
          res.vertex_dist_grid[idx] = std::sqrt(d2d * d2d + dz * dz);
          res.vertex_angles_grid[idx * 2] = std::atan2(dy, dx);
          res.vertex_angles_grid[idx * 2 + 1] = std::atan2(dz, d2d);
        }
      }
    }
  }

  return res;
}

} // namespace los2d
