#include "core/scene2d.hpp"
#include <cmath>
#include <omp.h>

namespace los2d {

Scene2D::TXResults
Scene2D::compute_tx_perspective(const std::vector<Point2D> &tx_sources,
                                float freq_ghz) {
  (void)freq_ghz;
  size_t num_tx = tx_sources.size();
  size_t num_v = all_vertices.size();

  TXResults res;
  res.num_tx = num_tx;
  res.m = num_v;
  res.feature_points.reserve(num_v * 2);
  for (const auto &v : all_vertices) {
    res.feature_points.push_back(v.x);
    res.feature_points.push_back(v.y);
  }

  res.adjacency_matrix.assign(num_tx * (num_v + 1) * (num_v + 1), -1);
  res.angle_matrix.assign(num_tx * num_v * 2, 0.0f);
  res.dist_matrix.assign(num_tx * num_v, 0.0f);
  res.visibility_mask.assign(num_tx * 257 * 257, 0);

#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < static_cast<int>(num_tx); ++i) {
    Point2D tx = tx_sources[i];

    for (int j = 0; j < static_cast<int>(num_v); ++j) {
      Point2D v = all_vertices[j];
      size_t adj_base = static_cast<size_t>(i) * (num_v + 1) * (num_v + 1);
      if (is_visible(tx, v, -1, -1, j)) {
        res.adjacency_matrix[adj_base + (j + 1)] = 1;
        res.adjacency_matrix[adj_base + (j + 1) * (num_v + 1)] = 1;
      }

      float dx = v.x - tx.x;
      float dy = v.y - tx.y;
      float d2d = std::sqrt(dx * dx + dy * dy);
      float dz = building_height - device_height;
      float d3d = std::sqrt(d2d * d2d + dz * dz);
      res.dist_matrix[static_cast<size_t>(i) * num_v + j] = d3d;
      res.angle_matrix[static_cast<size_t>(i) * num_v * 2 + j * 2] =
          std::atan2(dy, dx);
      res.angle_matrix[static_cast<size_t>(i) * num_v * 2 + j * 2 + 1] =
          std::atan2(dz, d2d);
    }

    for (int r = 0; r < 257; ++r) {
      for (int c = 0; c < 257; ++c) {
        if (is_visible(tx,
                       Point2D(static_cast<float>(c), static_cast<float>(r)),
                       -1, -1, -1)) {
          res.visibility_mask[static_cast<size_t>(i) * 257 * 257 +
                              static_cast<size_t>(r) * 257 + c] = 1;
        }
      }
    }
  }

  return res;
}

} // namespace los2d
