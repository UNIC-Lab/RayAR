#include "core/scene2d.hpp"
#include <algorithm>
#include <omp.h>

namespace los2d {

Scene2D::SelectedTXResults Scene2D::compute_selected_tx(const std::vector<Point2D> &tx_sources, const std::vector<int> &indices) {
  SelectedTXResults res;
  res.selected_indices = indices;
  size_t num_s = indices.size();
  size_t num_v = all_vertices.size();
  res.street_mask.assign(num_s * 257 * 257, 0);
  res.vertex_mask.assign(num_s * 257 * 257, -1);

#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < static_cast<int>(num_s); ++i) {
    Point2D tx = tx_sources[indices[i]];

    int tx_x = std::max(0, std::min(256, static_cast<int>(tx.x)));
    int tx_y = std::max(0, std::min(256, static_cast<int>(tx.y)));
    res.vertex_mask[static_cast<size_t>(i) * 257 * 257 + static_cast<size_t>(tx_y) * 257 + tx_x] = 0;

    for (int r = 0; r < 257; ++r) {
      for (int c = 0; c < 257; ++c) {
        Point2D street_pt(static_cast<float>(c), static_cast<float>(r));
        if (is_visible(tx, street_pt)) {
          res.street_mask[static_cast<size_t>(i) * 257 * 257 + static_cast<size_t>(r) * 257 + c] = 1;
        }
      }
    }

    for (int j = 0; j < static_cast<int>(num_v); ++j) {
      Point2D v = all_vertices[j];
      if (is_visible(tx, v)) {
        int vx = std::max(0, std::min(256, static_cast<int>(v.x)));
        int vy = std::max(0, std::min(256, static_cast<int>(v.y)));
        res.vertex_mask[static_cast<size_t>(i) * 257 * 257 + static_cast<size_t>(vy) * 257 + vx] = 1;
      }
    }
  }

  return res;
}

} // namespace los2d
