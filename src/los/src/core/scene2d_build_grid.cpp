#include "core/scene2d.hpp"
#include <algorithm>

namespace los2d {

void Scene2D::build_grid() {
  spatial_grid.init(map_size, 10.0f);
  for (size_t i = 0; i < all_segments.size(); ++i) {
    const auto &seg = all_segments[i];
    float min_x = std::min(seg.p1.x, seg.p2.x);
    float max_x = std::max(seg.p1.x, seg.p2.x);
    float min_y = std::min(seg.p1.y, seg.p2.y);
    float max_y = std::max(seg.p1.y, seg.p2.y);

    int start_c = std::max(0, static_cast<int>(min_x / spatial_grid.cell_size));
    int end_c = std::min(spatial_grid.cols - 1, static_cast<int>(max_x / spatial_grid.cell_size));
    int start_r = std::max(0, static_cast<int>(min_y / spatial_grid.cell_size));
    int end_r = std::min(spatial_grid.rows - 1, static_cast<int>(max_y / spatial_grid.cell_size));

    for (int r = start_r; r <= end_r; ++r) {
      for (int c = start_c; c <= end_c; ++c) {
        spatial_grid.cells[r * spatial_grid.cols + c].push_back(static_cast<int>(i));
      }
    }
  }
}

} // namespace los2d
