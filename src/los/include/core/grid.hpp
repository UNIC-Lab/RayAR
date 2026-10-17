#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace los2d {

struct Grid {
  float cell_size{10.0f};
  int cols{0};
  int rows{0};
  std::vector<std::vector<int>> cells;

  void init(float map_size, float c_size) {
    cell_size = c_size;
    cols = rows = static_cast<int>(std::ceil(map_size / cell_size));
    cells.assign(cols * rows, std::vector<int>());
  }

  int get_cell_idx(float x, float y) const {
    int c = std::max(0, std::min(cols - 1, static_cast<int>(x / cell_size)));
    int r = std::max(0, std::min(rows - 1, static_cast<int>(y / cell_size)));
    return r * cols + c;
  }
};

} // namespace los2d
