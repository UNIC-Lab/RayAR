#include "pipeline/stages.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace los2d {
namespace pipeline {

bool stage_compute_tx_results(const RunConfig &cfg, RunContext &ctx) {
  (void)cfg;
  ctx.tx_results = ctx.scene.compute_tx_perspective(ctx.source_points, 3.5f);

  size_t n = ctx.tx_results.num_tx;
  size_t m = ctx.tx_results.m;
  ctx.source_visibility_vertex_mask.assign(n * 257 * 257, -1);

  for (size_t i = 0; i < n; ++i) {
    const auto &src = ctx.source_points[i];
    int sx = std::max(0, std::min(256, static_cast<int>(src.x)));
    int sy = std::max(0, std::min(256, static_cast<int>(src.y)));
    ctx.source_visibility_vertex_mask[i * 257 * 257 + static_cast<size_t>(sy) * 257 + sx] = 0;

    size_t adj_base = i * (m + 1) * (m + 1);
    const auto &verts = ctx.scene.get_vertices();
    for (size_t v = 0; v < m; ++v) {
      if (ctx.tx_results.adjacency_matrix[adj_base + (v + 1)] == 1) {
        int vx = std::max(0, std::min(256, static_cast<int>(verts[v].x)));
        int vy = std::max(0, std::min(256, static_cast<int>(verts[v].y)));
        ctx.source_visibility_vertex_mask[i * 257 * 257 + static_cast<size_t>(vy) * 257 + vx] = 1;
      }
    }
  }

  std::cout << "Computed TX results for " << n << " sources, vertices=" << m << "\n";
  return true;
}

} // namespace pipeline
} // namespace los2d
