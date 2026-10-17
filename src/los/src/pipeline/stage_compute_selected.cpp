#include "pipeline/stages.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace los2d {
namespace pipeline {

static int64_t vertex_key(const Point2D &p) {
  int32_t x = static_cast<int32_t>(std::lround(p.x * 1000.0f));
  int32_t y = static_cast<int32_t>(std::lround(p.y * 1000.0f));
  return (static_cast<int64_t>(x) << 32) | static_cast<uint32_t>(y);
}

bool stage_compute_selected(const RunConfig &cfg, RunContext &ctx) {
  ctx.selected_results = ctx.scene.compute_selected_tx(ctx.source_points, ctx.selected_indices);

  size_t s = ctx.selected_indices.size();
  size_t m = ctx.tx_results.m;

  std::unordered_set<int64_t> visible_union;
  const auto &verts = ctx.scene.get_vertices();

  for (size_t i = 0; i < s; ++i) {
    size_t src_idx = static_cast<size_t>(ctx.selected_indices[i]);
    size_t adj_base = src_idx * (m + 1) * (m + 1);
    for (size_t v = 0; v < m; ++v) {
      if (ctx.tx_results.adjacency_matrix[adj_base + (v + 1)] == 1) {
        visible_union.insert(vertex_key(verts[v]));
      }
    }
  }

  std::vector<Point2D> union_vertices;
  union_vertices.reserve(visible_union.size());
  for (const auto &v : verts) {
    if (visible_union.find(vertex_key(v)) != visible_union.end()) {
      union_vertices.push_back(v);
    }
  }

  size_t u = union_vertices.size();
  ctx.selected_visible_vertex_points.assign(u * 2, 0.0f);
  for (size_t i = 0; i < u; ++i) {
    ctx.selected_visible_vertex_points[i * 2] = union_vertices[i].x;
    ctx.selected_visible_vertex_points[i * 2 + 1] = union_vertices[i].y;
  }

  std::unordered_map<int64_t, size_t> union_idx;
  for (size_t i = 0; i < u; ++i) {
    union_idx[vertex_key(union_vertices[i])] = i;
  }

  ctx.selected_angle_matrix.assign(s * u * 2, 0.0f);
  for (size_t i = 0; i < s; ++i) {
    size_t src_idx = static_cast<size_t>(ctx.selected_indices[i]);
    const auto &src = ctx.source_points[src_idx];
    size_t adj_base = src_idx * (m + 1) * (m + 1);
    for (size_t v = 0; v < m; ++v) {
      if (ctx.tx_results.adjacency_matrix[adj_base + (v + 1)] == 1) {
        auto it = union_idx.find(vertex_key(verts[v]));
        if (it == union_idx.end()) continue;
        size_t uv = it->second;
        float dx = verts[v].x - src.x;
        float dy = verts[v].y - src.y;
        float d2d = std::sqrt(dx * dx + dy * dy);
        float dz = cfg.vertex_height_m - cfg.street_height_m;
        ctx.selected_angle_matrix[i * u * 2 + uv * 2] = std::atan2(dy, dx);
        ctx.selected_angle_matrix[i * u * 2 + uv * 2 + 1] = std::atan2(dz, d2d);
      }
    }
  }

  return true;
}

} // namespace pipeline
} // namespace los2d
