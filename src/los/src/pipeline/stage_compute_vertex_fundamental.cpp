#include "pipeline/stages.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <unordered_map>
#include <vector>

namespace los2d {
namespace pipeline {

namespace {

constexpr int kGridSize = 257;
constexpr float kCoordQuantScale = 1000000.0f;
constexpr float kBoundaryTol = 1e-6f;

int clamp_coord(float v) {
  return std::max(0, std::min(256, static_cast<int>(v)));
}

size_t grid_index(size_t src_idx, int x, int y) {
  return src_idx * static_cast<size_t>(kGridSize) * static_cast<size_t>(kGridSize) +
         static_cast<size_t>(x) * static_cast<size_t>(kGridSize) +
         static_cast<size_t>(y);
}

int64_t point_key(const Point2D &p) {
  int32_t x = static_cast<int32_t>(std::lround(p.x * kCoordQuantScale));
  int32_t y = static_cast<int32_t>(std::lround(p.y * kCoordQuantScale));
  return (static_cast<int64_t>(x) << 32) | static_cast<uint32_t>(y);
}

bool are_adjacent(int idx_a, int idx_b, int n) {
  if (idx_a < 0 || idx_b < 0 || n < 2) {
    return false;
  }
  int diff = std::abs(idx_a - idx_b);
  return diff == 1 || diff == n - 1;
}

bool point_on_segment_with_tolerance(const Point2D &pt, const Point2D &a,
                                     const Point2D &b, float tolerance) {
  float cross = (pt.y - a.y) * (b.x - a.x) - (pt.x - a.x) * (b.y - a.y);
  if (std::abs(cross) > tolerance) {
    return false;
  }

  float dot = (pt.x - a.x) * (b.x - a.x) + (pt.y - a.y) * (b.y - a.y);
  float len_sq = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
  if (len_sq < 1e-10f) {
    return pt.distance_to(a) < tolerance;
  }

  float t = dot / len_sq;
  return t >= 0.0f && t <= 1.0f;
}

bool point_on_polygon_boundary(const Point2D &pt, const std::vector<Point2D> &poly,
                               float tolerance) {
  if (poly.size() < 2) {
    return false;
  }
  for (size_t i = 0; i < poly.size(); ++i) {
    const Point2D &a = poly[i];
    const Point2D &b = poly[(i + 1) % poly.size()];
    if (point_on_segment_with_tolerance(pt, a, b, tolerance)) {
      return true;
    }
  }
  return false;
}

std::vector<Point2D>
sanitize_polygon(const std::vector<std::vector<float>> &raw_poly) {
  std::vector<Point2D> poly;
  poly.reserve(raw_poly.size());
  for (const auto &p : raw_poly) {
    if (p.size() >= 2) {
      poly.emplace_back(p[0], p[1]);
    }
  }

  if (poly.size() >= 2 && poly.front().distance_to(poly.back()) < kBoundaryTol) {
    poly.pop_back();
  }

  std::vector<Point2D> deduped;
  deduped.reserve(poly.size());
  for (const auto &p : poly) {
    if (deduped.empty() || deduped.back().distance_to(p) >= kBoundaryTol) {
      deduped.push_back(p);
    }
  }

  if (deduped.size() >= 2 &&
      deduped.front().distance_to(deduped.back()) < kBoundaryTol) {
    deduped.pop_back();
  }

  return deduped;
}

struct BuildingLookup {
  std::vector<Point2D> polygon;
  std::unordered_map<int64_t, int> vertex_to_local;
};

} // namespace

bool stage_compute_vertex_fundamental(const RunConfig &cfg, RunContext &ctx) {
  if (!cfg.emit_fundamental_vertex) {
    ctx.raw_vertex_visibility_mask.clear();
    ctx.vertex_visibility_mask.clear();
    ctx.street_visibility_mask.clear();
    ctx.vertex_distance_mask.clear();
    ctx.vertex_angle_mask.clear();
    ctx.street_distance_mask.clear();
    ctx.street_angle_mask.clear();
    return true;
  }

  const auto &verts = ctx.scene.get_vertices();
  const size_t m = verts.size();
  const size_t grid_cells =
      m * static_cast<size_t>(kGridSize) * static_cast<size_t>(kGridSize);

  ctx.raw_vertex_visibility_mask.assign(grid_cells, static_cast<int8_t>(-1));
  ctx.vertex_visibility_mask.assign(grid_cells, static_cast<int8_t>(-1));
  ctx.street_visibility_mask.assign(grid_cells, static_cast<int8_t>(-1));
  ctx.vertex_distance_mask.assign(grid_cells, -1.0f);
  ctx.vertex_angle_mask.assign(grid_cells, -1.0f);
  ctx.street_distance_mask.assign(grid_cells, -1.0f);
  ctx.street_angle_mask.assign(grid_cells * 2, -1.0f);

  if (m == 0) {
    return true;
  }

  std::vector<BuildingLookup> lookup(ctx.polygons.size());
  for (size_t b = 0; b < ctx.polygons.size(); ++b) {
    lookup[b].polygon = sanitize_polygon(ctx.polygons[b]);
    lookup[b].vertex_to_local.reserve(lookup[b].polygon.size());
    for (size_t i = 0; i < lookup[b].polygon.size(); ++i) {
      lookup[b].vertex_to_local.emplace(point_key(lookup[b].polygon[i]),
                                        static_cast<int>(i));
    }
  }

  const std::vector<Point2D> street_points =
      ctx.scene.generate_street_mask(1.0f, cfg.street_height_m);
  const float street_dz = cfg.street_height_m - cfg.vertex_height_m;

#pragma omp parallel for schedule(dynamic, 1)
  for (int k = 0; k < static_cast<int>(m); ++k) {
    const size_t src_idx = static_cast<size_t>(k);
    const Point2D &src = verts[src_idx];
    const int src_building = ctx.scene.get_vertex_building(src_idx);

    const int sx = clamp_coord(src.x);
    const int sy = clamp_coord(src.y);
    const size_t self_idx = grid_index(src_idx, sx, sy);
    ctx.raw_vertex_visibility_mask[self_idx] = 0;
    ctx.vertex_visibility_mask[self_idx] = 0;
    ctx.street_visibility_mask[self_idx] = 0;
    ctx.vertex_distance_mask[self_idx] = 0.0f;
    ctx.vertex_angle_mask[self_idx] = 0.0f;
    ctx.street_distance_mask[self_idx] = 0.0f;
    ctx.street_angle_mask[self_idx * 2] = 0.0f;
    ctx.street_angle_mask[self_idx * 2 + 1] = 0.0f;

    std::vector<uint8_t> visible_filtered(m, 0);

    for (size_t j = 0; j < m; ++j) {
      if (j == src_idx) {
        continue;
      }
      const Point2D &dst = verts[j];
      if (!ctx.scene.is_visible(src, dst, src_building, k, static_cast<int>(j))) {
        continue;
      }

      visible_filtered[j] = 1;

      const int dx = clamp_coord(dst.x);
      const int dy = clamp_coord(dst.y);
      ctx.raw_vertex_visibility_mask[grid_index(src_idx, dx, dy)] = 1;
    }

    int src_local = -1;
    if (src_building >= 0 && static_cast<size_t>(src_building) < lookup.size()) {
      const auto &map = lookup[static_cast<size_t>(src_building)].vertex_to_local;
      auto it = map.find(point_key(src));
      if (it != map.end()) {
        src_local = it->second;
      }
    }

    for (size_t j = 0; j < m; ++j) {
      if (!visible_filtered[j]) {
        continue;
      }
      if (ctx.scene.get_vertex_building(j) != src_building) {
        continue;
      }
      if (src_building < 0 || static_cast<size_t>(src_building) >= lookup.size()) {
        continue;
      }

      const auto &bld = lookup[static_cast<size_t>(src_building)];
      if (bld.polygon.size() < 3) {
        continue;
      }

      int dst_local = -1;
      auto dst_it = bld.vertex_to_local.find(point_key(verts[j]));
      if (dst_it != bld.vertex_to_local.end()) {
        dst_local = dst_it->second;
      }

      if (are_adjacent(src_local, dst_local, static_cast<int>(bld.polygon.size()))) {
        continue;
      }

      Point2D mid((src.x + verts[j].x) * 0.5f, (src.y + verts[j].y) * 0.5f);
      if (ctx.scene.point_in_polygon(mid, bld.polygon)) {
        visible_filtered[j] = 0;
      }
    }

    for (size_t j = 0; j < m; ++j) {
      if (!visible_filtered[j]) {
        continue;
      }

      const Point2D &dst = verts[j];
      const int dx = clamp_coord(dst.x);
      const int dy = clamp_coord(dst.y);
      const size_t out_idx = grid_index(src_idx, dx, dy);
      ctx.vertex_visibility_mask[out_idx] = 1;

      bool keep_for_feature = true;
      if (ctx.scene.get_vertex_building(j) == src_building && src_building >= 0 &&
          static_cast<size_t>(src_building) < lookup.size()) {
        const auto &bld = lookup[static_cast<size_t>(src_building)];
        if (bld.polygon.size() >= 3) {
          int dst_local = -1;
          auto dst_it = bld.vertex_to_local.find(point_key(dst));
          if (dst_it != bld.vertex_to_local.end()) {
            dst_local = dst_it->second;
          }

          if (!are_adjacent(src_local, dst_local, static_cast<int>(bld.polygon.size()))) {
            Point2D mid((src.x + dst.x) * 0.5f, (src.y + dst.y) * 0.5f);
            if (ctx.scene.point_in_polygon(mid, bld.polygon) ||
                point_on_polygon_boundary(mid, bld.polygon, kBoundaryTol)) {
              keep_for_feature = false;
            }
          }
        }
      }

      if (!keep_for_feature) {
        continue;
      }

      const float vx = dst.x - src.x;
      const float vy = dst.y - src.y;
      ctx.vertex_distance_mask[out_idx] = std::sqrt(vx * vx + vy * vy);
      ctx.vertex_angle_mask[out_idx] = std::atan2(vy, vx);
    }

    for (const auto &street : street_points) {
      if (!ctx.scene.is_visible(src, street, src_building, k, -1)) {
        continue;
      }

      const int dx = clamp_coord(street.x);
      const int dy = clamp_coord(street.y);
      const size_t out_idx = grid_index(src_idx, dx, dy);
      ctx.street_visibility_mask[out_idx] = 1;

      const float vx = street.x - src.x;
      const float vy = street.y - src.y;
      const float d2d = std::sqrt(vx * vx + vy * vy);
      ctx.street_distance_mask[out_idx] =
          std::sqrt(d2d * d2d + street_dz * street_dz);
      ctx.street_angle_mask[out_idx * 2] = std::atan2(vy, vx);
      ctx.street_angle_mask[out_idx * 2 + 1] = std::atan2(street_dz, d2d);
    }
  }

  std::cout << "Computed fundamental vertex outputs for " << m
            << " vertices\n";
  return true;
}

} // namespace pipeline
} // namespace los2d
