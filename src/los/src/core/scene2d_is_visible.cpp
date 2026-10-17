#include "core/scene2d.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace los2d {

namespace {

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

  float param = dot / len_sq;
  return param >= 0.0f && param <= 1.0f;
}

} // namespace

bool Scene2D::sample_path_pass(const Point2D &p1, const Point2D &p2,
                               int ignore_building_id,
                               int force_interior_building_id) const {
  constexpr float kSampleStep = 0.5f;

  float dx = p2.x - p1.x;
  float dy = p2.y - p1.y;
  float dist = std::sqrt(dx * dx + dy * dy);
  if (dist <= kSampleStep) {
    return true;
  }

  float min_x = std::min(p1.x, p2.x) - kPointOnEdgeEps;
  float max_x = std::max(p1.x, p2.x) + kPointOnEdgeEps;
  float min_y = std::min(p1.y, p2.y) - kPointOnEdgeEps;
  float max_y = std::max(p1.y, p2.y) + kPointOnEdgeEps;

  std::vector<int> candidate_buildings;
  candidate_buildings.reserve(original_buildings.size());
  for (size_t i = 0; i < original_buildings.size(); ++i) {
    const auto &bld = original_buildings[i];
    bool is_ignored = (bld.id == ignore_building_id);
    bool force_interior = (bld.id == force_interior_building_id);
    if ((is_ignored && !force_interior) || bld.points.size() < 3) {
      continue;
    }

    float b_min_x = bld.points[0].x;
    float b_max_x = bld.points[0].x;
    float b_min_y = bld.points[0].y;
    float b_max_y = bld.points[0].y;
    for (const auto &pt : bld.points) {
      b_min_x = std::min(b_min_x, pt.x);
      b_max_x = std::max(b_max_x, pt.x);
      b_min_y = std::min(b_min_y, pt.y);
      b_max_y = std::max(b_max_y, pt.y);
    }

    bool overlap = !(b_max_x < min_x || b_min_x > max_x || b_max_y < min_y ||
                     b_min_y > max_y);
    if (overlap) {
      candidate_buildings.push_back(static_cast<int>(i));
    }
  }

  if (candidate_buildings.empty()) {
    return true;
  }

  int num_steps = static_cast<int>(std::ceil(dist / kSampleStep));
  if (num_steps <= 1) {
    return true;
  }

  for (int i = 1; i < num_steps; ++i) {
    float t = static_cast<float>(i) / static_cast<float>(num_steps);
    Point2D sample(p1.x + t * dx, p1.y + t * dy);

    for (int b_idx : candidate_buildings) {
      const auto &bld = original_buildings[b_idx];
      bool is_ignored = (bld.id == ignore_building_id);
      const auto &poly = bld.points;
      size_t n = poly.size();
      if (!is_ignored) {
        for (size_t e = 0; e < n; ++e) {
          const Point2D &a = poly[e];
          const Point2D &b = poly[(e + 1) % n];
          if (point_on_segment_with_tolerance(sample, a, b, kPointOnEdgeEps)) {
            return false;
          }
        }
      }

      if (point_in_polygon(sample, poly)) {
        return false;
      }
    }
  }

  return true;
}

bool Scene2D::is_visible(const Point2D &p1, const Point2D &p2,
                         int ignore_building_id, int p1_v_idx,
                         int p2_v_idx) const {
  int effective_ignore_building_id = ignore_building_id;
  int force_interior_building_id =
      (p1_v_idx != -1) ? vertex_to_building[p1_v_idx] : -1;

  if (p1_v_idx != -1 && p2_v_idx != -1) {
    int b1 = vertex_to_building[p1_v_idx];
    int b2 = vertex_to_building[p2_v_idx];
    if (b1 == b2) {
      int l1 = vertex_local_idx[p1_v_idx];
      int l2 = vertex_local_idx[p2_v_idx];
      int num_pts = static_cast<int>(buildings[b1].points.size());
      int diff = std::abs(l1 - l2);
      if (diff == 1 || diff == num_pts - 1) {
        // Hard rule: adjacent neighbors on the same building are always visible.
        return true;
      } else {
        Point2D mid((p1.x + p2.x) * 0.5f, (p1.y + p2.y) * 0.5f);
        if (point_in_polygon(mid, buildings[b1].points)) {
          return false;
        }
      }
    }
  }

  float dx = p2.x - p1.x;
  float dy = p2.y - p1.y;
  float dist = std::sqrt(dx * dx + dy * dy);
  if (dist < kRayForwardEps) {
    return true;
  }

  bool edge_pass = true;
  float cell_size = spatial_grid.cell_size;
  float t_max_x, t_max_y, t_delta_x, t_delta_y;
  if (std::abs(dx) < kLineParallelEps) {
    t_max_x = t_delta_x = 1e30f;
  } else {
    float inv_dx = 1.0f / dx;
    t_max_x = (dx > 0)
                  ? (std::floor(p1.x / cell_size + 1.0f) * cell_size - p1.x) *
                        inv_dx
                  : (p1.x - std::floor(p1.x / cell_size) * cell_size) *
                        std::abs(inv_dx);
    t_delta_x = cell_size * std::abs(inv_dx);
  }

  if (std::abs(dy) < kLineParallelEps) {
    t_max_y = t_delta_y = 1e30f;
  } else {
    float inv_dy = 1.0f / dy;
    t_max_y = (dy > 0)
                  ? (std::floor(p1.y / cell_size + 1.0f) * cell_size - p1.y) *
                        inv_dy
                  : (p1.y - std::floor(p1.y / cell_size) * cell_size) *
                        std::abs(inv_dy);
    t_delta_y = cell_size * std::abs(inv_dy);
  }

  int cur_c = std::max(
      0, std::min(spatial_grid.cols - 1, static_cast<int>(p1.x / cell_size)));
  int cur_r = std::max(
      0, std::min(spatial_grid.rows - 1, static_cast<int>(p1.y / cell_size)));

  float t_curr = 0.0f;
  const float t_max = 1.0f;

  while (t_curr < t_max) {
    const auto &cell_segs = spatial_grid.cells[cur_r * spatial_grid.cols + cur_c];
    for (int seg_idx : cell_segs) {
      const auto &seg = all_segments[seg_idx];
      if (seg.building_id == effective_ignore_building_id) {
        continue;
      }
      if (segments_intersect(p1, p2, seg.p1, seg.p2)) {
        edge_pass = false;
        break;
      }
    }

    if (!edge_pass) {
      break;
    }

    if (t_max_x < t_max_y) {
      t_curr = t_max_x;
      t_max_x += t_delta_x;
      cur_c += (dx > 0) ? 1 : -1;
    } else {
      t_curr = t_max_y;
      t_max_y += t_delta_y;
      cur_r += (dy > 0) ? 1 : -1;
    }

    if (cur_c < 0 || cur_c >= spatial_grid.cols || cur_r < 0 ||
        cur_r >= spatial_grid.rows) {
      break;
    }
  }

  if (!edge_pass) {
    return false;
  }

  bool sample_pass =
      sample_path_pass(p1, p2, effective_ignore_building_id,
                       force_interior_building_id);
  return edge_pass && sample_pass;
}

} // namespace los2d
