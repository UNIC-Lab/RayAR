#include "core/scene2d.hpp"

namespace los2d {

void Scene2D::preprocess_geometry() {
  original_buildings = buildings;
  // filter_boundary_vertices();  // 已禁用：保留边界上的建筑顶点

  all_vertices.clear();
  all_segments.clear();
  vertex_to_building.clear();
  vertex_local_idx.clear();

  for (size_t b = 0; b < buildings.size(); ++b) {
    auto &bld = buildings[b];
    if (bld.points.size() < 3) {
      continue;
    }
    bld.global_vertex_indices.clear();
    for (size_t i = 0; i < bld.points.size(); ++i) {
      bld.global_vertex_indices.push_back(static_cast<int>(all_vertices.size()));
      vertex_to_building.push_back(bld.id);
      vertex_local_idx.push_back(static_cast<int>(i));
      all_vertices.push_back(bld.points[i]);
    }

    const auto &orig_pts = original_buildings[b].points;
    if (orig_pts.size() < 2) {
      continue;
    }

    for (size_t i = 0; i < orig_pts.size(); ++i) {
      const Point2D &p1 = orig_pts[i];
      const Point2D &p2 = orig_pts[(i + 1) % orig_pts.size()];
      if (p1.distance_to(p2) < kRayForwardEps) {
        continue;
      }

      Segment seg;
      seg.p1 = p1;
      seg.p2 = p2;
      seg.building_id = bld.id;
      seg.v1_idx = static_cast<int>(i);
      seg.v2_idx = static_cast<int>((i + 1) % orig_pts.size());
      all_segments.push_back(seg);
    }
  }

  build_grid();
}

} // namespace los2d
