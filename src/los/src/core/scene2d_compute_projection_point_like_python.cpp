#include "core/scene2d.hpp"
#include <algorithm>
#include <cmath>

namespace los2d {

Scene2D::ProjectionResult Scene2D::compute_projection_point_like_python(
    const Point2D &source,
    const Point2D &vertex,
    int vertex_building_id,
    const std::vector<Point2D> &visibility_polygon,
    const std::vector<float> &visibility_angles) const {

  constexpr float kPi = 3.14159265358979323846f;
  const float distance_threshold = kProjectionDistanceThreshold;
  float angle = std::atan2(vertex.y - source.y, vertex.x - source.x);

  if ((std::abs(vertex.x - 0.0f) < distance_threshold || std::abs(vertex.x - map_size) < distance_threshold ||
       std::abs(vertex.y - 0.0f) < distance_threshold || std::abs(vertex.y - map_size) < distance_threshold) &&
      is_path_clear(source, vertex)) {
    return {vertex.x, vertex.y, 0};
  }

  bool is_vertex_visible = false;
  for (const auto &vp : visibility_polygon) {
    if (vp.distance_to(vertex) < distance_threshold) {
      is_vertex_visible = true;
      break;
    }
  }

  float closest_angle = angle;
  float min_angle_diff = 1e9f;
  for (float va : visibility_angles) {
    float diff = std::min(std::abs(va - angle), 2.0f * kPi - std::abs(va - angle));
    if (diff < min_angle_diff) {
      min_angle_diff = diff;
      closest_angle = va;
    }
  }

  if (min_angle_diff > kAngleEps || !is_vertex_visible) {
    return {vertex.x, vertex.y, 2};
  }

  Intersection first = cast_ray_angle(source, closest_angle, -1);
  if (first.hit && first.pt.distance_to(vertex) < distance_threshold) {
    Intersection next = cast_ray_angle_excluding_vertex(source, closest_angle, vertex, -1);
    if (next.hit && is_path_clear(vertex, next.pt)) {
      Point2D proj = next.pt;

      if (std::abs(proj.x - 0.0f) < distance_threshold || std::abs(proj.x - map_size) < distance_threshold ||
          std::abs(proj.y - 0.0f) < distance_threshold || std::abs(proj.y - map_size) < distance_threshold) {
        return {proj.x, proj.y, 0};
      }

      for (const auto &v : all_vertices) {
        if (proj.distance_to(v) < distance_threshold) {
          return {proj.x, proj.y, 2};
        }
      }

      if (point_on_obstacle_boundary(proj, kPointOnEdgeEps)) {
        return {proj.x, proj.y, 1};
      }
      return {proj.x, proj.y, 1};
    }

    return {vertex.x, vertex.y, 2};
  }

  for (const auto &vp : visibility_polygon) {
    float a = std::atan2(vp.y - source.y, vp.x - source.x);
    float diff = std::min(std::abs(a - angle), 2.0f * kPi - std::abs(a - angle));
    if (diff < kAngleEps) {
      if ((std::abs(vp.x - 0.0f) < distance_threshold || std::abs(vp.x - map_size) < distance_threshold ||
           std::abs(vp.y - 0.0f) < distance_threshold || std::abs(vp.y - map_size) < distance_threshold) &&
          is_path_clear(source, vp)) {
        return {vp.x, vp.y, 0};
      }

      bool is_building_vertex = false;
      for (const auto &v : all_vertices) {
        if (vp.distance_to(v) < distance_threshold) {
          is_building_vertex = true;
          break;
        }
      }

      if (!is_building_vertex && is_path_clear(source, vp)) {
        if (point_on_obstacle_boundary(vp, kPointOnEdgeEps)) {
          return {vp.x, vp.y, 1};
        }
        return {vp.x, vp.y, 1};
      }
    }
  }

  (void)vertex_building_id;
  return {vertex.x, vertex.y, 2};
}

} // namespace los2d
