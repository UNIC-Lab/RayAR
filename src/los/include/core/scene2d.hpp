#pragma once
#include "core/building.hpp"
#include "core/grid.hpp"
#include "core/intersection.hpp"
#include "core/segment.hpp"
#include <cstdint>
#include <vector>

namespace los2d {

class Scene2D {
public:
  struct ProjectionResult {
    float x{0.0f};
    float y{0.0f};
    int type{2}; // 0 boundary, 1 obstacle edge, 2 vertex/self
  };

  struct TXResults {
    std::vector<int8_t> adjacency_matrix;
    std::vector<float> angle_matrix;
    std::vector<float> dist_matrix;
    std::vector<int8_t> visibility_mask;
    std::vector<float> feature_points;
    size_t num_tx{0};
    size_t m{0};
  };

  struct SelectedTXResults {
    std::vector<int8_t> street_mask;
    std::vector<int8_t> vertex_mask;
    std::vector<int> selected_indices;
  };

  struct VertexResults {
    std::vector<int8_t> vertex_visibility_grid;
    std::vector<int8_t> vertex_adjacency;
    std::vector<float> vertex_dist_grid;
    std::vector<float> vertex_angles_grid;
    std::vector<float> feature_points;
    size_t m{0};
  };

  Scene2D();
  ~Scene2D();

  bool
  load_polygons(const std::vector<std::vector<std::vector<float>>> &poly_data);

  void preprocess_geometry();
  void filter_boundary_vertices();
  void build_grid();
  Intersection find_intersection(Point2D origin, Point2D dir,
                                 int ignore_building_id = -1) const;
  bool is_visible(const Point2D &p1, const Point2D &p2,
                  int ignore_building_id = -1, int p1_v_idx = -1,
                  int p2_v_idx = -1) const;
  bool segments_intersect(const Point2D &a, const Point2D &b, const Point2D &c,
                          const Point2D &d) const;
  bool point_in_polygon(const Point2D &pt,
                        const std::vector<Point2D> &polygon) const;
  bool point_on_obstacle_boundary(const Point2D &pt,
                                  float tolerance = 1e-6f) const;
  bool is_path_clear(const Point2D &start, const Point2D &end) const;
  Intersection cast_ray_angle(const Point2D &origin, float angle,
                              int ignore_building_id = -1) const;
  Intersection
  cast_ray_angle_excluding_vertex(const Point2D &origin, float angle,
                                  const Point2D &exclude_vertex,
                                  int ignore_building_id = -1) const;
  std::vector<float> get_visibility_angles(const Point2D &observer) const;
  std::vector<Point2D>
  calculate_visibility_polygon(const Point2D &observer) const;
  ProjectionResult compute_projection_point_like_python(
      const Point2D &source, const Point2D &vertex, int vertex_building_id,
      const std::vector<Point2D> &visibility_polygon,
      const std::vector<float> &visibility_angles) const;
  TXResults compute_tx_perspective(const std::vector<Point2D> &tx_sources,
                                   float freq_ghz);
  SelectedTXResults compute_selected_tx(const std::vector<Point2D> &tx_sources,
                                        const std::vector<int> &indices);
  VertexResults compute_vertex_perspective(float freq_ghz);
  std::vector<Point2D> generate_street_mask(float grid_step = 1.0f,
                                            float device_height = 1.5f);

  float get_map_size() const { return map_size; }
  size_t get_num_vertices() const { return all_vertices.size(); }
  const std::vector<Point2D> &get_vertices() const { return all_vertices; }
  int get_vertex_building(size_t idx) const { return vertex_to_building[idx]; }
  void set_heights(float vertex_height_m, float street_height_m) {
    building_height = vertex_height_m;
    device_height = street_height_m;
  }
  float get_vertex_height() const { return building_height; }
  float get_street_height() const { return device_height; }

private:
  bool sample_path_pass(const Point2D &p1, const Point2D &p2,
                        int ignore_building_id,
                        int force_interior_building_id) const;

  std::vector<Building> buildings;
  std::vector<Building> original_buildings;
  std::vector<Point2D> all_vertices;
  std::vector<Segment> all_segments;
  std::vector<int> vertex_to_building;
  std::vector<int> vertex_local_idx;
  Grid spatial_grid;

  float map_size{256.0f};
  float building_height{20.0f};
  float device_height{1.5f};

  static constexpr float kBoundaryEps = 1e-6f;
  static constexpr float kAngleEps = 1e-6f;
  static constexpr float kProjectionDistanceThreshold = 0.1f;
  static constexpr float kRayForwardEps = 1e-10f;
  static constexpr float kPointOnEdgeEps = 1e-6f;
  static constexpr float kLineParallelEps = 1e-8f;
};

} // namespace los2d
