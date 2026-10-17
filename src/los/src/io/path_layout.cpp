#include "io/path_layout.hpp"
#include <sstream>

namespace los2d {
namespace io {

std::string output_dir(const std::string &data_root, int map_id,
                       const std::string &output_root) {
  const std::string &root = output_root.empty() ? data_root : output_root;
  std::ostringstream oss;
  oss << root << "/map_" << map_id;
  return oss.str();
}

std::string output_file(const std::string &dir, const std::string &name,
                        int map_id) {
  std::ostringstream oss;
  oss << dir << "/" << name << "_" << map_id << ".npy";
  return oss.str();
}

std::vector<std::string> expected_outputs(const std::string &dir, int map_id) {
  return {
      output_file(dir, "source_points", map_id),
      output_file(dir, "vertex_points", map_id),
      output_file(dir, "adjacency_matrix", map_id),
      output_file(dir, "angle_matrix", map_id),
      output_file(dir, "source_to_vertex_distance_matrix", map_id),
      output_file(dir, "source_visibility_vertex_mask", map_id),
      output_file(dir, "source_visibility_mask", map_id),
      output_file(dir, "selected_visibility_vertex_mask", map_id),
      output_file(dir, "selected_angle_matrix", map_id),
      output_file(dir, "selected_visible_vertex_points", map_id),
      output_file(dir, "selected_source_points", map_id),
      output_file(dir, "selected_visibility_street_mask", map_id),
  };
}

} // namespace io
} // namespace los2d
