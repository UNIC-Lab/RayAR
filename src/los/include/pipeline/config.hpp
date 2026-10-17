#pragma once
#include <string>

namespace los2d {
namespace pipeline {

struct RunConfig {
  int map_id{0};
  std::string data_root{"assets"};
  std::string output_root; // if empty, defaults to data_root
  std::string source_points_path;
  int num_source{5000};
  int num_selected{100};
  int seed_source{20};
  int seed_selected{42};
  int threads{0};
  bool force{false};
  float vertex_height_m{20.0f};
  float street_height_m{1.5f};
  bool emit_fundamental_vertex{true};
};

} // namespace pipeline
} // namespace los2d
