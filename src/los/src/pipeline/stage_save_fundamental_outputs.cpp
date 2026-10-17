#include "io/npy_writer.hpp"
#include "io/path_layout.hpp"
#include "pipeline/stages.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

namespace los2d {
namespace pipeline {

bool stage_save_fundamental_outputs(const RunConfig &cfg, RunContext &ctx) {
  if (!cfg.emit_fundamental_vertex) {
    return true;
  }

  const size_t m = ctx.scene.get_vertices().size();
  std::string fundamental_dir =
      io::output_dir(cfg.data_root, cfg.map_id, cfg.output_root) +
      "/fundamental";
  std::filesystem::create_directories(fundamental_dir);

  std::vector<std::string> expected = {
      io::output_file(fundamental_dir, "raw_vertex_visibility_mask",
                      cfg.map_id),
      io::output_file(fundamental_dir, "vertex_visibility_mask", cfg.map_id),
      io::output_file(fundamental_dir, "street_visibility_mask", cfg.map_id),
      io::output_file(fundamental_dir, "vertex_distance_mask", cfg.map_id),
      io::output_file(fundamental_dir, "vertex_angle_mask", cfg.map_id),
      io::output_file(fundamental_dir, "street_distance_mask", cfg.map_id),
      io::output_file(fundamental_dir, "street_angle_mask", cfg.map_id),
  };

  bool all_exists = true;
  for (const auto &f : expected) {
    if (!std::filesystem::exists(f)) {
      all_exists = false;
      break;
    }
  }
  if (all_exists && !cfg.force) {
    std::cout << "Fundamental output files exist, skip map " << cfg.map_id
              << "\n";
    return true;
  }

  bool ok = true;
  ok &= io::write_npy_i8(expected[0], {m, 257, 257},
                         ctx.raw_vertex_visibility_mask.data());
  ok &= io::write_npy_i8(expected[1], {m, 257, 257},
                         ctx.vertex_visibility_mask.data());
  ok &= io::write_npy_i8(expected[2], {m, 257, 257},
                         ctx.street_visibility_mask.data());
  ok &= io::write_npy_f32(expected[3], {m, 257, 257},
                          ctx.vertex_distance_mask.data());
  ok &= io::write_npy_f32(expected[4], {m, 257, 257},
                          ctx.vertex_angle_mask.data());
  ok &= io::write_npy_f32(expected[5], {m, 257, 257},
                          ctx.street_distance_mask.data());
  ok &= io::write_npy_f32(expected[6], {m, 257, 257, 2},
                          ctx.street_angle_mask.data());

  if (!ok) {
    std::cerr << "Failed to write one or more fundamental output files\n";
    return false;
  }
  return true;
}

} // namespace pipeline
} // namespace los2d
