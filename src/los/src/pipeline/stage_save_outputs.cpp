#include "io/npy_writer.hpp"
#include "io/path_layout.hpp"
#include "pipeline/stages.hpp"
#include <filesystem>
#include <iostream>

namespace los2d {
namespace pipeline {

bool stage_save_outputs(const RunConfig &cfg, RunContext &ctx) {
  std::string out_dir =
      io::output_dir(cfg.data_root, cfg.map_id, cfg.output_root);
  std::filesystem::create_directories(out_dir);

  auto expected = io::expected_outputs(out_dir, cfg.map_id);
  bool all_exists = true;
  for (const auto &f : expected) {
    if (!std::filesystem::exists(f)) {
      all_exists = false;
      break;
    }
  }
  if (all_exists && !cfg.force) {
    std::cout << "All output files exist, skip map " << cfg.map_id << "\n";
    return true;
  }

  const auto &verts = ctx.scene.get_vertices();
  size_t n = ctx.tx_results.num_tx;
  size_t m = ctx.tx_results.m;
  size_t s = ctx.selected_indices.size();
  size_t u = ctx.selected_visible_vertex_points.size() / 2;

  std::vector<float> source_points(n * 2, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    source_points[i * 2] = ctx.source_points[i].x;
    source_points[i * 2 + 1] = ctx.source_points[i].y;
  }

  std::vector<float> vertex_points(m * 2, 0.0f);
  for (size_t i = 0; i < m; ++i) {
    vertex_points[i * 2] = verts[i].x;
    vertex_points[i * 2 + 1] = verts[i].y;
  }

  std::vector<float> selected_source_points(s * 2, 0.0f);
  for (size_t i = 0; i < s; ++i) {
    selected_source_points[i * 2] = ctx.selected_source_points[i].x;
    selected_source_points[i * 2 + 1] = ctx.selected_source_points[i].y;
  }

  bool ok = true;
  ok &= io::write_npy_f32(io::output_file(out_dir, "source_points", cfg.map_id),
                          {n, 2}, source_points.data());
  ok &= io::write_npy_f32(io::output_file(out_dir, "vertex_points", cfg.map_id),
                          {m, 2}, vertex_points.data());
  ok &= io::write_npy_i8(
      io::output_file(out_dir, "adjacency_matrix", cfg.map_id),
      {n, m + 1, m + 1}, ctx.tx_results.adjacency_matrix.data());
  ok &= io::write_npy_f32(io::output_file(out_dir, "angle_matrix", cfg.map_id),
                          {n, m, 2}, ctx.tx_results.angle_matrix.data());
  ok &= io::write_npy_f32(
      io::output_file(out_dir, "source_to_vertex_distance_matrix", cfg.map_id),
      {n, m}, ctx.tx_results.dist_matrix.data());
  ok &= io::write_npy_i8(
      io::output_file(out_dir, "source_visibility_vertex_mask", cfg.map_id),
      {n, 257, 257}, ctx.source_visibility_vertex_mask.data());
  ok &= io::write_npy_i8(
      io::output_file(out_dir, "source_visibility_mask", cfg.map_id),
      {n, 257, 257}, ctx.tx_results.visibility_mask.data());
  ok &= io::write_npy_i8(
      io::output_file(out_dir, "selected_visibility_vertex_mask", cfg.map_id),
      {s, 257, 257}, ctx.selected_results.vertex_mask.data());
  ok &= io::write_npy_f32(
      io::output_file(out_dir, "selected_angle_matrix", cfg.map_id), {s, u, 2},
      ctx.selected_angle_matrix.data());
  ok &= io::write_npy_f32(
      io::output_file(out_dir, "selected_visible_vertex_points", cfg.map_id),
      {u, 2}, ctx.selected_visible_vertex_points.data());
  ok &= io::write_npy_f32(
      io::output_file(out_dir, "selected_source_points", cfg.map_id), {s, 2},
      selected_source_points.data());
  ok &= io::write_npy_i8(
      io::output_file(out_dir, "selected_visibility_street_mask", cfg.map_id),
      {s, 257, 257}, ctx.selected_results.street_mask.data());

  if (!ok) {
    std::cerr << "Failed to write one or more output files\n";
    return false;
  }
  return true;
}

} // namespace pipeline
} // namespace los2d
