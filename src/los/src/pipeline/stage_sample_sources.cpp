#include "pipeline/stages.hpp"
#include "io/npy_reader.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

namespace los2d {
namespace pipeline {

bool stage_sample_sources(const RunConfig &cfg, RunContext &ctx) {
  if (!cfg.source_points_path.empty()) {
    io::NpyArray input;
    if (!io::read_npy(cfg.source_points_path, input) ||
        input.shape.size() != 2 ||
        (input.shape[1] != 2 && input.shape[1] != 3)) {
      std::cerr << "Invalid fixed source file: " << cfg.source_points_path
                << "\n";
      return false;
    }

    const size_t count = input.shape[0];
    const size_t stride = input.shape[1];
    const float *values = input.f32();
    ctx.source_points.clear();
    ctx.source_points.reserve(count);
    for (size_t row = 0; row < count; ++row) {
      if (stride == 3 &&
          std::abs(values[row * stride + 2] - cfg.street_height_m) > 1e-4f) {
        std::cerr << "Fixed source height does not match street height\n";
        return false;
      }
      ctx.source_points.emplace_back(
          values[row * stride], values[row * stride + 1]);
    }
    ctx.selected_indices.resize(count);
    std::iota(ctx.selected_indices.begin(), ctx.selected_indices.end(), 0);
    ctx.selected_source_points = ctx.source_points;
    std::cout << "Loaded fixed source points: " << count << "\n";
    return true;
  }

  ctx.street_points = ctx.scene.generate_street_mask(1.0f, cfg.street_height_m);
  if (static_cast<int>(ctx.street_points.size()) < cfg.num_source) {
    std::cerr << "Not enough street points: " << ctx.street_points.size() << " < " << cfg.num_source << "\n";
    return false;
  }

  std::vector<int> indices(ctx.street_points.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng_source(cfg.seed_source);
  std::shuffle(indices.begin(), indices.end(), rng_source);

  ctx.source_points.clear();
  ctx.source_points.reserve(cfg.num_source);
  for (int i = 0; i < cfg.num_source; ++i) {
    ctx.source_points.push_back(ctx.street_points[indices[i]]);
  }

  int nsel = std::min(cfg.num_selected, cfg.num_source);
  std::vector<int> idx2(cfg.num_source);
  std::iota(idx2.begin(), idx2.end(), 0);
  std::mt19937 rng_sel(cfg.seed_selected);
  std::shuffle(idx2.begin(), idx2.end(), rng_sel);

  ctx.selected_indices.assign(idx2.begin(), idx2.begin() + nsel);
  ctx.selected_source_points.clear();
  ctx.selected_source_points.reserve(nsel);
  for (int i = 0; i < nsel; ++i) {
    ctx.selected_source_points.push_back(ctx.source_points[ctx.selected_indices[i]]);
  }

  std::cout << "Sampled source points: " << ctx.source_points.size() << " selected=" << ctx.selected_source_points.size() << "\n";
  return true;
}

} // namespace pipeline
} // namespace los2d
