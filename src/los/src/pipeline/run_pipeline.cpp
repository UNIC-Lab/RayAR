#include "pipeline/run_pipeline.hpp"
#include "pipeline/stages.hpp"
#include <omp.h>

namespace los2d {
namespace pipeline {

int run_pipeline(const RunConfig &cfg) {
  if (cfg.threads > 0) {
    omp_set_num_threads(cfg.threads);
  }

  RunContext ctx;
  if (!stage_load_buildings(cfg, ctx)) return 1;
  if (!stage_build_scene(cfg, ctx)) return 2;
  if (!stage_collect_vertices(cfg, ctx)) return 3;
  if (!stage_sample_sources(cfg, ctx)) return 4;
  if (!stage_compute_tx_results(cfg, ctx)) return 5;
  if (!stage_compute_selected(cfg, ctx)) return 6;
  if (!stage_compute_vertex_fundamental(cfg, ctx)) return 7;
  if (!stage_save_outputs(cfg, ctx)) return 8;
  if (!stage_save_fundamental_outputs(cfg, ctx)) return 9;
  return 0;
}

} // namespace pipeline
} // namespace los2d
