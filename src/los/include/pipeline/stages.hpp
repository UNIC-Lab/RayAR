#pragma once
#include "pipeline/config.hpp"
#include "pipeline/context.hpp"

namespace los2d {
namespace pipeline {

bool stage_load_buildings(const RunConfig &cfg, RunContext &ctx);
bool stage_build_scene(const RunConfig &cfg, RunContext &ctx);
bool stage_collect_vertices(const RunConfig &cfg, RunContext &ctx);
bool stage_sample_sources(const RunConfig &cfg, RunContext &ctx);
bool stage_compute_tx_results(const RunConfig &cfg, RunContext &ctx);
bool stage_compute_selected(const RunConfig &cfg, RunContext &ctx);
bool stage_compute_vertex_fundamental(const RunConfig &cfg, RunContext &ctx);
bool stage_save_outputs(const RunConfig &cfg, RunContext &ctx);
bool stage_save_fundamental_outputs(const RunConfig &cfg, RunContext &ctx);

} // namespace pipeline
} // namespace los2d
