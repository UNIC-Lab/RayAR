#include "pipeline/stages.hpp"

namespace los2d {
namespace pipeline {

bool stage_build_scene(const RunConfig &cfg, RunContext &ctx) {
  ctx.scene.set_heights(cfg.vertex_height_m, cfg.street_height_m);
  return ctx.scene.load_polygons(ctx.polygons);
}

} // namespace pipeline
} // namespace los2d
