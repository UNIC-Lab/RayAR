#include "pipeline/stages.hpp"
#include "io/json_loader.hpp"
#include <iostream>

namespace los2d {
namespace pipeline {

bool stage_load_buildings(const RunConfig &cfg, RunContext &ctx) {
  std::string json_path = cfg.data_root + "/buildings/" + std::to_string(cfg.map_id) + ".json";
  if (io::load_buildings_json(json_path, ctx.polygons)) {
    std::cout << "Loaded buildings: " << json_path << " count=" << ctx.polygons.size() << "\n";
    return true;
  }

  std::cout << "Building json missing/unparseable, using fallback polygons\n";
  ctx.polygons = {
      {{237, 103}, {250, 103}, {250, 120}, {237, 120}},
      {{100, 232}, {120, 232}, {120, 235}, {100, 235}},
      {{150, 237}, {170, 237}, {170, 241}, {150, 241}},
      {{85, 238}, {95, 238}, {95, 246}, {85, 246}},
  };
  return true;
}

} // namespace pipeline
} // namespace los2d
