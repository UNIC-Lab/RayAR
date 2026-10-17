#include "pipeline/run_pipeline.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  los2d::pipeline::RunConfig cfg;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need_value = [&](const std::string &name) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(2);
      }
      return std::string(argv[++i]);
    };

    if (a == "--map-id")
      cfg.map_id = std::stoi(need_value(a));
    else if (a == "--data-root")
      cfg.data_root = need_value(a);
    else if (a == "--output-root")
      cfg.output_root = need_value(a);
    else if (a == "--source-points")
      cfg.source_points_path = need_value(a);
    else if (a == "--num-source")
      cfg.num_source = std::stoi(need_value(a));
    else if (a == "--num-selected")
      cfg.num_selected = std::stoi(need_value(a));
    else if (a == "--seed-source")
      cfg.seed_source = std::stoi(need_value(a));
    else if (a == "--seed-selected")
      cfg.seed_selected = std::stoi(need_value(a));
    else if (a == "--threads")
      cfg.threads = std::stoi(need_value(a));
    else if (a == "--vertex-height")
      cfg.vertex_height_m = std::stof(need_value(a));
    else if (a == "--street-height")
      cfg.street_height_m = std::stof(need_value(a));
    else if (a == "--force")
      cfg.force = true;
    else if (a == "--no-fundamental")
      cfg.emit_fundamental_vertex = false;
    else if (a == "--help") {
      std::cout << "los2d_main --map-id N [--data-root PATH] "
                   "[--output-root PATH] [--source-points NPY] "
                   "[--num-source N] "
                   "[--num-selected N] [--seed-source N] [--seed-selected N] "
                   "[--threads N] [--vertex-height M] [--street-height M] "
                   "[--no-fundamental] [--force]\n";
      return 0;
    }
  }

  return los2d::pipeline::run_pipeline(cfg);
}
