#pragma once
#include <string>
#include <vector>

namespace los2d {
namespace io {

bool load_buildings_json(const std::string &path,
                        std::vector<std::vector<std::vector<float>>> &polygons);

} // namespace io
} // namespace los2d
