#pragma once
#include <string>
#include <vector>

namespace los2d {
namespace io {

std::string output_dir(const std::string &data_root, int map_id,
                       const std::string &output_root = "");
std::string output_file(const std::string &dir, const std::string &name,
                        int map_id);
std::vector<std::string> expected_outputs(const std::string &dir, int map_id);

} // namespace io
} // namespace los2d
