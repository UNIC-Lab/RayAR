#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace los2d {
namespace io {

bool write_npy_f32(const std::string &path, const std::vector<size_t> &shape, const float *data);
bool write_npy_i8(const std::string &path, const std::vector<size_t> &shape, const int8_t *data);
bool write_npy_i32(const std::string &path, const std::vector<size_t> &shape, const int32_t *data);

} // namespace io
} // namespace los2d
