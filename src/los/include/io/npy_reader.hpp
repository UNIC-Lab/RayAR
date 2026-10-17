#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace los2d {
namespace io {

struct NpyArray {
  std::vector<size_t> shape;
  std::vector<uint8_t> data;
  std::string descr;

  size_t size() const;
  const float *f32() const;
};

bool read_npy(const std::string &path, NpyArray &out);

} // namespace io
} // namespace los2d
