#include "io/npy_writer.hpp"
#include <fstream>
#include <sstream>

namespace los2d {
namespace io {

namespace {

template <typename T>
bool write_npy_impl(const std::string &path,
                    const std::vector<size_t> &shape,
                    const T *data,
                    const char *descr) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    return false;
  }

  std::ostringstream hdr;
  hdr << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    hdr << shape[i];
    if (shape.size() == 1) {
      hdr << ",";
    } else if (i + 1 < shape.size()) {
      hdr << ", ";
    }
  }
  hdr << "), }";
  std::string header = hdr.str();

  const std::string magic = "\x93NUMPY";
  uint8_t major = 1;
  uint8_t minor = 0;

  size_t preamble = magic.size() + 2 + 2;
  size_t padding = 16 - ((preamble + header.size() + 1) % 16);
  if (padding == 16) {
    padding = 0;
  }
  header.append(padding, ' ');
  header.push_back('\n');

  uint16_t hlen = static_cast<uint16_t>(header.size());
  ofs.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  ofs.write(reinterpret_cast<const char *>(&major), 1);
  ofs.write(reinterpret_cast<const char *>(&minor), 1);
  ofs.write(reinterpret_cast<const char *>(&hlen), 2);
  ofs.write(header.data(), static_cast<std::streamsize>(header.size()));

  size_t count = 1;
  for (size_t d : shape) {
    count *= d;
  }
  ofs.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(count * sizeof(T)));
  return static_cast<bool>(ofs);
}

} // namespace

bool write_npy_f32(const std::string &path, const std::vector<size_t> &shape, const float *data) {
  return write_npy_impl(path, shape, data, "<f4");
}

bool write_npy_i8(const std::string &path, const std::vector<size_t> &shape, const int8_t *data) {
  return write_npy_impl(path, shape, data, "|i1");
}

bool write_npy_i32(const std::string &path, const std::vector<size_t> &shape, const int32_t *data) {
  return write_npy_impl(path, shape, data, "<i4");
}

} // namespace io
} // namespace los2d
