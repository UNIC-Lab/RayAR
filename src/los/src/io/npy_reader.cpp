#include "io/npy_reader.hpp"

#include <fstream>

namespace los2d {
namespace io {

namespace {

std::string header_value(const std::string &header, const std::string &key) {
  size_t pos = header.find("'" + key + "'");
  if (pos == std::string::npos) return {};
  pos = header.find(':', pos);
  if (pos == std::string::npos) return {};
  pos = header.find_first_not_of(" \t", pos + 1);
  if (pos == std::string::npos) return {};
  if (header[pos] == '\'') {
    const size_t end = header.find('\'', pos + 1);
    return end == std::string::npos ? std::string()
                                    : header.substr(pos + 1, end - pos - 1);
  }
  const size_t end = header.find_first_of(",}", pos);
  return header.substr(pos, end - pos);
}

bool parse_shape(const std::string &header, std::vector<size_t> &shape) {
  size_t pos = header.find("'shape'");
  if (pos == std::string::npos) return false;
  pos = header.find('(', pos);
  if (pos == std::string::npos) return false;
  ++pos;
  shape.clear();
  while (pos < header.size()) {
    pos = header.find_first_not_of(" \t,", pos);
    if (pos == std::string::npos) return false;
    if (header[pos] == ')') return true;
    size_t end = pos;
    while (end < header.size() && header[end] >= '0' && header[end] <= '9') {
      ++end;
    }
    if (end == pos) return false;
    shape.push_back(static_cast<size_t>(std::stoull(header.substr(pos, end - pos))));
    pos = end;
  }
  return false;
}

} // namespace

size_t NpyArray::size() const {
  size_t count = 1;
  for (size_t dim : shape) count *= dim;
  return count;
}

const float *NpyArray::f32() const {
  return reinterpret_cast<const float *>(data.data());
}

bool read_npy(const std::string &path, NpyArray &out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;

  char magic[6];
  input.read(magic, 6);
  if (!input || std::string(magic, 6) != "\x93NUMPY") return false;

  uint8_t major = 0;
  uint8_t minor = 0;
  input.read(reinterpret_cast<char *>(&major), 1);
  input.read(reinterpret_cast<char *>(&minor), 1);
  (void)minor;

  uint32_t header_size = 0;
  if (major == 1) {
    uint16_t size16 = 0;
    input.read(reinterpret_cast<char *>(&size16), 2);
    header_size = size16;
  } else if (major == 2 || major == 3) {
    input.read(reinterpret_cast<char *>(&header_size), 4);
  } else {
    return false;
  }

  std::string header(header_size, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!input || header_value(header, "fortran_order") == "True") return false;

  out.descr = header_value(header, "descr");
  if (out.descr != "<f4" || !parse_shape(header, out.shape)) return false;

  out.data.resize(out.size() * sizeof(float));
  input.read(reinterpret_cast<char *>(out.data.data()),
             static_cast<std::streamsize>(out.data.size()));
  return static_cast<bool>(input);
}

} // namespace io
} // namespace los2d
