#include "io/json_loader.hpp"
#include <fstream>
#include <string>
#include <vector>

namespace los2d {
namespace io {

namespace {

void skip_ws(const std::string &s, size_t &i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) {
    ++i;
  }
}

bool parse_number(const std::string &s, size_t &i, float &out) {
  skip_ws(s, i);
  size_t start = i;
  if (i < s.size() && (s[i] == '-' || s[i] == '+')) {
    ++i;
  }
  bool has_digit = false;
  while (i < s.size() && (s[i] >= '0' && s[i] <= '9')) {
    has_digit = true;
    ++i;
  }
  if (i < s.size() && s[i] == '.') {
    ++i;
    while (i < s.size() && (s[i] >= '0' && s[i] <= '9')) {
      has_digit = true;
      ++i;
    }
  }
  if (!has_digit) {
    return false;
  }
  out = std::stof(s.substr(start, i - start));
  return true;
}

bool parse_point(const std::string &s, size_t &i, std::vector<float> &pt) {
  skip_ws(s, i);
  if (i >= s.size() || s[i] != '[') return false;
  ++i;
  float x = 0.0f, y = 0.0f;
  if (!parse_number(s, i, x)) return false;
  skip_ws(s, i);
  if (i >= s.size() || s[i] != ',') return false;
  ++i;
  if (!parse_number(s, i, y)) return false;
  skip_ws(s, i);
  if (i >= s.size() || s[i] != ']') return false;
  ++i;
  pt = {x, y};
  return true;
}

bool parse_polygon(const std::string &s, size_t &i, std::vector<std::vector<float>> &poly) {
  skip_ws(s, i);
  if (i >= s.size() || s[i] != '[') return false;
  ++i;

  skip_ws(s, i);
  if (i < s.size() && s[i] == ']') {
    ++i;
    return true;
  }

  while (i < s.size()) {
    std::vector<float> pt;
    if (!parse_point(s, i, pt)) return false;
    poly.push_back(pt);
    skip_ws(s, i);
    if (i < s.size() && s[i] == ',') {
      ++i;
      continue;
    }
    if (i < s.size() && s[i] == ']') {
      ++i;
      return true;
    }
    return false;
  }
  return false;
}

} // namespace

bool load_buildings_json(const std::string &path, std::vector<std::vector<std::vector<float>>> &polygons) {
  std::ifstream ifs(path);
  if (!ifs) {
    return false;
  }
  std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  size_t i = 0;
  skip_ws(content, i);
  if (i >= content.size() || content[i] != '[') {
    return false;
  }
  ++i;

  skip_ws(content, i);
  if (i < content.size() && content[i] == ']') {
    return true;
  }

  while (i < content.size()) {
    std::vector<std::vector<float>> poly;
    if (!parse_polygon(content, i, poly)) {
      return false;
    }
    polygons.push_back(poly);
    skip_ws(content, i);
    if (i < content.size() && content[i] == ',') {
      ++i;
      continue;
    }
    if (i < content.size() && content[i] == ']') {
      return true;
    }
    return false;
  }

  return false;
}

} // namespace io
} // namespace los2d
