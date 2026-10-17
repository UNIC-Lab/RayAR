#include "core/scene2d.hpp"
#include <omp.h>

namespace los2d {

std::vector<Point2D> Scene2D::generate_street_mask(float grid_step, float dev_height) {
  (void)dev_height;
  std::vector<Point2D> valid_points;
  int grid_size = static_cast<int>(map_size / grid_step) + 1;
  valid_points.reserve(grid_size * grid_size / 2);

#pragma omp parallel
  {
    std::vector<Point2D> local_points;
    local_points.reserve(grid_size * grid_size / 8);

#pragma omp for schedule(dynamic, 16)
    for (int r = 0; r < grid_size; ++r) {
      for (int c = 0; c < grid_size; ++c) {
        Point2D pt(c * grid_step, r * grid_step);
        bool inside_building = false;
        for (const auto &bld : original_buildings) {
          if (point_in_polygon(pt, bld.points)) {
            inside_building = true;
            break;
          }
        }
        if (!inside_building && !point_on_obstacle_boundary(pt, 1.0f)) {
          local_points.push_back(pt);
        }
      }
    }

#pragma omp critical
    {
      valid_points.insert(valid_points.end(), local_points.begin(), local_points.end());
    }
  }

  return valid_points;
}

} // namespace los2d
