cmake_minimum_required(VERSION 3.18)
project(los2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_PYBIND "Build pybind module" ON)

find_package(OpenMP REQUIRED)

add_library(los2d_core STATIC
  src/core/scene2d_ctor.cpp
  src/core/scene2d_dtor.cpp
  src/core/scene2d_load_polygons.cpp
  src/core/scene2d_preprocess_geometry.cpp
  src/core/scene2d_filter_boundary_vertices.cpp
  src/core/scene2d_build_grid.cpp
  src/core/scene2d_find_intersection.cpp
  src/core/scene2d_cast_ray_angle.cpp
  src/core/scene2d_cast_ray_angle_excluding_vertex.cpp
  src/core/scene2d_get_visibility_angles.cpp
  src/core/scene2d_calculate_visibility_polygon.cpp
  src/core/scene2d_is_path_clear.cpp
  src/core/scene2d_point_on_obstacle_boundary.cpp
  src/core/scene2d_compute_projection_point_like_python.cpp
  src/core/scene2d_is_visible.cpp
  src/core/scene2d_segments_intersect.cpp
  src/core/scene2d_point_in_polygon.cpp
  src/core/scene2d_compute_tx_perspective.cpp
  src/core/scene2d_compute_selected_tx.cpp
  src/core/scene2d_compute_vertex_perspective.cpp
  src/core/scene2d_generate_street_mask.cpp
  src/io/npy_writer.cpp
  src/io/npy_reader.cpp
  src/io/json_loader.cpp
  src/io/path_layout.cpp
  src/pipeline/stage_load_buildings.cpp
  src/pipeline/stage_build_scene.cpp
  src/pipeline/stage_collect_vertices.cpp
  src/pipeline/stage_sample_sources.cpp
  src/pipeline/stage_compute_tx_results.cpp
  src/pipeline/stage_compute_selected.cpp
  src/pipeline/stage_compute_vertex_fundamental.cpp
  src/pipeline/stage_save_outputs.cpp
  src/pipeline/stage_save_fundamental_outputs.cpp
  src/pipeline/run_pipeline.cpp
)

set_target_properties(los2d_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(los2d_core PUBLIC include)
target_link_libraries(los2d_core PUBLIC OpenMP::OpenMP_CXX)

add_executable(los2d_main src/main.cpp)
target_link_libraries(los2d_main PRIVATE los2d_core)

if(BUILD_PYBIND)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE pybind11_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE pybind11_RESULT
  )
  if(pybind11_RESULT EQUAL 0)
    find_package(pybind11 CONFIG REQUIRED HINTS "${pybind11_DIR}")
    pybind11_add_module(los2d_py src/bindings.cpp)
    target_link_libraries(los2d_py PRIVATE los2d_core)
    set_target_properties(los2d_py PROPERTIES PREFIX "")
  endif()
endif()
