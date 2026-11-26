cmake_minimum_required(VERSION 3.22)
project(spatial LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -Wall -Wextra -Wno-unused-parameter
    $<$<CONFIG:Release>:-O3>
)

# Shader directory macro so binaries locate .metal files reliably
set(SPATIAL_SHADER_DIR "${CMAKE_SOURCE_DIR}/src/backend/metal/shaders")

# --- Core library (pure C++20) ---
add_library(spatial_core STATIC
    src/core/spat_format.cpp
    src/core/profiler.cpp
)
target_include_directories(spatial_core PUBLIC include)

# --- Metal backend (Objective-C++20 + MSL runtime compilation) ---
add_library(spatial_metal STATIC
    src/backend/metal/metal_backend.mm
)
target_include_directories(spatial_metal PUBLIC include)
target_compile_options(spatial_metal PRIVATE -fobjc-arc)
target_compile_definitions(spatial_metal PUBLIC SPATIAL_SHADER_DIR="${SPATIAL_SHADER_DIR}")
target_link_libraries(spatial_metal PUBLIC
    spatial_core
    "-framework Metal"
    "-framework Foundation"
    "-framework Accelerate"
)

# --- Models library ---
add_library(spatial_models STATIC
    src/cache/spatial_kv_cache.cpp
    src/models/taehv.cpp
    src/models/waypoint.cpp
)
target_include_directories(spatial_models PUBLIC include)
target_link_libraries(spatial_models PUBLIC spatial_metal)

# --- Unit & parity tests ---
add_executable(spatial_tests
    tests/test_core.cpp
)
target_link_libraries(spatial_tests PRIVATE spatial_metal)

add_executable(test_taehv
    tests/test_taehv.cpp
)
target_link_libraries(test_taehv PRIVATE spatial_models)

add_executable(test_dit_kernels
    tests/test_dit_kernels.cpp
)
target_link_libraries(test_dit_kernels PRIVATE spatial_models)

# --- M3 Headless CLI (parity verification + world rollout + profiler) ---
add_executable(spatial_cli
    src/app/spatial_cli.cpp
)
target_link_libraries(spatial_cli PRIVATE spatial_models)

# --- M4 Native macOS Interactive Metal App (WASD + Mouse -> Live World!) ---
add_executable(spatial_app
    src/app/spatial_app.mm
)
target_compile_options(spatial_app PRIVATE -fobjc-arc)
target_link_libraries(spatial_app PRIVATE
    spatial_models
    "-framework Cocoa"
    "-framework QuartzCore"
)
