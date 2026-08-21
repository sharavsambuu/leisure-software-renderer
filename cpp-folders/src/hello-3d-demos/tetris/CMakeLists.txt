# =============================================================================
# Hello 3D Tetris Demo — CMake Target Definition
# Location: cpp-folders/src/hello-3d-demos/tetris/CMakeLists.txt
# Parent aggregator: cpp-folders/src/hello-3d-demos/CMakeLists.txt (includes this subdir)
#
# NOTE: All shared dependencies and alias definitions are handled in the parent
# CMakeLists.txt to avoid "target already exists" errors. This file only defines
# its own executable, compile options, link libraries, and platform flags.
# =============================================================================

cmake_minimum_required(VERSION 3.28)
project(Hello3DTetris VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies (shared across all demos — defined in parent CMakeLists.txt) ---
find_package(SDL2 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})
# Include the shared software renderer (shs_renderer.hpp)
include_directories(${HelloSHSRenderer_SOURCE_DIR}/)

include_directories(${stb_SOURCE_DIR}/)

# --- Main executable: Hello3DTetris (multi-file domain pod architecture) ---
add_executable(Hello3DTetris
    hello_3d_tetris.cpp
    domains/matrix/matrix.contract.hpp
    domains/matrix/matrix.action.hpp
    domains/matrix/matrix.event.hpp
    domains/matrix/matrix.reducer.hpp
    domains/matrix/matrix.plan.hpp
    domains/progression/progression.contract.hpp
    domains/progression/progression.action.hpp
    domains/progression/progression.event.hpp
    domains/progression/progression.reducer.hpp
    domains/spatial_fx/spatial_fx.contract.hpp
    domains/spatial_fx/spatial_fx.reducer.hpp
)

# --- Compile options (shared across demos for consistency) ---
target_compile_options(Hello3DTetris PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<NOT:$<CONFIG:Debug>>>:-O3 -march=native -flto>
    $<$<OR:$<CXX_COMPILER_ID:GNU,Clang>,$<CXX_COMPILER_ID:Intel>>:-Wextra -Wall -Wpedantic -Wshadow -Wcast-qual -Wcast-align -Wunused -Wconversion -Wsign-conversion -Wfloat-conversion>
)

target_compile_definitions(Hello3DTetris PRIVATE GLM_ENABLE_EXPERIMENTAL)

# --- Link libraries (SDL2_image::Main is defined once globally in parent CMakeLists.txt) ---
target_link_libraries(Hello3DTetris PRIVATE
    SDL2::SDL2
    SDL2_image::Main
    glm::glm
    assimp::assimp
)

# --- Platform-specific optimizations ---
if (MSVC)
    target_compile_options(Hello3DTetris PRIVATE /O2 /fp:fast /arch:AVX2)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(Hello3DTetris PRIVATE -O3 -ffast-math)
else()
    target_compile_options(Hello3DTetris PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()

message(STATUS ">>> Building Hello3DTetris (multi-file domain pod architecture)...")
