# =============================================================================
# Hello 3D Demos — Aggregator CMake File
# Location: cpp-folders/src/hello-3d-demos/CMakeLists.txt
# Description: Includes all demo subdirectories (fps, plane, tetris, snake).
# Each demo has its own independent CMakeLists.txt in its respective subdir.
# =============================================================================

cmake_minimum_required(VERSION 3.28)
project(Hello3DDemos VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Shared dependencies (defined once in parent, exposed to all subdirs via add_subdirectory) ---
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG QUIET)   # optional: no demo uses image loading; make it non-fatal
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

# SDL2_image ships either a shared import target (SDL2_image::SDL2_image) or, when built static,
# an imported "SDL2_image-static" target. The demos link the unified scoped alias SDL2_image::Main.
# Guard ALL branches: on some vcpkg toolchains neither branch's fallback exists, and add_library ALIAS
# crashes if it aliases a non-existent target (see issue log). When SDL2_image is not installed we
# define an unscoped INTERFACE library "SDL2_image_Main" and alias it to the scoped name demos expect.
if(TARGET SDL2_image::SDL2_image)
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image)
elseif(TARGET SDL2_image-static)
    add_library(SDL2_image::Main ALIAS SDL2_image-static)
else()
    message(STATUS "SDL2_image not found — treating as optional (no demo uses image loading).")
    add_library(SDL2_image_Main INTERFACE)
    add_library(SDL2_image::Main ALIAS SDL2_image_Main)
endif()

# Include the shared software renderer library (shs_renderer.hpp) from hello-3d-primitives
include_directories(${HelloSHSRenderer_SOURCE_DIR}/)

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${stb_SOURCE_DIR}/)

# --- Demo pod roots: add each demo's root to the global include path so that
# within-pod bare includes (e.g. "snake.contract.hpp", "shs_renderer.hpp") resolve
# from any subdir, mirroring how tetris resolves cross-pod types via namespaces +
# global include dirs. No relative ("../../../...") paths are used in source headers.
include_directories(${CMAKE_CURRENT_LIST_DIR}/fps)
include_directories(${CMAKE_CURRENT_LIST_DIR}/plane)
include_directories(${CMAKE_CURRENT_LIST_DIR}/tetris)
include_directories(${CMAKE_CURRENT_LIST_DIR}/snake)

# --- Include all demo subdirectories (each is self-contained with its own CMakeLists.txt) ---
add_subdirectory(fps)          # fps/CMakeLists.txt  → HelloFPSDemo
add_subdirectory(plane)        # plane/CMakeLists.txt → HelloPlaneDemo  
add_subdirectory(tetris)       # tetris/CMakeLists.txt → Hello3DTetris
add_subdirectory(snake)        # snake/CMakeLists.txt   → Hello3DSnake

# Optional: create a combined "AllDemos" target for convenience
add_custom_target(all_demos ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Building all Hello 3D demos..."
    DEPENDS 
        HelloFPSDemo 
        HelloPlaneDemo 
        Hello3DTetris 
        Hello3DSnake
)

message(STATUS "")
message(STATUS "=== Hello 3D Demos ===")
message(STATUS "   fps/CMakeLists.txt          → HelloFPSDemo")
message(STATUS "   plane/CMakeLists.txt        → HelloPlaneDemo")
message(STATUS "   tetris/CMakeLists.txt       → Hello3DTetris")
message(STATUS "   snake/CMakeLists.txt        → Hello3DSnake")
message(STATUS "")
