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
find_package(SDL2_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

if(TARGET SDL2_image::SDL2_image)
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image)
else()
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image-static)
endif()

# Include the shared software renderer library (shs_renderer.hpp) from hello-3d-primitives
include_directories(${HelloSHSRenderer_SOURCE_DIR}/)

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${stb_SOURCE_DIR}/)

# --- Include all demo subdirectories (each is self-contained with its own CMakeLists.txt) ---
add_subdirectory(fps)          # fps/CMakeLists.txt  → HelloFPSDemo
add_subdirectory(plane)        # plane/CMakeLists.txt → HelloPlaneDemo  
add_subdirectory(tetris)       # tetris/CMakeLists.txt → Hello3DTetris
#add_subdirectory(snake)        # snake/CMakeLists.txt   → Hello3DSnake

# Optional: create a combined "AllDemos" target for convenience
add_custom_target(all_demos ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Building all Hello 3D demos..."
    DEPENDS 
        HelloFPSDemo 
        HelloPlaneDemo 
        Hello3DTetris 
        #Hello3DSnake
)

message(STATUS "")
message(STATUS "=== Hello 3D Demos ===")
message(STATUS "   fps/CMakeLists.txt          → HelloFPSDemo")
message(STATUS "   plane/CMakeLists.txt        → HelloPlaneDemo")
message(STATUS "   tetris/CMakeLists.txt       → Hello3DTetris")
#message(STATUS "   snake/CMakeLists.txt        → Hello3DSnake")
message(STATUS "")
