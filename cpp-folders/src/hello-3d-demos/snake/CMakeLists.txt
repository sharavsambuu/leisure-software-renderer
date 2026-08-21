# =============================================================================
# Hello 3D Snake Demo — CMake Target Definition
# Location: cpp-folders/src/hello-3d-demos/snake/CMakeLists.txt
# Parent aggregator: cpp-folders/src/hello-3d-demos/CMakeLists.txt (includes this subdir)
#
# NOTE: All shared dependencies and alias definitions are handled in the parent
# CMakeLists.txt to avoid "target already exists" errors. This file only defines
# its own executable, compile options, link libraries, and platform flags.
# =============================================================================

cmake_minimum_required(VERSION 3.28)
project(Hello3DSnake VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies (shared across all demos — defined in parent CMakeLists.txt) ---
find_package(SDL2 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR})
# Include the shared software renderer (shs_renderer.hpp) from hello-3d-primitives
include_directories(${HelloSHSRenderer_SOURCE_DIR}/)

include_directories(${stb_SOURCE_DIR}/)

# --- Main executable: Hello3DSnake (multi-file domain pod architecture) ---
# Note: include_directories("${CMAKE_CURRENT_LIST_DIR}/../") adds the parent of CMakeLists.txt,
# so that relative includes like "snake.contract.hpp" resolve correctly from any file
# under snake/ (e.g., from domains/config/difficulty.hpp → ../matrix/snake.contract.hpp).
include_directories("${CMAKE_CURRENT_LIST_DIR}/../")

add_executable(Hello3DSnake
    hello_3d_snake.cpp
    domains/matrix/snake.contract.hpp
    domains/matrix/snake.action.hpp
    domains/matrix/snake.event.hpp
    domains/matrix/snake.reducer.hpp
    domains/spatial_fx/snake.plan.hpp
)

# --- Compile options (shared across demos for consistency) ---
target_compile_options(Hello3DSnake PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<NOT:$<CONFIG:Debug>>>:-O3 -march=native -flto>
    $<$<OR:$<CXX_COMPILER_ID:GNU,Clang>,$<CXX_COMPILER_ID:Intel>>:-Wextra -Wall -Wpedantic -Wshadow -Wcast-qual -Wcast-align -Wunused -Wconversion -Wsign-conversion -Wfloat-conversion>
)

target_compile_definitions(Hello3DSnake PRIVATE GLM_ENABLE_EXPERIMENTAL)

# --- Link libraries ---
target_link_libraries(Hello3DSnake PRIVATE
    SDL2::SDL2
    glm::glm
    assimp::assimp
)

# --- Platform-specific optimizations ---
if (MSVC)
    target_compile_options(Hello3DSnake PRIVATE /O2 /fp:fast /arch:AVX2)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(Hello3DSnake PRIVATE -O3 -ffast-math)
else()
    target_compile_options(Hello3DSnake PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()

message(STATUS ">>> Building Hello3DSnake (Retro Arcade Tunnel scene)...")
