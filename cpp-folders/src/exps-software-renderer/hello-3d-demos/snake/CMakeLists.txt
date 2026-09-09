# =============================================================================
# Hello 3D Snake Demo — CMake Target Definition
# Location: cpp-folders/src/hello-3d-demos/snake/CMakeLists.txt
# Parent aggregator: cpp-folders/src/hello-3d-demos/CMakeLists.txt (includes this subdir)
#
# NOTE: All shared dependencies and alias definitions are handled in the parent
# CMakeLists.txt to avoid "target already exists" errors. This file only defines
# its own executable, compile options, link libraries, and platform flags.
# Include paths for within-pod headers (e.g. "snake.contract.hpp", "shs_renderer.hpp")
# are provided globally by the parent aggregator — no relative ("../../../...") includes
# are used in source headers.
# =============================================================================

cmake_minimum_required(VERSION 3.28)
project(Hello3DSnake VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies (shared across all demos — defined in parent CMakeLists.txt) ---
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_mixer CONFIG QUIET)   # optional: audio pod is standalone; only linked if available
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR})
# Include the shared software renderer (shs_renderer.hpp) from hello-3d-primitives
include_directories(${HelloSHSRenderer_SOURCE_DIR}/)

include_directories(${stb_SOURCE_DIR}/)

# --- Snake internal subdirs: add each snake subdir to the global include path so that
# bare cross-subdir includes resolve from any file under snake/, mirroring tetris's
# within-pod flat-include convention. Only dirs actually referenced by a source header
# are listed (verified via grep of #include directives):
#   - "snake.contract.hpp"  -> domains/matrix (shared command vocabulary for edges + pods)
#   - "difficulty.hpp"      -> config        (snake::config::Difficulty)
#   - "snake_level_01.hpp"  -> config/levels (SnakeLevel01 arena + food table)
# Purged 2026-08-22 (see docs/STATUS.md §0): domains/config dup, domains/matrix/snake.plan.hpp,
# domains/spatial_fx/snake.reducer.hpp, domains/environment/ — none were included by anything.
# edges/rasterizer + edges/audio remain on disk as origin-structure slots but stay unwired.
include_directories(${CMAKE_CURRENT_LIST_DIR}/config)
include_directories(${CMAKE_CURRENT_LIST_DIR}/config/levels)
include_directories(${CMAKE_CURRENT_LIST_DIR}/domains/matrix)

# --- Main executable: Hello3DSnake (multi-file domain pod architecture) ---
add_executable(Hello3DSnake
    hello_3d_snake.cpp
    domains/matrix/snake.contract.hpp
    domains/matrix/snake.action.hpp
    domains/matrix/snake.event.hpp
    domains/matrix/snake.reducer.hpp
    domains/spatial_fx/spatial_fx.contract.hpp  # render vocabulary (PipelineExecutionPlan, ShatterParticleSoA)
    domains/spatial_fx/snake.plan.hpp
    edges/input/snake.input.hpp          # input edge (SDL boundary -> SnakeCommand)
    domains/progression/snake.contract.hpp  # ScoreState
)

# --- Compile options (shared across demos for consistency) ---
target_compile_options(Hello3DSnake PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<NOT:$<CONFIG:Debug>>>:-O3 -march=native -flto>
    $<$<OR:$<CXX_COMPILER_ID:GNU,Clang>,$<CXX_COMPILER_ID:Intel>>:-Wextra -Wall -Wpedantic -Wshadow -Wcast-qual -Wcast-align -Wunused -Wconversion -Wsign-conversion -Wfloat-conversion>
)

target_compile_definitions(Hello3DSnake PRIVATE GLM_ENABLE_EXPERIMENTAL)

# --- Link libraries ---
# SDL2_mixer is optional: the audio pod (edges/audio/snake.audio.hpp) uses SDL_LoadWAV/SDL_SaveWAV/
# SDL_PlayAudioBuffer, but main does not include it. Only link if vcpkg provided the package so the
# demo builds even when mixer is unavailable; add "SDL2_mixer::SDL2_mixer" to the list once installed.
if (TARGET SDL2_mixer::SDL2_mixer)
    target_link_libraries(Hello3DSnake PRIVATE SDL2_mixer::SDL2_mixer)   # audio edge SFX (optional)
endif()

target_link_libraries(Hello3DSnake PRIVATE
    SDL2::SDL2
    glm::glm
    assimp::assimp
)

# --- Platform-specific optimizations ---
if (MSVC)
    # /GL + /LTCG = whole-program optimization (MSVC analogue of -flto);
    # without these a Release software rasterizer runs far slower than Linux.
    target_compile_options(Hello3DSnake PRIVATE 
        $<$<NOT:$<CONFIG:Debug>>:/O2 /fp:fast /arch:AVX2 /GL>
        /MP
    )
    target_link_options(Hello3DSnake PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:/LTCG>
    )
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(Hello3DSnake PRIVATE -O3 -ffast-math)
else()
    target_compile_options(Hello3DSnake PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()

message(STATUS ">>> Building Hello3DSnake (Retro Arcade Tunnel scene)...")
