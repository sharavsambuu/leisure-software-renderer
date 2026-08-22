# =============================================================================
# Hello 3D FPS Demo — CMake Target Definition
# Location: cpp-folders/src/hello-3d-demos/fps/CMakeLists.txt
# Parent aggregator: cpp-folders/src/hello-3d-demos/CMakeLists.txt (includes this subdir)
#
# NOTE: All shared dependencies and alias definitions are handled in the parent
# CMakeLists.txt to avoid "target already exists" errors. This file only defines
# its own executable, compile options, link libraries, and platform flags.
# =============================================================================

cmake_minimum_required(VERSION 3.28)
project(HelloFPSDemo VERSION 0.1 LANGUAGES CXX)

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

# --- Include convention ---
# ONE project root on the include path. Cross-pod headers are included by
# their root-relative path in angle brackets, so the include line states its
# location in the architecture:
#     #include <domains/matrix/fps.contract.hpp>
#     #include <config/difficulty.hpp>
# Angle brackets  = cross-pod / rooted path (architecture-visible dependency).
# Quote includes  = same-directory siblings only ("fps.meshes.hpp").
# Namespace identity lives in the code (fps::matrix, fps::spatial_fx, ...);
# include-path identity lives here. Never use "../../" relative chains.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# --- Main executable: HelloFPSDemo ---
add_executable(HelloFPSDemo
    hello_fps_demo.cpp
)

# --- Compile options (shared across demos for consistency) ---
target_compile_options(HelloFPSDemo PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<NOT:$<CONFIG:Debug>>>:-O3 -march=native -flto>
    $<$<OR:$<CXX_COMPILER_ID:GNU,Clang>,$<CXX_COMPILER_ID:Intel>>:-Wextra -Wall -Wpedantic -Wshadow -Wcast-qual -Wcast-align -Wunused -Wconversion -Wsign-conversion -Wfloat-conversion>
    $<$<CXX_COMPILER_ID:MSVC>:/MP>                       # parallel compilation (speeds up VS builds)
    $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
)

target_compile_definitions(HelloFPSDemo PRIVATE GLM_ENABLE_EXPERIMENTAL)

# --- Link libraries (SDL2_image::Main is defined once globally in parent CMakeLists.txt) ---
target_link_libraries(HelloFPSDemo PRIVATE
    SDL2::SDL2
    SDL2_image::Main
    glm::glm
    assimp::assimp
)

# --- Platform-specific optimizations ---
if (MSVC)
    # Appends the flags only if the configuration is NOT Debug.
    # /GL + /LTCG = whole-program optimization, the MSVC analogue of the
    # -O3 -march=native -flto used on Linux — without these, a Release
    # software rasterizer runs far slower than the Linux build.
    target_compile_options(HelloFPSDemo PRIVATE 
        $<$<NOT:$<CONFIG:Debug>>:/O2 /fp:fast /arch:AVX2 /GL>
    )
    target_link_options(HelloFPSDemo PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:/LTCG>
    )
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(HelloFPSDemo PRIVATE -O3 -ffast-math)
else()
    target_compile_options(HelloFPSDemo PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()

message(STATUS ">>> Building HelloFPSDemo (VOP/DOD domain pods)...")
