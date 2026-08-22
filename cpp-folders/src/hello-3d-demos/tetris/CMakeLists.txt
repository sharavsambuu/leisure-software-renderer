# =============================================================================
# Hello 3D Tetris Demo — CMake Target Definition
# Location: cpp-folders/src/hello-3d-demos/tetris/CMakeLists.txt
# Parent aggregator: cpp-folders/src/hello-3d-demos/CMakeLists.txt
#
# Domain-pod architecture (see docs/REFACTOR_PROPOSAL.md):
#   config/ + domains/{matrix,progression,spatial_fx} + edges/{input,audio,rasterizer,ui,lua}
# Include convention: ONE project root (tetris/) on the include path; cross-pod
# headers use root-relative angle brackets (<domains/matrix/matrix.contract.hpp>),
# quotes only for same-directory siblings. Never "../../" chains.
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

# --- Lua (optional): rule-scripting edge (L2 Blitz 120) ----------------------
# Found ⇒ TETRIS_LUA_ENABLED is defined and the evaluator edge compiles in.
# Missing ⇒ pods fall back to native C++ rules; the build stays green either way.
find_package(Lua QUIET)
if (LUA_FOUND)
    message(STATUS ">>> Hello3DTetris: Lua ${LUA_VERSION_STRING} found — rule scripting ENABLED")
else()
    message(STATUS ">>> Hello3DTetris: Lua NOT found — native C++ rules only")
endif()

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${HelloSHSRenderer_SOURCE_DIR}/)
include_directories(${stb_SOURCE_DIR}/)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


# --- Main executable: Hello3DTetris (domain pod architecture) ---
add_executable(Hello3DTetris
    hello_3d_tetris.cpp
    config/rules.hpp
    config/levels/marathon_01.hpp
    config/levels/blitz_120.hpp
    config/campaign/main_campaign.hpp
    domains/matrix/matrix.contract.hpp
    domains/matrix/matrix.action.hpp
    domains/matrix/matrix.event.hpp
    domains/matrix/matrix.reducer.hpp
    domains/progression/progression.contract.hpp
    domains/progression/progression.event.hpp
    domains/progression/progression.reducer.hpp
    domains/progression/scripts/blitz_mode.lua
    domains/spatial_fx/spatial_fx.contract.hpp
    domains/spatial_fx/spatial_fx.reducer.hpp
    domains/spatial_fx/spatial_fx.plan.hpp
    edges/input/tetris.input.hpp
    edges/audio/tetris.audio.hpp
    edges/rasterizer/tetris.rasterizer.hpp
    edges/ui/tetris.hud.hpp
    edges/lua/lua.edge.hpp
)

if (LUA_FOUND)
    target_include_directories(Hello3DTetris PRIVATE ${LUA_INCLUDE_DIR})
    target_link_libraries(Hello3DTetris PRIVATE ${LUA_LIBRARIES})
    target_compile_definitions(Hello3DTetris PRIVATE TETRIS_LUA_ENABLED=1)
endif()

# Absolute path baked in so the binary finds scripts from any working dir.
target_compile_definitions(Hello3DTetris PRIVATE
    TETRIS_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

# --- Compile options (shared across demos for consistency) ---
target_compile_options(Hello3DTetris PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<NOT:$<CONFIG:Debug>>>:-O3 -march=native -flto>
    $<$<OR:$<CXX_COMPILER_ID:GNU,Clang>,$<CXX_COMPILER_ID:Intel>>:-Wextra -Wall -Wpedantic -Wshadow -Wcast-qual -Wcast-align -Wunused -Wconversion -Wsign-conversion -Wfloat-conversion>
)

target_compile_definitions(Hello3DTetris PRIVATE GLM_ENABLE_EXPERIMENTAL)

# --- Link libraries ---
target_link_libraries(Hello3DTetris PRIVATE
    SDL2::SDL2
    glm::glm
    assimp::assimp
)

# --- Platform-specific optimizations ---
if (MSVC)
    # /GL + /LTCG = whole-program optimization (MSVC analogue of -flto);
    # without these a Release software rasterizer runs far slower than Linux.
    target_compile_options(Hello3DTetris PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:/O2 /fp:fast /arch:AVX2 /GL>
        /MP
    )
    target_link_options(Hello3DTetris PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:/LTCG>
    )
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(Hello3DTetris PRIVATE -O3 -ffast-math)
else()
    target_compile_options(Hello3DTetris PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()

# Target-scoped include order: tetris root FIRST so <domains/...>, <config/...>,
# <edges/...> resolve to THIS project even though the aggregator also exposes
# sibling demo roots globally (fps defines domains/spatial_fx too).
set_property(TARGET Hello3DTetris PROPERTY INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HelloSHSRenderer_SOURCE_DIR}
    ${ASSIMP_INCLUDE_DIR}
    ${SDL2_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
    ${stb_SOURCE_DIR}
)

message(STATUS ">>> Building Hello3DTetris (domain pods: matrix/progression/spatial_fx)...")
