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
find_package(unofficial-lua CONFIG QUIET)
if (unofficial-lua_FOUND AND NOT LUA_FOUND)
    set(LUA_FOUND TRUE)
    set(LUA_LIBRARIES unofficial::lua::lua)
    message(STATUS ">>> tetris: Lua via vcpkg unofficial-lua target")
endif()
find_package(Lua QUIET)

# vcpkg fallback root/triplet (cross-platform; used by include dirs below).
if (WIN32)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64)$")
        set(_TETRIS_VCPKG_TRIPLET arm64-windows)
    else()
        set(_TETRIS_VCPKG_TRIPLET x64-windows)
    endif()
elseif(APPLE)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        set(_TETRIS_VCPKG_TRIPLET arm64-osx)
    else()
        set(_TETRIS_VCPKG_TRIPLET x64-osx)
    endif()
else()
    set(_TETRIS_VCPKG_TRIPLET x64-linux)
endif()
set(_TETRIS_VCPKG_ROOT /opt/vcpkg)
set(_TETRIS_VCPKG_INSTALLED_DIR "${_TETRIS_VCPKG_ROOT}/installed/${_TETRIS_VCPKG_TRIPLET}")

if (NOT LUA_FOUND)
    # Explicit fallback: vcpkg installed tree, per-platform triplet.
    if (EXISTS "${_TETRIS_VCPKG_INSTALLED_DIR}/include/lua.hpp")
        set(LUA_FOUND TRUE)
        set(LUA_INCLUDE_DIR "${_TETRIS_VCPKG_INSTALLED_DIR}/include")
        file(GLOB LUA_LIBRARIES
             "${_TETRIS_VCPKG_INSTALLED_DIR}/lib/lua*.lib"
             "${_TETRIS_VCPKG_INSTALLED_DIR}/lib/liblua*.a"
             "${_TETRIS_VCPKG_INSTALLED_DIR}/debug/lib/lua*.lib")
        list(GET LUA_LIBRARIES 0 LUA_LIBRARIES)
        message(STATUS ">>> tetris: Lua via vcpkg fallback (${LUA_LIBRARIES})")
    endif()
endif()

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR})
if(NOT DEFINED HelloSHSRenderer_SOURCE_DIR)
    get_filename_component(_TETRIS_SHS_RENDERER "${CMAKE_CURRENT_SOURCE_DIR}/../../../exps-other/hello-shs-renderer" ABSOLUTE)
    if(EXISTS "${_TETRIS_SHS_RENDERER}/shs_renderer.hpp")
        set(HelloSHSRenderer_SOURCE_DIR "${_TETRIS_SHS_RENDERER}")
    endif()
endif()
include_directories(${HelloSHSRenderer_SOURCE_DIR}/)
include_directories(${stb_SOURCE_DIR}/)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/game/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/engine/include)


# --- Main executable: Hello3DTetris (domain pod architecture) ---
add_executable(Hello3DTetris
    hello_3d_tetris.cpp
    game/include/game/world.hpp
    game/include/game/step.hpp
    game/include/game/stage.hpp
    assets/campaign/campaign.lua
    config/rules.hpp
    config/levels/marathon_01.hpp
    config/levels/blitz_120.hpp
    config/campaign/main_campaign.hpp
    config/levels/encore_finale.hpp
    domains/environment/environment.contract.hpp
    domains/environment/environment.reducer.hpp
    domains/environment/environment.plan.hpp
    domains/environment/scripts/encounter_overseer.lua
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
    domains/session/session.contract.hpp
    domains/session/session.action.hpp
    domains/session/session.reducer.hpp
    edges/input/tetris.input.hpp
    edges/audio/tetris.audio.hpp
    edges/rasterizer/tetris.rasterizer.hpp
    edges/ui/tetris.hud.hpp
    edges/lua/lua.edge.hpp
)

if (LUA_FOUND)
    target_include_directories(Hello3DTetris PRIVATE ${LUA_INCLUDE_DIR})
    target_link_libraries(Hello3DTetris PRIVATE ${LUA_LIBRARIES})
    target_link_libraries(Hello3DTetris PRIVATE shs::engine)
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
if(NOT DEFINED HelloSHSRenderer_SOURCE_DIR OR HelloSHSRenderer_SOURCE_DIR STREQUAL "" OR NOT EXISTS "${HelloSHSRenderer_SOURCE_DIR}/shs_renderer.hpp")
    foreach(_cand
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../exps-other/hello-shs-renderer"
        "${CMAKE_CURRENT_SOURCE_DIR}/hello-shs-renderer"
        "/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/exps-other/hello-shs-renderer")
        if(EXISTS "${_cand}/shs_renderer.hpp")
            set(HelloSHSRenderer_SOURCE_DIR "${_cand}")
            break()
        endif()
    endforeach()
endif()
set_property(TARGET Hello3DTetris PROPERTY INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/game/include
    ${HelloSHSRenderer_SOURCE_DIR}
    ${ASSIMP_INCLUDE_DIR}
    ${SDL2_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
    ${stb_SOURCE_DIR}
    ${LUA_INCLUDE_DIR}
)

message(STATUS ">>> Building Hello3DTetris (domain pods: matrix/progression/spatial_fx)...")

# --- P0: unit tests (plain-main binary, house style) -------------------------
enable_testing()
add_executable(tetris_reducer_tests
    tests/reducer_tests.cpp
)
target_include_directories(tetris_reducer_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(tetris_reducer_tests PRIVATE cxx_std_20)
if (MSVC)
    target_compile_options(tetris_reducer_tests PRIVATE /W4)
else()
    target_compile_options(tetris_reducer_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
if (LUA_FOUND)
    target_include_directories(tetris_reducer_tests PRIVATE ${LUA_INCLUDE_DIR})
    target_link_libraries(tetris_reducer_tests PRIVATE ${LUA_LIBRARIES})
    target_compile_definitions(tetris_reducer_tests PRIVATE TETRIS_LUA_ENABLED=1)
endif()
set_target_properties(tetris_reducer_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tetris)
add_test(NAME tetris_reducer_tests
         COMMAND $<TARGET_FILE:tetris_reducer_tests>)

# --- P2: whole-frame + input harness tests -----------------------------------
# Needs SDL2 (input edge includes SDL.h) and the shs renderer headers (hud
# floaters use shs::Color). Links nothing new beyond what the main target uses.
add_executable(tetris_step_tests
    tests/step_tests.cpp
)
# Target-scoped include order (REPLACES globals so fps/plane/snake roots can't
# shadow tetris pods - same pattern as Hello3DTetris above).
set_property(TARGET tetris_step_tests PROPERTY INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/game/include
    ${HelloSHSRenderer_SOURCE_DIR}
    ${ASSIMP_INCLUDE_DIR}
    ${SDL2_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
    ${stb_SOURCE_DIR}
    ${_TETRIS_VCPKG_INSTALLED_DIR}/include
)
target_compile_definitions(tetris_step_tests PRIVATE
    TETRIS_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
if (WIN32)
    # SDL_main.h #defines main -> SDL_main; these tests use plain int main().
    target_compile_definitions(tetris_step_tests PRIVATE SDL_MAIN_HANDLED)
endif()
target_compile_features(tetris_step_tests PRIVATE cxx_std_20)
target_link_libraries(tetris_step_tests PRIVATE SDL2::SDL2 glm::glm shs::engine)
target_include_directories(tetris_step_tests PRIVATE ${_TETRIS_VCPKG_INSTALLED_DIR}/include)
if (MSVC)
    target_compile_options(tetris_step_tests PRIVATE /W4)
else()
    target_compile_options(tetris_step_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
if (LUA_FOUND)
    target_include_directories(tetris_step_tests PRIVATE ${LUA_INCLUDE_DIR})
    target_link_libraries(tetris_step_tests PRIVATE ${LUA_LIBRARIES})
    target_compile_definitions(tetris_step_tests PRIVATE TETRIS_LUA_ENABLED=1)
endif()
set_tests_properties(tetris_reducer_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(tetris_step_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tetris)
add_test(NAME tetris_step_tests
         COMMAND $<TARGET_FILE:tetris_step_tests>)
set_tests_properties(tetris_step_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# --- G2: mission pod tests ----------------------------------------------------
add_executable(tetris_mission_tests
    tests/mission_tests.cpp
    domains/mission/mission.contract.hpp
    domains/mission/mission.reducer.hpp
)
set_property(TARGET tetris_mission_tests PROPERTY INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(tetris_mission_tests PRIVATE cxx_std_20)
if (MSVC)
    target_compile_options(tetris_mission_tests PRIVATE /W4)
else()
    target_compile_options(tetris_mission_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
set_target_properties(tetris_mission_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tetris)
add_test(NAME tetris_mission_tests
         COMMAND $<TARGET_FILE:tetris_mission_tests>)

# --- P2c: script purity ctest -------------------------------------------------
# --- G1: scripted goal bridge tests ------------------------------------------
add_executable(tetris_goal_bridge_tests
    tests/goal_bridge_tests.cpp
)
set_property(TARGET tetris_goal_bridge_tests PROPERTY INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/game/include
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/include
    ${HelloSHSRenderer_SOURCE_DIR}
    ${LUA_INCLUDE_DIR}
    ${ASSIMP_INCLUDE_DIR}
    ${SDL2_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
    ${stb_SOURCE_DIR}
    ${_TETRIS_VCPKG_INSTALLED_DIR}/include
)
target_compile_features(tetris_goal_bridge_tests PRIVATE cxx_std_20)
target_compile_definitions(tetris_goal_bridge_tests PRIVATE
    TETRIS_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
if (LUA_FOUND)
    target_link_libraries(tetris_goal_bridge_tests PRIVATE ${LUA_LIBRARIES})
    target_compile_definitions(tetris_goal_bridge_tests PRIVATE TETRIS_LUA_ENABLED=1)
endif()
target_link_libraries(tetris_goal_bridge_tests PRIVATE glm::glm)
set_target_properties(tetris_goal_bridge_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tetris)
add_test(NAME tetris_goal_bridge_tests
         COMMAND $<TARGET_FILE:tetris_goal_bridge_tests>)

add_test(NAME tetris_script_purity
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/script_purity.sh)
# --- P3: stage loader tests ---------------------------------------------------
add_executable(tetris_stage_tests
    tests/stage_loader_tests.cpp
)
set_property(TARGET tetris_stage_tests PROPERTY INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/game/include
    ${HelloSHSRenderer_SOURCE_DIR}
    ${ASSIMP_INCLUDE_DIR}
    ${SDL2_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
    ${stb_SOURCE_DIR}
    ${_TETRIS_VCPKG_INSTALLED_DIR}/include
)
target_compile_features(tetris_stage_tests PRIVATE cxx_std_20)
target_compile_definitions(tetris_stage_tests PRIVATE
    TETRIS_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
if (LUA_FOUND)
    target_compile_definitions(tetris_stage_tests PRIVATE TETRIS_LUA_ENABLED=1)
endif()
if (LUA_FOUND)
    target_include_directories(tetris_stage_tests PRIVATE ${LUA_INCLUDE_DIR})
    target_link_libraries(tetris_stage_tests PRIVATE ${LUA_LIBRARIES})
    target_compile_definitions(tetris_stage_tests PRIVATE TETRIS_LUA_ENABLED=1)
endif()
add_test(NAME tetris_stage_tests COMMAND $<TARGET_FILE:tetris_stage_tests>)