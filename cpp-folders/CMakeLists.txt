cmake_minimum_required(VERSION 3.28)

project("leisure-software-renderer")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

# Keep MSVC runtime selection consistent across all targets (including Jolt).
# This avoids LNK2038 / LNK2005 mismatches when Jolt is built as /MT[d]
# while app targets are /MD[d] (or vice versa).
if(MSVC)
    option(SHS_USE_STATIC_MSVC_RUNTIME "Use static MSVC runtime (/MT,/MTd) for all targets" OFF)
    if(SHS_USE_STATIC_MSVC_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
        set(USE_STATIC_MSVC_RUNTIME_LIBRARY ON CACHE BOOL "" FORCE)
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)
        set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
    endif()
endif()


FIND_PACKAGE(Threads COMPONENTS REQUIRED)
#FIND_PACKAGE(Lua 5.5 REQUIRED)

include(FetchContent)

message(STATUS "Fetching stb library...")
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
)
FetchContent_MakeAvailable(stb)


#message(STATUS "Fetching Entt")
#FetchContent_Declare(
#    entt
#    GIT_REPOSITORY https://github.com/skypjack/entt.git
#    GIT_TAG        main
#)
#FetchContent_MakeAvailable(entt)

message(STATUS "Fetching XSIMD")
FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG 14.0.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(xsimd)


message(STATUS "Fetching JoltPhysics")
FetchContent_Declare(
    JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG        v5.2.0
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  Build
)
set(DOUBLE_PRECISION OFF CACHE BOOL "" FORCE)
set(GENERATE_DEBUG_SYMBOLS ON CACHE BOOL "" FORCE)
set(CROSS_PLATFORM_DETERMINISTIC OFF CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(JoltPhysics)

# VulkanMemoryAllocator via vcpkg (Run 2 / P3: optional — only consumed when
# Vulkan is found; shs-renderer-lib gates usage on the GPUOpen:: target
# existing, so GPU-free machines configure without it).
message(STATUS "Looking for VulkanMemoryAllocator library...")
find_package(VulkanMemoryAllocator CONFIG QUIET)


# Lua 5.5
message(STATUS "Fetching Lua 5.5 library...")
find_package(Lua 5.5)

#
# Too slow to fetch on my machine
#
#message(STATUS "Fetching flecs library...")
#FetchContent_Declare(
#    flecs
#    GIT_REPOSITORY https://github.com/SanderMertens/flecs.git
#    GIT_TAG ae0bed2815127d3c8315e9e4c3edb253ea8a91c4 # v3.2.7
#)
#FetchContent_MakeAvailable(flecs)


message(STATUS "Adding sub folders...")

# Demos/probes parked (2026-09): exps trees are commented out at the root so
# the build covers core libs + lib tests only. Source is kept on disk —
# demos will be restarted fresh later; re-enable by uncommenting these lines
# (note: exps CMake/TUs still reference retired aliases + facade include
# paths, so they need a pass before they build again).
#
# New layout (2026-09): single renderer library — shs-gpu-lib absorbed into
# shs-core-lib (Vulkan runtime edge gated behind SHS_HAS_VULKAN).
add_subdirectory("src/shs-renderer-lib")
# add_subdirectory("src/exps-other")
# add_subdirectory("src/exps-software-renderer")
# add_subdirectory("src/exps-gpu-renderer")
