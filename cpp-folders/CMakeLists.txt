cmake_minimum_required(VERSION 3.28)

project("leisure-software-renderer")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


FIND_PACKAGE(Threads COMPONENTS REQUIRED)

include(FetchContent)

message(STATUS "Fetching stb library...")
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
)
FetchContent_MakeAvailable(stb)


message(STATUS "Fetching Entt")
FetchContent_Declare(
    entt
    GIT_REPOSITORY https://github.com/skypjack/entt.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(entt)

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

add_subdirectory("src/hello-shs-renderer")
add_subdirectory("src/hello-pixel-primitives")
add_subdirectory("src/hello-shaders")
add_subdirectory("src/hello-parallelization")
add_subdirectory("src/hello-3d-primitives")
add_subdirectory("src/hello-render-target")
add_subdirectory("src/hello-other-exps")

add_subdirectory("src/shs-renderer-lib")
add_subdirectory("src/exp-plumbing")
