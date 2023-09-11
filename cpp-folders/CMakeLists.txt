cmake_minimum_required(VERSION 3.18)

project("leisure-software-renderer")

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sdl2)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/assimp)

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
)
FetchContent_MakeAvailable(entt)

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

add_subdirectory("src/shs-renderer")
add_subdirectory("src/hello-pixel-primitives")
add_subdirectory("src/hello-shaders")
add_subdirectory("src/hello-parallelization")
add_subdirectory("src/hello-3d-primitives")
