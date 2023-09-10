cmake_minimum_required(VERSION 3.18)

project("leisure-software-renderer")

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sdl2)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/assimp)

FIND_PACKAGE(Threads COMPONENTS REQUIRED)


add_subdirectory("src/shs-renderer")
add_subdirectory("src/hello-pixel-primitives")
add_subdirectory("src/hello-shaders")
add_subdirectory("src/hello-parallelization")
add_subdirectory("src/hello-3d-primitives")
