cmake_minimum_required(VERSION 3.18)

project("leisure-software-renderer")

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sdl2)

FIND_PACKAGE(Threads COMPONENTS REQUIRED)

#if(WIN32)
#    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/windows-cmakes/")
#endif()

# actions for rapidobj loader library
include(FetchContent)

FetchContent_Declare(
    rapidobj
    GIT_REPOSITORY https://github.com/guybrush77/rapidobj.git
    GIT_TAG v1.0.1
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(rapidobj)

add_subdirectory("src/hello-pixel-primitives")
