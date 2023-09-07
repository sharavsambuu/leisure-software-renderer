cmake_minimum_required(VERSION 3.18)

project("leisure-software-renderer")

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sdl2)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/assimp)

FIND_PACKAGE(Threads COMPONENTS REQUIRED)

#if(WIN32)
#    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/windows-cmakes/")
#endif()

include(FetchContent)

# C++20 coroutine based job system
FetchContent_Declare(
    pal_tasks
    GIT_REPOSITORY https://github.com/tgfrerer/pal_tasks.git
    GIT_TAG main
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(pal_tasks)

# C++20 coroutine library for general use
FetchContent_Declare(
    libcoro
    GIT_REPOSITORY https://github.com/jbaldwin/libcoro.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(libcoro)

add_subdirectory("src/shs-renderer")
add_subdirectory("src/hello-pixel-primitives")
add_subdirectory("src/hello-shaders")
add_subdirectory("src/hello-parallelization")
