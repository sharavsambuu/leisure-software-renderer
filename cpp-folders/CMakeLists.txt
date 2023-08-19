cmake_minimum_required(VERSION 3.18)

project("cpp-software-renderer-exps")

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sdl2)

FIND_PACKAGE(Threads COMPONENTS REQUIRED)

#if(WIN32)
#    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/windows-cmakes/")
#endif()

add_subdirectory("src/hello-pixel")
