# cpp-folders/libs/engine - shared game-engine skeleton (P5)
# Header-only for now (loop.hpp fixed-step accumulator). Future: event bus,
# command buffer, clock abstractions as a compiled static lib.
#
# Consumers:
#   add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../libs/engine engine.build)
#   target_link_libraries(my_target PRIVATE shs::engine)
cmake_minimum_required(VERSION 3.28)
project(shs_engine LANGUAGES CXX)

add_library(shs_engine INTERFACE)
add_library(shs::engine ALIAS shs_engine)
target_include_directories(shs_engine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(shs_engine INTERFACE cxx_std_20)
