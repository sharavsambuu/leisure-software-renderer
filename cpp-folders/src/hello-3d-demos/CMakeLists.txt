project(Hello3DDemos)

find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

if(TARGET SDL2_image::SDL2_image)
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image)
else()
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image-static)
endif()

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${stb_SOURCE_DIR}/)
include_directories(${HelloSHSRenderer_SOURCE_DIR}/)
include_directories(fps plane tetris)

set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(COPY ../assets DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# 1. Plane Demo
add_executable(HelloPlaneDemo plane/hello_plane_demo.cpp)
target_compile_definitions(HelloPlaneDemo PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_link_libraries(HelloPlaneDemo PRIVATE SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)
if (MSVC)
    target_compile_options(HelloPlaneDemo PRIVATE /O2 /fp:fast /arch:AVX2)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(HelloPlaneDemo PRIVATE -O3 -ffast-math)
else()
    target_compile_options(HelloPlaneDemo PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()

# 2. FPS Demo
add_executable(HelloFPSDemo fps/hello_fps_demo.cpp)
target_compile_definitions(HelloFPSDemo PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_link_libraries(HelloFPSDemo PRIVATE SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)
if (MSVC)
    target_compile_options(HelloFPSDemo PRIVATE /O2 /fp:fast /arch:AVX2)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(HelloFPSDemo PRIVATE -O3 -ffast-math)
else()
    target_compile_options(HelloFPSDemo PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()

# 3. Semi-3D Tetris Demo (Multi-file Domain Pod)
add_executable(Hello3DTetris
    tetris/hello_3d_tetris.cpp
    tetris/tetris.contract.hpp
    tetris/tetris.action.hpp
    tetris/tetris.event.hpp
    tetris/tetris.reducer.hpp
    tetris/tetris.plan.hpp
)
target_compile_definitions(Hello3DTetris PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_link_libraries(Hello3DTetris PRIVATE SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)
if (MSVC)
    target_compile_options(Hello3DTetris PRIVATE /O2 /fp:fast /arch:AVX2)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(Hello3DTetris PRIVATE -O3 -ffast-math)
else()
    target_compile_options(Hello3DTetris PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()