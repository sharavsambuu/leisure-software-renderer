project(HelloPhysics)

find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(unofficial-omniverse-physx-sdk CONFIG REQUIRED)

if(TARGET SDL2_image::SDL2_image)
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image)
else()
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image-static)
endif()

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${stb_SOURCE_DIR}/)
#include_directories(${entt_SOURCE_DIR}/src/)

include_directories(${SHSRenderer_SOURCE_DIR}/)

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


# 3D model wireframe rendering related commands
file(COPY ../assets DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(HelloPhysics hello_physics.cpp)
target_link_libraries(HelloPhysics PRIVATE xsimd SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp unofficial::omniverse-physx-sdk::sdk)
