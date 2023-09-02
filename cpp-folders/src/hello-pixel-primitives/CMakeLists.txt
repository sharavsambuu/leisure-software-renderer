project(HelloPixel)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(glm REQUIRED)

find_package(Assimp REQUIRED)
if(ASSIMP_FOUND)
  include_directories(${ASSIMP_INCLUDE_DIR})
endif() 

#include_directories(src/shs-renderer/)
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../shs-renderer)

set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


add_executable(HelloPixel hello_pixel.cpp ../shs-renderer/shs_renderer.hpp)
target_link_libraries(HelloPixel PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES})

add_executable(HelloSDLPixel hello_sdl_pixel.cpp ../shs-renderer/shs_renderer.hpp)
target_link_libraries(HelloSDLPixel Threads::Threads ${SDL2_LIBRARIES})

add_executable(Hello2DTransformation hello_2d_transformation.cpp ../shs-renderer/shs_renderer.hpp)
target_link_libraries(Hello2DTransformation Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloLine hello_line.cpp ../shs-renderer/shs_renderer.hpp)
target_link_libraries(HelloLine PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES})

# 3D model wireframe rendering related commands
file(COPY obj DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
add_executable(HelloWireframe hello_wireframe.cpp ../shs-renderer/shs_renderer.hpp)
target_link_libraries(HelloWireframe PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)


