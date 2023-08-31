project(HelloShaders)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(glm REQUIRED)

find_package(Assimp REQUIRED)
if(ASSIMP_FOUND)
  include_directories(${ASSIMP_INCLUDE_DIR})
endif() 

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

add_executable(HelloFragmentShader hello_fragment_shader.cpp ../shs-renderer/shs_renderer.hpp)
target_link_libraries(HelloFragmentShader Threads::Threads ${SDL2_LIBRARIES})

add_executable(HelloColorMix hello_color_mix.cpp ../shs-renderer/shs_renderer.hpp)
target_link_libraries(HelloColorMix Threads::Threads ${SDL2_LIBRARIES})

add_executable(HelloColorGradient hello_color_gradient.cpp ../shs-renderer/shs_renderer.hpp)
target_link_libraries(HelloColorGradient Threads::Threads ${SDL2_LIBRARIES})

add_executable(HelloColorHSB hello_color_hsb.cpp ../shs-renderer/shs_renderer.hpp)
target_link_libraries(HelloColorHSB Threads::Threads ${SDL2_LIBRARIES})