project(HelloShaders)

find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${stb_SOURCE_DIR}/)
include_directories(${HelloSHSRenderer_SOURCE_DIR}/)

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


add_executable(HelloColorMix hello_color_mix.cpp)
target_link_libraries(HelloColorMix PRIVATE ${SDL2_LIBRARIES} Threads::Threads )

add_executable(HelloColorGradient hello_color_gradient.cpp)
target_link_libraries(HelloColorGradient PRIVATE Threads::Threads ${SDL2_LIBRARIES})

add_executable(HelloColorHSB hello_color_hsb.cpp)
target_link_libraries(HelloColorHSB PRIVATE Threads::Threads ${SDL2_LIBRARIES})

add_executable(HelloColorHSBPolar hello_color_hsb_polar.cpp)
target_link_libraries(HelloColorHSBPolar PRIVATE Threads::Threads ${SDL2_LIBRARIES})

add_executable(HelloFractalBrownianMotion hello_fbm.cpp)
target_link_libraries(HelloFractalBrownianMotion PRIVATE Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloGreenNoise hello_green_noise.cpp)
target_link_libraries(HelloGreenNoise PRIVATE Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloMongolianFlag hello_mongolian_flag.cpp)
target_link_libraries(HelloMongolianFlag PRIVATE Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloWormHole hello_worm_hole.cpp)
target_link_libraries(HelloWormHole PRIVATE Threads::Threads ${SDL2_LIBRARIES} glm::glm)