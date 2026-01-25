project(HelloRenderTarget)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(glm REQUIRED)

find_package(Assimp REQUIRED)
if(ASSIMP_FOUND)
  include_directories(${ASSIMP_INCLUDE_DIR})
endif() 

include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${stb_SOURCE_DIR}/)
#include_directories(${entt_SOURCE_DIR}/src/)

include_directories(${SHSRenderer_SOURCE_DIR}/)

set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


# 3D model wireframe rendering related commands
file(COPY obj DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(HelloPingPong hello_ping_pong.cpp)
target_link_libraries(HelloPingPong PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloDepthOfField hello_depth_of_field.cpp)
target_link_libraries(HelloDepthOfField PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloPerObjectMotionBlur hello_per_object_motion_blur.cpp)
target_link_libraries(HelloPerObjectMotionBlur PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloMultiPass hello_multi_pass.cpp)
target_link_libraries(HelloMultiPass PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloGlowingStar hello_glowing_star.cpp)
target_link_libraries(HelloGlowingStar PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloCameraAndPerobjectMotionBlur hello_camera_and_perobject_motion_blur.cpp)
target_link_libraries(HelloCameraAndPerobjectMotionBlur PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloShadowMapping hello_shadow_mapping.cpp)
target_link_libraries(HelloShadowMapping PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloWater hello_water.cpp)
target_link_libraries(HelloWater PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)