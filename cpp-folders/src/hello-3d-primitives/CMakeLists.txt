project(Hello3D)

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


add_executable(HelloWireframe hello_wireframe.cpp)
target_link_libraries(HelloWireframe PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)


add_executable(Hello3DObject hello_3d_object.cpp)
target_link_libraries(Hello3DObject PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)



