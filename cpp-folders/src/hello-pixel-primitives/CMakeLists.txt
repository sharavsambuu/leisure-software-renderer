project(HelloPixel)

find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})

include_directories(${stb_SOURCE_DIR}/)
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

file(COPY images DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(HelloPixel hello_pixel.cpp)
target_link_libraries(HelloPixel PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES})

add_executable(HelloLine hello_line.cpp)
target_link_libraries(HelloLine PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES})

add_executable(HelloTriangle hello_triangle.cpp)
target_link_libraries(HelloTriangle Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloCirclePoly hello_circle_poly.cpp)
target_link_libraries(HelloCirclePoly Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloWatch hello_watch.cpp)
target_link_libraries(HelloWatch Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloTexture hello_texture.cpp)
target_link_libraries(HelloTexture PRIVATE Threads::Threads SDL2::SDL2 SDL2_image::SDL2_image-static)