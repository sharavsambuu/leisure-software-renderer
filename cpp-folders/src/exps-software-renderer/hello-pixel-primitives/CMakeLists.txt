project(HelloPixel)

find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

if(NOT TARGET SDL3_image::Main)
    if(TARGET SDL3_image::SDL3_image)
        add_library(SDL3_image::Main ALIAS SDL3_image::SDL3_image)
    else()
        add_library(SDL3_image::Main ALIAS SDL3_image::SDL3_image-static)
    endif()
endif()

include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL3_INCLUDE_DIRS} ${SDL3_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})

include_directories(${stb_SOURCE_DIR}/)
include_directories(${HelloSHSRenderer_SOURCE_DIR}/)

set(CMAKE_CXX_STANDARD 23)
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


file(COPY ../../assets/images/box_texture.png DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/images/box_texture.png")


add_executable(HelloPixel hello_pixel.cpp)
target_link_libraries(HelloPixel PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES})

add_executable(HelloLine hello_line.cpp)
target_link_libraries(HelloLine PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES})

add_executable(HelloTriangle hello_triangle.cpp)
target_link_libraries(HelloTriangle Threads::Threads ${SDL3_LIBRARIES} glm::glm)

add_executable(HelloCirclePoly hello_circle_poly.cpp)
target_link_libraries(HelloCirclePoly Threads::Threads ${SDL3_LIBRARIES} glm::glm)

add_executable(HelloWatch hello_watch.cpp)
target_link_libraries(HelloWatch Threads::Threads ${SDL3_LIBRARIES} glm::glm)


add_executable(HelloTexture hello_texture.cpp)
target_link_libraries(HelloTexture PRIVATE Threads::Threads SDL3::SDL3 SDL3_image::Main)