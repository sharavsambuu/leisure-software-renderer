project(HelloPixel)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS "")


add_executable(HelloPixel hello_pixel.cpp shs_renderer.hpp)
target_link_libraries(HelloPixel PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES})

add_executable(HelloRectangle hello_rectangle.cpp shs_renderer.hpp)
target_link_libraries(HelloRectangle PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES})

add_executable(HelloSDLPixel hello_sdl_pixel.cpp)
target_link_libraries(HelloSDLPixel Threads::Threads ${SDL2_LIBRARIES})

