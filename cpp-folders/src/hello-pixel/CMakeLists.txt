project(HelloPixel)

#find_package(Boost COMPONENTS REQUIRED)
#find_package(Threads COMPONENTS REQUIRED)
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS "")

#include_directories(${Boost_INCLUDE_DIR})
#link_directories(${Boost_LIBRARY_DIR})

add_executable(HelloPixel hello_pixel.cpp tgaimage.cpp)
target_link_libraries(HelloPixel Threads::Threads)

add_executable(HelloLine hello_line.cpp tgaimage.cpp)
target_link_libraries(HelloLine Threads::Threads)

add_executable(HelloSDLPixel hello_sdl_pixel.cpp)
target_link_libraries(HelloSDLPixel Threads::Threads ${SDL2_LIBRARIES})

