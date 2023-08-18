project(HelloPixel)

#FIND_PACKAGE(Boost COMPONENTS REQUIRED)
#FIND_PACKAGE(Threads COMPONENTS REQUIRED)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS "")

#include_directories(${Boost_INCLUDE_DIR})
#link_directories(${Boost_LIBRARY_DIR})

add_executable(HelloPixel hello_pixel.cpp tgaimage.cpp)
target_link_libraries(HelloPixel Threads::Threads)

add_executable(HelloLine hello_line.cpp tgaimage.cpp)
target_link_libraries(HelloLine Threads::Threads)
