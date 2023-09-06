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
include_directories(${SHSRenderer_SOURCE_DIR}/)

set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


add_executable(HelloColorMix hello_color_mix.cpp 
  ${SHSRenderer_SOURCE_DIR}/shs_renderer.hpp
  )
target_link_libraries(HelloColorMix Threads::Threads ${SDL2_LIBRARIES})

add_executable(HelloColorGradient hello_color_gradient.cpp 
  ${SHSRenderer_SOURCE_DIR}/shs_renderer.hpp
  )
target_link_libraries(HelloColorGradient Threads::Threads ${SDL2_LIBRARIES})

add_executable(HelloColorHSB hello_color_hsb.cpp 
  ${SHSRenderer_SOURCE_DIR}/shs_renderer.hpp
  )
target_link_libraries(HelloColorHSB Threads::Threads ${SDL2_LIBRARIES})

add_executable(HelloColorHSBPolar hello_color_hsb_polar.cpp 
  ${SHSRenderer_SOURCE_DIR}/shs_renderer.hpp
  )
target_link_libraries(HelloColorHSBPolar Threads::Threads ${SDL2_LIBRARIES})

add_executable(HelloFractalBrownianMotion hello_fbm.cpp 
  ${SHSRenderer_SOURCE_DIR}/shs_renderer.hpp
  )
target_link_libraries(HelloFractalBrownianMotion Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloGreenNoise hello_green_noise.cpp 
  ${SHSRenderer_SOURCE_DIR}/shs_renderer.hpp
  )
target_link_libraries(HelloGreenNoise Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloMongolianFlag hello_mongolian_flag.cpp 
  ${SHSRenderer_SOURCE_DIR}/shs_renderer.hpp
  )
target_link_libraries(HelloMongolianFlag Threads::Threads ${SDL2_LIBRARIES} glm::glm)