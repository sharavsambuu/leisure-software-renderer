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

add_executable(HelloFlatShading hello_flat_shading.cpp)
target_link_libraries(HelloFlatShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloPipelineFlatShading hello_pipeline_flat_shading.cpp)
target_link_libraries(HelloPipelineFlatShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloPipelinePhongShading hello_pipeline_phong_shading.cpp)
target_link_libraries(HelloPipelinePhongShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloPipelineBlinnPhongShading hello_pipeline_blinn_phong_shading.cpp)
target_link_libraries(HelloPipelineBlinnPhongShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloPipelineGouraudShading hello_pipeline_gouraud_shading.cpp)
target_link_libraries(HelloPipelineGouraudShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloPipelineToonShading hello_pipeline_toon_shading.cpp)
target_link_libraries(HelloPipelineToonShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloPipelineGoochShading hello_pipeline_gooch_shading.cpp)
target_link_libraries(HelloPipelineGoochShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)

add_executable(HelloPipelineOrenNayarShading hello_pipeline_oren_nayar_shading.cpp)
target_link_libraries(HelloPipelineOrenNayarShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm -lassimp)