project(Hello3D)

find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

if(TARGET SDL2_image::SDL2_image)
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image)
else()
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image-static)
endif()


include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${stb_SOURCE_DIR}/)
#include_directories(${entt_SOURCE_DIR}/src/)

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


# 3D model wireframe rendering related commands
file(COPY obj DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(HelloWireframe hello_wireframe.cpp)
target_link_libraries(HelloWireframe PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)



add_executable(Hello3DObject hello_3d_object.cpp)
target_link_libraries(Hello3DObject PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)


add_executable(HelloFlatShading hello_flat_shading.cpp)
target_link_libraries(HelloFlatShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloFlatShadingXSIMD hello_flat_shading_xsimd.cpp)
target_compile_features(HelloFlatShadingXSIMD PRIVATE cxx_std_20)
target_link_libraries(HelloFlatShadingXSIMD PRIVATE xsimd ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)
if (MSVC)
  target_compile_options(HelloFlatShadingXSIMD PRIVATE /O2 /fp:fast /arch:AVX2)
else()
  target_compile_options(HelloFlatShadingXSIMD PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()


add_executable(HelloPipelineFlatShading hello_pipeline_flat_shading.cpp)
target_link_libraries(HelloPipelineFlatShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelinePhongShading hello_pipeline_phong_shading.cpp)
target_link_libraries(HelloPipelinePhongShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineBlinnPhongShading hello_pipeline_blinn_phong_shading.cpp)
target_link_libraries(HelloPipelineBlinnPhongShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineGouraudShading hello_pipeline_gouraud_shading.cpp)
target_link_libraries(HelloPipelineGouraudShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineToonShading hello_pipeline_toon_shading.cpp)
target_link_libraries(HelloPipelineToonShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineGoochShading hello_pipeline_gooch_shading.cpp)
target_link_libraries(HelloPipelineGoochShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineOrenNayarShading hello_pipeline_oren_nayar_shading.cpp)
target_link_libraries(HelloPipelineOrenNayarShading PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineNormalZbufferDebug hello_pipeline_normal_zbuffer_debug.cpp)
target_link_libraries(HelloPipelineNormalZbufferDebug PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineTextureMapping hello_pipeline_texture_mapping.cpp)
target_link_libraries(HelloPipelineTextureMapping PRIVATE SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)