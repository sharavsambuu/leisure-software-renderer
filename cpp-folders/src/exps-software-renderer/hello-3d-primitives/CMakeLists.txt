project(Hello3D)

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
#include_directories(${entt_SOURCE_DIR}/src/)

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


file(COPY ../../assets DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")



add_executable(HelloWireframe hello_wireframe.cpp)
target_link_libraries(HelloWireframe PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)



add_executable(Hello3DObject hello_3d_object.cpp)
target_link_libraries(Hello3DObject PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)


add_executable(HelloFlatShading hello_flat_shading.cpp)
target_link_libraries(HelloFlatShading PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloFlatShadingXSIMD hello_flat_shading_xsimd.cpp)
target_compile_features(HelloFlatShadingXSIMD PRIVATE cxx_std_23)
target_link_libraries(HelloFlatShadingXSIMD PRIVATE xsimd ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)
if (MSVC)
    target_compile_options(HelloFlatShadingXSIMD PRIVATE /O2 /fp:fast /arch:AVX2)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    # M1/M2/M3 specific flags
    # -O3 is usually enough as Clang optimizes for NEON automatically on ARM64
    target_compile_options(HelloFlatShadingXSIMD PRIVATE -O3 -ffast-math)
else()
    # Standard Linux/Intel Mac x86_64 flags
    target_compile_options(HelloFlatShadingXSIMD PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()

add_executable(HelloPipelineFlatShading hello_pipeline_flat_shading.cpp)
target_link_libraries(HelloPipelineFlatShading PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelinePhongShading hello_pipeline_phong_shading.cpp)
target_link_libraries(HelloPipelinePhongShading PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineBlinnPhongShading hello_pipeline_blinn_phong_shading.cpp)
target_link_libraries(HelloPipelineBlinnPhongShading PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineGouraudShading hello_pipeline_gouraud_shading.cpp)
target_link_libraries(HelloPipelineGouraudShading PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineToonShading hello_pipeline_toon_shading.cpp)
target_link_libraries(HelloPipelineToonShading PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineGoochShading hello_pipeline_gooch_shading.cpp)
target_link_libraries(HelloPipelineGoochShading PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineOrenNayarShading hello_pipeline_oren_nayar_shading.cpp)
target_link_libraries(HelloPipelineOrenNayarShading PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineNormalZbufferDebug hello_pipeline_normal_zbuffer_debug.cpp)
target_link_libraries(HelloPipelineNormalZbufferDebug PRIVATE ${SDL3_IMAGE_LIBRARIES} ${SDL3_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPipelineTextureMapping hello_pipeline_texture_mapping.cpp)
target_link_libraries(HelloPipelineTextureMapping PRIVATE SDL3::SDL3 SDL3_image::Main glm::glm assimp::assimp)