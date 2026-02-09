project(HelloRenderTarget)

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

include_directories(${HelloSHSRenderer_SOURCE_DIR}/)
#include_directories(${CMAKE_SOURCE_DIR}/src/shs-renderer-lib/include)

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



file(COPY ../assets DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")


add_executable(HelloPingPong hello_ping_pong.cpp)
target_link_libraries(HelloPingPong PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloDepthOfField hello_depth_of_field.cpp)
target_link_libraries(HelloDepthOfField PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloPerObjectMotionBlur hello_per_object_motion_blur.cpp)
target_link_libraries(HelloPerObjectMotionBlur PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloMultiPass hello_multi_pass.cpp)
target_link_libraries(HelloMultiPass PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloGlowingStar hello_glowing_star.cpp)
target_link_libraries(HelloGlowingStar PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloCameraAndPerobjectMotionBlur hello_camera_and_perobject_motion_blur.cpp)
target_link_libraries(HelloCameraAndPerobjectMotionBlur PRIVATE ${SDL2_IMAGE_LIBRARIES} ${SDL2_LIBRARIES} glm::glm assimp::assimp)

add_executable(HelloShadowMapping hello_shadow_mapping.cpp)
target_link_libraries(HelloShadowMapping PRIVATE SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)

add_executable(HelloShadowMappingSoft hello_shadow_mapping_soft.cpp)
target_link_libraries(HelloShadowMappingSoft PRIVATE SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)

add_executable(HelloWater hello_water.cpp)
target_link_libraries(HelloWater PRIVATE SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)


add_executable(HelloIblSkybox hello_ibl_skybox.cpp)
target_link_libraries(HelloIblSkybox PRIVATE SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)


add_executable(HelloIblSkyboxOpt hello_ibl_skybox_optimized.cpp)
target_compile_features(HelloIblSkyboxOpt PRIVATE cxx_std_20)
target_link_libraries(HelloIblSkyboxOpt PRIVATE xsimd SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)
if (MSVC)
    target_compile_options(HelloIblSkyboxOpt PRIVATE /O2 /fp:fast /arch:AVX2)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    # M1/M2/M3 specific flags
    # -O3 is usually enough as Clang optimizes for NEON automatically on ARM64
    target_compile_options(HelloIblSkyboxOpt PRIVATE -O3 -ffast-math)
else()
    # Standard Linux/Intel Mac x86_64 flags
    target_compile_options(HelloIblSkyboxOpt PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()


#add_executable(HelloIblSkyboxXSIMD hello_ibl_skybox_xsimd.cpp)
#target_compile_features(HelloIblSkyboxXSIMD PRIVATE cxx_std_20)
#target_link_libraries(HelloIblSkyboxXSIMD PRIVATE xsimd SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)
#if (MSVC)
#    target_compile_options(HelloIblSkyboxXSIMD PRIVATE /O2 /fp:fast /arch:AVX2)
#elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
#    target_compile_options(HelloIblSkyboxXSIMD PRIVATE -O3 -ffast-math)
#else()
#    target_compile_options(HelloIblSkyboxXSIMD PRIVATE -O3 -ffast-math -mavx2 -mfma)
#endif()


#add_executable(HelloPBR hello_pbr.cpp)
#target_compile_features(HelloPBR PRIVATE cxx_std_20)
#target_link_libraries(HelloPBR PRIVATE xsimd SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)
#if (MSVC)
#    target_compile_options(HelloPBR PRIVATE /O2 /fp:fast /arch:AVX2)
#elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
#    target_compile_options(HelloPBR PRIVATE -O3 -ffast-math)
#else()
#    target_compile_options(HelloPBR PRIVATE -O3 -ffast-math -mavx2 -mfma)
#endif()


#add_executable(HelloPBRLightShafts hello_pbr_light_shafts.cpp)
#target_compile_features(HelloPBRLightShafts PRIVATE cxx_std_20)
#target_link_libraries(HelloPBRLightShafts PRIVATE xsimd SDL2::SDL2 SDL2_image::Main glm::glm assimp::assimp)
#if (MSVC)
#    target_compile_options(HelloPBRLightShafts PRIVATE /O2 /fp:fast /arch:AVX2)
#elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
#    target_compile_options(HelloPBRLightShafts PRIVATE -O3 -ffast-math)
#else()
#    target_compile_options(HelloPBRLightShafts PRIVATE -O3 -ffast-math -mavx2 -mfma)
#endif()

