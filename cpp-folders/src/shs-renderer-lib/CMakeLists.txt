#project(SHSRendererLib)
#find_package(SDL2 CONFIG REQUIRED)
#find_package(SDL2_image CONFIG REQUIRED)
#find_package(glm CONFIG REQUIRED)
#find_package(assimp CONFIG REQUIRED)
#include_directories(${ASSIMP_INCLUDE_DIR})
#include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
#include_directories(${CMAKE_BINARY_DIR})
#include_directories(${stb_SOURCE_DIR}/)
#cmake_minimum_required(VERSION 3.20)


project(SHSRenderer LANGUAGES CXX)

find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

add_library(shs_renderer INTERFACE)
add_library(shs::renderer ALIAS shs_renderer)

# C++ standard for everyone who links this
target_compile_features(shs_renderer INTERFACE cxx_std_20)

# Packages (vcpkg config mode)
find_package(SDL2 CONFIG REQUIRED)
if(TARGET SDL2_image::SDL2_image)
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image)
else()
    add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image-static)
endif()
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

# Include paths that your headers need
# Adjust these to your real layout:
#   src/shs-renderer/include/...
#   src/shs-renderer/ (if headers still live here)
target_include_directories(shs_renderer INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${stb_SOURCE_DIR}
)

# Link deps that demos will inherit automatically
target_link_libraries(shs_renderer INTERFACE
    SDL2::SDL2
    SDL2_image::Main
    glm::glm
    assimp::assimp
)


# If you need ASSIMP_INCLUDE_DIR for some reason, prefer target includes:
# target_include_directories(shs_renderer INTERFACE ${ASSIMP_INCLUDE_DIR})
