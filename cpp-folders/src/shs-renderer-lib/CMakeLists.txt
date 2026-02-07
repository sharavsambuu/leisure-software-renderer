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

if(APPLE AND DEFINED ENV{VULKAN_SDK})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
endif()
find_package(Vulkan QUIET)
if(APPLE AND NOT Vulkan_FOUND AND DEFINED ENV{VULKAN_SDK})
    set(_shs_vk_sdk "$ENV{VULKAN_SDK}")
    set(_shs_vk_header "${_shs_vk_sdk}/include/vulkan/vulkan.h")
    set(_shs_vk_lib "${_shs_vk_sdk}/lib/libvulkan.dylib")
    if(EXISTS "${_shs_vk_header}" AND EXISTS "${_shs_vk_lib}" AND NOT TARGET Vulkan::Vulkan)
        add_library(Vulkan::Vulkan SHARED IMPORTED GLOBAL)
        set_target_properties(Vulkan::Vulkan PROPERTIES
            IMPORTED_LOCATION "${_shs_vk_lib}"
            INTERFACE_INCLUDE_DIRECTORIES "${_shs_vk_sdk}/include"
        )
        set(Vulkan_FOUND TRUE)
    endif()
endif()

add_library(shs_renderer INTERFACE)
add_library(shs::renderer ALIAS shs_renderer)

set(SHS_SDL2_HAS_VULKAN OFF)
set(_shs_sdl2_config "")
if(TARGET SDL2::SDL2)
    get_target_property(_shs_sdl2_inc_dirs SDL2::SDL2 INTERFACE_INCLUDE_DIRECTORIES)
    foreach(_inc IN LISTS _shs_sdl2_inc_dirs)
        if("${_inc}" MATCHES "^\\$<")
            continue()
        endif()
        if(EXISTS "${_inc}/SDL2/SDL_config.h")
            set(_shs_sdl2_config "${_inc}/SDL2/SDL_config.h")
            break()
        endif()
        if(EXISTS "${_inc}/SDL_config.h")
            set(_shs_sdl2_config "${_inc}/SDL_config.h")
            break()
        endif()
    endforeach()
endif()
if(_shs_sdl2_config)
    file(READ "${_shs_sdl2_config}" _shs_sdl2_config_text)
    if(_shs_sdl2_config_text MATCHES "#define[ \t]+SDL_VIDEO_VULKAN[ \t]+1")
        set(SHS_SDL2_HAS_VULKAN ON)
    endif()
endif()

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

set(SHS_RENDERER_HAS_VULKAN OFF)
if(Vulkan_FOUND AND TARGET Vulkan::Vulkan)
    set(SHS_RENDERER_HAS_VULKAN ON)
    target_compile_definitions(shs_renderer INTERFACE SHS_HAS_VULKAN=1)
    target_link_libraries(shs_renderer INTERFACE Vulkan::Vulkan)
endif()
set(SHS_RENDERER_HAS_VULKAN "${SHS_RENDERER_HAS_VULKAN}" PARENT_SCOPE)
set(SHS_SDL2_HAS_VULKAN "${SHS_SDL2_HAS_VULKAN}" PARENT_SCOPE)


# If you need ASSIMP_INCLUDE_DIR for some reason, prefer target includes:
# target_include_directories(shs_renderer INTERFACE ${ASSIMP_INCLUDE_DIR})
