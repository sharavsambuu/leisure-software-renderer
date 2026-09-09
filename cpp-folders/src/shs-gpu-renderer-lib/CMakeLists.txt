# SHS GPU renderer lib — Vulkan backend extracted from shs-renderer-lib.
# Location: cpp-folders/src/shs-gpu-renderer-lib/CMakeLists.txt
# Owns: include/shs/rhi/drivers/vulkan/, include/shs/pipeline/vk_*,
#       shaders/vulkan/, src/shs_vma.cpp.
# Links shs::software-renderer PUBLIC so consumers get both backends.

project(SHSGpuRenderer LANGUAGES CXX)

find_package(SDL2 CONFIG REQUIRED)
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

option(SHS_RENDERER_BUILD_SHARED "Build shs_gpu_renderer as a shared library" OFF)
set(_shs_gpu_lib_type STATIC)
if(SHS_RENDERER_BUILD_SHARED)
    set(_shs_gpu_lib_type SHARED)
endif()

add_library(shs_gpu_renderer ${_shs_gpu_lib_type}
    src/shs_gpu_lib.cpp
    src/shs_vma.cpp
)
add_library(shs::gpu-renderer ALIAS shs_gpu_renderer)

target_compile_features(shs_gpu_renderer PUBLIC cxx_std_20)

target_include_directories(shs_gpu_renderer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Software lib provides IRenderBackend + core types; GPU extends them.
target_link_libraries(shs_gpu_renderer PUBLIC shs::software-renderer)

# SDL2 -> Vulkan surface capability probe (same logic as software lib).
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

set(SHS_RENDERER_HAS_VULKAN OFF)
if(Vulkan_FOUND AND TARGET Vulkan::Vulkan)
    set(SHS_RENDERER_HAS_VULKAN ON)
    target_compile_definitions(shs_gpu_renderer PUBLIC SHS_HAS_VULKAN=1)
    target_link_libraries(shs_gpu_renderer PUBLIC Vulkan::Vulkan)
    if(TARGET GPUOpen::VulkanMemoryAllocator)
        target_link_libraries(shs_gpu_renderer PUBLIC GPUOpen::VulkanMemoryAllocator)
    endif()
    target_compile_definitions(shs_gpu_renderer PUBLIC SHS_HAS_VMA=1)
endif()
set(SHS_RENDERER_HAS_VULKAN "${SHS_RENDERER_HAS_VULKAN}" PARENT_SCOPE)
set(SHS_SDL2_HAS_VULKAN "${SHS_SDL2_HAS_VULKAN}" PARENT_SCOPE)
