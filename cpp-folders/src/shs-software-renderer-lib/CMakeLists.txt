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


project(SHSSoftwareRenderer LANGUAGES CXX)

find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

# NOTE: Vulkan lives in src/shs-gpu-renderer-lib (rhi/drivers/vulkan/,
# pipeline/vk_*, shaders/vulkan/, src/shs_vma.cpp). This lib is Vulkan-free.

option(SHS_RENDERER_BUILD_SHARED "Build shs_software_renderer as a shared library" OFF)
set(_shs_renderer_lib_type STATIC)
if(SHS_RENDERER_BUILD_SHARED)
    set(_shs_renderer_lib_type SHARED)
endif()

add_library(shs_software_renderer ${_shs_renderer_lib_type}
    src/shs_renderer_lib.cpp
)
add_library(shs::software-renderer ALIAS shs_software_renderer)
# Compat: existing exps link shs::renderer; keep it pointing at software.
# GPU consumers link shs::gpu-renderer (which links software PUBLIC).
add_library(shs::renderer ALIAS shs_software_renderer)

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
target_compile_features(shs_software_renderer PUBLIC cxx_std_20)

# Packages (vcpkg config mode)
find_package(SDL2 CONFIG REQUIRED)
if(NOT TARGET SDL2_image::Main)
    if(TARGET SDL2_image::SDL2_image)
        add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image)
    elseif(TARGET SDL2_image::SDL2_image-static)
        add_library(SDL2_image::Main ALIAS SDL2_image::SDL2_image-static)
    elseif(TARGET SDL2_image-static)
        add_library(SDL2_image::Main ALIAS SDL2_image-static)
    endif()
endif()
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

# Include paths that your headers need
target_include_directories(shs_software_renderer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${stb_SOURCE_DIR}
)

# Link deps that demos will inherit automatically
target_link_libraries(shs_software_renderer PUBLIC
    SDL2::SDL2
    SDL2_image::Main
    glm::glm
    assimp::assimp
)

set(SHS_RENDERER_HAS_XSIMD OFF)
if(TARGET xsimd)
    set(SHS_RENDERER_HAS_XSIMD ON)
    target_link_libraries(shs_software_renderer PUBLIC xsimd)
    target_compile_definitions(shs_software_renderer PUBLIC SHS_HAS_XSIMD=1)
elseif(TARGET xsimd::xsimd)
    set(SHS_RENDERER_HAS_XSIMD ON)
    target_link_libraries(shs_software_renderer PUBLIC xsimd::xsimd)
    target_compile_definitions(shs_software_renderer PUBLIC SHS_HAS_XSIMD=1)
endif()
set(SHS_RENDERER_HAS_XSIMD "${SHS_RENDERER_HAS_XSIMD}" PARENT_SCOPE)

set(SHS_RENDERER_HAS_JOLT OFF)
if(TARGET Jolt)
    set(SHS_RENDERER_HAS_JOLT ON)
    target_link_libraries(shs_software_renderer PUBLIC Jolt)
    target_compile_definitions(shs_software_renderer PUBLIC SHS_HAS_JOLT=1)
endif()
set(SHS_RENDERER_HAS_JOLT "${SHS_RENDERER_HAS_JOLT}" PARENT_SCOPE)

# Vulkan ownership moved to src/shs-gpu-renderer-lib. Report OFF here;
# the GPU lib re-exports ON to the parent scope when Vulkan is found.
set(SHS_RENDERER_HAS_VULKAN OFF PARENT_SCOPE)
set(SHS_SDL2_HAS_VULKAN "${SHS_SDL2_HAS_VULKAN}" PARENT_SCOPE)

# Header-only value-layer target: include dirs + header-only deps ONLY.
# Pure value code (pods, reducers, planners) must build and test against this
# with zero SDL/assimp/Vulkan links (VOP Constitution §7 headless-test law).
add_library(shs_renderer_values INTERFACE)
add_library(shs::renderer-values ALIAS shs_renderer_values)
target_include_directories(shs_renderer_values INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${stb_SOURCE_DIR}
)
target_compile_features(shs_renderer_values INTERFACE cxx_std_20)
# glm is header-only; safe to propagate without linking any compiled lib.
target_link_libraries(shs_renderer_values INTERFACE glm::glm)
if(TARGET xsimd::xsimd)
    target_link_libraries(shs_renderer_values INTERFACE xsimd::xsimd)
    target_compile_definitions(shs_renderer_values INTERFACE SHS_HAS_XSIMD=1)
elseif(TARGET xsimd)
    target_link_libraries(shs_renderer_values INTERFACE xsimd)
    target_compile_definitions(shs_renderer_values INTERFACE SHS_HAS_XSIMD=1)
endif()

option(SHS_RENDERER_BUILD_TESTS "Build shs_renderer core tests" ON)
if(SHS_RENDERER_BUILD_TESTS)
    enable_testing()

    add_executable(shs_renderer_vop_tests
        tests/vop_core_tests.cpp
    )
    target_link_libraries(shs_renderer_vop_tests PRIVATE shs::renderer)
    target_compile_features(shs_renderer_vop_tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(shs_renderer_vop_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_vop_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # P1 renderpath pod tests: pure value tests, frame-arena events, no
    # platform links (only shs::renderer-values + glm headers).
    add_executable(shs_renderer_vop_renderpath_tests
        tests/vop_renderpath_tests.cpp
    )
    target_link_libraries(shs_renderer_vop_renderpath_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_vop_renderpath_tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(shs_renderer_vop_renderpath_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_vop_renderpath_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_custom_target(shs_renderer_vop_boundary_check
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_vop_boundaries.sh"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Running VOP boundary checks for planner layer"
        VERBATIM
    )

    add_test(
        NAME shs_renderer_vop_boundary_check
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_vop_boundaries.sh"
    )
    set_tests_properties(shs_renderer_vop_boundary_check PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME shs_renderer_vop_tests
        COMMAND shs_renderer_vop_tests
    )
    set_tests_properties(shs_renderer_vop_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # P1.5 container/arena tests: header-only value tests, no platform links.
    add_executable(shs_renderer_vop_containers_tests
        tests/vop_containers_tests.cpp
    )
    target_link_libraries(shs_renderer_vop_containers_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_vop_containers_tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(shs_renderer_vop_containers_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_vop_containers_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_vop_renderpath_tests
        COMMAND shs_renderer_vop_renderpath_tests
    )
    set_tests_properties(shs_renderer_vop_renderpath_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_vop_containers_tests
        COMMAND shs_renderer_vop_containers_tests
    )
    set_tests_properties(shs_renderer_vop_containers_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()


# If you need ASSIMP_INCLUDE_DIR for some reason, prefer target includes:
# target_include_directories(shs_software_renderer INTERFACE ${ASSIMP_INCLUDE_DIR})
