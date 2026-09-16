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

# Vulkan discovery (absorbed from shs-gpu-lib): QUIET — the lib configures and
# builds GPU-free when Vulkan is absent; the runtime edge sources are simply
# not compiled and SHS_HAS_VULKAN stays unset.
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

# NOTE: since P2 the value-desc Vulkan driver pod lives HERE under
# include/shs/execution/rhi/drivers/vulkan/ (header-only, pure translation +
# hash-keyed explicit caches, GPU-free testable). The lib stays Vulkan-free by
# default — the driver compiles only where Vulkan headers are used (test target
# above, backend_factory under SHS_HAS_VULKAN).
# Since 2026-09 the absorbed runtime edge from shs-gpu-lib also lives here:
# include/shs/rhi/drivers/vulkan/ (SDL/windowed monolith backend) +
# include/shs/pipeline/vk_* — gated behind SHS_HAS_VULKAN; its P3 pod
# decomposition into the pod-aligned driver remains future work.

option(SHS_RENDERER_BUILD_SHARED "Build shs_renderer as a shared library" OFF)
set(_shs_renderer_lib_type STATIC)
if(SHS_RENDERER_BUILD_SHARED)
    set(_shs_renderer_lib_type SHARED)
endif()

add_library(shs_renderer ${_shs_renderer_lib_type}
    src/shs_renderer_lib.cpp
)
add_library(shs::renderer ALIAS shs_renderer)
# Converged naming (2026-09): single library — primary target shs_renderer,
# alias shs::renderer. Software vs GPU is a driver-pod + capability-gate
# selection (SHS_HAS_VULKAN), not a library split.

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
target_compile_features(shs_renderer PUBLIC cxx_std_23)

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
target_include_directories(shs_renderer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${stb_SOURCE_DIR}
)

# Link deps that demos will inherit automatically
target_link_libraries(shs_renderer PUBLIC
    SDL2::SDL2
    SDL2_image::Main
    glm::glm
    assimp::assimp
)

set(SHS_RENDERER_HAS_XSIMD OFF)
if(TARGET xsimd)
    set(SHS_RENDERER_HAS_XSIMD ON)
    target_link_libraries(shs_renderer PUBLIC xsimd)
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_XSIMD=1)
elseif(TARGET xsimd::xsimd)
    set(SHS_RENDERER_HAS_XSIMD ON)
    target_link_libraries(shs_renderer PUBLIC xsimd::xsimd)
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_XSIMD=1)
endif()
set(SHS_RENDERER_HAS_XSIMD "${SHS_RENDERER_HAS_XSIMD}" PARENT_SCOPE)

set(SHS_RENDERER_HAS_JOLT OFF)
if(TARGET Jolt)
    set(SHS_RENDERER_HAS_JOLT ON)
    target_link_libraries(shs_renderer PUBLIC Jolt)
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_JOLT=1)
endif()
set(SHS_RENDERER_HAS_JOLT "${SHS_RENDERER_HAS_JOLT}" PARENT_SCOPE)

# Vulkan runtime edge (absorbed from shs-gpu-lib, 2026-09): the monolith
# backend + VMA implementation compile into shs_renderer only when Vulkan was
# found; otherwise the lib stays GPU-free and the pod-aligned driver
# (shs/execution/rhi/drivers/vulkan/) remains the only Vulkan code.
set(SHS_RENDERER_HAS_VULKAN OFF)
if(Vulkan_FOUND AND TARGET Vulkan::Vulkan)
    set(SHS_RENDERER_HAS_VULKAN ON)
    target_sources(shs_renderer PRIVATE
        src/shs_gpu_runtime_edge.cpp
        src/shs_vma.cpp
    )
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_VULKAN=1)
    target_link_libraries(shs_renderer PUBLIC Vulkan::Vulkan)
    if(TARGET GPUOpen::VulkanMemoryAllocator)
        target_link_libraries(shs_renderer PUBLIC GPUOpen::VulkanMemoryAllocator)
        target_compile_definitions(shs_renderer PUBLIC SHS_HAS_VMA=1)
    endif()
endif()
set(SHS_RENDERER_HAS_VULKAN "${SHS_RENDERER_HAS_VULKAN}" PARENT_SCOPE)
set(SHS_SDL2_HAS_VULKAN "${SHS_SDL2_HAS_VULKAN}" PARENT_SCOPE)

# Header-only value-layer target: include dirs + header-only deps ONLY.
# Pure value code (pods, reducers, planners) must build and test against this
# with zero SDL/assimp/Vulkan links (KDBA Constitution II §7 headless-test law).
add_library(shs_renderer_values INTERFACE)
add_library(shs::renderer-values ALIAS shs_renderer_values)
target_include_directories(shs_renderer_values INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${stb_SOURCE_DIR}
)
target_compile_features(shs_renderer_values INTERFACE cxx_std_23)
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

    add_executable(shs_renderer_tests
        tests/core_tests.cpp
    )
    target_link_libraries(shs_renderer_tests PRIVATE shs::renderer)
    target_compile_features(shs_renderer_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # P1 renderpath pod tests: pure value tests, frame-arena events, no
    # platform links (only shs::renderer-values + glm headers).
    add_executable(shs_renderer_renderpath_tests
        tests/renderpath_tests.cpp
    )
    target_link_libraries(shs_renderer_renderpath_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_renderpath_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_renderpath_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_renderpath_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_custom_target(shs_renderer_boundary_check
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_kdba_boundaries.sh"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Running KDBA boundary checks for planner layer"
        VERBATIM
    )

    add_test(
        NAME shs_renderer_boundary_check
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_kdba_boundaries.sh"
    )
    set_tests_properties(shs_renderer_boundary_check PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME shs_renderer_tests
        COMMAND shs_renderer_tests
    )
    set_tests_properties(shs_renderer_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # P1.5 container/arena tests: header-only value tests, no platform links.
    add_executable(shs_renderer_containers_tests
        tests/containers_tests.cpp
    )
    target_link_libraries(shs_renderer_containers_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_containers_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_containers_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_containers_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_renderpath_tests
        COMMAND shs_renderer_renderpath_tests
    )
    set_tests_properties(shs_renderer_renderpath_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_containers_tests
        COMMAND shs_renderer_containers_tests
    )
    set_tests_properties(shs_renderer_containers_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # R3 pod tests: input + frame Core 4, header-only value tests, no platform links.
    add_executable(shs_renderer_input_tests
        tests/input_tests.cpp
    )
    target_link_libraries(shs_renderer_input_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_input_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_input_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_input_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_frame_tests
        tests/frame_tests.cpp
    )
    target_link_libraries(shs_renderer_frame_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_frame_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_frame_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_frame_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_input_tests
        COMMAND shs_renderer_input_tests
    )
    set_tests_properties(shs_renderer_input_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_executable(shs_renderer_geometry_tests
        tests/geometry_tests.cpp
    )
    target_link_libraries(shs_renderer_geometry_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_geometry_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_geometry_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_geometry_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_lighting_tests
        tests/lighting_tests.cpp
    )
    target_link_libraries(shs_renderer_lighting_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_lighting_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_lighting_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_lighting_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_geometry_tests
        COMMAND shs_renderer_geometry_tests
    )
    set_tests_properties(shs_renderer_geometry_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_executable(shs_renderer_camera_tests
        tests/camera_tests.cpp
    )
    target_link_libraries(shs_renderer_camera_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_camera_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_camera_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_camera_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_resources_tests
        tests/resources_tests.cpp
    )
    target_link_libraries(shs_renderer_resources_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_resources_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_resources_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_resources_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_camera_tests
        COMMAND shs_renderer_camera_tests
    )
    set_tests_properties(shs_renderer_camera_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_executable(shs_renderer_sky_tests
        tests/sky_tests.cpp
    )
    target_link_libraries(shs_renderer_sky_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_sky_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_sky_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_sky_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_logic_tests
        tests/logic_tests.cpp
    )
    target_link_libraries(shs_renderer_logic_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_logic_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_logic_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_logic_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_scene_tests
        tests/scene_tests.cpp
    )
    target_link_libraries(shs_renderer_scene_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_scene_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_scene_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_scene_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_gfx_tests
        tests/gfx_tests.cpp
    )

    add_executable(shs_renderer_saga_tests
        tests/saga_tests.cpp
    )
    target_link_libraries(shs_renderer_saga_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_saga_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_saga_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_saga_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    target_link_libraries(shs_renderer_gfx_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_gfx_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_gfx_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_gfx_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_gfx_tests
        COMMAND shs_renderer_gfx_tests
    )

    add_test(
        NAME shs_renderer_saga_tests
        COMMAND shs_renderer_saga_tests
    )
    set_tests_properties(shs_renderer_saga_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    set_tests_properties(shs_renderer_gfx_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_scene_tests
        COMMAND shs_renderer_scene_tests
    )
    set_tests_properties(shs_renderer_scene_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_logic_tests
        COMMAND shs_renderer_logic_tests
    )
    set_tests_properties(shs_renderer_logic_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_sky_tests
        COMMAND shs_renderer_sky_tests
    )
    set_tests_properties(shs_renderer_sky_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_resources_tests
        COMMAND shs_renderer_resources_tests
    )
    set_tests_properties(shs_renderer_resources_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_lighting_tests
        COMMAND shs_renderer_lighting_tests
    )
    set_tests_properties(shs_renderer_lighting_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_frame_tests
        COMMAND shs_renderer_frame_tests
    )
    set_tests_properties(shs_renderer_frame_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # P2 Vulkan driver pod tests: pure translation/bookkeeping tests. The
    # binary never EXECUTES vk* entry points at runtime (headless, no ICD
    # needed) but links the loader so the never-taken GPU code paths resolve.
    # Skipped cleanly when Vulkan is absent (lib stays Vulkan-free by default).
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        add_executable(shs_renderer_vk_driver_tests
            tests/vk_driver_tests.cpp
        )
        target_link_libraries(shs_renderer_vk_driver_tests PRIVATE
            shs::renderer-values
            Vulkan::Vulkan
        )
        target_compile_features(shs_renderer_vk_driver_tests PRIVATE cxx_std_23)
        if(MSVC)
            target_compile_options(shs_renderer_vk_driver_tests PRIVATE /W4)
        else()
            target_compile_options(shs_renderer_vk_driver_tests PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        add_test(
            NAME shs_renderer_vk_driver_tests
            COMMAND shs_renderer_vk_driver_tests
        )
        set_tests_properties(shs_renderer_vk_driver_tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
    else()
        message(STATUS "Vulkan headers not found — skipping shs_renderer_vk_driver_tests")
    endif()
endif()


# If you need ASSIMP_INCLUDE_DIR for some reason, prefer target includes:
# target_include_directories(shs_renderer INTERFACE ${ASSIMP_INCLUDE_DIR})
