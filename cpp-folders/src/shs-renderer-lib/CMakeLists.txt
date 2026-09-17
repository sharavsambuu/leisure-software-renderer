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

# Adapter selection (migration step 5 — selectable dependency ownership):
# SDL2/SDL2_image and Assimp are optional integration adapters. Defaults keep
# the historical behavior (all ON); consumer configurations may disable them
# to build the GPU-free, windowing-free core (the compiled TUs are anchors
# only, so nothing in the library itself needs these SDKs). Vulkan stays
# auto-discovered (QUIET); Jolt/xsimd stay auto-discovered target-conditional.
option(SHS_RENDERER_WITH_SDL2 "Discover and link SDL2/SDL2_image (windowed runtime + texture adapters)" ON)
option(SHS_RENDERER_WITH_ASSIMP "Discover and link Assimp (mesh import adapters)" ON)

find_package(glm CONFIG REQUIRED)
if(SHS_RENDERER_WITH_SDL2)
    find_package(SDL2 CONFIG REQUIRED)
    find_package(SDL2_image CONFIG REQUIRED)
endif()
if(SHS_RENDERER_WITH_ASSIMP)
    find_package(assimp CONFIG REQUIRED)
endif()

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
# include/shs/rhi/vulkan/value/ (header-only, pure translation +
# hash-keyed explicit caches, GPU-free testable). The lib stays Vulkan-free by
# default — the driver compiles only where Vulkan headers are used (test target
# above, backend_factory under SHS_HAS_VULKAN).
# Since 2026-09 the absorbed runtime edge from shs-gpu-lib also lives here:
# include/shs/rhi/vulkan/runtime/ (SDL/windowed monolith backend) +
# include/shs/renderpath/execution/vk_* — gated behind SHS_HAS_VULKAN; its P3 pod
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

# Packages (vcpkg config mode) — conditional per SHS_RENDERER_WITH_SDL2 /
# SHS_RENDERER_WITH_ASSIMP (see adapter-selection options at top).
if(SHS_RENDERER_WITH_SDL2)
    find_package(SDL2 CONFIG REQUIRED)
endif()
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
if(SHS_RENDERER_WITH_ASSIMP)
    find_package(assimp CONFIG REQUIRED)
endif()

# Include paths that your headers need. Migration step 5 (packaging): the
# source-tree, build-tree and stb dirs stay BUILD_INTERFACE-only so in-tree
# behavior is unchanged while the exported target carries exactly the package
# include dir (self-contained headers; no source-tree/build-tree leakage in
# installed packages — Rule 15 closed exception set).
target_include_directories(shs_renderer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<BUILD_INTERFACE:${stb_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link deps that demos will inherit automatically. SDL2/Assimp are optional
# adapters (step 5): when disabled, the library builds and links with zero
# windowing/asset-import SDKs; SHS_HAS_SDL2 / SHS_HAS_ASSIMP let headers
# feature-guard adapter entry points (same shape as SHS_HAS_JOLT).
target_link_libraries(shs_renderer PUBLIC
    glm::glm
)
if(SHS_RENDERER_WITH_SDL2)
    target_link_libraries(shs_renderer PUBLIC
        SDL2::SDL2
        SDL2_image::Main
    )
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_SDL2=1)
endif()
if(SHS_RENDERER_WITH_ASSIMP)
    target_link_libraries(shs_renderer PUBLIC assimp::assimp)
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_ASSIMP=1)
endif()

# Jolt and xsimd are FetchContent-owned in-tree (parent CMakeLists), so their
# linkage is BUILD_INTERFACE-only: the compiled target keeps linking them
# in-tree unchanged, but the installed package does not re-export them
# (consumers enable those features via their own Jolt/xsimd targets —
# documented binary-compatibility limitation, migration step 5).
set(SHS_RENDERER_HAS_XSIMD OFF)
if(TARGET xsimd)
    set(SHS_RENDERER_HAS_XSIMD ON)
    target_link_libraries(shs_renderer PUBLIC $<BUILD_INTERFACE:xsimd>)
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_XSIMD=1)
elseif(TARGET xsimd::xsimd)
    set(SHS_RENDERER_HAS_XSIMD ON)
    target_link_libraries(shs_renderer PUBLIC $<BUILD_INTERFACE:xsimd::xsimd>)
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_XSIMD=1)
endif()
set(SHS_RENDERER_HAS_XSIMD "${SHS_RENDERER_HAS_XSIMD}" PARENT_SCOPE)

set(SHS_RENDERER_HAS_JOLT OFF)
if(TARGET Jolt)
    set(SHS_RENDERER_HAS_JOLT ON)
    target_link_libraries(shs_renderer PUBLIC $<BUILD_INTERFACE:Jolt>)
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_JOLT=1)
endif()
set(SHS_RENDERER_HAS_JOLT "${SHS_RENDERER_HAS_JOLT}" PARENT_SCOPE)

# Vulkan runtime edge (absorbed from shs-gpu-lib, 2026-09): the monolith
# backend + VMA implementation compile into shs_renderer only when Vulkan was
# found; otherwise the lib stays GPU-free and the pod-aligned driver
# (shs/rhi/vulkan/value/) remains the only Vulkan code.
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
        target_link_libraries(shs_renderer PUBLIC
            $<BUILD_INTERFACE:GPUOpen::VulkanMemoryAllocator>)
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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<BUILD_INTERFACE:${stb_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(shs_renderer_values INTERFACE cxx_std_23)
# glm is header-only; safe to propagate without linking any compiled lib.
target_link_libraries(shs_renderer_values INTERFACE glm::glm)
if(TARGET xsimd::xsimd)
    target_link_libraries(shs_renderer_values INTERFACE $<BUILD_INTERFACE:xsimd::xsimd>)
    target_compile_definitions(shs_renderer_values INTERFACE SHS_HAS_XSIMD=1)
elseif(TARGET xsimd)
    target_link_libraries(shs_renderer_values INTERFACE $<BUILD_INTERFACE:xsimd>)
    target_compile_definitions(shs_renderer_values INTERFACE SHS_HAS_XSIMD=1)
endif()

# ---------------------------------------------------------------------------
# Install / export (migration step 5: make the library consumable as an
# installed package while preserving `shs_renderer` / `shs::renderer` and
# `shs::renderer-values` as the aggregate compatibility seams).
#
# Design (Rule 14 adapter value seam / Rule 15 closed exception sets):
# - in-tree behavior unchanged: source-tree, build-tree, stb include dirs and
#   FetchContent-owned Jolt/xsimd/VMA links are all $<BUILD_INTERFACE:...>,
#   so nothing they carry enters the export;
# - the exported interface is minimal: `include/` + glm, plus SDL2/Assimp/
#   Vulkan only when THIS build enabled them (Config re-discovers exactly
#   those); SHS_HAS_* feature guards travel with the binaries as
#   INTERFACE_COMPILE_DEFINITIONS, so package flags always match the build;
# - Config.cmake recreates the aggregate aliases (`shs::renderer`,
#   `shs::renderer-values`) for installed consumers.
include(GNUInstallDirs)
install(TARGETS shs_renderer shs_renderer_values
    EXPORT shs_rendererTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(DIRECTORY include/shs DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
set(SHS_RENDERER_PKG_WITH_SDL2 "${SHS_RENDERER_WITH_SDL2}")
set(SHS_RENDERER_PKG_WITH_ASSIMP "${SHS_RENDERER_WITH_ASSIMP}")
set(SHS_RENDERER_PKG_WITH_VULKAN "${SHS_RENDERER_HAS_VULKAN}")
set(SHS_RENDERER_CONFIG_OUT
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/shs_rendererConfig.cmake")
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/shs_rendererConfig.cmake.in"
    "${SHS_RENDERER_CONFIG_OUT}"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/shs_renderer"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/shs_rendererConfigVersion.cmake"
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${SHS_RENDERER_CONFIG_OUT}"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/shs_rendererConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/shs_renderer"
)
install(EXPORT shs_rendererTargets
    FILE shs_rendererTargets.cmake
    NAMESPACE shs::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/shs_renderer"
)

option(SHS_RENDERER_BUILD_TESTS "Build shs_renderer core tests" ON)
if(SHS_RENDERER_BUILD_TESTS)
    enable_testing()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME shs_renderer_header_migration_tests
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/header_migration_tests.py")

    # Migration step 5 — install/export/package-consumer tests: install this
    # build into a scratch prefix, reject source-tree/build-tree path leakage
    # in the installed package, then configure/build/run a minimal headless
    # consumer through find_package + the aggregate `shs::renderer` /
    # `shs::renderer-values` compatibility seams.
    set(_shs_pkg_consumer_args
        "${CMAKE_BINARY_DIR}"
        "${CMAKE_SOURCE_DIR}")
    if(CMAKE_TOOLCHAIN_FILE)
        list(APPEND _shs_pkg_consumer_args "${CMAKE_TOOLCHAIN_FILE}")
    endif()
    if(CMAKE_CXX_COMPILER)
        list(APPEND _shs_pkg_consumer_args "${CMAKE_CXX_COMPILER}")
    endif()
    add_test(NAME shs_renderer_package_consumer_test
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/package_consumer_test.sh"
            ${_shs_pkg_consumer_args})

    # Migration step 5 — self-contained public headers gate: every public
    # header compiles standalone with only the exported dependency include
    # dirs, no SHS_HAS_* defines and no build-tree/stb include dirs.
    set(_shs_selfcont_targets
        glm::glm SDL2::SDL2 SDL2_image::SDL2_image assimp::assimp
        Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator Jolt xsimd xsimd::xsimd)
    set(_shs_selfcont_inc "")
    foreach(_shs_t IN LISTS _shs_selfcont_targets)
        if(TARGET ${_shs_t})
            get_target_property(_shs_t_inc "${_shs_t}" INTERFACE_INCLUDE_DIRECTORIES)
            foreach(_shs_d IN LISTS _shs_t_inc)
                if(_shs_d AND NOT "${_shs_d}" MATCHES "^\\$<" AND EXISTS "${_shs_d}")
                    list(APPEND _shs_selfcont_inc "${_shs_d}")
                endif()
            endforeach()
            unset(_shs_t_inc)
        endif()
    endforeach()
    list(APPEND _shs_selfcont_inc "${CMAKE_CURRENT_SOURCE_DIR}/include")
    list(REMOVE_DUPLICATES _shs_selfcont_inc)
    set(_shs_selfcont_flags
        "${CMAKE_CURRENT_BINARY_DIR}/shs_selfcont_include_flags.txt")
    file(WRITE "${_shs_selfcont_flags}" "")
    foreach(_shs_d IN LISTS _shs_selfcont_inc)
        file(APPEND "${_shs_selfcont_flags}" "-I${_shs_d}\n")
    endforeach()
    add_test(NAME shs_renderer_header_self_containment_test
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/header_self_containment_test.sh"
            "${_shs_selfcont_flags}"
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CXX_COMPILER}")
    add_test(NAME shs_renderer_header_inventory_check
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/inventory_headers.py")
    add_test(NAME shs_renderer_include_graph_tests
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/include_graph_tests.py")
    add_test(NAME shs_renderer_include_graph_gate
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_include_graph.py")

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

    # Canonical/compatibility include ordering must preserve one definition.
    foreach(order IN ITEMS canonical legacy)
        set(camera_include_target shs_renderer_camera_include_${order}_tests)
        add_executable(${camera_include_target} tests/camera_include_compatibility_tests.cpp)
        target_link_libraries(${camera_include_target} PRIVATE shs::renderer-values)
        if(order STREQUAL "legacy")
            target_compile_definitions(${camera_include_target} PRIVATE SHS_TEST_LEGACY_INCLUDE_FIRST=1)
        endif()
        if(MSVC)
            target_compile_options(${camera_include_target} PRIVATE /W4)
        else()
            target_compile_options(${camera_include_target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        add_test(NAME ${camera_include_target} COMMAND ${camera_include_target})
    endforeach()

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
        COMMAND "${CMAKE_COMMAND}" -E env "PYTHON=${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_kdba_boundaries.sh"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Running KDBA boundary checks for planner layer"
        VERBATIM
    )

    add_test(
        NAME shs_renderer_boundary_check
        COMMAND "${CMAKE_COMMAND}" -E env "PYTHON=${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_kdba_boundaries.sh"
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

    # Contract guardrails bridge tests (Constitution II Rule 17, C1): the
    # enforcement twin observes the violation handler; the release twin
    # compiles the assume path with checks discarded.
    add_executable(shs_renderer_contract_guardrails_tests
        tests/contract_guardrails_tests.cpp
    )
    target_link_libraries(shs_renderer_contract_guardrails_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_contract_guardrails_tests PRIVATE cxx_std_23)
    target_compile_definitions(shs_renderer_contract_guardrails_tests PRIVATE SHS_CONTRACTS_ENFORCED=1)
    if(MSVC)
        target_compile_options(shs_renderer_contract_guardrails_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_contract_guardrails_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_contract_guardrails_release_tests
        tests/contract_guardrails_tests.cpp
    )
    target_link_libraries(shs_renderer_contract_guardrails_release_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_contract_guardrails_release_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_contract_guardrails_release_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_contract_guardrails_release_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_contract_guardrails_tests
        COMMAND shs_renderer_contract_guardrails_tests
    )
    set_tests_properties(shs_renderer_contract_guardrails_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_contract_guardrails_release_tests
        COMMAND shs_renderer_contract_guardrails_release_tests
    )
    set_tests_properties(shs_renderer_contract_guardrails_release_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # Replay parity (C1.4): identical probe source, only the contract mode
    # differs; replay-relevant output must be byte-identical across the pair.
    add_executable(shs_renderer_contract_replay_probe_enforced
        tests/contract_guardrails_replay_probe.cpp
    )
    target_link_libraries(shs_renderer_contract_replay_probe_enforced PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_contract_replay_probe_enforced PRIVATE cxx_std_23)
    target_compile_definitions(shs_renderer_contract_replay_probe_enforced PRIVATE SHS_CONTRACTS_ENFORCED=1)
    if(MSVC)
        target_compile_options(shs_renderer_contract_replay_probe_enforced PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_contract_replay_probe_enforced PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_contract_replay_probe_assume
        tests/contract_guardrails_replay_probe.cpp
    )
    target_link_libraries(shs_renderer_contract_replay_probe_assume PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_contract_replay_probe_assume PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_contract_replay_probe_assume PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_contract_replay_probe_assume PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_contract_replay_parity
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/contract_guardrails_replay_parity_test.py"
            "$<TARGET_FILE:shs_renderer_contract_replay_probe_enforced>"
            "$<TARGET_FILE:shs_renderer_contract_replay_probe_assume>"
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

        # G1 prerequisite coverage uses a real device; unavailability is a skip,
        # never a successful rendering claim. No GPU work is submitted.
        add_executable(shs_renderer_vk_recording_prerequisite_tests
            tests/vk_recording_prerequisite_tests.cpp)
        target_link_libraries(shs_renderer_vk_recording_prerequisite_tests PRIVATE
            shs::renderer-values Vulkan::Vulkan)
        target_compile_features(shs_renderer_vk_recording_prerequisite_tests PRIVATE cxx_std_23)
        if(MSVC)
            target_compile_options(shs_renderer_vk_recording_prerequisite_tests PRIVATE /W4)
        else()
            target_compile_options(shs_renderer_vk_recording_prerequisite_tests PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        add_test(NAME shs_renderer_vk_recording_prerequisite_tests
            COMMAND shs_renderer_vk_recording_prerequisite_tests)
        set_tests_properties(shs_renderer_vk_recording_prerequisite_tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            SKIP_RETURN_CODE 77)

        # G2 attachment/recording slice: actual objects and command recording,
        # but no queue submission or pixel claim yet.
        add_executable(shs_renderer_vk_offscreen_pass_tests tests/vk_offscreen_pass_tests.cpp)
        target_link_libraries(shs_renderer_vk_offscreen_pass_tests PRIVATE
            shs::renderer-values Vulkan::Vulkan)
        target_compile_features(shs_renderer_vk_offscreen_pass_tests PRIVATE cxx_std_23)
        if(MSVC)
            target_compile_options(shs_renderer_vk_offscreen_pass_tests PRIVATE /W4)
        else()
            target_compile_options(shs_renderer_vk_offscreen_pass_tests PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        # Vulkan 1.1 core: use SPIR-V 1.3, not the adventures' 1.5 profile.
        find_program(SHS_OFFSCREEN_SLANGC NAMES slangc HINTS "$ENV{VULKAN_SDK}/bin")
        if(SHS_OFFSCREEN_SLANGC)
            set(_offscreen_src "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/offscreen_pipeline.slang")
            set(_offscreen_spv "${CMAKE_CURRENT_BINARY_DIR}/offscreen_spv")
            add_custom_command(
                OUTPUT "${_offscreen_spv}/offscreen_vs.spv" "${_offscreen_spv}/offscreen_fs.spv"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_offscreen_spv}"
                COMMAND "${SHS_OFFSCREEN_SLANGC}" "${_offscreen_src}"
                    -target spirv -profile spirv_1_3 -matrix-layout-column-major -fvk-use-entrypoint-name
                    -entry vs_main -stage vertex -o "${_offscreen_spv}/offscreen_vs.spv"
                COMMAND "${SHS_OFFSCREEN_SLANGC}" "${_offscreen_src}"
                    -target spirv -profile spirv_1_3 -matrix-layout-column-major -fvk-use-entrypoint-name
                    -entry fs_main -stage fragment -o "${_offscreen_spv}/offscreen_fs.spv"
                DEPENDS "${_offscreen_src}" VERBATIM)
            add_custom_command(
                OUTPUT "${_offscreen_spv}/uploaded_vs.spv"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_offscreen_spv}"
                COMMAND "${SHS_OFFSCREEN_SLANGC}" "${_offscreen_src}"
                    -target spirv -profile spirv_1_3 -matrix-layout-column-major -fvk-use-entrypoint-name
                    -entry vs_uploaded -stage vertex -o "${_offscreen_spv}/uploaded_vs.spv"
                DEPENDS "${_offscreen_src}" VERBATIM)
            add_custom_target(shs_renderer_offscreen_shaders DEPENDS
                "${_offscreen_spv}/offscreen_vs.spv" "${_offscreen_spv}/offscreen_fs.spv"
                "${_offscreen_spv}/uploaded_vs.spv")
            add_dependencies(shs_renderer_vk_offscreen_pass_tests shs_renderer_offscreen_shaders)
            target_compile_definitions(shs_renderer_vk_offscreen_pass_tests PRIVATE
                SHS_OFFSCREEN_SHADER_DIR="${_offscreen_spv}")
        else()
            message(STATUS "slangc unavailable: offscreen test covers attachments only, not pipelines")
        endif()
        add_test(NAME shs_renderer_vk_offscreen_pass_tests COMMAND shs_renderer_vk_offscreen_pass_tests)
        set_tests_properties(shs_renderer_vk_offscreen_pass_tests PROPERTIES
            FAIL_REGULAR_EXPRESSION "Validation Error")
        set_tests_properties(shs_renderer_vk_offscreen_pass_tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            SKIP_RETURN_CODE 77)

    else()
        message(STATUS "Vulkan headers not found — skipping shs_renderer_vk_driver_tests")
    endif()
endif()


# If you need ASSIMP_INCLUDE_DIR for some reason, prefer target includes:
# target_include_directories(shs_renderer INTERFACE ${ASSIMP_INCLUDE_DIR})
