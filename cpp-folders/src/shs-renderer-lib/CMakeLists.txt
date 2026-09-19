#project(SHSRendererLib)
#find_package(SDL3 CONFIG REQUIRED)
#find_package(SDL3_image CONFIG REQUIRED)
#find_package(glm CONFIG REQUIRED)
#find_package(assimp CONFIG REQUIRED)
#include_directories(${ASSIMP_INCLUDE_DIR})
#include_directories(${SDL3_INCLUDE_DIRS} ${SDL3_IMAGE_INCLUDE_DIR})
#include_directories(${CMAKE_BINARY_DIR})
#include_directories(${stb_SOURCE_DIR}/)
#cmake_minimum_required(VERSION 3.20)


project(SHSRenderer LANGUAGES CXX)

# Adapter selection (migration step 5 — selectable dependency ownership):
# SDL3/SDL3_image, SDL2/SDL2_image and Assimp are optional integration
# adapters. Defaults keep the historical behavior (all ON); consumer
# configurations may disable them to build the GPU-free, windowing-free core
# (the compiled TUs are anchors only, so nothing in the library itself needs
# these SDKs). Vulkan stays auto-discovered (QUIET); Jolt/xsimd stay
# auto-discovered target-conditional.
# Platform-agnostic seam ruling (2026-09-17): the SDL3-only stance is
# superseded — SDL2 and SDL3 are BOTH supported windowing backends behind
# IPlatformRuntime + window_backend.hpp (SFML/GLFW plug into the same seam
# later). SDL2/SDL3 headers must never share a TU: each backend owns its own
# anchor TU (src/platform_sdl3_anchor.cpp / src/platform_sdl2_anchor.cpp).
# SDL2 discovery is best-effort QUIET: missing SDL2 downgrades to an
# SDL3-only build (honest status message); SDL3 stays FATAL when requested.
option(SHS_RENDERER_WITH_SDL3 "Discover and link SDL3/SDL3_image (windowed runtime + texture adapters)" ON)
option(SHS_RENDERER_WITH_SDL2 "Discover and link SDL2/SDL2_image (legacy windowed runtime + texture adapters)" ON)
option(SHS_RENDERER_WITH_ASSIMP "Discover and link Assimp (mesh import adapters)" ON)

find_package(glm CONFIG REQUIRED)
if(SHS_RENDERER_WITH_SDL3)
    find_package(SDL3 CONFIG REQUIRED)
    find_package(SDL3_image CONFIG REQUIRED)
endif()
set(SHS_RENDERER_HAS_SDL2 OFF)
if(SHS_RENDERER_WITH_SDL2)
    # Dual-dispatch constraint (docs/outdated/sdl3_cutover_runbook.md (archived 2026-09-18) §7): a binary that
    # references BOTH backends' factories (seam consumers such as the
    # adventure demos' --window front-end) must link at least one SDK as a
    # shared library — static SDL2 + static SDL3 collide on the dynapi
    # symbols (SDL_Init, SDL_AddEventWatch, …). Probe the SYSTEM SHARED SDL2
    # dev install first (find_library prefers .so over .a); fall back to
    # pkg-config / CMake config, which under the vcpkg toolchain resolve to
    # the static archives (then dual-factory binaries are the documented
    # limitation, not a build error).
    find_library(SHS_SYSTEM_SDL2_LIB NAMES SDL2-2.0 SDL2
        HINTS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" /usr/lib /usr/local/lib /usr/lib64
        NO_DEFAULT_PATH)
    find_path(SHS_SYSTEM_SDL2_INC SDL2/SDL.h
        HINTS /usr/include /usr/local/include NO_DEFAULT_PATH)
    if(SHS_SYSTEM_SDL2_LIB AND SHS_SYSTEM_SDL2_INC)
        if(NOT TARGET SDL2::SDL2-system)
            add_library(SDL2::SDL2-system SHARED IMPORTED GLOBAL)
            set_target_properties(SDL2::SDL2-system PROPERTIES
                IMPORTED_LOCATION "${SHS_SYSTEM_SDL2_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${SHS_SYSTEM_SDL2_INC}")
        endif()
        message(STATUS "SHS: system shared SDL2 found (${SHS_SYSTEM_SDL2_LIB}) — dual-dispatch friendly")
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND AND NOT TARGET PkgConfig::SDL2)
            pkg_check_modules(SDL2 QUIET IMPORTED_TARGET sdl2)
        endif()
        if(NOT TARGET PkgConfig::SDL2)
            find_package(SDL2 CONFIG QUIET)
        endif()
    endif()
    if(TARGET SDL2::SDL2-system OR TARGET SDL2::SDL2 OR TARGET PkgConfig::SDL2)
        find_package(SDL2_image CONFIG QUIET)
        if(NOT TARGET SDL2_image::SDL2_image)
            message(STATUS "SHS: SDL2_image CMake config did not provide SDL2_image::SDL2_image — falling back to system probe/pkg-config")
            # System shared SDL2_image first (same dual-dispatch reasoning as
            # the base SDL2 probe: the vcpkg pkg-config data under the
            # toolchain resolves to the static debug archive, which drags
            # static SDL2d into dual-dispatch binaries).
            find_library(SHS_SYSTEM_SDL2_IMAGE_LIB NAMES SDL2_image-2.0 SDL2_image
                HINTS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" /usr/lib /usr/local/lib /usr/lib64
                NO_DEFAULT_PATH)
            find_path(SHS_SYSTEM_SDL2_IMAGE_INC SDL2/SDL_image.h
                HINTS /usr/include /usr/local/include NO_DEFAULT_PATH)
            if(SHS_SYSTEM_SDL2_IMAGE_LIB AND SHS_SYSTEM_SDL2_IMAGE_INC)
                if(NOT TARGET SDL2_image::SDL2_image)
                    add_library(SDL2_image::SDL2_image SHARED IMPORTED GLOBAL)
                    set_target_properties(SDL2_image::SDL2_image PROPERTIES
                        IMPORTED_LOCATION "${SHS_SYSTEM_SDL2_IMAGE_LIB}"
                        INTERFACE_INCLUDE_DIRECTORIES "${SHS_SYSTEM_SDL2_IMAGE_INC}")
                endif()
                message(STATUS "SHS: system shared SDL2_image found (${SHS_SYSTEM_SDL2_IMAGE_LIB})")
            else()
                find_package(PkgConfig QUIET)
                if(PkgConfig_FOUND AND NOT TARGET PkgConfig::SDL2_image)
                    pkg_check_modules(SDL2_image QUIET IMPORTED_TARGET SDL2_image)
                endif()
            endif()
        endif()
        set(SHS_RENDERER_HAS_SDL2 ON)
    else()
        message(STATUS "SHS: SDL2 requested but not found — SDL2 windowing backend unavailable (SDL3-only build)")
    endif()
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
    src/platform_sdl3_anchor.cpp
    src/platform_sdl2_anchor.cpp
)
add_library(shs::renderer ALIAS shs_renderer)
# Converged naming (2026-09): single library — primary target shs_renderer,
# alias shs::renderer. Software vs GPU is a driver-pod + capability-gate
# selection (SHS_HAS_VULKAN), not a library split.

set(SHS_SDL3_HAS_VULKAN OFF)
set(_shs_sdl3_target "")
if(TARGET SDL3::SDL3)
    set(_shs_sdl3_target SDL3::SDL3)
elseif(TARGET SDL3::SDL3-static)
    set(_shs_sdl3_target SDL3::SDL3-static)
endif()
set(_shs_sdl3_config "")
if(_shs_sdl3_target)
    get_target_property(_shs_sdl3_inc_dirs ${_shs_sdl3_target} INTERFACE_INCLUDE_DIRECTORIES)
    foreach(_inc IN LISTS _shs_sdl3_inc_dirs)
        if("${_inc}" MATCHES "^\\$<")
            continue()
        endif()
        # SDL3 ships no SDL_config.h/SDL_build_config.h in its public headers
        # (build config is private). SDL_vulkan.h, however, is installed
        # unconditionally — its presence is the capability probe.
        if(EXISTS "${_inc}/SDL3/SDL_vulkan.h")
            set(_shs_sdl3_config "${_inc}/SDL3/SDL_vulkan.h")
            break()
        endif()
        if(EXISTS "${_inc}/SDL_vulkan.h")
            set(_shs_sdl3_config "${_inc}/SDL_vulkan.h")
            break()
        endif()
    endforeach()
endif()
if(_shs_sdl3_config)
    set(SHS_SDL3_HAS_VULKAN ON)
endif()

# C++ standard for everyone who links this
target_compile_features(shs_renderer PUBLIC cxx_std_23)

# Packages (vcpkg config mode) — conditional per SHS_RENDERER_WITH_SDL3 /
# SHS_RENDERER_WITH_ASSIMP (see adapter-selection options at top).
if(SHS_RENDERER_WITH_SDL3)
    find_package(SDL3 CONFIG REQUIRED)
endif()
if(NOT TARGET SDL3_image::Main)
    # SDL3_image's config may expose SDL3_image::SDL3_image as an ALIAS of the
    # -static/-shared real target; CMake forbids alias-of-alias, so resolve to
    # the real target name first.
    set(_shs_sdl3_image_real "")
    foreach(_shs_cand SDL3_image::SDL3_image SDL3_image::SDL3_image-static SDL3_image-static)
        if(TARGET ${_shs_cand})
            get_target_property(_shs_aliased ${_shs_cand} ALIASED_TARGET)
            if(_shs_aliased)
                set(_shs_sdl3_image_real "${_shs_aliased}")
            else()
                set(_shs_sdl3_image_real "${_shs_cand}")
            endif()
            break()
        endif()
    endforeach()
    if(_shs_sdl3_image_real)
        add_library(SDL3_image::Main ALIAS ${_shs_sdl3_image_real})
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

# Link deps that demos will inherit automatically. SDL3/SDL2/Assimp are
# optional adapters (step 5): when disabled, the library builds and links with
# zero windowing/asset-import SDKs; SHS_HAS_SDL3 / SHS_HAS_SDL2 / SHS_HAS_ASSIMP
# let headers and anchor TUs feature-guard adapter entry points (same shape as
# SHS_HAS_JOLT).
target_link_libraries(shs_renderer PUBLIC
    glm::glm
)
# CMake >= 3.29 SDL version-conflict detection (INTERFACE_SDL_VERSION) rejects
# linking SDL2 and SDL3 into one target. Here the dual link is intentional
# (platform-agnostic seam, R4 ruling): both SDKs stay fully isolated per TU
# via the backend anchor TUs, so the guard is cleared for the SDL targets.
foreach(_shs_sdl_prop_tgt SDL3::SDL3 SDL3::SDL3-static SDL2::SDL2 SDL2::SDL2-static)
    if(TARGET ${_shs_sdl_prop_tgt})
        # vcpkg SDL2 config exposes SDL2::SDL2 as an ALIAS of the real target;
        # properties can only be set on the real one.
        get_target_property(_shs_sdl_real ${_shs_sdl_prop_tgt} ALIASED_TARGET)
        if(_shs_sdl_real)
            set_target_properties(${_shs_sdl_real} PROPERTIES INTERFACE_SDL_VERSION "")
        else()
            set_target_properties(${_shs_sdl_prop_tgt} PROPERTIES INTERFACE_SDL_VERSION "")
        endif()
        unset(_shs_sdl_real)
    endif()
endforeach()
if(SHS_RENDERER_WITH_SDL3)
    if(TARGET SDL3::SDL3)
        target_link_libraries(shs_renderer PUBLIC SDL3::SDL3)
    elseif(TARGET SDL3::SDL3-static)
        target_link_libraries(shs_renderer PUBLIC SDL3::SDL3-static)
    else()
        message(FATAL_ERROR "SHS_RENDERER_WITH_SDL3=ON but no SDL3::SDL3 target found")
    endif()
    target_link_libraries(shs_renderer PUBLIC
        SDL3_image::Main
    )
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_SDL3=1)
endif()
set(SHS_SDL2_HAS_VULKAN OFF)
if(SHS_RENDERER_HAS_SDL2)
    set(_shs_sdl2_link "")
    if(TARGET SDL2::SDL2-system)
        set(_shs_sdl2_link SDL2::SDL2-system) # system shared — dual-dispatch friendly
    elseif(TARGET SDL2::SDL2)
        set(_shs_sdl2_link SDL2::SDL2)
    elseif(TARGET PkgConfig::SDL2)
        set(_shs_sdl2_link PkgConfig::SDL2)
    endif()
    if(_shs_sdl2_link)
        # BUILD_INTERFACE only: the installed package keeps its minimal export
        # interface (SDL2 stays a consumer-side find_dependency, as documented —
        # pkg-config-only discovery was already a documented limitation).
        target_link_libraries(shs_renderer PUBLIC $<BUILD_INTERFACE:${_shs_sdl2_link}>)
    endif()
    if(TARGET SDL2_image::SDL2_image)
        # BUILD_INTERFACE only (same reason as the base SDL2 link above).
        target_link_libraries(shs_renderer PUBLIC $<BUILD_INTERFACE:SDL2_image::SDL2_image>)
    elseif(TARGET PkgConfig::SDL2_image)
        target_link_libraries(shs_renderer PUBLIC $<BUILD_INTERFACE:PkgConfig::SDL2_image>)
    endif()
    target_compile_definitions(shs_renderer PUBLIC SHS_HAS_SDL2=1)
    # SDL2 Vulkan capability probe (same shape as the SDL3 probe): SDL2 ships
    # SDL_vulkan.h unconditionally in its include dirs.
    get_target_property(_shs_sdl2_inc_dirs ${_shs_sdl2_link} INTERFACE_INCLUDE_DIRECTORIES)
    foreach(_inc IN LISTS _shs_sdl2_inc_dirs)
        if("${_inc}" MATCHES "^\\$<")
            continue()
        endif()
        if(EXISTS "${_inc}/SDL2/SDL_vulkan.h" OR EXISTS "${_inc}/SDL_vulkan.h")
            set(SHS_SDL2_HAS_VULKAN ON)
            break()
        endif()
    endforeach()
    unset(_shs_sdl2_link)
    unset(_shs_sdl2_inc_dirs)
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
set(SHS_SDL3_HAS_VULKAN "${SHS_SDL3_HAS_VULKAN}" PARENT_SCOPE)
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
# - the exported interface is minimal: `include/` + glm, plus SDL3/Assimp/
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
set(SHS_RENDERER_PKG_WITH_SDL3 "${SHS_RENDERER_WITH_SDL3}")
set(SHS_RENDERER_PKG_WITH_SDL2 "${SHS_RENDERER_HAS_SDL2}")
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
    # Forward extra dependency roots (e.g. a user-local vcpkg holding SDL3)
    # so the installed package's find_dependency(SDL3) can resolve.
    if(CMAKE_PREFIX_PATH)
        list(APPEND _shs_pkg_consumer_args "${CMAKE_PREFIX_PATH}")
    endif()
    add_test(NAME shs_renderer_package_consumer_test
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/package_consumer_test.sh"
            ${_shs_pkg_consumer_args})

    # Migration step 5 — self-contained public headers gate: every public
    # header compiles standalone with only the exported dependency include
    # dirs, no SHS_HAS_* defines and no build-tree/stb include dirs.
    set(_shs_selfcont_targets
        glm::glm SDL3::SDL3 SDL3_image::SDL3_image SDL2::SDL2 SDL2_image::SDL2_image
        SDL2::SDL2-system SDL2_image::SDL2_image-system
        PkgConfig::SDL2 PkgConfig::SDL2_image assimp::assimp
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
    # ROP-4.3 (owner ruling 2026-09-18): doc-path gate — every `docs/....md`
    # citation (in C++ and in live markdown) must exist, and relative markdown
    # links must resolve. Archives are exempt: T2 makes `docs/outdated/` and
    # `kdba_history/` read-only, so the gate must never be the reason an
    # archive edits. The negative fixture proves each rule FAILs and that the
    # documented exemptions (bare filenames, embedded paths, `file://` links)
    # stay legal.
    add_test(NAME shs_renderer_doc_paths_gate
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_doc_paths.py")
    add_test(NAME shs_renderer_doc_paths_negative_test
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_doc_paths_negative_test.sh")
    # ROP-2.4 (owner ruling 2026-09-18): pass-execution shape gate — the pass
    # rim's `(payload, bool valid)` shape (Constitution II §8) must not return:
    # no stored `bool executed` / `bool valid`, no reason-free `not_executed()`,
    # and `PassOutcome` stays a closed string-free `enum class`. The negative
    # fixture proves each rule FAILs and that the ruled shape PASSES.
    add_test(NAME shs_renderer_pass_execution_shape_gate
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_pass_execution_shape.sh")
    add_test(NAME shs_renderer_pass_execution_shape_negative_test
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_pass_execution_shape_negative_test.sh")
    # ROP-1.3 (owner ruling 2026-09-18): claimed-primitive gate — a core/
    # primitive must have a live consumer or a recorded disposition. This is
    # the guard whose absence let core/result.hpp rot unclaimed; the negative
    # fixture proves R1/R2/R3 FAIL and that a recorded entry still reports KNOWN.
    add_test(NAME shs_renderer_claimed_primitives_gate
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_claimed_primitives.py")
    add_test(NAME shs_renderer_claimed_primitives_negative_test
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_claimed_primitives_negative_test.sh")
    # P1 placement law (Rule 17 as amended): gate 8 negative fixture — a
    # contract macro in a math leaf / non-seam header FAILs, legal homes PASS.
    add_test(NAME shs_renderer_contract_placement_negative_test
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_contract_placement_negative_test.sh")
    # G2.1 positive gate. Found dormant on 2026-09-18 (ROP sweep): the
    # classification gate existed and passed on the real tree, but only its
    # negative fixture was registered, so the actual header tree was never
    # policed. Activated — a gate that never runs enforces nothing.
    add_test(NAME shs_renderer_pure_value_library_check
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_pure_value_libraries.sh"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/shs")
    # G2.1 (governance review 2026-09-18): Pure Domain Value Library
    # classification gate negative fixture — stateful machinery in a
    # stateless leaf fails; conforming leaf passes; empty/missing/vacuous
    # classifications fail loudly.
    add_test(NAME shs_renderer_pure_value_library_negative_test
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_pure_value_libraries_negative_test.sh")

    # G3.2 (governance review 2026-09-18): golden-value tests for the math
    # laws — the screen/canvas discrete row law (Constitution I §2:
    # row_canvas = (H-1) - row_screen, continuous form y_canvas = H - y_screen,
    # and the rasterizer NDC mapping pinned beside them) and the Jolt bridge
    # involution S·M·S with S = diag(1,1,-1,1) (Constitution I §3, roundtrips
    # + homomorphism property). Named gates per the G2.3 law-budget norm.
    if(TARGET Jolt)
        add_executable(shs_renderer_math_law_golden_tests
            tests/math_law_golden_tests.cpp
        )
        target_link_libraries(shs_renderer_math_law_golden_tests PRIVATE shs::renderer-values Jolt)
        target_compile_definitions(shs_renderer_math_law_golden_tests PRIVATE SHS_HAS_JOLT=1)
        target_compile_features(shs_renderer_math_law_golden_tests PRIVATE cxx_std_23)
        if(MSVC)
            target_compile_options(shs_renderer_math_law_golden_tests PRIVATE /W4)
        else()
            target_compile_options(shs_renderer_math_law_golden_tests PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        add_test(
            NAME shs_renderer_math_law_golden_tests
            COMMAND shs_renderer_math_law_golden_tests
        )
        set_tests_properties(shs_renderer_math_law_golden_tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
    endif()

    # R2 (renderer-lib review 2026-09-18): golden pixel-coverage gate — one
    # known screen triangle's exact covered pixels pinned against an
    # independent double-precision analytic oracle plus literal golden
    # values (count, positional checksum, per-row spans).
    add_executable(shs_renderer_rasterizer_coverage_golden_tests
        tests/rasterizer_coverage_golden_tests.cpp
    )
    target_link_libraries(shs_renderer_rasterizer_coverage_golden_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_rasterizer_coverage_golden_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_rasterizer_coverage_golden_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_rasterizer_coverage_golden_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(
        NAME shs_renderer_rasterizer_coverage_golden_tests
        COMMAND shs_renderer_rasterizer_coverage_golden_tests
    )
    set_tests_properties(shs_renderer_rasterizer_coverage_golden_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # R3 (renderer-lib review 2026-09-18): tile-path parity gate — the R3
    # tile rasterization path (one wait-free job per 32px-class tile) must be
    # BYTE-IDENTICAL to the streaming path across worker counts and tile
    # sizes, for color, depth, and motion buffers, with identical stats.
    add_executable(shs_renderer_rasterizer_tile_parity_tests
        tests/rasterizer_tile_parity_tests.cpp
    )
    target_link_libraries(shs_renderer_rasterizer_tile_parity_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_rasterizer_tile_parity_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_rasterizer_tile_parity_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_rasterizer_tile_parity_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(
        NAME shs_renderer_rasterizer_tile_parity_tests
        COMMAND shs_renderer_rasterizer_tile_parity_tests
    )
    set_tests_properties(shs_renderer_rasterizer_tile_parity_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # W-C gateway rails (P2 + P5 owner rulings): negative fixture — a throw /
    # bool-rim gateway FAILs, a dispatch without a static_assert tail FAILs,
    # a conformant gateway PASSES.
    add_test(NAME shs_renderer_gateway_rails_negative_test
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_gateway_rails_negative_test.sh")

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

    # Step-7 namespace cutover compatibility: old (shs::X) vs new
    # (shs::<owner>::X) spellings must denote the same entities; overload
    # lookup, ADL and serialization ids keep working. (The legacy
    # compatibility include-order tests were retired together with the
    # shs/domains/ + shs/execution/ forwarder trees — migration step-7
    # item 4.) Namespaces can break ABI: aliases do not guarantee
    # binary compatibility (see migration doc step 7).
    add_executable(shs_renderer_namespace_cutover_tests
        tests/namespace_cutover_tests.cpp
    )
    target_link_libraries(shs_renderer_namespace_cutover_tests PRIVATE shs::renderer-values)
    if(MSVC)
        target_compile_options(shs_renderer_namespace_cutover_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_namespace_cutover_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME shs_renderer_namespace_cutover_tests COMMAND shs_renderer_namespace_cutover_tests)

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

    # Constitution I §7 — No User Lock-In (arch/render_path_architecture.md §4
    # graduation req 1): open pass-id registry. Consumer/demo-owned passes get
    # typed ids from an open registered range with zero core edits, while every
    # builtin plan path must keep its exact previous behavior (parity guards
    # inside). Header-only: links shs::renderer-values only, GPU-free.
    add_executable(shs_renderer_pass_id_open_tests
        tests/pass_id_open_registry_tests.cpp
    )
    target_link_libraries(shs_renderer_pass_id_open_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_pass_id_open_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_pass_id_open_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_pass_id_open_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME shs_renderer_pass_id_open_tests COMMAND shs_renderer_pass_id_open_tests)

    # Shader identity gate (Slang plan P1.5): one ShaderId names one authored
    # shader, each backend resolves its own realization (software program vs
    # Vulkan module + entry points), and a backend with no realization refuses
    # loudly instead of silently approximating. GPU-free; the authored-source
    # check reads the .slang file that is actually compiled.
    add_executable(shs_renderer_shader_identity_tests
        tests/shader_identity_tests.cpp
    )
    target_link_libraries(shs_renderer_shader_identity_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_shader_identity_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_shader_identity_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_shader_identity_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    target_compile_definitions(shs_renderer_shader_identity_tests PRIVATE
        SHS_SHADER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders")
    add_test(NAME shs_renderer_shader_identity_tests COMMAND shs_renderer_shader_identity_tests)

    # Open shader-id registry gate (Constitution I §7 — No User Lock-In;
    # arch §4 graduation req 6): ShaderId gains a builtin + open registered
    # range, a consumer mints a shader identity from a name (content-addressed,
    # like PassIdRegistry), and the manifest resolves it through the same
    # backend-blind path a builtin uses. Also pins that the shader and pass
    # registries derive offsets from the SAME shared law. GPU-free, value-tier.
    add_executable(shs_renderer_shader_id_open_tests
        tests/shader_id_open_registry_tests.cpp
    )
    target_link_libraries(shs_renderer_shader_id_open_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_shader_id_open_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_shader_id_open_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_shader_id_open_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME shs_renderer_shader_id_open_tests COMMAND shs_renderer_shader_id_open_tests)
    # Open pass-semantic registry gate (Constitution I §7 — No User Lock-In;
    # arch §4 graduation req 7): PassSemantic gains a builtin + open registered
    # range so a technique names its own channel (custom G-buffer layout, tiled
    # or mobile-lighting payload) with zero core edits, over the SAME
    # content-addressed law as the pass-id and shader-id registries. Pins the
    # four switch surfaces (name, descriptor, resolution class, resource id),
    # that the registry-aware resource id stays DISTINGUISHING where the bare
    # spelling would alias, and that every builtin path — including Unknown,
    # which the open-range guard must not catch — is byte-for-byte unchanged.
    # GPU-free, value-tier.
    add_executable(shs_renderer_semantic_id_open_tests
        tests/semantic_id_open_registry_tests.cpp
    )
    target_link_libraries(shs_renderer_semantic_id_open_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_semantic_id_open_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_semantic_id_open_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_semantic_id_open_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME shs_renderer_semantic_id_open_tests COMMAND shs_renderer_semantic_id_open_tests)



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
    # Backend seam symbol-hygiene gate (docs/outdated/sdl3_cutover_runbook.md (archived 2026-09-18) §7.5; governance
    # todo G1.2): the SDL2 anchor object must carry zero undefined SDL_*/IMG_*
    # symbols — direct SDL2 calls in the dlopen-dispatch anchor TU are exactly
    # the flat-namespace hijack the dispatch exists to prevent. The SDL3
    # anchor is reported, not policed (normal SDL3 linkage is legal). Runs on
    # the built library's object files; assumes `ctest` executes after the
    # library target is built (house discipline: full build before CTest).
    set(_shs_seam_obj_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/shs_renderer.dir/src")
    add_test(NAME shs_renderer_backend_seam_symbols_check
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_backend_seam_symbols.sh"
            --scan-root "${_shs_seam_obj_dir}")
    set_tests_properties(shs_renderer_backend_seam_symbols_check PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    # Negative fixture: a TU referencing SDL_Init must trip the gate; missing
    # anchor object must fail honestly (not silently pass).
    add_test(NAME shs_renderer_backend_seam_symbols_negative_test
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_backend_seam_symbols_negative_test.sh"
            "${CMAKE_CXX_COMPILER}")



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

    # C2 pilot negative tests (renderpath pod, W-B): the enforcement twin
    # observes the handler on hand-broken plans / illegal transition-table
    # rows / aliased spans; the release twin compiles the assume path.
    add_executable(shs_renderer_contract_pilot_tests
        tests/contract_guardrails_pilot_tests.cpp
    )
    target_link_libraries(shs_renderer_contract_pilot_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_contract_pilot_tests PRIVATE cxx_std_23)
    target_compile_definitions(shs_renderer_contract_pilot_tests PRIVATE SHS_CONTRACTS_ENFORCED=1)
    if(MSVC)
        target_compile_options(shs_renderer_contract_pilot_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_contract_pilot_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_contract_pilot_release_tests
        tests/contract_guardrails_pilot_tests.cpp
    )
    target_link_libraries(shs_renderer_contract_pilot_release_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_contract_pilot_release_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_contract_pilot_release_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_contract_pilot_release_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_contract_pilot_tests
        COMMAND shs_renderer_contract_pilot_tests
    )
    set_tests_properties(shs_renderer_contract_pilot_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_contract_pilot_release_tests
        COMMAND shs_renderer_contract_pilot_release_tests
    )
    set_tests_properties(shs_renderer_contract_pilot_release_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # W-D logic-pod guardrails (slice 1): the enforcement twin observes the
    # handler on a hand-broken transition table (kind=pre) and a caller-owned
    # unregistered resting state (kind=post); the release twin compiles the
    # assume path.
    add_executable(shs_renderer_logic_guardrail_tests
        tests/logic_guardrail_tests.cpp
    )
    target_link_libraries(shs_renderer_logic_guardrail_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_logic_guardrail_tests PRIVATE cxx_std_23)
    target_compile_definitions(shs_renderer_logic_guardrail_tests PRIVATE SHS_CONTRACTS_ENFORCED=1)
    if(MSVC)
        target_compile_options(shs_renderer_logic_guardrail_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_logic_guardrail_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_logic_guardrail_release_tests
        tests/logic_guardrail_tests.cpp
    )
    target_link_libraries(shs_renderer_logic_guardrail_release_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_logic_guardrail_release_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_logic_guardrail_release_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_logic_guardrail_release_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_logic_guardrail_tests
        COMMAND shs_renderer_logic_guardrail_tests
    )
    set_tests_properties(shs_renderer_logic_guardrail_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_logic_guardrail_release_tests
        COMMAND shs_renderer_logic_guardrail_release_tests
    )
    set_tests_properties(shs_renderer_logic_guardrail_release_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # W-D frame-pod guardrails (slice 2): the pinned identity transition
    # (empty §6.1 vocabulary) has no negative leg reachable through the
    # public seam — the vocabulary pins are compile-time (the visit's
    # static_assert + the event-count pin), so the twins prove the silence
    # side: valid batches pass with the rim POST observed and the handler
    # untouched (enforced) / checks discarded (release twin).
    add_executable(shs_renderer_frame_guardrail_tests
        tests/frame_guardrail_tests.cpp
    )
    target_link_libraries(shs_renderer_frame_guardrail_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_frame_guardrail_tests PRIVATE cxx_std_23)
    target_compile_definitions(shs_renderer_frame_guardrail_tests PRIVATE SHS_CONTRACTS_ENFORCED=1)
    if(MSVC)
        target_compile_options(shs_renderer_frame_guardrail_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_frame_guardrail_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_frame_guardrail_release_tests
        tests/frame_guardrail_tests.cpp
    )
    target_link_libraries(shs_renderer_frame_guardrail_release_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_frame_guardrail_release_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_frame_guardrail_release_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_frame_guardrail_release_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_frame_guardrail_tests
        COMMAND shs_renderer_frame_guardrail_tests
    )
    set_tests_properties(shs_renderer_frame_guardrail_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_frame_guardrail_release_tests
        COMMAND shs_renderer_frame_guardrail_release_tests
    )
    set_tests_properties(shs_renderer_frame_guardrail_release_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # W-D input-pod guardrails (slice 9): the input pod owns translation
    # only; its intents are applied at the app orchestrator rim
    # (session_orchestrate). That rim is infallible by law (every intent in
    # the closed 5-alternative vocabulary is valid), so the twins prove the
    # silence side: a mixed valid batch applies every command with one fact
    # per command and the handler untouched (enforced) / checks discarded
    # (release twin). The P5 tail is compile-time.
    add_executable(shs_renderer_input_guardrail_tests
        tests/input_guardrail_tests.cpp
    )
    target_link_libraries(shs_renderer_input_guardrail_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_input_guardrail_tests PRIVATE cxx_std_23)
    target_compile_definitions(shs_renderer_input_guardrail_tests PRIVATE SHS_CONTRACTS_ENFORCED=1)
    if(MSVC)
        target_compile_options(shs_renderer_input_guardrail_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_input_guardrail_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(shs_renderer_input_guardrail_release_tests
        tests/input_guardrail_tests.cpp
    )
    target_link_libraries(shs_renderer_input_guardrail_release_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_input_guardrail_release_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_input_guardrail_release_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_input_guardrail_release_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_input_guardrail_tests
        COMMAND shs_renderer_input_guardrail_tests
    )
    set_tests_properties(shs_renderer_input_guardrail_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_test(
        NAME shs_renderer_input_guardrail_release_tests
        COMMAND shs_renderer_input_guardrail_release_tests
    )
    set_tests_properties(shs_renderer_input_guardrail_release_tests PROPERTIES
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

    # Step 4.1: explicit app orchestration regression suite (state ownership,
    # host-instance independence, recorded-input replay), value-tier only.
    add_executable(shs_renderer_session_orchestration_tests
        tests/session_orchestration_tests.cpp
    )
    target_link_libraries(shs_renderer_session_orchestration_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_session_orchestration_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_session_orchestration_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_session_orchestration_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_session_orchestration_tests
        COMMAND shs_renderer_session_orchestration_tests
    )
    set_tests_properties(shs_renderer_session_orchestration_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # Step 6: vertical-slice host suite (engine_domain_separation_migration.md)
    # — public-API-only engine integration: deterministic headless replay,
    # independent host instances, resize, backend-unavailable rejection,
    # asset deletion/recreation, shutdown, known-pixel software output, and
    # teardown with outstanding work. Links only shs::renderer-values + glm.
    add_executable(shs_renderer_vertical_slice_tests
        tests/vertical_slice_tests.cpp
    )
    target_link_libraries(shs_renderer_vertical_slice_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_vertical_slice_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_vertical_slice_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_vertical_slice_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_vertical_slice_tests
        COMMAND shs_renderer_vertical_slice_tests
    )
    set_tests_properties(shs_renderer_vertical_slice_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # Governance review 2026-09-18 todo G1.1 (Tension 2 / P2): allocator-
    # interception frame gate. Overrides global operator new/delete, arms
    # around VerticalSliceHost::run_frame and fails when any heap allocation
    # escapes the caller's designated frame arena; also self-checks that the
    # interception counts a deliberate allocation. Links shs::renderer-values.
    add_executable(shs_renderer_frame_allocator_interception_tests
        tests/frame_allocator_interception_tests.cpp
    )
    target_link_libraries(shs_renderer_frame_allocator_interception_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_frame_allocator_interception_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_frame_allocator_interception_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_frame_allocator_interception_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_frame_allocator_interception_tests
        COMMAND shs_renderer_frame_allocator_interception_tests
    )
    set_tests_properties(shs_renderer_frame_allocator_interception_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # Step 4.3: scene/resource identity policy regression suite (object_id
    # uniqueness/stability, name first-wins, handle null/append-only/
    # generation rules, renderer projection copy rules), value-tier only.
    add_executable(shs_renderer_scene_identity_tests
        tests/scene_identity_tests.cpp
    )
    target_link_libraries(shs_renderer_scene_identity_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_scene_identity_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shs_renderer_scene_identity_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_scene_identity_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_scene_identity_tests
        COMMAND shs_renderer_scene_identity_tests
    )
    set_tests_properties(shs_renderer_scene_identity_tests PROPERTIES
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

    # Step 4.4 lifecycle-semantics tests (engine_domain_separation_migration
    # .md): thread access, arena lifetimes, shutdown ordering, rejection
    # preservation and partial-batch/event-allocation failure semantics;
    # value-tier only (shs::renderer-values + std::thread).
    add_executable(shs_renderer_lifecycle_semantics_tests
        tests/lifecycle_semantics_tests.cpp
    )
    target_link_libraries(shs_renderer_lifecycle_semantics_tests PRIVATE shs::renderer-values)
    target_compile_features(shs_renderer_lifecycle_semantics_tests PRIVATE cxx_std_23)
    find_package(Threads REQUIRED)
    target_link_libraries(shs_renderer_lifecycle_semantics_tests PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(shs_renderer_lifecycle_semantics_tests PRIVATE /W4)
    else()
        target_compile_options(shs_renderer_lifecycle_semantics_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(
        NAME shs_renderer_lifecycle_semantics_tests
        COMMAND shs_renderer_lifecycle_semantics_tests
    )
    set_tests_properties(shs_renderer_lifecycle_semantics_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

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

        # Step 6: bounded software/Vulkan parity (engine_domain_separation_
        # migration.md). Same flat NDC triangle through the real GPU offscreen
        # path and the CPU rasterizer; per-channel tolerance 1/255 with a
        # small edge-pixel budget. Explicit SKIP + 77 when the device or the
        # offscreen SPIR-V is unavailable — never a silent pass.
        add_executable(shs_renderer_sw_vk_parity_tests tests/sw_vk_parity_tests.cpp)
        target_link_libraries(shs_renderer_sw_vk_parity_tests PRIVATE
            shs::renderer-values Vulkan::Vulkan)
        target_compile_features(shs_renderer_sw_vk_parity_tests PRIVATE cxx_std_23)
        if(MSVC)
            target_compile_options(shs_renderer_sw_vk_parity_tests PRIVATE /W4)
        else()
            target_compile_options(shs_renderer_sw_vk_parity_tests PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        if(SHS_OFFSCREEN_SLANGC)
            add_dependencies(shs_renderer_sw_vk_parity_tests shs_renderer_offscreen_shaders)
            target_compile_definitions(shs_renderer_sw_vk_parity_tests PRIVATE
                SHS_OFFSCREEN_SHADER_DIR="${_offscreen_spv}")
        endif()
        target_compile_definitions(shs_renderer_sw_vk_parity_tests PRIVATE SHS_HAS_VULKAN=1)
        add_test(NAME shs_renderer_sw_vk_parity_tests COMMAND shs_renderer_sw_vk_parity_tests)
        set_tests_properties(shs_renderer_sw_vk_parity_tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            SKIP_RETURN_CODE 77)

        # G3 factory-facing execution: the same minimal scene, but obtained from
        # create_render_backend() and driven through the generic
        # IRenderBackend/IOffscreenExecution surface — the test never names the
        # concrete backend. Known pixels are asserted independently of parity; a
        # missing device, pipeline or generic surface is a SKIP (77), never a pass.
        add_executable(shs_renderer_vk_factory_offscreen_tests tests/vk_factory_offscreen_tests.cpp)
        target_link_libraries(shs_renderer_vk_factory_offscreen_tests PRIVATE
            shs::renderer-values Vulkan::Vulkan)
        target_compile_features(shs_renderer_vk_factory_offscreen_tests PRIVATE cxx_std_23)
        if(MSVC)
            target_compile_options(shs_renderer_vk_factory_offscreen_tests PRIVATE /W4)
        else()
            target_compile_options(shs_renderer_vk_factory_offscreen_tests PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        if(SHS_OFFSCREEN_SLANGC)
            add_dependencies(shs_renderer_vk_factory_offscreen_tests shs_renderer_offscreen_shaders)
            target_compile_definitions(shs_renderer_vk_factory_offscreen_tests PRIVATE
                SHS_OFFSCREEN_SHADER_DIR="${_offscreen_spv}")
        endif()
        target_compile_definitions(shs_renderer_vk_factory_offscreen_tests PRIVATE SHS_HAS_VULKAN=1)
        add_test(NAME shs_renderer_vk_factory_offscreen_tests
            COMMAND shs_renderer_vk_factory_offscreen_tests)
        set_tests_properties(shs_renderer_vk_factory_offscreen_tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            SKIP_RETURN_CODE 77)

        # G4 library SW/Vulkan equivalence: the SAME minimal scene, the SAME
        # consumer code and the SAME command stream, driven through the generic
        # IRenderBackend/IOffscreenExecution surface for BOTH backends — each
        # obtained from create_render_backend(). Known-answer checks run
        # independently per backend against the authored scene; only then are the
        # two readbacks compared, under documented tolerances (1/255 per channel
        # where both cover, plus a boundary-coverage budget). The software half is
        # always asserted (a CPU rasterizer needs no device); only the Vulkan half
        # may report itself unavailable, and then the process exits 77 after the
        # software half has passed — never a silent pass.
        add_executable(shs_renderer_sw_vk_equivalence_tests tests/sw_vk_equivalence_tests.cpp)
        target_link_libraries(shs_renderer_sw_vk_equivalence_tests PRIVATE
            shs::renderer-values Vulkan::Vulkan)
        target_compile_features(shs_renderer_sw_vk_equivalence_tests PRIVATE cxx_std_23)
        if(MSVC)
            target_compile_options(shs_renderer_sw_vk_equivalence_tests PRIVATE /W4)
        else()
            target_compile_options(shs_renderer_sw_vk_equivalence_tests PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        if(SHS_OFFSCREEN_SLANGC)
            add_dependencies(shs_renderer_sw_vk_equivalence_tests shs_renderer_offscreen_shaders)
            target_compile_definitions(shs_renderer_sw_vk_equivalence_tests PRIVATE
                SHS_OFFSCREEN_SHADER_DIR="${_offscreen_spv}")
        endif()
        target_compile_definitions(shs_renderer_sw_vk_equivalence_tests PRIVATE SHS_HAS_VULKAN=1)
        add_test(NAME shs_renderer_sw_vk_equivalence_tests
            COMMAND shs_renderer_sw_vk_equivalence_tests)
        set_tests_properties(shs_renderer_sw_vk_equivalence_tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            SKIP_RETURN_CODE 77)

    else()
        message(STATUS "Vulkan headers not found — skipping shs_renderer_vk_driver_tests")
    endif()
endif()


# If you need ASSIMP_INCLUDE_DIR for some reason, prefer target includes:
# target_include_directories(shs_renderer INTERFACE ${ASSIMP_INCLUDE_DIR})
