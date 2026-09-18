# exps-rendering-adventures — tier-based rendering-techniques curriculum demos.
# Location: cpp-folders/src/exps-rendering-adventures/CMakeLists.txt
#
# Unlike the parked exps trees (commented out at the root), this tree is
# active in the default build. Each tier folder is self-contained; every demo
# ships as a *_sw + *_vk pair (GLM C++ software rasterizer vs Vulkan + Slang),
# sharing vertex data from the tier's common scenes header for parity
# comparison. Both variants default to headless (render offscreen, write a
# PNG) so they validate in GPU-free CI; with --window they present the
# rendered frame through the platform seam (SDL3-preferred auto dispatch) and
# export PNGs from the window loop via F12 (common/adventures_window.hpp).
#
# Slang modules (.slang) are compiled to SPIR-V by slangc when found
# (bootstraps P0 of docs/roadmap/slang_utilization_plan.md); *_vk targets are
# additionally gated on Vulkan availability, symmetric to SHS_HAS_VULKAN.
# GPU-free configure (-DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE) stays green:
# the _sw half always builds.

project(ExpsRenderingAdventures)

find_package(glm CONFIG REQUIRED)
find_package(Vulkan QUIET)

# Slang P0: locate slangc (SDK ships it under $VULKAN_SDK/bin; the pinned
# standalone install lives in ~/slang per docs/dev/build_and_setup.md).
find_program(SLANGC_EXECUTABLE slangc
    HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin" "$ENV{HOME}/slang/slang/bin")
if(SLANGC_EXECUTABLE)
    message(STATUS "Adventures: slangc found — ${SLANGC_EXECUTABLE}")
    set(SHS_ADVENTURES_HAS_SLANG TRUE)
else()
    message(STATUS "Adventures: slangc NOT found — *_vk demo targets are skipped (shader logic still available as .slang sources)")
    set(SHS_ADVENTURES_HAS_SLANG FALSE)
endif()

add_subdirectory(tier0-rasterization-foundations)
add_subdirectory(tier1-classic-shading)
