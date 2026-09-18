# tier0-rasterization-foundations — 5 curriculum demos, each *_sw + *_vk.
# Location: cpp-folders/src/exps-rendering-adventures/tier0-rasterization-foundations/CMakeLists.txt
#
# Demo contract (same for both realizations):
#   t0_<name>_sw  — GLM C++ software rasterizer, headless PNG out.
#   t0_<name>_vk  — Vulkan offscreen render, Slang-compiled SPIR-V, PNG out.
#
# Slang flags are the load-bearing pins from docs/roadmap/slang_utilization_plan.md:
#   -matrix-layout-column-major   GLM storage compat (row-major default corrupts GLM math)
#   -profile spirv_1_5            baseline SPIR-V profile

find_package(glm CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Shared helper: framebuffer + PNG writer (one stb impl TU). The offscreen
# Vulkan harness joins in only when Vulkan is available.
# ---------------------------------------------------------------------------
add_library(t0_adventures_common STATIC common/adventures_stb.cpp)
target_include_directories(t0_adventures_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${stb_SOURCE_DIR})
target_link_libraries(t0_adventures_common PUBLIC glm::glm)
target_compile_definitions(t0_adventures_common PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE=1)
target_compile_features(t0_adventures_common PUBLIC cxx_std_23)
# Platform seam (window_backend.hpp) for the optional --window front-end of
# every demo: header-only when the demo stays headless; the factory dispatch
# pulls the compiled anchors at link time. Anchors are honest nullptr stubs
# when a windowing SDK was not found, so GPU-free/SDL-free CI stays green.
target_link_libraries(t0_adventures_common PUBLIC shs::renderer)

if(Vulkan_FOUND)
    target_sources(t0_adventures_common PRIVATE common/adventures_vk.cpp)
endif()

# ---------------------------------------------------------------------------
# Demos: dir:name pairs
# ---------------------------------------------------------------------------
set(T0_DEMOS
    "01_barycentric_interpolation:tri_barycentric:01_barycentric"
    "02_perspective_ortho_projection:projection:02_projection"
    "03_depth_test_alpha_blend:depth_blend:03_depth_blend"
    "04_texture_sampling_scissor:texture_sampling:04_texture_sampling"
    "05_stencil_operations:stencil:05_stencil"
)

foreach(_entry IN LISTS T0_DEMOS)
    string(REPLACE ":" ";" _pair "${_entry}")
    list(GET _pair 0 _dir)
    list(GET _pair 1 _name)
    list(GET _pair 2 _short)

    # ---- software variant (always built) ----
    add_executable(t0_${_name}_sw ${_dir}/${_name}_sw.cpp)
    target_link_libraries(t0_${_name}_sw PRIVATE t0_adventures_common)

    # ---- Slang -> SPIR-V compile for this demo's shaders ----
    set(_slang_src ${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/${_name}.slang)
    set(_spv_dir ${CMAKE_CURRENT_BINARY_DIR}/spv/${_dir})
    set(_spv_vs ${_spv_dir}/${_name}_vs.spv)
    set(_spv_fs ${_spv_dir}/${_name}_fs.spv)

    if(SHS_ADVENTURES_HAS_SLANG)
        add_custom_command(
            OUTPUT ${_spv_vs} ${_spv_fs}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${_spv_dir}
            COMMAND ${SLANGC_EXECUTABLE} ${_slang_src}
                    -target spirv -profile spirv_1_5 -matrix-layout-column-major -fvk-use-entrypoint-name
                    -entry vs_main -stage vertex -o ${_spv_vs}
            COMMAND ${SLANGC_EXECUTABLE} ${_slang_src}
                    -target spirv -profile spirv_1_5 -matrix-layout-column-major -fvk-use-entrypoint-name
                    -entry fs_main -stage fragment -o ${_spv_fs}
            DEPENDS ${_slang_src}
            COMMENT "slangc: ${_dir}/${_name}.slang -> SPIR-V"
            VERBATIM)
        add_custom_target(t0_${_name}_slang DEPENDS ${_spv_vs} ${_spv_fs})
    endif()

    # ---- Vulkan variant (needs Vulkan + slangc) ----
    if(SHS_ADVENTURES_HAS_SLANG AND Vulkan_FOUND)
        add_executable(t0_${_name}_vk ${_dir}/${_name}_vk.cpp)
        target_link_libraries(t0_${_name}_vk PRIVATE t0_adventures_common)
        add_dependencies(t0_${_name}_vk t0_${_name}_slang)
        # shader dir consumed at runtime via SHS_ADVENTURES_SHADER_DIR
        target_compile_definitions(t0_${_name}_vk PRIVATE
            SHS_ADVENTURES_SHADER_DIR="${_spv_dir}")
    endif()

    # ---- AD1: portable automated gates (CTest registration) ----
    #   * SW smoke: headless render into an isolated output dir, exit 0.
    #     Registered independently of Vulkan/slang so GPU-free CI still
    #     exercises demo coverage.
    #   * Parity twin: one t0_parity_suite run per pair (build dirs supplied
    #     by CMake — no checkout-specific paths), registered only when the
    #     Vulkan twin exists. A missing twin is a documented capability skip
    #     inside the suite; render/upload/PNG/compare failures remain
    #     failures.
    set(SHS_ADVENTURES_OUT_DIR ${CMAKE_BINARY_DIR}/adventure_out)
    file(MAKE_DIRECTORY ${SHS_ADVENTURES_OUT_DIR})

    add_test(NAME t0_${_name}_sw_smoke
        COMMAND t0_${_name}_sw ${SHS_ADVENTURES_OUT_DIR}/t0_${_short}_sw.png)

    if(SHS_ADVENTURES_HAS_SLANG AND Vulkan_FOUND AND Python3_Interpreter_FOUND)
        add_test(NAME t0_${_short}_parity
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/t0_parity_suite.py
                    --build-dir ${CMAKE_CURRENT_BINARY_DIR}
                    --tier1-build-dir ${CMAKE_BINARY_DIR}/src/exps-rendering-adventures/tier1-classic-shading
                    --only ${_short}
                    --scratch-dir ${SHS_ADVENTURES_OUT_DIR}/parity/t0_${_short})
    endif()
endforeach()

# ---- AD4: independent known-answer checks (GPU-free, always active) ----
#   Demos 02/03/05 run in-process through the shared SwRaster kernels so the
#   depth/stencil STORAGE can be inspected directly; demos 01/04 run the real
#   demo binaries and are checked against analytic oracles derived here
#   (independent barycentric solve / checkerboard + sampler model). No parity
#   agreement is accepted as evidence. The prove-fail entry corrupts one
#   expectation and requires the comparator to catch it, so a correlated
#   SW/VK mistake cannot pass through this tool silently.
add_executable(t0_known_answer_checks
    tools/t0_known_answer_checks.cpp
    common/adventures_stb_load.cpp)
target_link_libraries(t0_known_answer_checks PRIVATE t0_adventures_common)
target_compile_definitions(t0_known_answer_checks PRIVATE
    SHS_KA_BIN_DIR="${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(t0_known_answer_checks t0_tri_barycentric_sw t0_texture_sampling_sw)
add_test(NAME t0_known_answer_checks COMMAND t0_known_answer_checks)
add_test(NAME t0_known_answer_prove_fail
    COMMAND t0_known_answer_checks --prove-wrong-expected)

# ---- AD1: negative probes — the gate tooling must fail reliably (GPU-free) ----
if(Python3_Interpreter_FOUND)
    add_test(NAME t0_gate_negative_probes
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/t0_gate_negative_probes.py)
endif()

