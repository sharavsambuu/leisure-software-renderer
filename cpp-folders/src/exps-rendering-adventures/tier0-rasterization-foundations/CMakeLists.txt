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

if(Vulkan_FOUND)
    target_sources(t0_adventures_common PRIVATE common/adventures_vk.cpp)
    # vk_shader_utils.hpp + SHS_HAS_VULKAN + Vulkan::Vulkan come via the lib
    target_link_libraries(t0_adventures_common PUBLIC shs::renderer)
endif()

# ---------------------------------------------------------------------------
# Demos: dir:name pairs
# ---------------------------------------------------------------------------
set(T0_DEMOS
    "01_barycentric_interpolation:tri_barycentric"
    "02_perspective_ortho_projection:projection"
    "03_depth_test_alpha_blend:depth_blend"
    "04_texture_sampling_scissor:texture_sampling"
    "05_stencil_operations:stencil"
)

foreach(_entry IN LISTS T0_DEMOS)
    string(REPLACE ":" ";" _pair "${_entry}")
    list(GET _pair 0 _dir)
    list(GET _pair 1 _name)

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
endforeach()
