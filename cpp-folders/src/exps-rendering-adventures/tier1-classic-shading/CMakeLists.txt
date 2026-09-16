# tier1-classic-shading — Tier 1 curriculum demos (rung 08 first).
# Location: cpp-folders/src/exps-rendering-adventures/tier1-classic-shading/CMakeLists.txt
#
# Same demo contract as tier0: t1_<name>_sw (GLM C++ headless PNG) +
# t1_<name>_vk (Vulkan offscreen, Slang SPIR-V, PNG). Slang flags pinned
# per docs/roadmap/slang_utilization_plan.md. GPU-free stays green (_sw only).

find_package(glm CONFIG REQUIRED)

add_library(t1_adventures_common STATIC ../tier0-rasterization-foundations/common/adventures_stb.cpp)
target_include_directories(t1_adventures_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../tier0-rasterization-foundations/common
    ${stb_SOURCE_DIR})
target_link_libraries(t1_adventures_common PUBLIC glm::glm)
target_compile_definitions(t1_adventures_common PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE=1)
target_compile_features(t1_adventures_common PUBLIC cxx_std_23)

if(Vulkan_FOUND)
    target_sources(t1_adventures_common PRIVATE ../tier0-rasterization-foundations/common/adventures_vk.cpp)
    target_link_libraries(t1_adventures_common PUBLIC shs::renderer)
endif()

set(T1_DEMOS
    "08_normal_mapping:normal_mapping"
)

foreach(_entry IN LISTS T1_DEMOS)
    string(REPLACE ":" ";" _pair "${_entry}")
    list(GET _pair 0 _dir)
    list(GET _pair 1 _name)

    add_executable(t1_${_name}_sw ${_dir}/${_name}_sw.cpp)
    target_link_libraries(t1_${_name}_sw PRIVATE t1_adventures_common)

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
        add_custom_target(t1_${_name}_slang DEPENDS ${_spv_vs} ${_spv_fs})
    endif()

    if(SHS_ADVENTURES_HAS_SLANG AND Vulkan_FOUND)
        add_executable(t1_${_name}_vk ${_dir}/${_name}_vk.cpp)
        target_link_libraries(t1_${_name}_vk PRIVATE t1_adventures_common)
        add_dependencies(t1_${_name}_vk t1_${_name}_slang)
        target_compile_definitions(t1_${_name}_vk PRIVATE
            SHS_ADVENTURES_SHADER_DIR="${_spv_dir}")
    endif()
endforeach()
