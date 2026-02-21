project(ExpPlumbing LANGUAGES CXX)


file(COPY ../assets DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")


add_executable(HelloPassBasics hello_pass_basics.cpp)
target_link_libraries(HelloPassBasics PRIVATE shs::renderer)
if(MSVC)
    target_compile_options(HelloPassBasics PRIVATE /W4)
else()
    target_compile_options(HelloPassBasics PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(HelloPassBasics PRIVATE $<$<CONFIG:Release>:-O3>)
    target_compile_options(HelloPassBasics PRIVATE $<$<CONFIG:Debug>:-g>)
endif()
target_compile_features(HelloPassBasics PRIVATE cxx_std_20)

add_executable(HelloPassPlumbing hello_pass_plumbing.cpp)
target_link_libraries(HelloPassPlumbing PRIVATE shs::renderer)
if(MSVC)
    target_compile_options(HelloPassPlumbing PRIVATE /W4)
else()
    target_compile_options(HelloPassPlumbing PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(HelloPassPlumbing PRIVATE $<$<CONFIG:Release>:-O3>)
    target_compile_options(HelloPassPlumbing PRIVATE $<$<CONFIG:Debug>:-g>)
endif()
target_compile_features(HelloPassPlumbing PRIVATE cxx_std_20)

add_executable(HelloSoftwareTriangle hello_software_triangle.cpp)
target_link_libraries(HelloSoftwareTriangle PRIVATE shs::renderer)
if(MSVC)
    target_compile_options(HelloSoftwareTriangle PRIVATE /W4)
else()
    target_compile_options(HelloSoftwareTriangle PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(HelloSoftwareTriangle PRIVATE $<$<CONFIG:Release>:-O3>)
    target_compile_options(HelloSoftwareTriangle PRIVATE $<$<CONFIG:Debug>:-g>)
endif()
target_compile_features(HelloSoftwareTriangle PRIVATE cxx_std_20)

if(APPLE AND DEFINED ENV{VULKAN_SDK})
    find_program(GLSLANG_VALIDATOR
        NAMES glslangValidator glslang glslangValidator.exe
        HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin"
    )
else()
    find_program(GLSLANG_VALIDATOR
        NAMES glslangValidator glslang glslangValidator.exe
    )
endif()
if(GLSLANG_VALIDATOR)
    message(STATUS "Using glslang compiler: ${GLSLANG_VALIDATOR}")
endif()

if(SHS_SDL2_HAS_VULKAN)
    message(STATUS "SDL2 Vulkan support: ON")
else()
    message(STATUS "SDL2 Vulkan support: OFF (reinstall SDL2 with Vulkan feature)")
endif()

if(SHS_RENDERER_HAS_VULKAN AND GLSLANG_VALIDATOR AND SHS_SDL2_HAS_VULKAN)
    set(VK_TRI_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(VK_DEFAULT_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shs-renderer-lib/shaders/vulkan")
    set(VK_SHADER_COMMON_DIR "${VK_DEFAULT_SHADER_DIR}/common")
    set(VK_TRI_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${VK_TRI_OUT_DIR}")

    set(VK_COMMON_MATH "${VK_SHADER_COMMON_DIR}/math.glsl")
    set(VK_COMMON_LIGHT_CONSTANTS "${VK_SHADER_COMMON_DIR}/light_constants.glsl")
    set(VK_COMMON_LIGHT_MATH "${VK_SHADER_COMMON_DIR}/light_math.glsl")
    set(VK_COMMON_CULLING_LIGHT_STRUCT "${VK_SHADER_COMMON_DIR}/culling_light_struct.glsl")

    set(VK_TRI_VERT "${VK_TRI_SHADER_DIR}/vulkan_triangle.vert")
    set(VK_TRI_FRAG "${VK_TRI_SHADER_DIR}/vulkan_triangle.frag")
    set(VK_MODERN_VERT "${VK_TRI_SHADER_DIR}/modern_instanced.vert")
    set(VK_MODERN_FRAG "${VK_TRI_SHADER_DIR}/modern_instanced.frag")
    set(VK_CULLING_VERT "${VK_TRI_SHADER_DIR}/culling_vk.vert")
    set(VK_CULLING_FRAG "${VK_TRI_SHADER_DIR}/culling_vk.frag")
    set(VK_LIGHT_TYPES_CULLING_VERT "${VK_TRI_SHADER_DIR}/light_types_culling_vk.vert")
    set(VK_LIGHT_TYPES_CULLING_FRAG "${VK_TRI_SHADER_DIR}/light_types_culling_vk.frag")
    set(VK_SOFT_SHADOW_CULLING_VERT "${VK_TRI_SHADER_DIR}/soft_shadow_culling_vk.vert")
    set(VK_SOFT_SHADOW_CULLING_FRAG "${VK_TRI_SHADER_DIR}/soft_shadow_culling_vk.frag")
    set(VK_TRI_VERT_SPV "${VK_TRI_OUT_DIR}/vulkan_triangle.vert.spv")
    set(VK_TRI_FRAG_SPV "${VK_TRI_OUT_DIR}/vulkan_triangle.frag.spv")
    set(VK_MODERN_VERT_SPV "${VK_TRI_OUT_DIR}/modern_instanced.vert.spv")
    set(VK_MODERN_FRAG_SPV "${VK_TRI_OUT_DIR}/modern_instanced.frag.spv")
    set(VK_CULLING_VERT_SPV "${VK_TRI_OUT_DIR}/culling_vk.vert.spv")
    set(VK_CULLING_FRAG_SPV "${VK_TRI_OUT_DIR}/culling_vk.frag.spv")
    set(VK_LIGHT_TYPES_CULLING_VERT_SPV "${VK_TRI_OUT_DIR}/light_types_culling_vk.vert.spv")
    set(VK_LIGHT_TYPES_CULLING_FRAG_SPV "${VK_TRI_OUT_DIR}/light_types_culling_vk.frag.spv")
    set(VK_SOFT_SHADOW_CULLING_VERT_SPV "${VK_TRI_OUT_DIR}/soft_shadow_culling_vk.vert.spv")
    set(VK_SOFT_SHADOW_CULLING_FRAG_SPV "${VK_TRI_OUT_DIR}/soft_shadow_culling_vk.frag.spv")

    set(VK_PB_SHADOW_VERT "${VK_DEFAULT_SHADER_DIR}/pb_shadow.vert")
    set(VK_PB_SCENE_VERT "${VK_DEFAULT_SHADER_DIR}/pb_scene.vert")
    set(VK_PB_SCENE_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_scene.frag")
    set(VK_PB_POST_VERT "${VK_DEFAULT_SHADER_DIR}/pb_post.vert")
    set(VK_PB_SKY_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_sky.frag")
    set(VK_PB_BRIGHT_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_bright.frag")
    set(VK_PB_SHAFTS_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_shafts.frag")
    set(VK_PB_FLARE_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_flare.frag")
    set(VK_PB_COMPOSITE_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_composite.frag")
    set(VK_PB_FXAA_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_fxaa.frag")
    set(VK_FP_SCENE_VERT "${VK_DEFAULT_SHADER_DIR}/fp_stress_scene.vert")
    set(VK_FP_SCENE_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_scene.frag")
    set(VK_FP_GBUFFER_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_gbuffer.frag")
    set(VK_FP_DEFERRED_VERT "${VK_DEFAULT_SHADER_DIR}/fp_stress_deferred.vert")
    set(VK_FP_SSAO_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_ssao.frag")
    set(VK_FP_DEFERRED_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_deferred.frag")
    set(VK_FP_MOTION_BLUR_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_motion_blur.frag")
    set(VK_FP_DOF_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_dof.frag")
    set(VK_FP_SHADOW_VERT "${VK_DEFAULT_SHADER_DIR}/fp_stress_shadow.vert")
    set(VK_FP_LIGHT_CULL_COMP "${VK_DEFAULT_SHADER_DIR}/fp_stress_light_cull.comp")
    set(VK_FP_DEPTH_REDUCE_COMP "${VK_DEFAULT_SHADER_DIR}/fp_stress_depth_reduce.comp")

    set(VK_PB_SHADOW_VERT_SPV "${VK_TRI_OUT_DIR}/pb_shadow.vert.spv")
    set(VK_PB_SCENE_VERT_SPV "${VK_TRI_OUT_DIR}/pb_scene.vert.spv")
    set(VK_PB_SCENE_FRAG_SPV "${VK_TRI_OUT_DIR}/pb_scene.frag.spv")
    set(VK_PB_POST_VERT_SPV "${VK_TRI_OUT_DIR}/pb_post.vert.spv")
    set(VK_PB_SKY_FRAG_SPV "${VK_TRI_OUT_DIR}/pb_sky.frag.spv")
    set(VK_PB_BRIGHT_FRAG_SPV "${VK_TRI_OUT_DIR}/pb_bright.frag.spv")
    set(VK_PB_SHAFTS_FRAG_SPV "${VK_TRI_OUT_DIR}/pb_shafts.frag.spv")
    set(VK_PB_FLARE_FRAG_SPV "${VK_TRI_OUT_DIR}/pb_flare.frag.spv")
    set(VK_PB_COMPOSITE_FRAG_SPV "${VK_TRI_OUT_DIR}/pb_composite.frag.spv")
    set(VK_PB_FXAA_FRAG_SPV "${VK_TRI_OUT_DIR}/pb_fxaa.frag.spv")
    set(VK_FP_SCENE_VERT_SPV "${VK_TRI_OUT_DIR}/fp_stress_scene.vert.spv")
    set(VK_FP_SCENE_FRAG_SPV "${VK_TRI_OUT_DIR}/fp_stress_scene.frag.spv")
    set(VK_FP_GBUFFER_FRAG_SPV "${VK_TRI_OUT_DIR}/fp_stress_gbuffer.frag.spv")
    set(VK_FP_DEFERRED_VERT_SPV "${VK_TRI_OUT_DIR}/fp_stress_deferred.vert.spv")
    set(VK_FP_SSAO_FRAG_SPV "${VK_TRI_OUT_DIR}/fp_stress_ssao.frag.spv")
    set(VK_FP_DEFERRED_FRAG_SPV "${VK_TRI_OUT_DIR}/fp_stress_deferred.frag.spv")
    set(VK_FP_MOTION_BLUR_FRAG_SPV "${VK_TRI_OUT_DIR}/fp_stress_motion_blur.frag.spv")
    set(VK_FP_DOF_FRAG_SPV "${VK_TRI_OUT_DIR}/fp_stress_dof.frag.spv")
    set(VK_FP_SHADOW_VERT_SPV "${VK_TRI_OUT_DIR}/fp_stress_shadow.vert.spv")
    set(VK_FP_LIGHT_CULL_COMP_SPV "${VK_TRI_OUT_DIR}/fp_stress_light_cull.comp.spv")
    set(VK_FP_DEPTH_REDUCE_COMP_SPV "${VK_TRI_OUT_DIR}/fp_stress_depth_reduce.comp.spv")

    add_custom_command(
        OUTPUT "${VK_TRI_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_TRI_VERT}" -o "${VK_TRI_VERT_SPV}"
        DEPENDS "${VK_TRI_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_TRI_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_TRI_FRAG}" -o "${VK_TRI_FRAG_SPV}"
        DEPENDS "${VK_TRI_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_MODERN_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_MODERN_VERT}" -o "${VK_MODERN_VERT_SPV}"
        DEPENDS "${VK_MODERN_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_MODERN_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_MODERN_FRAG}" -o "${VK_MODERN_FRAG_SPV}"
        DEPENDS "${VK_MODERN_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_CULLING_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_CULLING_VERT}" -o "${VK_CULLING_VERT_SPV}"
        DEPENDS "${VK_CULLING_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_CULLING_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_CULLING_FRAG}" -o "${VK_CULLING_FRAG_SPV}"
        DEPENDS "${VK_CULLING_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_LIGHT_TYPES_CULLING_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_LIGHT_TYPES_CULLING_VERT}" -o "${VK_LIGHT_TYPES_CULLING_VERT_SPV}"
        DEPENDS "${VK_LIGHT_TYPES_CULLING_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_LIGHT_TYPES_CULLING_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_LIGHT_TYPES_CULLING_FRAG}" -o "${VK_LIGHT_TYPES_CULLING_FRAG_SPV}"
        DEPENDS
            "${VK_LIGHT_TYPES_CULLING_FRAG}"
            "${VK_COMMON_LIGHT_CONSTANTS}"
            "${VK_COMMON_MATH}"
            "${VK_COMMON_LIGHT_MATH}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_SOFT_SHADOW_CULLING_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_SOFT_SHADOW_CULLING_VERT}" -o "${VK_SOFT_SHADOW_CULLING_VERT_SPV}"
        DEPENDS "${VK_SOFT_SHADOW_CULLING_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_SOFT_SHADOW_CULLING_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_SOFT_SHADOW_CULLING_FRAG}" -o "${VK_SOFT_SHADOW_CULLING_FRAG_SPV}"
        DEPENDS "${VK_SOFT_SHADOW_CULLING_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_PB_SHADOW_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_PB_SHADOW_VERT}" -o "${VK_PB_SHADOW_VERT_SPV}"
        DEPENDS "${VK_PB_SHADOW_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_PB_SCENE_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_PB_SCENE_VERT}" -o "${VK_PB_SCENE_VERT_SPV}"
        DEPENDS "${VK_PB_SCENE_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_PB_SCENE_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_PB_SCENE_FRAG}" -o "${VK_PB_SCENE_FRAG_SPV}"
        DEPENDS "${VK_PB_SCENE_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_PB_POST_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_PB_POST_VERT}" -o "${VK_PB_POST_VERT_SPV}"
        DEPENDS "${VK_PB_POST_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_PB_SKY_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_PB_SKY_FRAG}" -o "${VK_PB_SKY_FRAG_SPV}"
        DEPENDS "${VK_PB_SKY_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_PB_BRIGHT_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_PB_BRIGHT_FRAG}" -o "${VK_PB_BRIGHT_FRAG_SPV}"
        DEPENDS "${VK_PB_BRIGHT_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_PB_SHAFTS_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_PB_SHAFTS_FRAG}" -o "${VK_PB_SHAFTS_FRAG_SPV}"
        DEPENDS "${VK_PB_SHAFTS_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_PB_FLARE_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_PB_FLARE_FRAG}" -o "${VK_PB_FLARE_FRAG_SPV}"
        DEPENDS "${VK_PB_FLARE_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_PB_COMPOSITE_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_PB_COMPOSITE_FRAG}" -o "${VK_PB_COMPOSITE_FRAG_SPV}"
        DEPENDS "${VK_PB_COMPOSITE_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_PB_FXAA_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_PB_FXAA_FRAG}" -o "${VK_PB_FXAA_FRAG_SPV}"
        DEPENDS "${VK_PB_FXAA_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_SCENE_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_SCENE_VERT}" -o "${VK_FP_SCENE_VERT_SPV}"
        DEPENDS "${VK_FP_SCENE_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_SCENE_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_SCENE_FRAG}" -o "${VK_FP_SCENE_FRAG_SPV}"
        DEPENDS
            "${VK_FP_SCENE_FRAG}"
            "${VK_COMMON_LIGHT_CONSTANTS}"
            "${VK_COMMON_LIGHT_MATH}"
            "${VK_COMMON_CULLING_LIGHT_STRUCT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_GBUFFER_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_GBUFFER_FRAG}" -o "${VK_FP_GBUFFER_FRAG_SPV}"
        DEPENDS "${VK_FP_GBUFFER_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_DEFERRED_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_DEFERRED_VERT}" -o "${VK_FP_DEFERRED_VERT_SPV}"
        DEPENDS "${VK_FP_DEFERRED_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_SSAO_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_SSAO_FRAG}" -o "${VK_FP_SSAO_FRAG_SPV}"
        DEPENDS "${VK_FP_SSAO_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_DEFERRED_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_DEFERRED_FRAG}" -o "${VK_FP_DEFERRED_FRAG_SPV}"
        DEPENDS
            "${VK_FP_DEFERRED_FRAG}"
            "${VK_COMMON_LIGHT_CONSTANTS}"
            "${VK_COMMON_LIGHT_MATH}"
            "${VK_COMMON_CULLING_LIGHT_STRUCT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_MOTION_BLUR_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_MOTION_BLUR_FRAG}" -o "${VK_FP_MOTION_BLUR_FRAG_SPV}"
        DEPENDS "${VK_FP_MOTION_BLUR_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_DOF_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_DOF_FRAG}" -o "${VK_FP_DOF_FRAG_SPV}"
        DEPENDS "${VK_FP_DOF_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_SHADOW_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_SHADOW_VERT}" -o "${VK_FP_SHADOW_VERT_SPV}"
        DEPENDS "${VK_FP_SHADOW_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_LIGHT_CULL_COMP_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_LIGHT_CULL_COMP}" -o "${VK_FP_LIGHT_CULL_COMP_SPV}"
        DEPENDS
            "${VK_FP_LIGHT_CULL_COMP}"
            "${VK_COMMON_LIGHT_CONSTANTS}"
            "${VK_COMMON_CULLING_LIGHT_STRUCT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_FP_DEPTH_REDUCE_COMP_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_FP_DEPTH_REDUCE_COMP}" -o "${VK_FP_DEPTH_REDUCE_COMP_SPV}"
        DEPENDS "${VK_FP_DEPTH_REDUCE_COMP}"
        VERBATIM
    )
    add_custom_target(HelloVulkanTriangleShaders DEPENDS "${VK_TRI_VERT_SPV}" "${VK_TRI_FRAG_SPV}")
    add_custom_target(HelloCullingVkShaders DEPENDS "${VK_CULLING_VERT_SPV}" "${VK_CULLING_FRAG_SPV}")
    add_custom_target(HelloLightTypesCullingVkShaders
        DEPENDS
            "${VK_LIGHT_TYPES_CULLING_VERT_SPV}"
            "${VK_LIGHT_TYPES_CULLING_FRAG_SPV}"
    )
    add_custom_target(HelloModernVulkanShaders DEPENDS "${VK_MODERN_VERT_SPV}" "${VK_MODERN_FRAG_SPV}")
    add_custom_target(HelloSoftShadowCullingVkShaders
        DEPENDS
            "${VK_CULLING_VERT_SPV}"
            "${VK_CULLING_FRAG_SPV}"
            "${VK_SOFT_SHADOW_CULLING_VERT_SPV}"
            "${VK_SOFT_SHADOW_CULLING_FRAG_SPV}"
            "${VK_PB_SHADOW_VERT_SPV}"
    )
    add_custom_target(HelloPassBasicsVulkanShaders
        DEPENDS
            "${VK_PB_SHADOW_VERT_SPV}"
            "${VK_PB_SCENE_VERT_SPV}"
            "${VK_PB_SCENE_FRAG_SPV}"
            "${VK_PB_POST_VERT_SPV}"
            "${VK_PB_SKY_FRAG_SPV}"
            "${VK_PB_BRIGHT_FRAG_SPV}"
            "${VK_PB_SHAFTS_FRAG_SPV}"
            "${VK_PB_FLARE_FRAG_SPV}"
            "${VK_PB_COMPOSITE_FRAG_SPV}"
            "${VK_PB_FXAA_FRAG_SPV}"
    )
    add_custom_target(HelloRenderingPathsShaders
        DEPENDS
            "${VK_FP_SCENE_VERT_SPV}"
            "${VK_FP_SCENE_FRAG_SPV}"
            "${VK_FP_GBUFFER_FRAG_SPV}"
            "${VK_FP_DEFERRED_VERT_SPV}"
            "${VK_FP_SSAO_FRAG_SPV}"
            "${VK_FP_DEFERRED_FRAG_SPV}"
            "${VK_FP_MOTION_BLUR_FRAG_SPV}"
            "${VK_FP_DOF_FRAG_SPV}"
            "${VK_FP_SHADOW_VERT_SPV}"
            "${VK_FP_LIGHT_CULL_COMP_SPV}"
            "${VK_FP_DEPTH_REDUCE_COMP_SPV}"
    )

    add_executable(HelloVulkanTriangle hello_vulkan_triangle.cpp)
    add_dependencies(HelloVulkanTriangle HelloVulkanTriangleShaders)
    target_link_libraries(HelloVulkanTriangle PRIVATE shs::renderer)
    target_compile_definitions(HelloVulkanTriangle PRIVATE
        SHS_VK_TRIANGLE_VERT_SPV="${VK_TRI_VERT_SPV}"
        SHS_VK_TRIANGLE_FRAG_SPV="${VK_TRI_FRAG_SPV}"
    )
    if(MSVC)
        target_compile_options(HelloVulkanTriangle PRIVATE /W4)
    else()
        target_compile_options(HelloVulkanTriangle PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(HelloVulkanTriangle PRIVATE $<$<CONFIG:Release>:-O3>)
        target_compile_options(HelloVulkanTriangle PRIVATE $<$<CONFIG:Debug>:-g>)
    endif()
    target_compile_features(HelloVulkanTriangle PRIVATE cxx_std_20)

    add_executable(HelloPassBasicsVulkan hello_pass_basics_vulkan.cpp)
    add_dependencies(HelloPassBasicsVulkan HelloPassBasicsVulkanShaders)
    target_link_libraries(HelloPassBasicsVulkan PRIVATE shs::renderer)
    target_compile_definitions(HelloPassBasicsVulkan PRIVATE
        SHS_VK_PB_SHADOW_VERT_SPV="${VK_PB_SHADOW_VERT_SPV}"
        SHS_VK_PB_SCENE_VERT_SPV="${VK_PB_SCENE_VERT_SPV}"
        SHS_VK_PB_SCENE_FRAG_SPV="${VK_PB_SCENE_FRAG_SPV}"
        SHS_VK_PB_POST_VERT_SPV="${VK_PB_POST_VERT_SPV}"
        SHS_VK_PB_SKY_FRAG_SPV="${VK_PB_SKY_FRAG_SPV}"
        SHS_VK_PB_BRIGHT_FRAG_SPV="${VK_PB_BRIGHT_FRAG_SPV}"
        SHS_VK_PB_SHAFTS_FRAG_SPV="${VK_PB_SHAFTS_FRAG_SPV}"
        SHS_VK_PB_FLARE_FRAG_SPV="${VK_PB_FLARE_FRAG_SPV}"
        SHS_VK_PB_COMPOSITE_FRAG_SPV="${VK_PB_COMPOSITE_FRAG_SPV}"
        SHS_VK_PB_FXAA_FRAG_SPV="${VK_PB_FXAA_FRAG_SPV}"
    )
    if(MSVC)
        target_compile_options(HelloPassBasicsVulkan PRIVATE /W4)
    else()
        target_compile_options(HelloPassBasicsVulkan PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(HelloPassBasicsVulkan PRIVATE $<$<CONFIG:Release>:-O3>)
        target_compile_options(HelloPassBasicsVulkan PRIVATE $<$<CONFIG:Debug>:-g>)
    endif()
    target_compile_features(HelloPassBasicsVulkan PRIVATE cxx_std_20)

    add_executable(HelloRenderingPaths hello_rendering_paths.cpp)
    add_dependencies(HelloRenderingPaths HelloRenderingPathsShaders)
    target_link_libraries(HelloRenderingPaths PRIVATE shs::renderer)
    target_compile_definitions(HelloRenderingPaths PRIVATE
        SHS_VK_FP_SCENE_VERT_SPV="${VK_FP_SCENE_VERT_SPV}"
        SHS_VK_FP_SCENE_FRAG_SPV="${VK_FP_SCENE_FRAG_SPV}"
        SHS_VK_FP_GBUFFER_FRAG_SPV="${VK_FP_GBUFFER_FRAG_SPV}"
        SHS_VK_FP_DEFERRED_VERT_SPV="${VK_FP_DEFERRED_VERT_SPV}"
        SHS_VK_FP_SSAO_FRAG_SPV="${VK_FP_SSAO_FRAG_SPV}"
        SHS_VK_FP_DEFERRED_FRAG_SPV="${VK_FP_DEFERRED_FRAG_SPV}"
        SHS_VK_FP_MOTION_BLUR_FRAG_SPV="${VK_FP_MOTION_BLUR_FRAG_SPV}"
        SHS_VK_FP_DOF_FRAG_SPV="${VK_FP_DOF_FRAG_SPV}"
        SHS_VK_FP_SHADOW_VERT_SPV="${VK_FP_SHADOW_VERT_SPV}"
        SHS_VK_FP_LIGHT_CULL_COMP_SPV="${VK_FP_LIGHT_CULL_COMP_SPV}"
        SHS_VK_FP_DEPTH_REDUCE_COMP_SPV="${VK_FP_DEPTH_REDUCE_COMP_SPV}"
    )
    if(MSVC)
        target_compile_options(HelloRenderingPaths PRIVATE /W4)
    else()
        target_compile_options(HelloRenderingPaths PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(HelloRenderingPaths PRIVATE $<$<CONFIG:Release>:-O3>)
        target_compile_options(HelloRenderingPaths PRIVATE $<$<CONFIG:Debug>:-g>)
    endif()
    target_compile_features(HelloRenderingPaths PRIVATE cxx_std_20)
else()
    if(NOT SHS_RENDERER_HAS_VULKAN)
        message(STATUS "HelloVulkanTriangle skipped (shs::renderer built without Vulkan support)")
    elseif(NOT GLSLANG_VALIDATOR)
        message(STATUS "HelloVulkanTriangle skipped (glslangValidator not found)")
    elseif(NOT SHS_SDL2_HAS_VULKAN)
        message(STATUS "HelloVulkanTriangle skipped (SDL2 was built without Vulkan support)")
    endif()
    if(NOT SHS_RENDERER_HAS_VULKAN)
        message(STATUS "HelloPassBasicsVulkan skipped (shs::renderer built without Vulkan support)")
    elseif(NOT GLSLANG_VALIDATOR)
        message(STATUS "HelloPassBasicsVulkan skipped (glslangValidator not found)")
    elseif(NOT SHS_SDL2_HAS_VULKAN)
        message(STATUS "HelloPassBasicsVulkan skipped (SDL2 was built without Vulkan support)")
    endif()
    if(NOT SHS_RENDERER_HAS_VULKAN)
        message(STATUS "HelloRenderingPaths skipped (shs::renderer built without Vulkan support)")
    elseif(NOT GLSLANG_VALIDATOR)
        message(STATUS "HelloRenderingPaths skipped (glslangValidator not found)")
    elseif(NOT SHS_SDL2_HAS_VULKAN)
        message(STATUS "HelloRenderingPaths skipped (SDL2 was built without Vulkan support)")
    endif()
endif()

# -----------------------------------------------------------------------------
#  Jolt Integration Demo
# -----------------------------------------------------------------------------
if(SHS_RENDERER_HAS_VULKAN AND GLSLANG_VALIDATOR AND SHS_SDL2_HAS_VULKAN)
    add_executable(hello_jolt_integration hello_jolt_integration.cpp)
    add_dependencies(hello_jolt_integration HelloRenderingPathsShaders)
    target_link_libraries(hello_jolt_integration PRIVATE shs::renderer)
    set_target_properties(hello_jolt_integration PROPERTIES FOLDER "Demos")
    target_compile_definitions(hello_jolt_integration PRIVATE
        SHS_VK_FP_SCENE_VERT_SPV="${VK_FP_SCENE_VERT_SPV}"
        SHS_VK_FP_SCENE_FRAG_SPV="${VK_FP_SCENE_FRAG_SPV}"
        SHS_VK_FP_SHADOW_VERT_SPV="${VK_FP_SHADOW_VERT_SPV}"
        SHS_VK_FP_LIGHT_CULL_COMP_SPV="${VK_FP_LIGHT_CULL_COMP_SPV}"
        SHS_VK_FP_DEPTH_REDUCE_COMP_SPV="${VK_FP_DEPTH_REDUCE_COMP_SPV}"
    )

    add_executable(hello_culling_vk hello_culling_vk.cpp)
    add_dependencies(hello_culling_vk HelloCullingVkShaders)
    target_link_libraries(hello_culling_vk PRIVATE shs::renderer)
    set_target_properties(hello_culling_vk PROPERTIES FOLDER "Demos")
    target_compile_definitions(hello_culling_vk PRIVATE
        SHS_VK_CULLING_VERT_SPV="${VK_CULLING_VERT_SPV}"
        SHS_VK_CULLING_FRAG_SPV="${VK_CULLING_FRAG_SPV}"
    )

    add_executable(hello_occlusion_culling_vk hello_occlusion_culling_vk.cpp)
    add_dependencies(hello_occlusion_culling_vk HelloCullingVkShaders)
    target_link_libraries(hello_occlusion_culling_vk PRIVATE shs::renderer)
    set_target_properties(hello_occlusion_culling_vk PROPERTIES FOLDER "Demos")
    target_compile_definitions(hello_occlusion_culling_vk PRIVATE
        SHS_VK_CULLING_VERT_SPV="${VK_CULLING_VERT_SPV}"
        SHS_VK_CULLING_FRAG_SPV="${VK_CULLING_FRAG_SPV}"
    )

    add_executable(hello_light_types_culling_vk hello_light_types_culling_vk.cpp)
    add_dependencies(hello_light_types_culling_vk HelloLightTypesCullingVkShaders)
    target_link_libraries(hello_light_types_culling_vk PRIVATE shs::renderer)
    set_target_properties(hello_light_types_culling_vk PROPERTIES FOLDER "Demos")
    target_compile_definitions(hello_light_types_culling_vk PRIVATE
        SHS_VK_LIGHT_TYPES_CULLING_VERT_SPV="${VK_LIGHT_TYPES_CULLING_VERT_SPV}"
        SHS_VK_LIGHT_TYPES_CULLING_FRAG_SPV="${VK_LIGHT_TYPES_CULLING_FRAG_SPV}"
    )

    add_executable(hello_soft_shadow_culling_vk hello_soft_shadow_culling_vk.cpp)
    add_dependencies(hello_soft_shadow_culling_vk HelloSoftShadowCullingVkShaders)
    target_link_libraries(hello_soft_shadow_culling_vk PRIVATE shs::renderer)
    set_target_properties(hello_soft_shadow_culling_vk PROPERTIES FOLDER "Demos")
    target_compile_definitions(hello_soft_shadow_culling_vk PRIVATE
        SHS_VK_CULLING_VERT_SPV="${VK_CULLING_VERT_SPV}"
        SHS_VK_CULLING_FRAG_SPV="${VK_CULLING_FRAG_SPV}"
        SHS_VK_SOFT_SHADOW_CULLING_VERT_SPV="${VK_SOFT_SHADOW_CULLING_VERT_SPV}"
        SHS_VK_SOFT_SHADOW_CULLING_FRAG_SPV="${VK_SOFT_SHADOW_CULLING_FRAG_SPV}"
        SHS_VK_PB_SHADOW_VERT_SPV="${VK_PB_SHADOW_VERT_SPV}"
    )

    add_executable(hello_modern_vulkan hello_modern_vulkan.cpp)
    add_dependencies(hello_modern_vulkan HelloModernVulkanShaders)
    target_link_libraries(hello_modern_vulkan PRIVATE shs::renderer)
    set_target_properties(hello_modern_vulkan PROPERTIES FOLDER "Demos")
    target_compile_definitions(hello_modern_vulkan PRIVATE
        SHS_VK_MODERN_VERT_SPV="${VK_MODERN_VERT_SPV}"
        SHS_VK_MODERN_FRAG_SPV="${VK_MODERN_FRAG_SPV}"
    )
endif()

# Software culling demo (always available)
add_executable(hello_culling_sw hello_culling_sw.cpp)
target_link_libraries(hello_culling_sw PRIVATE shs::renderer)
set_target_properties(hello_culling_sw PROPERTIES FOLDER "Demos")
target_compile_features(hello_culling_sw PRIVATE cxx_std_20)

add_executable(hello_occlusion_culling_sw hello_occlusion_culling_sw.cpp)
target_link_libraries(hello_occlusion_culling_sw PRIVATE shs::renderer)
set_target_properties(hello_occlusion_culling_sw PROPERTIES FOLDER "Demos")
target_compile_features(hello_occlusion_culling_sw PRIVATE cxx_std_20)

add_executable(hello_soft_shadow_culling_sw hello_soft_shadow_culling_sw.cpp)
target_link_libraries(hello_soft_shadow_culling_sw PRIVATE shs::renderer)
set_target_properties(hello_soft_shadow_culling_sw PROPERTIES FOLDER "Demos")
target_compile_features(hello_soft_shadow_culling_sw PRIVATE cxx_std_20)

add_executable(hello_light_types_culling_sw hello_light_types_culling_sw.cpp)
target_link_libraries(hello_light_types_culling_sw PRIVATE shs::renderer)
set_target_properties(hello_light_types_culling_sw PROPERTIES FOLDER "Demos")
target_compile_features(hello_light_types_culling_sw PRIVATE cxx_std_20)
