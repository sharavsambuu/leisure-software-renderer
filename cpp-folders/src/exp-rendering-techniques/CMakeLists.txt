cmake_minimum_required(VERSION 3.28)
project(ExpRenderingTechniques LANGUAGES CXX)

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

if(SHS_RENDERER_HAS_VULKAN AND GLSLANG_VALIDATOR AND SHS_SDL2_HAS_VULKAN)
    set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${OUT_DIR}")

    set(VK_DEFAULT_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shs-renderer-lib/shaders/vulkan")
    set(VK_SHADER_COMMON_DIR "${VK_DEFAULT_SHADER_DIR}/common")
    set(VK_COMMON_MATH "${VK_SHADER_COMMON_DIR}/math.glsl")
    set(VK_COMMON_LIGHT_CONSTANTS "${VK_SHADER_COMMON_DIR}/light_constants.glsl")
    set(VK_COMMON_LIGHT_MATH "${VK_SHADER_COMMON_DIR}/light_math.glsl")

    # 1. Classic Forward (RenderPath Implementation) - Standard Shaders
    
    # Standard Shaders for SSAO/FXAA
    set(FP_GBUFFER_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_gbuffer.frag")
    set(FP_GBUFFER_FRAG_SPV "${OUT_DIR}/fp_stress_gbuffer.frag.spv")
    set(FP_SSAO_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_ssao.frag")
    set(FP_SSAO_FRAG_SPV "${OUT_DIR}/fp_stress_ssao.frag.spv")
    set(PB_FXAA_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_fxaa.frag")
    set(PB_FXAA_FRAG_SPV "${OUT_DIR}/pb_fxaa.frag.spv")
    set(FP_DEFERRED_VERT "${VK_DEFAULT_SHADER_DIR}/fp_stress_deferred.vert")
    set(FP_DEFERRED_VERT_SPV "${OUT_DIR}/fp_stress_deferred.vert.spv")

    add_custom_command(
        OUTPUT "${FP_GBUFFER_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_GBUFFER_FRAG}" -o "${FP_GBUFFER_FRAG_SPV}"
        DEPENDS "${FP_GBUFFER_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FP_SSAO_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_SSAO_FRAG}" -o "${FP_SSAO_FRAG_SPV}"
        DEPENDS "${FP_SSAO_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${PB_FXAA_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${PB_FXAA_FRAG}" -o "${PB_FXAA_FRAG_SPV}"
        DEPENDS "${PB_FXAA_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FP_DEFERRED_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_DEFERRED_VERT}" -o "${FP_DEFERRED_VERT_SPV}"
        DEPENDS "${FP_DEFERRED_VERT}"
        VERBATIM
    )
    # Post-Process Shaders
    set(PB_SKY_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_sky.frag")
    set(PB_SKY_FRAG_SPV "${OUT_DIR}/pb_sky.frag.spv")
    set(PB_BRIGHT_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_bright.frag")
    set(PB_BRIGHT_FRAG_SPV "${OUT_DIR}/pb_bright.frag.spv")
    set(PB_SHAFTS_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_shafts.frag")
    set(PB_SHAFTS_FRAG_SPV "${OUT_DIR}/pb_shafts.frag.spv")
    set(PB_FLARE_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_flare.frag")
    set(PB_FLARE_FRAG_SPV "${OUT_DIR}/pb_flare.frag.spv")
    set(PB_COMPOSITE_FRAG "${VK_DEFAULT_SHADER_DIR}/pb_composite.frag")
    set(PB_COMPOSITE_FRAG_SPV "${OUT_DIR}/pb_composite.frag.spv")
    set(FP_DEFERRED_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_deferred.frag")
    set(FP_DEFERRED_FRAG_SPV "${OUT_DIR}/fp_stress_deferred.frag.spv")
    set(FP_MOTION_BLUR_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_motion_blur.frag")
    set(FP_MOTION_BLUR_FRAG_SPV "${OUT_DIR}/fp_stress_motion_blur.frag.spv")
    set(FP_DOF_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_dof.frag")
    set(FP_DOF_FRAG_SPV "${OUT_DIR}/fp_stress_dof.frag.spv")
    set(FP_SHADOW_VERT "${VK_DEFAULT_SHADER_DIR}/fp_stress_shadow.vert")
    set(FP_SHADOW_VERT_SPV "${OUT_DIR}/fp_stress_shadow.vert.spv")
    set(FP_SCENE_VERT "${VK_DEFAULT_SHADER_DIR}/fp_stress_scene.vert")
    set(FP_SCENE_VERT_SPV "${OUT_DIR}/fp_stress_scene.vert.spv")
    set(FP_SCENE_FRAG "${VK_DEFAULT_SHADER_DIR}/fp_stress_scene.frag")
    set(FP_SCENE_FRAG_SPV "${OUT_DIR}/fp_stress_scene.frag.spv")
    set(FP_LIGHT_CULL_COMP "${VK_DEFAULT_SHADER_DIR}/fp_stress_light_cull.comp")
    set(FP_LIGHT_CULL_COMP_SPV "${OUT_DIR}/fp_stress_light_cull.comp.spv")
    set(FP_DEPTH_REDUCE_COMP "${VK_DEFAULT_SHADER_DIR}/fp_stress_depth_reduce.comp")
    set(FP_DEPTH_REDUCE_COMP_SPV "${OUT_DIR}/fp_stress_depth_reduce.comp.spv")

    add_custom_command(
        OUTPUT "${PB_SKY_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${PB_SKY_FRAG}" -o "${PB_SKY_FRAG_SPV}"
        DEPENDS "${PB_SKY_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${PB_BRIGHT_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${PB_BRIGHT_FRAG}" -o "${PB_BRIGHT_FRAG_SPV}"
        DEPENDS "${PB_BRIGHT_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${PB_SHAFTS_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${PB_SHAFTS_FRAG}" -o "${PB_SHAFTS_FRAG_SPV}"
        DEPENDS "${PB_SHAFTS_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${PB_FLARE_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${PB_FLARE_FRAG}" -o "${PB_FLARE_FRAG_SPV}"
        DEPENDS "${PB_FLARE_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${PB_COMPOSITE_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${PB_COMPOSITE_FRAG}" -o "${PB_COMPOSITE_FRAG_SPV}"
        DEPENDS "${PB_COMPOSITE_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FP_DEFERRED_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_DEFERRED_FRAG}" -o "${FP_DEFERRED_FRAG_SPV}"
        DEPENDS "${FP_DEFERRED_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FP_MOTION_BLUR_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_MOTION_BLUR_FRAG}" -o "${FP_MOTION_BLUR_FRAG_SPV}"
        DEPENDS "${FP_MOTION_BLUR_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FP_DOF_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_DOF_FRAG}" -o "${FP_DOF_FRAG_SPV}"
        DEPENDS "${FP_DOF_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FP_SHADOW_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_SHADOW_VERT}" -o "${FP_SHADOW_VERT_SPV}"
        DEPENDS "${FP_SHADOW_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FP_SCENE_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_SCENE_VERT}" -o "${FP_SCENE_VERT_SPV}"
        DEPENDS "${FP_SCENE_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FP_SCENE_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_SCENE_FRAG}" -o "${FP_SCENE_FRAG_SPV}"
        DEPENDS "${FP_SCENE_FRAG}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FP_LIGHT_CULL_COMP_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_LIGHT_CULL_COMP}" -o "${FP_LIGHT_CULL_COMP_SPV}"
        DEPENDS "${FP_LIGHT_CULL_COMP}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FP_DEPTH_REDUCE_COMP_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${FP_DEPTH_REDUCE_COMP}" -o "${FP_DEPTH_REDUCE_COMP_SPV}"
        DEPENDS "${FP_DEPTH_REDUCE_COMP}"
        VERBATIM
    )

    add_custom_target(DemoStandardShaders DEPENDS 
        "${FP_GBUFFER_FRAG_SPV}" 
        "${FP_SSAO_FRAG_SPV}" 
        "${PB_FXAA_FRAG_SPV}"
        "${FP_DEFERRED_VERT_SPV}"
        "${FP_DEFERRED_FRAG_SPV}"
        "${FP_MOTION_BLUR_FRAG_SPV}"
        "${FP_DOF_FRAG_SPV}"
        "${FP_SHADOW_VERT_SPV}"
        "${FP_SCENE_VERT_SPV}"
        "${FP_SCENE_FRAG_SPV}"
        "${PB_SKY_FRAG_SPV}"
        "${PB_BRIGHT_FRAG_SPV}"
        "${PB_SHAFTS_FRAG_SPV}"
        "${PB_FLARE_FRAG_SPV}"
        "${PB_COMPOSITE_FRAG_SPV}"
        "${FP_LIGHT_CULL_COMP_SPV}"
        "${FP_DEPTH_REDUCE_COMP_SPV}")

    # 1.1 Classic Forward (RenderPath Implementation)
    add_executable(demo_forward_classic_renderpath demo_forward_classic_renderpath.cpp)
    add_dependencies(demo_forward_classic_renderpath DemoStandardShaders)
    target_link_libraries(demo_forward_classic_renderpath PRIVATE shs::renderer)
    set_target_properties(demo_forward_classic_renderpath PROPERTIES FOLDER "Demos/RenderingTechniques")
    target_compile_definitions(demo_forward_classic_renderpath PRIVATE
        SHS_VK_FP_SCENE_VERT_SPV="${FP_SCENE_VERT_SPV}"
        SHS_VK_FP_SCENE_FRAG_SPV="${FP_SCENE_FRAG_SPV}"
        SHS_VK_FP_GBUFFER_FRAG_SPV="${FP_GBUFFER_FRAG_SPV}"
        SHS_VK_FP_DEFERRED_VERT_SPV="${FP_DEFERRED_VERT_SPV}"
        SHS_VK_FP_SSAO_FRAG_SPV="${FP_SSAO_FRAG_SPV}"
        SHS_VK_FP_DEFERRED_FRAG_SPV="${FP_DEFERRED_FRAG_SPV}"
        SHS_VK_FP_MOTION_BLUR_FRAG_SPV="${FP_MOTION_BLUR_FRAG_SPV}"
        SHS_VK_FP_DOF_FRAG_SPV="${FP_DOF_FRAG_SPV}"
        SHS_VK_FP_SHADOW_VERT_SPV="${FP_SHADOW_VERT_SPV}"
        SHS_VK_FP_LIGHT_CULL_COMP_SPV="${FP_LIGHT_CULL_COMP_SPV}"
        SHS_VK_FP_DEPTH_REDUCE_COMP_SPV="${FP_DEPTH_REDUCE_COMP_SPV}"
    )

    if(MSVC)
        target_compile_options(demo_forward_classic_renderpath PRIVATE /W4)
    else()
        target_compile_options(demo_forward_classic_renderpath PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(demo_forward_classic_renderpath PRIVATE $<$<CONFIG:Release>:-O3>)
        target_compile_options(demo_forward_classic_renderpath PRIVATE $<$<CONFIG:Debug>:-g>)
    endif()
    target_compile_features(demo_forward_classic_renderpath PRIVATE cxx_std_20)

endif()
