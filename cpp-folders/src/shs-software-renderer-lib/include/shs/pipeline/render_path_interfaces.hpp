#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_interfaces.hpp
    MODULE: pipeline
    PURPOSE: Strategy interfaces and shared frame data bundles for dynamic render paths.
*/


#include <cstdint>
#include <vector>

#include "shs/core/context.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_registry.hpp"
#include "shs/pipeline/render_path_runtime_state.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs
{
    struct FrameSceneData
    {
        Context* ctx = nullptr;
        const Scene* scene = nullptr;
        const FrameParams* frame_params = nullptr;
        RTRegistry* rt_registry = nullptr;
    };

    struct FrameCameraData
    {
        float near_plane = 0.1f;
        float far_plane = 1000.0f;
    };

    struct FrameLightData
    {
        uint32_t active_light_count = 0;
        std::vector<uint32_t> light_indices{};
    };

    struct FrameCullData
    {
        uint32_t visible_object_count = 0;
        uint32_t visible_shadow_caster_count = 0;
        std::vector<uint32_t> visible_objects{};
        std::vector<uint32_t> visible_shadow_casters{};
    };

    struct FramePassResources
    {
        RTRegistry* rt_registry = nullptr;
    };

    struct FrameStats
    {
        uint32_t pass_count = 0;
        uint32_t draw_calls = 0;
        uint32_t culled_objects = 0;
    };

    struct ILightVolumeProvider
    {
        virtual ~ILightVolumeProvider() = default;
        virtual void build(const FrameSceneData& scene, FrameLightData& lights, const RenderPathRuntimeState& runtime_state) = 0;
    };

    struct ICullingStrategy
    {
        virtual ~ICullingStrategy() = default;
        virtual void run_view(FrameCullData& cull, const RenderPathRuntimeState& runtime_state) = 0;
        virtual void run_shadow(FrameCullData& cull, const RenderPathRuntimeState& runtime_state) = 0;
    };

    struct IRenderTechnique
    {
        virtual ~IRenderTechnique() = default;
        virtual void record(
            FramePassResources& resources,
            const FrameSceneData& scene,
            const FrameCullData& cull,
            const RenderPathRuntimeState& runtime_state) = 0;
    };

    struct IPassNode
    {
        virtual ~IPassNode() = default;
        virtual void execute(
            FramePassResources& resources,
            const FrameSceneData& scene,
            const FrameCullData& cull,
            const RenderPathRuntimeState& runtime_state) = 0;
    };
}

