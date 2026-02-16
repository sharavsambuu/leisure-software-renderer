#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_standard_pass_routing.hpp
    MODULE: pipeline
    PURPOSE: Shared standard pass-routing helper so hosts can bind pass handlers
             without duplicating PassId-to-handler wiring.
*/


#include <functional>

#include "shs/pipeline/pass_id.hpp"
#include "shs/pipeline/render_path_compiler.hpp"
#include "shs/pipeline/render_path_pass_dispatch.hpp"

namespace shs
{
    template <typename TContext>
    struct StandardRenderPathPassHandlers
    {
        using Handler = typename RenderPathPassDispatcher<TContext>::Handler;

        Handler shadow_map{};
        Handler depth_prepass{};
        Handler light_culling{};
        Handler cluster_build{};
        Handler scene_forward{};
        Handler gbuffer{};
        Handler ssao{};
        Handler deferred_lighting{};
        Handler tonemap{};
        Handler motion_blur{};
        Handler depth_of_field{};
        Handler taa{};
        Handler fallback_noop{};
    };

    template <typename TContext>
    inline bool register_standard_render_path_handlers(
        RenderPathPassDispatcher<TContext>& dispatcher,
        const StandardRenderPathPassHandlers<TContext>& handlers)
    {
        using Handler = typename RenderPathPassDispatcher<TContext>::Handler;

        const Handler noop = handlers.fallback_noop
            ? handlers.fallback_noop
            : Handler([](TContext&, const RenderPathCompiledPass&) {
                  return true;
              });

        const Handler shadow = handlers.shadow_map ? handlers.shadow_map : noop;
        const Handler depth = handlers.depth_prepass ? handlers.depth_prepass : noop;
        const Handler cull = handlers.light_culling ? handlers.light_culling : noop;
        const Handler cluster = handlers.cluster_build ? handlers.cluster_build : noop;
        const Handler scene = handlers.scene_forward ? handlers.scene_forward : noop;
        const Handler gbuffer = handlers.gbuffer ? handlers.gbuffer : noop;
        const Handler ssao = handlers.ssao ? handlers.ssao : noop;
        const Handler deferred = handlers.deferred_lighting
            ? handlers.deferred_lighting
            : (handlers.scene_forward ? handlers.scene_forward : noop);
        const Handler tonemap = handlers.tonemap ? handlers.tonemap : noop;
        const Handler motion_blur = handlers.motion_blur ? handlers.motion_blur : noop;
        const Handler depth_of_field = handlers.depth_of_field ? handlers.depth_of_field : noop;
        const Handler taa = handlers.taa ? handlers.taa : noop;

        dispatcher.clear();

        bool ok = true;
        ok = dispatcher.register_handler(PassId::ShadowMap, shadow) && ok;
        ok = dispatcher.register_handler(PassId::DepthPrepass, depth) && ok;
        ok = dispatcher.register_handler(PassId::LightCulling, cull) && ok;
        ok = dispatcher.register_handler(PassId::ClusterBuild, cluster) && ok;
        ok = dispatcher.register_handler(PassId::ClusterLightAssign, cull) && ok;

        ok = dispatcher.register_handler(PassId::PBRForward, scene) && ok;
        ok = dispatcher.register_handler(PassId::PBRForwardPlus, scene) && ok;
        ok = dispatcher.register_handler(PassId::PBRForwardClustered, scene) && ok;

        ok = dispatcher.register_handler(PassId::GBuffer, gbuffer) && ok;
        ok = dispatcher.register_handler(PassId::SSAO, ssao) && ok;
        ok = dispatcher.register_handler(PassId::DeferredLighting, deferred) && ok;
        ok = dispatcher.register_handler(PassId::DeferredLightingTiled, deferred) && ok;

        ok = dispatcher.register_handler(PassId::Tonemap, tonemap) && ok;
        ok = dispatcher.register_handler(PassId::TAA, taa) && ok;
        ok = dispatcher.register_handler(PassId::MotionBlur, motion_blur) && ok;
        ok = dispatcher.register_handler(PassId::DepthOfField, depth_of_field) && ok;
        return ok;
    }
}
