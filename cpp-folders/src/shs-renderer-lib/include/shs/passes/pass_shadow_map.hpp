#pragma once

#include "shs/passes/pass_common.hpp"
#include "shs/scene/scene_types.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"

namespace shs
{
    struct Context;

    class PassShadowMap
    {
    public:
        struct Inputs
        {
            const Scene* scene = nullptr;
            const FrameParams* fp = nullptr;
            RTRegistry* rtr = nullptr;

            RT_Shadow rt_shadow{};
        };

        void execute(Context& ctx, const Inputs& in)
        {
            (void)ctx;
            if (!in.scene || !in.fp || !in.rtr) return;
            if (!in.rt_shadow.valid()) return;

            void* shadow = in.rtr->get(in.rt_shadow);
            if (!shadow) return;

            // TODO:
            // - Build light view/proj
            // - Render depth
        }
    };
}

