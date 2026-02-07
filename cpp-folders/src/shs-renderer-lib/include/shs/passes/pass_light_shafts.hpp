// File: src/shs-renderer-lib/include/shs/passes/pass_light_shafts.hpp
#pragma once
/*
    SHS RENDERER LIB - PASS: LIGHT SHAFTS (SCREEN-SPACE)

    ЗОРИЛГО:
    - LDR дээр нэмэлт god rays additive эффект хийх
    - Depth-aware / occlusion-aware байхаар wiring хийх боломжтой

    NOTE:
    - Одоогоор opaque RT pointers (түр). Дараа нь RT төрөлтэйгээ холбож ажиллуулна.
*/

#include "shs/passes/pass_common.hpp"
#include "shs/scene/scene_types.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"

namespace shs
{
    struct Context;

    class PassLightShafts
    {
    public:
        struct Inputs
        {
            const Scene* scene = nullptr;
            const FrameParams* fp = nullptr;
            RTRegistry* rtr = nullptr;

            RTHandle rt_input_ldr{};
            RTHandle rt_output_ldr{};

            RTHandle rt_depth_like{};
            RTHandle rt_shafts_tmp{};
        };

        void execute(Context& ctx, const Inputs& in)
        {
            (void)ctx;
            if (!in.scene || !in.fp || !in.rtr) return;
            if (!in.rt_input_ldr.valid() || !in.rt_output_ldr.valid()) return;

            void* inldr = in.rtr->get(in.rt_input_ldr);
            void* outldr = in.rtr->get(in.rt_output_ldr);
            if (!inldr || !outldr) return;

            void* depth_like = in.rt_depth_like.valid() ? in.rtr->get(in.rt_depth_like) : nullptr;
            void* tmp = in.rt_shafts_tmp.valid() ? in.rtr->get(in.rt_shafts_tmp) : nullptr;

            (void)depth_like;
            (void)tmp;

            // TODO: real shafts
        }
    };
}