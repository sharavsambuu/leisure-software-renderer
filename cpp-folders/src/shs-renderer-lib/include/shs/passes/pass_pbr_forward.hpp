// File: src/shs-renderer-lib/include/shs/passes/pass_pbr_forward.hpp
#pragma once
/*
    SHS RENDERER LIB - PASS: PBR FORWARD

    ЗОРИЛГО:
    - Scene-г HDR RT рүү PBR + IBL + shadow ашиглан зурна
    - Motion vectors шаардлагатай бол эндээс гаргана

    - Одоогоор plumbing demo-д тааруулж байгаа тул RT-үүд opaque void* байна.
      Дараа нь RT төрөл/struct-уудтайгаа шууд холбож бодит ажил болгоно.
*/

#include "shs/passes/pass_common.hpp"
#include "shs/scene/scene_types.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"

namespace shs
{
    struct Context;

    class PassPBRForward
    {
    public:
        struct Inputs
        {
            const Scene*       scene = nullptr;
            const FrameParams* fp    = nullptr;
            RTRegistry*        rtr   = nullptr;

            RTHandle rt_hdr{};
            RTHandle rt_motion{};
            RTHandle rt_shadow{};
        };

        void execute(Context& ctx, const Inputs& in)
        {
            (void)ctx;
            if (!in.scene || !in.fp || !in.rtr) return;
            if (!in.rt_hdr.valid()) return;

            void* hdr = in.rtr->get(in.rt_hdr);
            if (!hdr) return;

            // optional
            void* motion = in.rt_motion.valid() ? in.rtr->get(in.rt_motion) : nullptr;
            void* shadow = in.rt_shadow.valid() ? in.rtr->get(in.rt_shadow) : nullptr;

            (void)motion;
            (void)shadow;

            // TODO: real draw
        }
    };
}