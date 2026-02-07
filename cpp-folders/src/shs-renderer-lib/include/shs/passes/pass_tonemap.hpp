// File: src/shs-renderer-lib/include/shs/passes/pass_tonemap.hpp
#pragma once
/*
    SHS RENDERER LIB - PASS: TONEMAP + GAMMA

    ЗОРИЛГО:
    - HDR RT -> LDR RT
    - Exposure + filmic/ACES (дараа нь) + gamma

    NOTE:
    - Plumbing-first: actual pixel loop later, одоогоор contract + wiring.
*/

#include "shs/passes/pass_common.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"

namespace shs
{
    struct Context;

    class PassTonemap
    {
    public:
        struct Inputs
        {
            const FrameParams* fp = nullptr;
            RTRegistry* rtr = nullptr;

            RTHandle rt_hdr{}; // input
            RTHandle rt_ldr{}; // output
        };

        void execute(Context& ctx, const Inputs& in)
        {
            (void)ctx;
            if (!in.fp || !in.rtr) return;
            if (!in.rt_hdr.valid() || !in.rt_ldr.valid()) return;

            void* hdr = in.rtr->get(in.rt_hdr);
            void* ldr = in.rtr->get(in.rt_ldr);
            if (!hdr || !ldr) return;

            // TODO:
            // - Sample HDR color
            // - Apply exposure
            // - Tonemap (filmic/ACES)
            // - Gamma
            // - Write LDR
        }
    };
}

