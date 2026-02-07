/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pipeline_light_shafts.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн pipeline модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/

// PATCH: replace FrameResources_LightShafts void* with handles + add tonemap pass

#pragma once

#include <cstdint>

#include "shs/scene/scene_types.hpp"
#include "shs/frame/frame_params.hpp"

#include "shs/passes/pass_context.hpp"

#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"

#include "shs/passes/pass_shadow_map.hpp"
#include "shs/passes/pass_pbr_forward.hpp"
#include "shs/passes/pass_tonemap.hpp"
#include "shs/passes/pass_light_shafts.hpp"

namespace shs
{
    struct Context;

    struct FrameResources_LightShafts
    {
        RT_Shadow rt_shadow{};
        RTHandle  rt_hdr{};     // HDR color+depth (one RT in current style)
        RT_Motion rt_motion{};
        RTHandle  rt_ldr{};
        RTHandle  rt_shafts{};
    };

    class PipelineLightShafts
    {
    public:
        void set_registry(RTRegistry* rtr) { rtr_ = rtr; }
        void set_resources(FrameResources_LightShafts* fr) { fr_ = fr; }

        void init(Context& ctx, int w, int h)
        {
            (void)ctx;
            w_ = w; h_ = h;
        }

        void render(Context& ctx, const Scene& scene, const FrameParams& fp)
        {
            if (!fr_ || !rtr_) return;

            // 0) Shadow map
            PassShadowMap::Inputs in_sm{};
            in_sm.scene     = &scene;
            in_sm.fp        = &fp;
            in_sm.rtr       = rtr_;
            in_sm.rt_shadow = fr_->rt_shadow;
            shadow_.execute(ctx, in_sm);

            // 1) PBR forward -> HDR
            PassPBRForward::Inputs in_pbr{};
            in_pbr.scene     = &scene;
            in_pbr.fp        = &fp;
            in_pbr.rtr       = rtr_;
            in_pbr.rt_hdr    = fr_->rt_hdr;
            in_pbr.rt_motion = fr_->rt_motion;
            in_pbr.rt_shadow = fr_->rt_shadow;
            pbr_.execute(ctx, in_pbr);

            // 2) Tonemap HDR -> LDR
            PassTonemap::Inputs in_tm{};
            in_tm.fp     = &fp;
            in_tm.rtr    = rtr_;
            in_tm.rt_hdr = fr_->rt_hdr;
            in_tm.rt_ldr = fr_->rt_ldr;
            tm_.execute(ctx, in_tm);

            // 3) Light shafts on LDR (in-place)
            if (fp.enable_light_shafts)
            {
                PassLightShafts::Inputs in_ls{};
                in_ls.scene         = &scene;
                in_ls.fp            = &fp;
                in_ls.rtr           = rtr_;
                in_ls.rt_input_ldr  = fr_->rt_ldr;
                in_ls.rt_output_ldr = fr_->rt_ldr;
                in_ls.rt_depth_like = fr_->rt_motion;
                in_ls.rt_shafts_tmp = fr_->rt_shafts;
                ls_.execute(ctx, in_ls);
            }
        }

    private:
        int w_ = 0;
        int h_ = 0;

        RTRegistry* rtr_ = nullptr;
        FrameResources_LightShafts* fr_ = nullptr;

        PassShadowMap   shadow_{};
        PassPBRForward  pbr_{};
        PassTonemap     tm_{};
        PassLightShafts ls_{};
    };
}
