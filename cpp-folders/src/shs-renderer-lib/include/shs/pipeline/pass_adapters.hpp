#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_adapters.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн pipeline модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/gfx/rt_handle.hpp"
#include "shs/passes/pass_light_shafts.hpp"
#include "shs/passes/pass_pbr_forward.hpp"
#include "shs/passes/pass_shadow_map.hpp"
#include "shs/passes/pass_tonemap.hpp"
#include "shs/pipeline/render_pass.hpp"

namespace shs
{
    class PassShadowMapAdapter final : public IRenderPass
    {
    public:
        explicit PassShadowMapAdapter(RT_Shadow rt_shadow)
            : rt_shadow_(rt_shadow)
        {}

        const char* id() const override { return "shadow_map"; }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            PassShadowMap::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_shadow = rt_shadow_;
            pass_.execute(ctx, in);
        }

    private:
        RT_Shadow rt_shadow_{};
        PassShadowMap pass_{};
    };

    class PassPBRForwardAdapter final : public IRenderPass
    {
    public:
        PassPBRForwardAdapter(RTHandle rt_hdr, RT_Motion rt_motion, RTHandle rt_shadow)
            : rt_hdr_(rt_hdr), rt_motion_(rt_motion), rt_shadow_(rt_shadow)
        {}

        const char* id() const override { return "pbr_forward"; }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            PassPBRForward::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_hdr = rt_hdr_;
            in.rt_motion = rt_motion_;
            in.rt_shadow = rt_shadow_;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_hdr_{};
        RT_Motion rt_motion_{};
        RTHandle rt_shadow_{};
        PassPBRForward pass_{};
    };

    class PassTonemapAdapter final : public IRenderPass
    {
    public:
        PassTonemapAdapter(RTHandle rt_hdr, RTHandle rt_ldr)
            : rt_hdr_(rt_hdr), rt_ldr_(rt_ldr)
        {}

        const char* id() const override { return "tonemap"; }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            (void)scene;
            PassTonemap::Inputs in{};
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_hdr = rt_hdr_;
            in.rt_ldr = rt_ldr_;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_hdr_{};
        RTHandle rt_ldr_{};
        PassTonemap pass_{};
    };

    class PassLightShaftsAdapter final : public IRenderPass
    {
    public:
        PassLightShaftsAdapter(RTHandle rt_ldr_inout, RTHandle rt_depth_like, RTHandle rt_shafts_tmp)
            : rt_ldr_(rt_ldr_inout), rt_depth_like_(rt_depth_like), rt_shafts_tmp_(rt_shafts_tmp)
        {}

        const char* id() const override { return "light_shafts"; }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            PassLightShafts::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_input_ldr = rt_ldr_;
            in.rt_output_ldr = rt_ldr_;
            in.rt_depth_like = rt_depth_like_;
            in.rt_shafts_tmp = rt_shafts_tmp_;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_ldr_{};
        RTHandle rt_depth_like_{};
        RTHandle rt_shafts_tmp_{};
        PassLightShafts pass_{};
    };
}

