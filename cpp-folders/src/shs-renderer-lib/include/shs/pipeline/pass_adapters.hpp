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
#include "shs/passes/pass_motion_blur.hpp"
#include "shs/passes/pass_pbr_forward.hpp"
#include "shs/passes/pass_shadow_map.hpp"
#include "shs/passes/pass_tonemap.hpp"
#include "shs/pipeline/pass_registry.hpp"
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
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.write(make_rt_resource_ref(static_cast<const RTHandle&>(rt_shadow_), PassResourceType::Shadow, "shadow", PassResourceDomain::Software));
            return io;
        }

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
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_rt_resource_ref(rt_shadow_, PassResourceType::Shadow, "shadow", PassResourceDomain::Software));
            io.write(make_rt_resource_ref(rt_hdr_, PassResourceType::ColorHDR, "hdr", PassResourceDomain::Software));
            io.write(make_rt_resource_ref(static_cast<const RTHandle&>(rt_motion_), PassResourceType::Motion, "motion", PassResourceDomain::Software));
            return io;
        }

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
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_rt_resource_ref(rt_hdr_, PassResourceType::ColorHDR, "hdr", PassResourceDomain::Software));
            io.write(make_rt_resource_ref(rt_ldr_, PassResourceType::ColorLDR, "ldr", PassResourceDomain::Software));
            return io;
        }

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
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read_write(make_rt_resource_ref(rt_ldr_, PassResourceType::ColorLDR, "ldr", PassResourceDomain::Software));
            io.read(make_rt_resource_ref(rt_depth_like_, PassResourceType::Motion, "motion", PassResourceDomain::Software));
            if (rt_shafts_tmp_.valid())
            {
                io.write(make_rt_resource_ref(rt_shafts_tmp_, PassResourceType::Temp, "shafts_tmp", PassResourceDomain::Software));
            }
            else
            {
                io.write(make_named_resource_ref("light_shafts.auto_tmp", PassResourceType::Temp, PassResourceDomain::Software));
            }
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            PassLightShafts::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_input_ldr = rt_ldr_;
            in.rt_output_ldr = rt_ldr_;
            in.rt_depth_like = rt_depth_like_;
            RTHandle tmp = rt_shafts_tmp_;
            if (!tmp.valid())
            {
                auto* ldr = static_cast<RT_ColorLDR*>(rtr.get(rt_ldr_));
                if (ldr) tmp = rtr.ensure_transient_color_ldr("light_shafts.auto_tmp", ldr->w, ldr->h);
            }
            in.rt_shafts_tmp = tmp;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_ldr_{};
        RTHandle rt_depth_like_{};
        RTHandle rt_shafts_tmp_{};
        PassLightShafts pass_{};
    };

    class PassMotionBlurAdapter final : public IRenderPass
    {
    public:
        PassMotionBlurAdapter(RTHandle rt_ldr_inout, RTHandle rt_motion, RTHandle rt_tmp)
            : rt_ldr_(rt_ldr_inout), rt_motion_(rt_motion), rt_tmp_(rt_tmp)
        {}

        const char* id() const override { return "motion_blur"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read_write(make_rt_resource_ref(rt_ldr_, PassResourceType::ColorLDR, "ldr", PassResourceDomain::Software));
            io.read(make_rt_resource_ref(rt_motion_, PassResourceType::Motion, "motion", PassResourceDomain::Software));
            if (rt_tmp_.valid())
            {
                io.write(make_rt_resource_ref(rt_tmp_, PassResourceType::Temp, "motion_tmp", PassResourceDomain::Software));
            }
            else
            {
                io.write(make_named_resource_ref("motion_blur.auto_tmp", PassResourceType::Temp, PassResourceDomain::Software));
            }
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            (void)scene;
            PassMotionBlur::Inputs in{};
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_input_ldr = rt_ldr_;
            in.rt_output_ldr = rt_ldr_;
            in.rt_motion = rt_motion_;
            RTHandle tmp = rt_tmp_;
            if (!tmp.valid())
            {
                auto* ldr = static_cast<RT_ColorLDR*>(rtr.get(rt_ldr_));
                if (ldr) tmp = rtr.ensure_transient_color_ldr("motion_blur.auto_tmp", ldr->w, ldr->h);
            }
            in.rt_tmp = tmp;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_ldr_{};
        RTHandle rt_motion_{};
        RTHandle rt_tmp_{};
        PassMotionBlur pass_{};
    };

    inline PassFactoryRegistry make_standard_pass_factory_registry(
        RT_Shadow rt_shadow,
        RTHandle rt_hdr,
        RT_Motion rt_motion,
        RTHandle rt_ldr,
        RTHandle rt_shafts_tmp,
        RTHandle rt_motion_blur_tmp
    )
    {
        PassFactoryRegistry reg{};
        reg.register_factory("shadow_map", [=]() {
            return std::make_unique<PassShadowMapAdapter>(rt_shadow);
        });
        reg.register_factory("pbr_forward", [=]() {
            return std::make_unique<PassPBRForwardAdapter>(rt_hdr, rt_motion, RTHandle{rt_shadow.id});
        });
        reg.register_factory("tonemap", [=]() {
            return std::make_unique<PassTonemapAdapter>(rt_hdr, rt_ldr);
        });
        reg.register_factory("light_shafts", [=]() {
            return std::make_unique<PassLightShaftsAdapter>(rt_ldr, rt_motion, rt_shafts_tmp);
        });
        reg.register_factory("motion_blur", [=]() {
            return std::make_unique<PassMotionBlurAdapter>(rt_ldr, rt_motion, rt_motion_blur_tmp);
        });
        return reg;
    }
}
