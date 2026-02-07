#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: render_pass.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн pipeline модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <memory>
#include <vector>

#include "shs/core/context.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_registry.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs
{
    class IRenderPass
    {
    public:
        virtual ~IRenderPass() = default;
        virtual const char* id() const = 0;

        virtual bool enabled() const { return enabled_; }
        virtual void set_enabled(bool v) { enabled_ = v; }

        virtual void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) = 0;

    protected:
        bool enabled_ = true;
    };
}

