#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: context.hpp
    МОДУЛЬ: core
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн core модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>
#include <glm/glm.hpp>

#include "shs/job/job_system.hpp"
#include "shs/gfx/rt_shadow.hpp"

namespace shs
{
    struct RenderDebugStats
    {
        uint64_t tri_input = 0;
        uint64_t tri_after_clip = 0;
        uint64_t tri_raster = 0;

        void reset()
        {
            tri_input = 0;
            tri_after_clip = 0;
            tri_raster = 0;
        }
    };

    struct ShadowRuntimeState
    {
        const RT_ShadowDepth* map = nullptr;
        glm::mat4 light_viewproj{1.0f};
        bool valid = false;

        void reset()
        {
            map = nullptr;
            light_viewproj = glm::mat4(1.0f);
            valid = false;
        }
    };

    struct Context
    {
        IJobSystem* job_system = nullptr;
        RenderDebugStats debug{};
        ShadowRuntimeState shadow{};
    };
}
