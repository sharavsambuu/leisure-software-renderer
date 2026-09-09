#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: skybox_renderer.hpp
    МОДУЛЬ: sky
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн sky модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "shs/gfx/rt_types.hpp"
#include "shs/job/parallel_for.hpp"
#include "shs/scene/scene_types.hpp"
#include "shs/sky/sky_model.hpp"

namespace shs
{
    inline void render_skybox_to_hdr(RT_ColorHDR& out_hdr, const Scene& scene, const ISkyModel& sky, IJobSystem* jobs = nullptr)
    {
        if (out_hdr.w <= 0 || out_hdr.h <= 0) return;

        const glm::mat4 inv_vp = glm::inverse(scene.cam.viewproj);
        const glm::vec3 cam_pos = scene.cam.pos;

        const int w = out_hdr.w;
        const int h = out_hdr.h;
        parallel_for_1d(jobs, 0, h, 8, [&](int yb, int ye)
        {
            for (int y = yb; y < ye; ++y)
            {
                const float ndc_y = (2.0f * ((float)y + 0.5f) / (float)h) - 1.0f;
                for (int x = 0; x < w; ++x)
                {
                    const float ndc_x = (2.0f * ((float)x + 0.5f) / (float)w) - 1.0f;
                    const glm::vec4 clip = glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
                    glm::vec4 world = inv_vp * clip;
                    if (std::abs(world.w) < 1e-8f)
                    {
                        out_hdr.color.at(x, y) = ColorF{0.0f, 0.0f, 0.0f, 1.0f};
                        continue;
                    }
                    world /= world.w;

                    const glm::vec3 dir_ws = glm::normalize(glm::vec3(world) - cam_pos);
                    const glm::vec3 c = sky.sample(dir_ws);
                    out_hdr.color.at(x, y) = ColorF{c.r, c.g, c.b, 1.0f};
                }
            }
        });
    }
}
