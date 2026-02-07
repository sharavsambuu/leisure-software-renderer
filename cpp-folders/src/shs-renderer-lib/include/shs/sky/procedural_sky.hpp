#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: procedural_sky.hpp
    МОДУЛЬ: sky
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн sky модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <glm/glm.hpp>

#include "shs/sky/sky_model.hpp"

namespace shs
{
    class ProceduralSky final : public ISkyModel
    {
    public:
        explicit ProceduralSky(glm::vec3 sun_dir_ws = glm::normalize(glm::vec3(0.4668f, -0.3487f, 0.8127f)))
            : sun_direction_ws_(glm::normalize(sun_dir_ws))
        {}

        glm::vec3 sample(const glm::vec3& direction_ws) const override
        {
            const glm::vec3 d = glm::normalize(direction_ws);
            const float t = glm::clamp(d.y * 0.5f + 0.5f, 0.0f, 1.0f);

            const glm::vec3 zenith = glm::vec3(0.05f, 0.20f, 0.50f);
            const glm::vec3 horizon = glm::vec3(0.30f, 0.60f, 1.00f);
            glm::vec3 sky = glm::mix(horizon, zenith, t);

            const float sun_dot = glm::dot(d, -sun_direction_ws_);
            if (sun_dot > 0.9998f)
            {
                sky = glm::vec3(15.0f);
            }
            else if (sun_dot > 0.9990f)
            {
                const float glow = (sun_dot - 0.9990f) / (0.9998f - 0.9990f);
                sky = glm::mix(sky, glm::vec3(10.0f, 8.0f, 4.0f), glow);
            }
            return sky;
        }

        void set_sun_direction(glm::vec3 sun_dir_ws)
        {
            sun_direction_ws_ = glm::normalize(sun_dir_ws);
        }

    private:
        glm::vec3 sun_direction_ws_{};
    };
}

