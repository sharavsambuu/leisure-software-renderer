#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: cubemap_sky.hpp
    МОДУЛЬ: sky
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн sky модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <algorithm>
#include <array>
#include <cmath>

#include <glm/glm.hpp>

#include "shs/resources/texture.hpp"
#include "shs/sky/sky_model.hpp"

namespace shs
{
    struct CubemapData
    {
        // 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
        std::array<Texture2DData, 6> face{};

        bool valid() const
        {
            for (int i = 0; i < 6; ++i)
            {
                if (!face[(size_t)i].valid()) return false;
            }
            return true;
        }
    };

    inline glm::vec3 srgb_to_linear_approx(const Color& c, float gamma = 2.2f)
    {
        const float r = std::pow((float)c.r / 255.0f, gamma);
        const float g = std::pow((float)c.g / 255.0f, gamma);
        const float b = std::pow((float)c.b / 255.0f, gamma);
        return glm::vec3(r, g, b);
    }

    inline glm::vec3 sample_face_bilinear_linear(const Texture2DData& tex, float u, float v)
    {
        if (!tex.valid()) return glm::vec3(0.0f);

        u = glm::clamp(u, 0.0f, 1.0f);
        v = glm::clamp(v, 0.0f, 1.0f);

        const float fx = u * (float)(tex.w - 1);
        const float fy = v * (float)(tex.h - 1);
        const int x0 = (int)std::floor(fx);
        const int y0 = (int)std::floor(fy);
        const int x1 = std::min(x0 + 1, tex.w - 1);
        const int y1 = std::min(y0 + 1, tex.h - 1);
        const float tx = fx - (float)x0;
        const float ty = fy - (float)y0;

        const glm::vec3 v00 = srgb_to_linear_approx(tex.at(x0, y0));
        const glm::vec3 v10 = srgb_to_linear_approx(tex.at(x1, y0));
        const glm::vec3 v01 = srgb_to_linear_approx(tex.at(x0, y1));
        const glm::vec3 v11 = srgb_to_linear_approx(tex.at(x1, y1));

        const glm::vec3 vx0 = glm::mix(v00, v10, tx);
        const glm::vec3 vx1 = glm::mix(v01, v11, tx);
        return glm::mix(vx0, vx1, ty);
    }

    class CubemapSky final : public ISkyModel
    {
    public:
        CubemapSky(CubemapData cubemap, float intensity = 1.0f)
            : cubemap_(std::move(cubemap)), intensity_(intensity)
        {}

        glm::vec3 sample(const glm::vec3& direction_ws) const override
        {
            if (!cubemap_.valid()) return glm::vec3(0.0f);

            glm::vec3 d = direction_ws;
            const float len = glm::length(d);
            if (len < 1e-8f) return glm::vec3(0.0f);
            d /= len;

            const float ax = std::abs(d.x);
            const float ay = std::abs(d.y);
            const float az = std::abs(d.z);

            int face = 0;
            float u = 0.5f;
            float v = 0.5f;

            if (ax >= ay && ax >= az)
            {
                if (d.x > 0.0f) { face = 0; u = (-d.z / ax); v = ( d.y / ax); }
                else            { face = 1; u = ( d.z / ax); v = ( d.y / ax); }
            }
            else if (ay >= ax && ay >= az)
            {
                if (d.y > 0.0f) { face = 2; u = ( d.x / ay); v = (-d.z / ay); }
                else            { face = 3; u = ( d.x / ay); v = ( d.z / ay); }
            }
            else
            {
                if (d.z > 0.0f) { face = 4; u = ( d.x / az); v = ( d.y / az); }
                else            { face = 5; u = (-d.x / az); v = ( d.y / az); }
            }

            u = 0.5f * (u + 1.0f);
            v = 0.5f * (v + 1.0f);
            return sample_face_bilinear_linear(cubemap_.face[(size_t)face], u, v) * intensity_;
        }

    private:
        CubemapData cubemap_{};
        float intensity_ = 1.0f;
    };
}

