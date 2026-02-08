#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: light_camera.hpp
    МОДУЛЬ: camera
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн camera модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <algorithm>
#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>

#include "shs/camera/convention.hpp"
#include "shs/geometry/aabb.hpp"

namespace shs
{
    struct LightCamera
    {
        glm::mat4 view{1.0f};
        glm::mat4 proj{1.0f};
        glm::mat4 viewproj{1.0f};
        glm::vec3 pos_ws{0.0f};
        glm::vec3 dir_ws{0.0f, -1.0f, 0.0f};
    };

    inline LightCamera build_dir_light_camera_aabb(
        const glm::vec3& sun_dir_ws_norm,
        const AABB& scene_aabb_ws,
        float extra_margin = 10.0f,
        uint32_t shadow_map_resolution = 2048u
    )
    {
        LightCamera lc{};
        lc.dir_ws = glm::normalize(sun_dir_ws_norm);

        glm::vec3 up = (std::abs(lc.dir_ws.y) > 0.95f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);

        const glm::vec3 c = scene_aabb_ws.center();
        const float scene_radius = glm::length(scene_aabb_ws.extent()) + extra_margin;
        lc.pos_ws = c - lc.dir_ws * (scene_radius * 2.0f);
        lc.view = look_at_lh(lc.pos_ws, c, up);

        const glm::vec3 mn = scene_aabb_ws.minv;
        const glm::vec3 mx = scene_aabb_ws.maxv;
        const glm::vec3 corners[8] = {
            {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z},
            {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z}, {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z},
        };

        float l = 1e30f, r = -1e30f;
        float b = 1e30f, t = -1e30f;
        float n = 1e30f, f = -1e30f;
        for (int i = 0; i < 8; ++i)
        {
            const glm::vec4 p_ls = lc.view * glm::vec4(corners[i], 1.0f);
            l = std::min(l, p_ls.x); r = std::max(r, p_ls.x);
            b = std::min(b, p_ls.y); t = std::max(t, p_ls.y);
            n = std::min(n, p_ls.z); f = std::max(f, p_ls.z);
        }

        const float m = extra_margin;
        l -= m; r += m; b -= m; t += m;
        n -= m; f += m;

        // Stabilize directional shadows by snapping the ortho XY center
        // to shadow texel increments in light space.
        if (shadow_map_resolution > 0u)
        {
            const float span_x = std::max(r - l, 1e-5f);
            const float span_y = std::max(t - b, 1e-5f);
            const float inv_res = 1.0f / static_cast<float>(shadow_map_resolution);
            const float texel_x = span_x * inv_res;
            const float texel_y = span_y * inv_res;

            float cx = 0.5f * (l + r);
            float cy = 0.5f * (b + t);

            if (texel_x > 1e-6f) cx = std::floor(cx / texel_x + 0.5f) * texel_x;
            if (texel_y > 1e-6f) cy = std::floor(cy / texel_y + 0.5f) * texel_y;

            const float hx = 0.5f * span_x;
            const float hy = 0.5f * span_y;
            l = cx - hx;
            r = cx + hx;
            b = cy - hy;
            t = cy + hy;
        }

        lc.proj = ortho_lh_no(l, r, b, t, n, f);
        lc.viewproj = lc.proj * lc.view;
        return lc;
    }
}
