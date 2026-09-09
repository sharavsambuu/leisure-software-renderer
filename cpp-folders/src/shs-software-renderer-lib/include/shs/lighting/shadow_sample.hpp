#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: shadow_sample.hpp
    МОДУЛЬ: lighting
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн lighting модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/

#include <glm/glm.hpp>
#include <algorithm>
#include <shs/gfx/rt_shadow.hpp>

namespace shs {

struct ShadowParams {
    glm::mat4 light_viewproj{1.0f};

    // Bias (slope-scale + constant)
    float bias_const = 0.0008f;
    float bias_slope = 0.0015f;

    // PCF
    int   pcf_radius = 1;     // 0 = hard shadow, 1 = 3x3, 2 = 5x5
    float pcf_step   = 1.0f;  // in texels
};

// world position -> (u,v,depth) in shadow space
inline bool shadow_project_uvz(
    const glm::mat4& light_vp,
    const glm::vec3& pos_ws,
    float& out_u,
    float& out_v,
    float& out_z
){
    glm::vec4 p = light_vp * glm::vec4(pos_ws, 1.0f);
    if (std::abs(p.w) < 1e-8f) return false;

    glm::vec3 ndc = glm::vec3(p) / p.w;        // [-1,1] range
    out_u = ndc.x * 0.5f + 0.5f;
    out_v = ndc.y * 0.5f + 0.5f;
    out_z = ndc.z * 0.5f + 0.5f;               // map [-1,1] -> [0,1]
    return true;
}

inline float shadow_bias(
    float ndotl,
    float bias_const,
    float bias_slope
){
    // ndotl бага үед илүү bias (acne багасгана)
    const float slope = (1.0f - std::clamp(ndotl, 0.0f, 1.0f));
    return bias_const + bias_slope * slope;
}

inline float shadow_fetch_depth_clamped(const RT_ShadowDepth& sm, int x, int y){
    x = std::clamp(x, 0, sm.w - 1);
    y = std::clamp(y, 0, sm.h - 1);
    return sm.at(x,y);
}

// returns visibility in [0..1] (1 = lit, 0 = fully shadowed)
inline float shadow_visibility_dir(
    const RT_ShadowDepth& sm,
    const ShadowParams& sp,
    const glm::vec3& pos_ws,
    float ndotl
){
    float u,v,z;
    if (!shadow_project_uvz(sp.light_viewproj, pos_ws, u, v, z)) return 1.0f;

    // outside shadow map -> treat as lit
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) return 1.0f;

    const float bias   = shadow_bias(ndotl, sp.bias_const, sp.bias_slope);
    const float z_test = z - bias;

    const float fx = u * (float)(sm.w - 1);
    const float fy = v * (float)(sm.h - 1);
    const int   cx = (int)std::round(fx);
    const int   cy = (int)std::round(fy);

    const int r = std::max(0, sp.pcf_radius);
    if (r == 0){
        const float z_ref = shadow_fetch_depth_clamped(sm, cx, cy);
        return (z_test <= z_ref) ? 1.0f : 0.0f;
    }

    const int step = std::max(1, (int)std::round(sp.pcf_step));
    int count = 0;
    int lit = 0;

    for(int oy=-r; oy<=r; oy++){
        for(int ox=-r; ox<=r; ox++){
            const float z_ref = shadow_fetch_depth_clamped(sm, cx + ox*step, cy + oy*step);
            lit += (z_test <= z_ref) ? 1 : 0;
            count++;
        }
    }

    return (count > 0) ? (float)lit / (float)count : 1.0f;
}

} // namespace shs
