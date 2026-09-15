#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pbr_sun_shadow.hpp
    МОДУЛЬ: lighting
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн lighting модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/

#include <glm/glm.hpp>
#include <algorithm>

#include <shs/lighting/shadow_sample.hpp>
#include <shs/gfx/rt_shadow.hpp>

namespace shs {

// PBR BRDF-ийн туслах функцууд.
inline glm::vec3 fresnel_schlick(float cosTheta, const glm::vec3& F0){
    return F0 + (1.0f - F0) * std::pow(std::clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

inline float ndf_ggx(float NdotH, float rough){
    const float a  = rough * rough;
    const float a2 = a * a;
    const float d  = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    return a2 / std::max(1e-6f, 3.14159265f * d * d);
}

inline float g_schlick_ggx(float NdotV, float rough){
    const float r = rough + 1.0f;
    const float k = (r * r) / 8.0f;
    return NdotV / std::max(1e-6f, NdotV * (1.0f - k) + k);
}

inline float g_smith(float NdotV, float NdotL, float rough){
    return g_schlick_ggx(NdotV, rough) * g_schlick_ggx(NdotL, rough);
}

// Direct sun PBR (Cook-Torrance) + shadow visibility
inline glm::vec3 pbr_direct_sun_shadowed(
    const glm::vec3& pos_ws,
    const glm::vec3& N,
    const glm::vec3& V,                 // view dir (from pos to camera) normalized
    const glm::vec3& sun_dir_ws,        // direction FROM sun TO scene
    const glm::vec3& sun_radiance,      // linear HDR radiance
    const glm::vec3& albedo,            // linear
    float metal,
    float rough,
    const RT_ShadowDepth* shadow_map,   // can be null
    const ShadowParams* shadow_params   // can be null
){
    // SHS convention: sun_dir_ws points from light toward scene.
    const glm::vec3 L = glm::normalize(-sun_dir_ws);

    const float NdotL = std::max(0.0f, glm::dot(N, L));
    if (NdotL <= 0.0f) return glm::vec3(0.0f);

    const float NdotV = std::max(0.0f, glm::dot(N, V));
    const glm::vec3 H = glm::normalize(V + L);

    const float NdotH = std::max(0.0f, glm::dot(N, H));
    const float VdotH = std::max(0.0f, glm::dot(V, H));

    // F0
    glm::vec3 F0 = glm::vec3(0.04f);
    F0           = F0 * (1.0f - metal) + albedo * metal;

    const float     D = ndf_ggx(NdotH, rough);
    const float     G = g_smith(NdotV, NdotL, rough);
    const glm::vec3 F = fresnel_schlick(VdotH, F0);

    const glm::vec3 kS = F;
    const glm::vec3 kD = (glm::vec3(1.0f) - kS) * (1.0f - metal);

    const glm::vec3 spec_num = D * G * F;
    const float spec_den     = std::max(1e-6f, 4.0f * NdotV * NdotL);
    const glm::vec3 spec     = spec_num / spec_den;

    const glm::vec3 diff = (kD * albedo) * (1.0f / 3.14159265f);

    // Shadow visibility (only for direct)
    float vis = 1.0f;
    if (shadow_map && shadow_params){
        vis = shadow_visibility_dir(*shadow_map, *shadow_params, pos_ws, NdotL);
    }

    return (diff + spec) * sun_radiance * (NdotL * vis);
}

} // namespace shs
