#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: builtin_shaders.hpp
    МОДУЛЬ: shader
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн shader модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "shs/frame/frame_params.hpp"
#include "shs/lighting/shadow_sample.hpp"
#include "shs/shader/program.hpp"

namespace shs
{
    inline glm::vec3 srgb_to_linear_rgb(const Color& c)
    {
        const float r = std::pow((float)c.r / 255.0f, 2.2f);
        const float g = std::pow((float)c.g / 255.0f, 2.2f);
        const float b = std::pow((float)c.b / 255.0f, 2.2f);
        return glm::vec3(r, g, b);
    }

    inline glm::vec3 sample_texture2d_bilinear_repeat_linear(const Texture2DData* tex, const glm::vec2& uv)
    {
        if (!tex || !tex->valid()) return glm::vec3(1.0f);
        const float u = uv.x - std::floor(uv.x);
        const float v = uv.y - std::floor(uv.y);

        const float fx = u * (float)(tex->w - 1);
        const float fy = v * (float)(tex->h - 1);
        const int x0 = (int)std::floor(fx);
        const int y0 = (int)std::floor(fy);
        const int x1 = std::min(x0 + 1, tex->w - 1);
        const int y1 = std::min(y0 + 1, tex->h - 1);
        const float tx = fx - (float)x0;
        const float ty = fy - (float)y0;

        const glm::vec3 c00 = srgb_to_linear_rgb(tex->at(x0, y0));
        const glm::vec3 c10 = srgb_to_linear_rgb(tex->at(x1, y0));
        const glm::vec3 c01 = srgb_to_linear_rgb(tex->at(x0, y1));
        const glm::vec3 c11 = srgb_to_linear_rgb(tex->at(x1, y1));
        const glm::vec3 cx0 = glm::mix(c00, c10, tx);
        const glm::vec3 cx1 = glm::mix(c01, c11, tx);
        return glm::mix(cx0, cx1, ty);
    }

    inline glm::vec3 eval_fake_ibl(const glm::vec3& N, const glm::vec3& V, const glm::vec3& base_color, float metallic, float roughness, float ao)
    {
        // LUT/PMREM-гүй нөхцөлд орчны гэрлийг ойролцоолсон хөнгөн IBL.
        const glm::vec3 n = glm::normalize(N);
        const glm::vec3 v = glm::normalize(V);
        const glm::vec3 r = glm::reflect(-v, n);

        const glm::vec3 sky_zenith = glm::vec3(0.32f, 0.46f, 0.72f);
        const glm::vec3 sky_horizon = glm::vec3(0.62f, 0.66f, 0.72f);
        const glm::vec3 ground_tint = glm::vec3(0.16f, 0.15f, 0.14f);

        const float up_n = std::clamp(n.y * 0.5f + 0.5f, 0.0f, 1.0f);
        const float up_r = std::clamp(r.y * 0.5f + 0.5f, 0.0f, 1.0f);
        const glm::vec3 env_n = glm::mix(ground_tint, glm::mix(sky_horizon, sky_zenith, up_n), up_n);
        const glm::vec3 env_r = glm::mix(ground_tint, glm::mix(sky_horizon, sky_zenith, up_r), up_r);

        const float m = std::clamp(metallic, 0.0f, 1.0f);
        const float rgh = std::clamp(roughness, 0.0f, 1.0f);
        const glm::vec3 F0 = glm::mix(glm::vec3(0.04f), glm::max(base_color, glm::vec3(0.0f)), m);
        const float fres = std::pow(1.0f - std::max(0.0f, glm::dot(n, v)), 5.0f);
        const glm::vec3 F = F0 + (glm::vec3(1.0f) - F0) * fres;

        const glm::vec3 kd = (glm::vec3(1.0f) - F) * (1.0f - m);
        // Ambient-ийг хэт өсгөхгүй барьж, plastic/floor гадаргуу цайрахаас сэргийлнэ.
        const glm::vec3 diffuse_ibl = kd * base_color * env_n * 0.12f;
        const float spec_strength = 0.02f + (1.0f - rgh) * 0.18f;
        const glm::vec3 spec_ibl = env_r * F * spec_strength;
        return (diffuse_ibl + spec_ibl) * std::clamp(ao, 0.0f, 1.0f);
    }

    inline VertexOut make_default_vertex_out(const ShaderVertex& vin, const ShaderUniforms& u)
    {
        VertexOut o{};
        const glm::vec4 wp4 = u.model * glm::vec4(vin.position, 1.0f);
        o.world_pos = glm::vec3(wp4);
        o.clip = u.viewproj * wp4;
        glm::mat3 nrm_m = glm::mat3(u.model);
        const float det = glm::determinant(nrm_m);
        if (std::abs(det) > 1e-8f) nrm_m = glm::transpose(glm::inverse(nrm_m));
        o.normal_ws = glm::normalize(nrm_m * vin.normal);
        o.uv = vin.uv;
        set_varying(o, VaryingSemantic::WorldPos, glm::vec4(o.world_pos, 1.0f));
        set_varying(o, VaryingSemantic::NormalWS, glm::vec4(o.normal_ws, 0.0f));
        set_varying(o, VaryingSemantic::UV0, glm::vec4(o.uv, 0.0f, 0.0f));
        set_varying(o, VaryingSemantic::Color0, vin.color);
        return o;
    }

    inline ShaderProgram make_blinn_phong_program()
    {
        ShaderProgram p{};
        p.vs = [](const ShaderVertex& vin, const ShaderUniforms& u) -> VertexOut {
            return make_default_vertex_out(vin, u);
        };
        p.fs = [](const FragmentIn& fin, const ShaderUniforms& u) -> FragmentOut {
            FragmentOut o{};
            const glm::vec3 albedo_tex = sample_texture2d_bilinear_repeat_linear(u.base_color_tex, fin.uv);
            const glm::vec3 albedo = glm::max(u.base_color * albedo_tex, glm::vec3(0.0f));
            const glm::vec3 N = glm::normalize(fin.normal_ws);
            const glm::vec3 L = glm::normalize(-u.light_dir_ws);
            const glm::vec3 V = glm::normalize(u.camera_pos - fin.world_pos);
            const glm::vec3 H = glm::normalize(L + V);

            const float NdotL = std::max(0.0f, glm::dot(N, L));
            const float NdotH = std::max(0.0f, glm::dot(N, H));
            const float rough = std::clamp(u.roughness, 0.0f, 1.0f);
            const float metal = std::clamp(u.metallic, 0.0f, 1.0f);
            const float spec_pow = std::max(4.0f, 8.0f + (1.0f - rough) * 120.0f);
            // Эрчим хүчийг тогтвортой барих normalize хийсэн Blinn-Phong.
            const float spec_norm = (spec_pow + 2.0f) / (2.0f * glm::pi<float>());
            const float spec_f0 = 0.04f + 0.96f * metal;
            const float spec = std::pow(NdotH, spec_pow) * spec_norm * spec_f0 * NdotL;
            const glm::vec3 kd = glm::vec3(1.0f - metal);
            const glm::vec3 diffuse = kd * albedo * (NdotL / glm::pi<float>());
            float shadow_vis = 1.0f;
            if (u.shadow_map && NdotL > 0.0f)
            {
                // Гэрэл объектын ар талд байвал shadow sampling хийх шаардлагагүй.
                ShadowParams sp{};
                sp.light_viewproj = u.light_viewproj;
                sp.bias_const = u.shadow_bias_const;
                sp.bias_slope = u.shadow_bias_slope;
                sp.pcf_radius = std::max(0, u.shadow_pcf_radius);
                sp.pcf_step = std::max(1.0f, u.shadow_pcf_step);
                shadow_vis = shadow_visibility_dir(*u.shadow_map, sp, fin.world_pos, NdotL);
                shadow_vis = glm::mix(1.0f, shadow_vis, std::clamp(u.shadow_strength, 0.0f, 1.0f));
            }
            const glm::vec3 direct = (diffuse + glm::vec3(spec)) * u.light_color * u.light_intensity * shadow_vis;
            const glm::vec3 ibl = eval_fake_ibl(N, V, albedo, u.metallic, u.roughness, u.ao);
            const glm::vec3 c = direct + ibl;

            o.color = ColorF{c.r, c.g, c.b, 1.0f};
            return o;
        };
        return p;
    }

    inline ShaderProgram make_pbr_mr_program()
    {
        ShaderProgram p{};
        p.vs = [](const ShaderVertex& vin, const ShaderUniforms& u) -> VertexOut {
            return make_default_vertex_out(vin, u);
        };
        p.fs = [](const FragmentIn& fin, const ShaderUniforms& u) -> FragmentOut {
            FragmentOut o{};
            const glm::vec3 albedo_tex = sample_texture2d_bilinear_repeat_linear(u.base_color_tex, fin.uv);
            const glm::vec3 N = glm::normalize(fin.normal_ws);
            const glm::vec3 V = glm::normalize(u.camera_pos - fin.world_pos);
            const glm::vec3 L = glm::normalize(-u.light_dir_ws);
            const glm::vec3 H = glm::normalize(V + L);

            const float NdotL = std::max(0.0f, glm::dot(N, L));
            const float NdotV = std::max(0.0f, glm::dot(N, V));
            const float NdotH = std::max(0.0f, glm::dot(N, H));
            const float VdotH = std::max(0.0f, glm::dot(V, H));
            const float rough = std::clamp(u.roughness, 0.04f, 1.0f);
            const float metal = std::clamp(u.metallic, 0.0f, 1.0f);
            const glm::vec3 albedo = glm::max(u.base_color * albedo_tex, glm::vec3(0.0f));
            const glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metal);

            const float a = rough * rough;
            const float a2 = a * a;
            const float denomD = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
            const float D = a2 / (glm::pi<float>() * denomD * denomD + 1e-7f);

            auto smith_ggx_g1 = [a](float ndotx) {
                const float k = ((a + 1.0f) * (a + 1.0f)) * 0.125f;
                return ndotx / (ndotx * (1.0f - k) + k + 1e-7f);
            };
            const float G = smith_ggx_g1(NdotV) * smith_ggx_g1(NdotL);

            const glm::vec3 F = F0 + (glm::vec3(1.0f) - F0) * std::pow(1.0f - VdotH, 5.0f);
            const glm::vec3 spec = (D * G) * F / std::max(4.0f * NdotL * NdotV, 1e-6f);

            const glm::vec3 kd = (glm::vec3(1.0f) - F) * (1.0f - metal);
            const glm::vec3 diff = kd * albedo * (1.0f / glm::pi<float>());
            const glm::vec3 radiance = u.light_color * u.light_intensity;
            float shadow_vis = 1.0f;
            if (u.shadow_map && NdotL > 0.0f)
            {
                // Direct lighting үүсэхгүй нөхцөлд shadow fetch хийлгүй skip.
                ShadowParams sp{};
                sp.light_viewproj = u.light_viewproj;
                sp.bias_const = u.shadow_bias_const;
                sp.bias_slope = u.shadow_bias_slope;
                sp.pcf_radius = std::max(0, u.shadow_pcf_radius);
                sp.pcf_step = std::max(1.0f, u.shadow_pcf_step);
                shadow_vis = shadow_visibility_dir(*u.shadow_map, sp, fin.world_pos, NdotL);
                shadow_vis = glm::mix(1.0f, shadow_vis, std::clamp(u.shadow_strength, 0.0f, 1.0f));
            }
            const glm::vec3 direct = (NdotL > 0.0f && NdotV > 0.0f) ? ((diff + spec) * radiance * NdotL * shadow_vis) : glm::vec3(0.0f);
            const glm::vec3 ibl = eval_fake_ibl(N, V, albedo, metal, rough, u.ao);
            const glm::vec3 c = direct + ibl;
            o.color = ColorF{c.r, c.g, c.b, 1.0f};
            return o;
        };
        return p;
    }

    inline ShaderProgram make_lit_shader_program()
    {
        return make_pbr_mr_program();
    }

    inline ShaderProgram make_debug_view_shader_program(DebugViewMode mode)
    {
        ShaderProgram p = make_lit_shader_program();

        p.fs = [mode](const FragmentIn& fin, const ShaderUniforms& u) -> FragmentOut {
            FragmentOut o{};
            if (mode == DebugViewMode::Albedo)
            {
                o.color = ColorF{u.base_color.r, u.base_color.g, u.base_color.b, 1.0f};
                return o;
            }
            if (mode == DebugViewMode::Normal)
            {
                const glm::vec3 n = glm::normalize(fin.normal_ws) * 0.5f + glm::vec3(0.5f);
                o.color = ColorF{n.r, n.g, n.b, 1.0f};
                return o;
            }
            // Depth debug: [0,1] мужид near=хар, far=цагаан.
            const float d = std::clamp(fin.depth01, 0.0f, 1.0f);
            o.color = ColorF{d, d, d, 1.0f};
            return o;
        };

        return p;
    }
}
