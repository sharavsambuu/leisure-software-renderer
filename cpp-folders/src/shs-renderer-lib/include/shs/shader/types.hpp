#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: types.hpp
    МОДУЛЬ: shader
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн shader модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <array>
#include <cstdint>

#include <glm/glm.hpp>

#include "shs/gfx/rt_shadow.hpp"
#include "shs/gfx/rt_types.hpp"
#include "shs/resources/texture.hpp"

namespace shs
{
    constexpr uint32_t SHS_MAX_VARYINGS = 12;
    constexpr uint32_t SHS_MAX_UNIFORM_VECS = 64;
    constexpr uint32_t SHS_MAX_UNIFORM_MATS = 16;

    enum class VaryingSemantic : uint32_t
    {
        WorldPos = 0,
        NormalWS = 1,
        UV0 = 2,
        Color0 = 3,
        TangentWS = 4,
        BitangentWS = 5,
        Custom0 = 6,
        Custom1 = 7,
        Custom2 = 8,
        Custom3 = 9,
        Custom4 = 10,
        Custom5 = 11
    };

    inline constexpr uint32_t varying_bit(uint32_t slot) { return (1u << slot); }

    struct ShaderVertex
    {
        glm::vec3 position{0.0f};
        glm::vec3 normal{0.0f, 1.0f, 0.0f};
        glm::vec2 uv{0.0f};
        glm::vec4 color{1.0f};
        glm::vec4 tangent{1.0f, 0.0f, 0.0f, 1.0f};
    };

    struct VertexOut
    {
        glm::vec4 clip{0.0f, 0.0f, 0.0f, 1.0f};
        std::array<glm::vec4, SHS_MAX_VARYINGS> varyings{};
        uint32_t varying_mask = 0u;

        // Fragment shader-д шууд ашиглагдах үндсэн өгөгдөл.
        glm::vec3 world_pos{0.0f};
        glm::vec3 normal_ws{0.0f, 1.0f, 0.0f};
        glm::vec2 uv{0.0f};
    };

    struct FragmentIn
    {
        std::array<glm::vec4, SHS_MAX_VARYINGS> varyings{};
        uint32_t varying_mask = 0u;

        // Pixel шатны шэйдингт хэрэглэгдэх үндсэн атрибутууд.
        glm::vec3 world_pos{0.0f};
        glm::vec3 normal_ws{0.0f, 1.0f, 0.0f};
        glm::vec2 uv{0.0f};
        float depth01 = 1.0f;
        int px = 0;
        int py = 0;
    };

    struct FragmentOut
    {
        ColorF color{0.0f, 0.0f, 0.0f, 1.0f};
        bool discard = false;
    };

    struct ShaderUniforms
    {
        std::array<glm::vec4, SHS_MAX_UNIFORM_VECS> vec4s{};
        std::array<glm::mat4, SHS_MAX_UNIFORM_MATS> mats{};

        glm::mat4 model{1.0f};
        glm::mat4 viewproj{1.0f};
        glm::vec3 light_dir_ws{0.0f, -1.0f, 0.0f};
        glm::vec3 light_color{1.0f, 1.0f, 1.0f};
        float light_intensity = 1.0f;
        glm::vec3 camera_pos{0.0f};

        glm::vec3 base_color{1.0f, 1.0f, 1.0f};
        float metallic = 0.0f;
        float roughness = 0.6f;
        float ao = 1.0f;
        const Texture2DData* base_color_tex = nullptr;

        const RT_ShadowDepth* shadow_map = nullptr;
        glm::mat4 light_viewproj{1.0f};
        float shadow_bias_const = 0.0008f;
        float shadow_bias_slope = 0.0015f;
        int shadow_pcf_radius = 2;
        float shadow_pcf_step = 1.0f;
        float shadow_strength = 1.0f;
    };

    inline void set_varying(VertexOut& out, VaryingSemantic semantic, const glm::vec4& v)
    {
        const uint32_t i = (uint32_t)semantic;
        out.varyings[i] = v;
        out.varying_mask |= varying_bit(i);
    }

    inline glm::vec4 get_varying(const FragmentIn& in, VaryingSemantic semantic, const glm::vec4& fallback = glm::vec4(0.0f))
    {
        const uint32_t i = (uint32_t)semantic;
        if ((in.varying_mask & varying_bit(i)) == 0u) return fallback;
        return in.varyings[i];
    }

    inline void set_uniform_vec4(ShaderUniforms& u, uint32_t slot, const glm::vec4& v)
    {
        if (slot < SHS_MAX_UNIFORM_VECS) u.vec4s[slot] = v;
    }

    inline glm::vec4 get_uniform_vec4(const ShaderUniforms& u, uint32_t slot, const glm::vec4& fallback = glm::vec4(0.0f))
    {
        if (slot < SHS_MAX_UNIFORM_VECS) return u.vec4s[slot];
        return fallback;
    }

    inline void set_uniform_mat4(ShaderUniforms& u, uint32_t slot, const glm::mat4& m)
    {
        if (slot < SHS_MAX_UNIFORM_MATS) u.mats[slot] = m;
    }

    inline glm::mat4 get_uniform_mat4(const ShaderUniforms& u, uint32_t slot, const glm::mat4& fallback = glm::mat4(1.0f))
    {
        if (slot < SHS_MAX_UNIFORM_MATS) return u.mats[slot];
        return fallback;
    }
}
