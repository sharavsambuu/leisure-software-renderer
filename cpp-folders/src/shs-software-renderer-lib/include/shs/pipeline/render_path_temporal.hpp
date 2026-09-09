#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_temporal.hpp
    MODULE: pipeline
    PURPOSE: Shared temporal state/jitter helpers for render-path hosts.
*/


#include <algorithm>
#include <cstdint>

#include <glm/glm.hpp>

namespace shs
{
    struct RenderPathTemporalSettings
    {
        bool jitter_enabled = false;
        float jitter_scale = 1.0f;
        bool accumulation_enabled = false;
        float history_blend = 0.12f;
    };

    struct RenderPathTemporalFrameState
    {
        uint64_t frame_index = 0u;
        glm::vec2 jitter_ndc{0.0f};
        glm::vec2 jitter_pixels{0.0f};
        glm::mat4 previous_view_proj{1.0f};
        glm::mat4 current_view_proj{1.0f};
    };

    inline float halton(uint64_t index, uint32_t base)
    {
        if (base < 2u) return 0.0f;
        float f = 1.0f;
        float r = 0.0f;
        uint64_t i = index;
        while (i > 0u)
        {
            f = f / static_cast<float>(base);
            r += f * static_cast<float>(i % base);
            i /= base;
        }
        return r;
    }

    inline glm::vec2 halton_2_3(uint64_t frame_index)
    {
        // +1 avoids zero jitter on frame 0 and produces stable cycling.
        const uint64_t idx = frame_index + 1u;
        return glm::vec2(halton(idx, 2u), halton(idx, 3u));
    }

    inline glm::vec2 compute_taa_jitter_ndc(
        uint64_t frame_index,
        uint32_t width,
        uint32_t height,
        float jitter_scale = 1.0f)
    {
        if (width == 0u || height == 0u) return glm::vec2(0.0f);

        const glm::vec2 h = halton_2_3(frame_index) - glm::vec2(0.5f);
        const glm::vec2 pixel = h * std::max(jitter_scale, 0.0f);
        return glm::vec2(
            (2.0f * pixel.x) / static_cast<float>(width),
            (2.0f * pixel.y) / static_cast<float>(height));
    }

    inline glm::mat4 add_projection_jitter_ndc(const glm::mat4& proj, const glm::vec2& jitter_ndc)
    {
        // Standard projection jitter: bias clip-space x/y before perspective divide.
        glm::mat4 out = proj;
        out[2][0] += jitter_ndc.x;
        out[2][1] += jitter_ndc.y;
        return out;
    }
}
