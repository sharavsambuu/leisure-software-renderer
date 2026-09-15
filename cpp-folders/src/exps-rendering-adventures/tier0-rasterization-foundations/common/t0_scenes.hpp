#pragma once

/*
    tier0 demo scenes — single source of vertex data shared by the *_sw and
    *_vk binaries of every tier0 demo, so the two realizations render the
    identical geometry (parity comparison, exp-plumbing style).

    Vertex layout is frozen across tier0: pos(3f) + color(4f) + uv(2f),
    matching the Slang VSInput (POSITION=0, COLOR0=1, TEXCOORD0=2).

    Push-constant block mirrors the Slang PushConstants
    (float4x4 mvp; float4 params; 80 bytes). GLM stores column-major, and the
    slangc pin -matrix-layout-column-major matches that, so raw memcpy works.
    Depth is Vulkan-style [0, 1]: demos define GLM_FORCE_DEPTH_ZERO_TO_ONE.
*/

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#error "tier0 demos must define GLM_FORCE_DEPTH_ZERO_TO_ONE (Vulkan NDC depth pin)"
#endif

namespace adventures
{
    struct T0Vertex
    {
        float pos[3];
        float col[4];
        float uv[2];
    };

    struct T0Push
    {
        float mvp[16];
        float params[4];
    };

    inline T0Vertex t0_vertex(const glm::vec3& p, const glm::vec4& c, const glm::vec2& uv = glm::vec2(0.0f))
    {
        return T0Vertex{ { p.x, p.y, p.z }, { c.r, c.g, c.b, c.a }, { uv.x, uv.y } };
    }

    inline T0Push make_push(const glm::mat4& mvp, const glm::vec4& params = glm::vec4(0.0f))
    {
        T0Push out{};
        static_assert(sizeof(out.mvp) == sizeof(glm::mat4), "mat4 layout drift");
        static_assert(sizeof(out.params) == sizeof(glm::vec4), "vec4 layout drift");
        __builtin_memcpy(out.mvp, &mvp, sizeof(mvp));
        __builtin_memcpy(out.params, &params, sizeof(params));
        return out;
    }

    // SPIR-V push constant member is RowMajor std430 (verified via spirv-dis);
    // decoding bytes as row i = b[4i..4i+3], the bytes of the intended matrix
    // M must be uploaded as M^T so that decode == M (plain column-major bytes
    // decode as M^T).
    inline T0Push make_push_row_major(const glm::mat4& mvp, const glm::vec4& params = glm::vec4(0.0f))
    {
        // RowMajor decode of plain GLM bytes yields M^T, which flings the cubes
        // offscreen (verified); uploading M^T makes the decode == M.
        return make_push(glm::transpose(mvp), params);
    }

    // --- demo 01 — single triangle, per-vertex colors, drawn directly in clip space.
    inline std::vector<T0Vertex> scene_tri_barycentric()
    {
        return {
            t0_vertex(glm::vec3(-0.75f, -0.70f, 0.5f), glm::vec4(1.0f, 0.20f, 0.20f, 1.0f), glm::vec2(0.0f, 0.0f)),
            t0_vertex(glm::vec3( 0.75f, -0.70f, 0.5f), glm::vec4(0.20f, 1.0f, 0.20f, 1.0f), glm::vec2(1.0f, 0.0f)),
            t0_vertex(glm::vec3( 0.00f,  0.85f, 0.5f), glm::vec4(0.25f, 0.35f, 1.0f, 1.0f), glm::vec2(0.5f, 1.0f)),
        };
    }



    // --- demo 02 — unit cube (36 verts), per-face colors, centered at origin.
    inline std::vector<T0Vertex> scene_projection_cube()
    {
        const glm::vec3 p[8] = {
            { -0.5f, -0.5f, -0.5f }, {  0.5f, -0.5f, -0.5f },
            {  0.5f,  0.5f, -0.5f }, { -0.5f,  0.5f, -0.5f },
            { -0.5f, -0.5f,  0.5f }, {  0.5f, -0.5f,  0.5f },
            {  0.5f,  0.5f,  0.5f }, { -0.5f,  0.5f,  0.5f },
        };
        const int faces[6][4] = {
            { 4, 5, 6, 7 }, // +Z front  (red)
            { 1, 0, 3, 2 }, // -Z back   (orange)
            { 5, 1, 2, 6 }, // +X right  (green)
            { 0, 4, 7, 3 }, // -X left   (yellow)
            { 7, 6, 2, 3 }, // +Y top    (blue)
            { 0, 1, 5, 4 }, // -Y bottom (magenta)
        };
        const glm::vec3 face_cols[6] = {
            { 0.90f, 0.20f, 0.20f }, { 0.95f, 0.55f, 0.10f }, { 0.20f, 0.75f, 0.30f },
            { 0.90f, 0.85f, 0.20f }, { 0.25f, 0.45f, 0.95f }, { 0.80f, 0.20f, 0.80f },
        };
        std::vector<T0Vertex> out;
        out.reserve(36);
        for (int f = 0; f < 6; ++f)
        {
            const int*      q     = faces[f];
            const glm::vec2 uv[4] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
            out.push_back(t0_vertex(p[q[0]], glm::vec4(face_cols[f], 1.0f), uv[0]));
            out.push_back(t0_vertex(p[q[1]], glm::vec4(face_cols[f], 1.0f), uv[1]));
            out.push_back(t0_vertex(p[q[2]], glm::vec4(face_cols[f], 1.0f), uv[2]));
            out.push_back(t0_vertex(p[q[3]], glm::vec4(face_cols[f], 1.0f), uv[3]));
        }
        return out;
    }

    // --- demo 03 — near red triangle, far blue triangle (overlap), front translucent quad.
    inline std::vector<T0Vertex> scene_depth_blend()
    {
        std::vector<T0Vertex> out;
        out.push_back(t0_vertex(glm::vec3(-0.6f, -0.6f, 0.20f), glm::vec4(0.95f, 0.15f, 0.15f, 1.0f)));
        out.push_back(t0_vertex(glm::vec3( 0.6f, -0.6f, 0.20f), glm::vec4(0.95f, 0.15f, 0.15f, 1.0f)));
        out.push_back(t0_vertex(glm::vec3( 0.0f,  0.7f, 0.20f), glm::vec4(0.95f, 0.15f, 0.15f, 1.0f)));
        out.push_back(t0_vertex(glm::vec3(-0.9f, -0.4f, 0.70f), glm::vec4(0.15f, 0.30f, 0.95f, 1.0f)));
        out.push_back(t0_vertex(glm::vec3( 0.9f, -0.4f, 0.70f), glm::vec4(0.15f, 0.30f, 0.95f, 1.0f)));
        out.push_back(t0_vertex(glm::vec3( 0.0f,  0.6f, 0.70f), glm::vec4(0.15f, 0.30f, 0.95f, 1.0f)));
        const glm::vec3 q[6] = {
            { -0.45f, -0.85f, 0.05f }, {  0.45f, -0.85f, 0.05f }, {  0.45f,  0.15f, 0.05f },
            { -0.45f, -0.85f, 0.05f }, {  0.45f,  0.15f, 0.05f }, { -0.45f,  0.15f, 0.05f },
        };
        for (const auto& v : q)
        {
            out.push_back(t0_vertex(v, glm::vec4(0.20f, 0.90f, 0.35f, 0.55f)));
        }
        return out;
    }

    // --- demo 04 — two full-viewport quads, identical UVs (0..8, wrapping);
    // left half shows NEAREST filtering, right half LINEAR.
    inline std::vector<T0Vertex> scene_texture_quads()
    {
        std::vector<T0Vertex> out;
        const glm::vec3 left[6] = {
            { -0.95f, -0.60f, 0.5f }, { -0.05f, -0.60f, 0.5f }, { -0.05f,  0.60f, 0.5f },
            { -0.95f, -0.60f, 0.5f }, { -0.05f,  0.60f, 0.5f }, { -0.95f,  0.60f, 0.5f },
        };
        const glm::vec3 right[6] = {
            { 0.05f, -0.6f, 0.5f }, { 0.95f, -0.6f, 0.5f }, { 0.95f,  0.6f, 0.5f },
            { 0.05f, -0.6f, 0.5f }, { 0.95f,  0.6f, 0.5f }, { 0.05f,  0.6f, 0.5f },
        };
        const glm::vec2 uv[6] = {
            { 0, 0 }, { 8, 0 }, { 8, 8 }, // uv > 1 exercises the wrap mode
            { 0, 0 }, { 8, 8 }, { 0, 8 },
        };
        for (int i = 0; i < 6; ++i)
        {
            out.push_back(t0_vertex(left[i], glm::vec4(1.0f), uv[i]));
        }
        for (int i = 0; i < 6; ++i)
        {
            out.push_back(t0_vertex(right[i], glm::vec4(1.0f), uv[i]));
        }
        return out;
    }

    // --- demo 05 — center triangle (stencil writer) + covering quad (stencil tester).
    inline std::vector<T0Vertex> scene_stencil_triangle()
    {
        return {
            t0_vertex(glm::vec3( 0.00f,  0.80f, 0.5f), glm::vec4(0.95f, 0.75f, 0.15f, 1.0f)),
            t0_vertex(glm::vec3(-0.80f, -0.75f, 0.5f), glm::vec4(0.95f, 0.75f, 0.15f, 1.0f)),
            t0_vertex(glm::vec3( 0.80f, -0.75f, 0.5f), glm::vec4(0.95f, 0.75f, 0.15f, 1.0f)),
        };
    }

    inline std::vector<T0Vertex> scene_stencil_quad()
    {
        std::vector<T0Vertex> out;
        const glm::vec3 q[6] = {
            { -0.95f, -0.90f, 0.5f }, { 0.95f, -0.90f, 0.5f }, { 0.95f,  0.90f, 0.5f },
            { -0.95f, -0.90f, 0.5f }, { 0.95f,  0.90f, 0.5f }, { -0.95f, 0.90f, 0.5f },
        };
        for (const auto& v : q)
        {
            out.push_back(t0_vertex(v, glm::vec4(0.55f, 0.15f, 0.85f, 1.0f)));
        }
        return out;
    }
}
