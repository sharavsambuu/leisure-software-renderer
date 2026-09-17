#pragma once

/*
    SHS RENDERER SAN

    FILE: tangent_frame.hpp
    MODULE: domains/geometry
    PURPOSE: The rung-08 ingested operator (R4 P3.4): tangent-space frame
             construction + normal-map decode/perturb as pure functions.
             The tier1 pair proves the math cross-backend; this header makes it
             library truth (Slang side mirrors the formulas, pinned by parity).
*/

#include <glm/glm.hpp>

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace geometry
    {
    struct TangentFrame
    {
        glm::vec3 tangent{1.0f, 0.0f, 0.0f};
        glm::vec3 bitangent{0.0f, 1.0f, 0.0f};
        glm::vec3 normal{0.0f, 0.0f, 1.0f};

        bool operator==(const TangentFrame&) const = default;
    };

    // Frame-from-normal construction (for unauthored tangents): T = normalize(cross(up, N)).
    inline TangentFrame compute_tangent_frame(const glm::vec3& n)
    {
        TangentFrame f{};
        f.normal    = glm::normalize(n);
        f.tangent   = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), f.normal));
        f.bitangent = glm::cross(f.normal, f.tangent);
        return f;
    }

    // Byte-normalized texel (0..1) -> tangent-space normal (-1..1).
    inline glm::vec3 decode_normal_texel(float r, float g, float b)
    {
        return glm::vec3(r * 2.0f - 1.0f, g * 2.0f - 1.0f, b * 2.0f - 1.0f);
    }

    // Perturb the geometric normal by a tangent-space sample.
    inline glm::vec3 perturb_normal(const TangentFrame& f, const glm::vec3& n_tangent)
    {
        return glm::normalize(f.tangent * n_tangent.x + f.bitangent * n_tangent.y + f.normal * n_tangent.z);
    }

    } // inline namespace geometry
} // namespace shs
