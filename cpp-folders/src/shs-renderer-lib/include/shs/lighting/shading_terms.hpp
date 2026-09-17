#pragma once

/*
    SHS RENDERER SAN

    FILE: shading_terms.hpp
    MODULE: domains/lighting
    PURPOSE: The rung-08 ingested operator (R4 P3.5): Lambert diffuse term +
             ambient composition as pure functions. The tier1 pair proves the
             formulas cross-backend; this header makes them library truth.
*/

#include <glm/glm.hpp>

namespace shs
{
    // Lambert diffuse factor: clamped cosine between normal and light dir.
    inline float lambert_diffuse(const glm::vec3& n, const glm::vec3& l)
    {
        return glm::max(glm::dot(n, l), 0.0f);
    }

    // Final composition used by the rung-08 pair (and its Slang mirror).
    inline glm::vec3 shade_lambert(const glm::vec3& albedo, float diffuse, float ambient)
    {
        return albedo * diffuse + glm::vec3(ambient);
    }
} // namespace shs
