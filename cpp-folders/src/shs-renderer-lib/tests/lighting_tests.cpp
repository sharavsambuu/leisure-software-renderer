#include <cstdio>

#include "shs/lighting/lighting.contract.hpp"

// Headless tests for the lighting pod (R4 P3.5: shading-term pins). The
// identity gateway scaffolding was retired (migration step 4.5): the lighting
// pod owns pure shading terms only until real intents arrive (empty command
// vocabulary by law §6.1). Links only shs::renderer-values + glm.
namespace
{
    // Facing light -> 1, perpendicular -> 0, behind -> clamped 0.
    bool test_lambert_known_answers()
    {
        if (shs::lighting::lambert_diffuse(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f)) != 1.0f) return false;
        if (shs::lighting::lambert_diffuse(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f)) != 0.0f) return false;
        return shs::lighting::lambert_diffuse(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, -1.0f)) == 0.0f;
    }

    // Composition matches the rung-08 pair formula exactly.
    bool test_shade_composition()
    {
        const glm::vec3 out = shs::lighting::shade_lambert(glm::vec3(0.85f, 0.87f, 0.90f), 0.5f, 0.08f);
        return out == glm::vec3(0.85f * 0.5f + 0.08f, 0.87f * 0.5f + 0.08f, 0.90f * 0.5f + 0.08f);
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_lambert_known_answers() && ok;
    ok = test_shade_composition() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[lighting-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[lighting-tests] all tests passed\n");
    return 0;
}
