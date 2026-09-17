#include <cmath>
#include <cstdio>

#include "shs/sky/sky.contract.hpp"

// Headless tests for the sky pod (R5a P3.9: procedural-sky pins). The identity
// gateway scaffolding was retired (migration step 4.5): the sky pod owns pure
// sampling only until real intents arrive (empty command vocabulary by law
// §6.1). Links only shs::renderer-values + glm.
namespace
{
    bool near_vec(const glm::vec3& a, const glm::vec3& b, float eps = 1e-5f)
    {
        return std::fabs(a.x - b.x) <= eps && std::fabs(a.y - b.y) <= eps && std::fabs(a.z - b.z) <= eps;
    }

    // Down == horizon exactly (mix weight 0 is exact); sun disk == 15.
    bool test_procedural_known_answers()
    {
        const shs::sky::ProceduralSky sky{};
        if (sky.sample(glm::vec3(0.0f, -1.0f, 0.0f)) != glm::vec3(0.30f, 0.60f, 1.00f)) return false;
        if (!near_vec(sky.sample(glm::vec3(0.0f, 1.0f, 0.0f)), glm::vec3(0.05f, 0.20f, 0.50f))) return false;
        return true;
    }

    // Looking back along the sun dir hits the disk (dot == 1 > 0.9998).
    bool test_sun_disk()
    {
        const glm::vec3 sun = glm::normalize(glm::vec3(0.4668f, -0.3487f, 0.8127f));
        const shs::sky::ProceduralSky sky{sun};
        return sky.sample(-sun) == glm::vec3(15.0f);
    }

    // Sampling is deterministic (double evaluation identical).
    bool test_sample_deterministic()
    {
        const shs::sky::ProceduralSky sky{};
        const glm::vec3 d = glm::normalize(glm::vec3(0.4f, 0.2f, 0.9f));
        return sky.sample(d) == sky.sample(d);
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_procedural_known_answers() && ok;
    ok = test_sun_disk() && ok;
    ok = test_sample_deterministic() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[sky-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[sky-tests] all tests passed\n");
    return 0;
}
