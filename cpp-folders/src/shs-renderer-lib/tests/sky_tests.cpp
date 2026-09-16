#include <cmath>
#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/domains/sky/sky.contract.hpp"
#include "shs/domains/sky/sky.gateway.hpp"
#include "shs/domains/pod_test_kit.hpp"

// Headless tests for the sky pod (R5a P3.9: procedural-sky pins + identity).
// Links only shs::renderer-values + glm.
namespace
{
    auto run_gateway = [](shs::sky::SkyState& s,
                             std::span<const shs::sky::SkyCommand> a,
                             const shs::sky::SkyContext& in,
                             std::pmr::vector<shs::sky::SkyEvent>& e)
    {
        shs::sky::sky_gateway(s, a, in, e);
    };

    bool near_vec(const glm::vec3& a, const glm::vec3& b, float eps = 1e-5f)
    {
        return std::fabs(a.x - b.x) <= eps && std::fabs(a.y - b.y) <= eps && std::fabs(a.z - b.z) <= eps;
    }

    // Down == horizon exactly (mix weight 0 is exact); sun disk == 15.
    bool test_procedural_known_answers()
    {
        const shs::ProceduralSky sky{};
        if (sky.sample(glm::vec3(0.0f, -1.0f, 0.0f)) != glm::vec3(0.30f, 0.60f, 1.00f)) return false;
        if (!near_vec(sky.sample(glm::vec3(0.0f, 1.0f, 0.0f)), glm::vec3(0.05f, 0.20f, 0.50f))) return false;
        return true;
    }

    // Looking back along the sun dir hits the disk (dot == 1 > 0.9998).
    bool test_sun_disk()
    {
        const glm::vec3 sun = glm::normalize(glm::vec3(0.4668f, -0.3487f, 0.8127f));
        const shs::ProceduralSky sky{sun};
        return sky.sample(-sun) == glm::vec3(15.0f);
    }

    // Sampling is deterministic (double evaluation identical).
    bool test_sample_deterministic()
    {
        const shs::ProceduralSky sky{};
        const glm::vec3 d = glm::normalize(glm::vec3(0.4f, 0.2f, 0.9f));
        return sky.sample(d) == sky.sample(d);
    }

    bool test_identity_stable()
    {
        return shs::pod_test::empty_log_is_stable<shs::sky::SkyState,
            shs::sky::SkyCommand, shs::sky::SkyContext,
            shs::sky::SkyEvent>(
            run_gateway, shs::sky::SkyState{}, shs::sky::SkyContext{});
    }

    bool test_replay_deterministic()
    {
        const shs::sky::SkyState s0{};
        const std::vector<shs::sky::SkyCommand> none{};
        return shs::pod_test::replay_is_deterministic<shs::sky::SkyState,
            shs::sky::SkyCommand, shs::sky::SkyContext,
            shs::sky::SkyEvent>(
            run_gateway, s0,
            std::span<const shs::sky::SkyCommand>{none.data(), none.size()},
            shs::sky::SkyContext{});
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_procedural_known_answers() && ok;
    ok = test_sun_disk() && ok;
    ok = test_sample_deterministic() && ok;
    ok = test_identity_stable() && ok;
    ok = test_replay_deterministic() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[sky-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[sky-tests] all tests passed\n");
    return 0;
}
