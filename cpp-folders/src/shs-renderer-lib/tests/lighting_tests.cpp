#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/lighting/lighting.contract.hpp"
#include "shs/lighting/lighting.gateway.hpp"
#include "shs/core/testing/pod_test_kit.hpp"
#include "identity_step_test.hpp"

// Headless tests for the lighting pod (R4 P3.5: shading-term pins + identity).
// Links only shs::renderer-values + glm.
namespace
{
    auto run_gateway = [](shs::lighting::LightingState& s,
                             std::span<const shs::lighting::LightingCommand> a,
                             const shs::lighting::LightingContext& in,
                             std::pmr::vector<shs::lighting::LightingEvent>& e)
    {
        shs::lighting::lighting_gateway(s, a, in, e);
    };

    // Facing light -> 1, perpendicular -> 0, behind -> clamped 0.
    bool test_lambert_known_answers()
    {
        if (shs::lambert_diffuse(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f)) != 1.0f) return false;
        if (shs::lambert_diffuse(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f)) != 0.0f) return false;
        return shs::lambert_diffuse(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, -1.0f)) == 0.0f;
    }

    // Composition matches the rung-08 pair formula exactly.
    bool test_shade_composition()
    {
        const glm::vec3 out = shs::shade_lambert(glm::vec3(0.85f, 0.87f, 0.90f), 0.5f, 0.08f);
        return out == glm::vec3(0.85f * 0.5f + 0.08f, 0.87f * 0.5f + 0.08f, 0.90f * 0.5f + 0.08f);
    }

    bool test_identity_stable()
    {
        return shs::pod_test::empty_log_is_stable<shs::lighting::LightingState,
            shs::lighting::LightingCommand, shs::lighting::LightingContext,
            shs::lighting::LightingEvent>(
            run_gateway, shs::lighting::LightingState{}, shs::lighting::LightingContext{});
    }

    bool test_replay_deterministic()
    {
        const shs::lighting::LightingState s0{};
        const std::vector<shs::lighting::LightingCommand> none{};
        return shs::pod_test::replay_is_deterministic<shs::lighting::LightingState,
            shs::lighting::LightingCommand, shs::lighting::LightingContext,
            shs::lighting::LightingEvent>(
            run_gateway, s0,
            std::span<const shs::lighting::LightingCommand>{none.data(), none.size()},
            shs::lighting::LightingContext{});
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_lambert_known_answers() && ok;
    ok = test_shade_composition() && ok;
    ok = identity_step_summary<shs::lighting::LightingState,
        shs::lighting::LightingCommand, shs::lighting::LightingContext,
        shs::lighting::LightingEvent, shs::lighting::LightingStep>(shs::lighting::lighting_gateway) && ok;
    ok = test_identity_stable() && ok;
    ok = test_replay_deterministic() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[lighting-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[lighting-tests] all tests passed\n");
    return 0;
}
