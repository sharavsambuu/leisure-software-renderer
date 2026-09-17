#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/resources/resources.contract.hpp"
#include "shs/resources/resources.gateway.hpp"
#include "shs/core/testing/pod_test_kit.hpp"
#include "identity_step_test.hpp"

// Headless tests for the resources pod (R5a P3.7: data-type + registry pins).
// Links only shs::renderer-values + glm. Registry API is exercised as-is
// (flagged edge-candidate until R5b); the pins below are its migration spec.
namespace
{
    auto run_gateway = [](shs::resources::ResourcesState& s,
                             std::span<const shs::resources::ResourcesCommand> a,
                             const shs::resources::ResourcesContext& in,
                             std::pmr::vector<shs::resources::ResourcesEvent>& e)
    {
        shs::resources::resources_gateway(s, a, in, e);
    };

    // Empty assets are invalid/empty; handles start null.
    bool test_data_basics()
    {
        const shs::Texture2DData tex{};
        if (tex.valid()) return false;
        const shs::MeshData mesh{};
        if (!mesh.empty()) return false;
        return true;
    }

    // Registry round-trip: add -> get -> same bytes; bad handle -> null.
    bool test_registry_round_trip()
    {
        shs::ResourceRegistry reg{};
        shs::Texture2DData tex{};
        tex.w = 2;
        tex.h = 2;
        if (reg.get_texture(0) != nullptr) return false;
        const shs::TextureAssetHandle h = reg.add_texture(std::move(tex), "t");
        if (h == 0) return false;
        const shs::Texture2DData* back = reg.get_texture(h);
        if (!back || back->w != 2 || back->h != 2) return false;
        return reg.get_texture(h + 100) == nullptr;
    }

    bool test_identity_stable()
    {
        return shs::pod_test::empty_log_is_stable<shs::resources::ResourcesState,
            shs::resources::ResourcesCommand, shs::resources::ResourcesContext,
            shs::resources::ResourcesEvent>(
            run_gateway, shs::resources::ResourcesState{}, shs::resources::ResourcesContext{});
    }

    bool test_replay_deterministic()
    {
        const shs::resources::ResourcesState s0{};
        const std::vector<shs::resources::ResourcesCommand> none{};
        return shs::pod_test::replay_is_deterministic<shs::resources::ResourcesState,
            shs::resources::ResourcesCommand, shs::resources::ResourcesContext,
            shs::resources::ResourcesEvent>(
            run_gateway, s0,
            std::span<const shs::resources::ResourcesCommand>{none.data(), none.size()},
            shs::resources::ResourcesContext{});
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_data_basics() && ok;
    ok = test_registry_round_trip() && ok;
    ok = identity_step_summary<shs::resources::ResourcesState,
        shs::resources::ResourcesCommand, shs::resources::ResourcesContext,
        shs::resources::ResourcesEvent, shs::resources::ResourcesStep>(shs::resources::resources_gateway) && ok;
    ok = test_identity_stable() && ok;
    ok = test_replay_deterministic() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[resources-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[resources-tests] all tests passed\n");
    return 0;
}
