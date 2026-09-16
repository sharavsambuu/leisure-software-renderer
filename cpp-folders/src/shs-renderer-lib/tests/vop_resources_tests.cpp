#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/domains/resources/resources.contract.hpp"
#include "shs/domains/resources/resources.reducer.hpp"
#include "shs/domains/pod_test_kit.hpp"

// Headless tests for the resources pod (R5a P3.7: data-type + registry pins).
// Links only shs::renderer-values + glm. Registry API is exercised as-is
// (flagged edge-candidate until R5b); the pins below are its migration spec.
namespace
{
    auto reduce_via_pod = [](shs::resources::ResourcesState& s,
                             std::span<const shs::resources::ResourcesAction> a,
                             const shs::resources::ResourcesReduceInputs& in,
                             std::pmr::vector<shs::resources::ResourcesEvent>& e)
    {
        shs::resources::reduce_resources(s, a, in, e);
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
            shs::resources::ResourcesAction, shs::resources::ResourcesReduceInputs,
            shs::resources::ResourcesEvent>(
            reduce_via_pod, shs::resources::ResourcesState{}, shs::resources::ResourcesReduceInputs{});
    }

    bool test_replay_deterministic()
    {
        const shs::resources::ResourcesState s0{};
        const std::vector<shs::resources::ResourcesAction> none{};
        return shs::pod_test::replay_is_deterministic<shs::resources::ResourcesState,
            shs::resources::ResourcesAction, shs::resources::ResourcesReduceInputs,
            shs::resources::ResourcesEvent>(
            reduce_via_pod, s0,
            std::span<const shs::resources::ResourcesAction>{none.data(), none.size()},
            shs::resources::ResourcesReduceInputs{});
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_data_basics() && ok;
    ok = test_registry_round_trip() && ok;
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
