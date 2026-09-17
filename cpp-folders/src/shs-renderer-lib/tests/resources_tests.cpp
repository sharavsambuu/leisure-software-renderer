#include <cstdio>
#include <vector>

#include "shs/resources/resources.contract.hpp"

// Headless tests for the resources pod (R5a P3.7: data-type + registry pins).
// The identity gateway scaffolding was retired (migration step 4.5): the
// resources pod owns value types + the registry edge until real intents
// arrive (empty command vocabulary by law §6.1). Links only
// shs::renderer-values + glm. Registry API is exercised as-is (flagged
// edge-candidate until R5b); the pins below are its migration spec.
namespace
{
    // Empty assets are invalid/empty; handles start null.
    bool test_data_basics()
    {
        const shs::resources::Texture2DData tex{};
        if (tex.valid()) return false;
        const shs::resources::MeshData mesh{};
        if (!mesh.empty()) return false;
        return true;
    }

    // Registry round-trip: add -> get -> same bytes; bad handle -> null.
    bool test_registry_round_trip()
    {
        shs::resources::ResourceRegistry reg{};
        shs::resources::Texture2DData tex{};
        tex.w = 2;
        tex.h = 2;
        if (reg.get_texture(0) != nullptr) return false;
        const shs::resources::TextureAssetHandle h = reg.add_texture(std::move(tex), "t");
        if (h == 0) return false;
        const shs::resources::Texture2DData* back = reg.get_texture(h);
        if (!back || back->w != 2 || back->h != 2) return false;
        return reg.get_texture(h + 100) == nullptr;
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_data_basics() && ok;
    ok = test_registry_round_trip() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[resources-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[resources-tests] all tests passed\n");
    return 0;
}
