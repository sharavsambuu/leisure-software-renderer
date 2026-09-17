#include <cstdio>
#include <string>
#include <vector>

#include "shs/scene/scene.contract.hpp"

// Headless tests for the scene pod (R5b P3.6: projection pins + identity).
// The identity gateway scaffolding was retired (migration step 4.5): the
// scene pod owns pure projections only (make_render_item, to_render_items)
// until real intents arrive — the command vocabulary stays empty by law
// (§6.1). Links only shs::renderer-values + glm. Store mutation is exercised
// only through its pure projection (migration spec for the R5b edge move).
namespace
{
    // make_render_item maps every field, no surprises.
    bool test_make_render_item()
    {
        const shs::scene::RenderItem it = shs::scene::make_render_item(
            7, 3, glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(2.0f), glm::vec3(0.0f, 1.0f, 0.0f), 42);
        return it.mesh == 7 && it.mat == 3 && it.object_id == 42
            && it.tr.pos == glm::vec3(1.0f, 2.0f, 3.0f)
            && it.tr.scl == glm::vec3(2.0f)
            && it.visible && it.casts_shadow;
    }

    // Projection: added objects project 1:1 with stable ids; names resolve.
    bool test_projection_round_trip()
    {
        shs::scene::SceneObjectSet set{};
        shs::scene::SceneObject a{};
        a.name = "alpha";
        a.mesh = 5;
        set.add(a);
        shs::scene::SceneObject b{};
        b.name = "beta";
        set.add(b);

        const std::vector<shs::scene::RenderItem> items = set.to_render_items();
        if (items.size() != 2) return false;
        if (items[0].mesh != 5 || items[0].object_id == 0) return false;
        if (items[0].object_id == items[1].object_id) return false; // stable + distinct
        if (!set.find("alpha") || set.find("missing")) return false;

        // Deterministic ids: a fresh set hashes identically.
        shs::scene::SceneObjectSet set2{};
        shs::scene::SceneObject a2{};
        a2.name = "alpha";
        set2.add(a2);
        return set2.to_render_items()[0].object_id == items[0].object_id;
    }
} // namespace

int main()
{
    bool ok = true;
    auto run = [&](const char* name, bool result)
    {
        std::fprintf(stderr, "[scene-tests] %s: %s\n", name, result ? "pass" : "FAIL");
        ok = result && ok;
    };

    run("make_render_item", test_make_render_item());
    run("projection_round_trip", test_projection_round_trip());

    if (!ok)
    {
        std::fprintf(stderr, "[scene-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[scene-tests] all tests passed\n");
    return 0;
}
