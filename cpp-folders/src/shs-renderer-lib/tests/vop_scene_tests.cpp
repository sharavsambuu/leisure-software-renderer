#include <cstdio>
#include <memory_resource>
#include <string>
#include <variant>
#include <vector>

#include "shs/domains/scene/scene.contract.hpp"
#include "shs/domains/scene/scene.reducer.hpp"
#include "shs/domains/pod_test_kit.hpp"

// Headless tests for the scene pod (R5b P3.6: projection pins + identity).
// Links only shs::renderer-values + glm. Store mutation is exercised only
// through its pure projection (migration spec for the R5b edge move).
namespace
{
    auto reduce_via_pod = [](shs::scene::SceneState& s,
                             std::span<const shs::scene::SceneAction> a,
                             const shs::scene::SceneReduceInputs& in,
                             std::pmr::vector<shs::scene::SceneEvent>& e)
    {
        shs::scene::reduce_scene(s, a, in, e);
    };

    // make_render_item maps every field, no surprises.
    bool test_make_render_item()
    {
        const shs::RenderItem it = shs::make_render_item(
            7, 3, glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(2.0f), glm::vec3(0.0f, 1.0f, 0.0f), 42);
        return it.mesh == 7 && it.mat == 3 && it.object_id == 42
            && it.tr.pos == glm::vec3(1.0f, 2.0f, 3.0f)
            && it.tr.scl == glm::vec3(2.0f)
            && it.visible && it.casts_shadow;
    }

    // Projection: added objects project 1:1 with stable ids; names resolve.
    bool test_projection_round_trip()
    {
        shs::SceneObjectSet set{};
        shs::SceneObject a{};
        a.name = "alpha";
        a.mesh = 5;
        set.add(a);
        shs::SceneObject b{};
        b.name = "beta";
        set.add(b);

        const std::vector<shs::RenderItem> items = set.to_render_items();
        if (items.size() != 2) return false;
        if (items[0].mesh != 5 || items[0].object_id == 0) return false;
        if (items[0].object_id == items[1].object_id) return false; // stable + distinct
        if (!set.find("alpha") || set.find("missing")) return false;

        // Deterministic ids: a fresh set hashes identically.
        shs::SceneObjectSet set2{};
        shs::SceneObject a2{};
        a2.name = "alpha";
        set2.add(a2);
        return set2.to_render_items()[0].object_id == items[0].object_id;
    }

    bool test_identity_stable()
    {
        return shs::pod_test::empty_log_is_stable<shs::scene::SceneState,
            shs::scene::SceneAction, shs::scene::SceneReduceInputs,
            shs::scene::SceneEvent>(
            reduce_via_pod, shs::scene::SceneState{}, shs::scene::SceneReduceInputs{});
    }

    bool test_replay_deterministic()
    {
        const shs::scene::SceneState s0{};
        const std::vector<shs::scene::SceneAction> none{};
        return shs::pod_test::replay_is_deterministic<shs::scene::SceneState,
            shs::scene::SceneAction, shs::scene::SceneReduceInputs,
            shs::scene::SceneEvent>(
            reduce_via_pod, s0,
            std::span<const shs::scene::SceneAction>{none.data(), none.size()},
            shs::scene::SceneReduceInputs{});
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
    run("identity_stable", test_identity_stable());
    run("replay_deterministic", test_replay_deterministic());

    if (!ok)
    {
        std::fprintf(stderr, "[scene-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[scene-tests] all tests passed\n");
    return 0;
}
