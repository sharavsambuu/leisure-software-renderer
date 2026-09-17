#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "shs/resources/material.hpp"
#include "shs/resources/mesh.hpp"
#include "shs/resources/storage/resource_registry.hpp"
#include "shs/resources/texture.hpp"
#include "shs/scene/scene_bindings.hpp"
#include "shs/scene/scene_identity.hpp"
#include "shs/scene/scene_objects.hpp"
#include "shs/scene/scene_resource_view.hpp"
#include "shs/scene/scene_types.hpp"

// Step 4.3 regression suite (engine_domain_separation_migration.md):
// scene/resource identity policy. Pins the written policy in
// shs/scene/scene_identity.hpp, shs/scene/scene_objects.hpp and
// shs/resources/storage/resource_registry.hpp:
//   - object_id: 0 reserved, unique within a projection, stable across
//     deletion/recreation (same name -> same id),
//   - name lookup first-wins; duplicates detectable; remove() deletes first,
//   - registry handles: 1-based, 0 = unbound -> nullptr; append-only
//     duplicate-key rebinding (last-wins) leaves old handles intact;
//     clear() bumps generation() and voids pre-clear handles,
//   - renderer projections: to_render_items() copies are independent of
//     later set mutations; SceneResourceView resolves per call.
// Links only shs::renderer-values + glm: no SDL, no Vulkan, no Context.
namespace
{
    // ------------------------------------------------------------------
    // SceneObjectSet: stable identity, deletion/recreation, name policy.
    bool test_scene_object_identity()
    {
        shs::SceneObjectSet set{};

        shs::SceneObject a{};
        a.name = "wall";
        a.mesh = 1;
        shs::SceneObject& stored_a = set.add(a);
        if (stored_a.object_id == 0) return false; // never zero by policy
        const uint64_t id_a = stored_a.object_id;

        // Explicit non-zero ids are preserved verbatim.
        shs::SceneObject b{};
        b.name = "other";
        b.object_id = 42;
        if (set.add(b).object_id != 42) return false;
        if (id_a == 42) return false; // distinct names -> distinct ids

        // Deletion/recreation preserves identity: same name -> same id.
        if (!set.remove("wall")) return false;
        if (set.remove("not-there")) return false; // missing name -> false
        shs::SceneObject recreated{};
        recreated.name = "wall";
        recreated.mesh = 5;
        if (set.add(recreated).object_id != id_a) return false;

        // Name lookup is first-wins; duplicates detectable.
        shs::SceneObject d1{};
        d1.name = "dupe";
        d1.mesh = 1;
        shs::SceneObject d2{};
        d2.name = "dupe";
        d2.mesh = 2;
        set.add(d1);
        set.add(d2);
        if (set.find("dupe") == nullptr) return false;
        if (set.find("dupe")->mesh != 1) return false; // first-wins
        if (set.count_duplicate_names() != 1) return false;
        if (!set.has_duplicate_names()) return false;
        // remove() deletes the FIRST match only.
        if (!set.remove("dupe")) return false;
        if (set.find("dupe") == nullptr) return false;
        if (set.find("dupe")->mesh != 2) return false;
        return true;
    }

    // ------------------------------------------------------------------
    // Renderer projection copy: to_render_items() never aliases live state.
    bool test_projection_copy_independence()
    {
        shs::SceneObjectSet set{};
        shs::SceneObject obj{};
        obj.name = "wall";
        obj.mesh = 7;
        obj.material = 3;
        obj.tr.pos = {1.0f, 2.0f, 3.0f};
        const uint64_t id = set.add(obj).object_id;

        const std::vector<shs::RenderItem> projection = set.to_render_items();
        if (projection.size() != 1) return false;
        if (projection[0].object_id != id || projection[0].mesh != 7) return false;
        if (projection[0].tr.pos.x != 1.0f) return false;

        // Mutate the set afterwards (add + mutate + remove): the projection
        // copy stays independent — renderer projections never alias live
        // SceneObjectSet storage.
        shs::SceneObject extra{};
        extra.name = "extra";
        extra.mesh = 9;
        set.add(extra);
        set.find("wall")->mesh = 100;
        set.remove("wall");
        if (projection.size() != 1) return false;
        if (projection[0].mesh != 7) return false;
        if (projection[0].tr.pos.x != 1.0f) return false;

        // A fresh projection reflects the new state instead.
        const std::vector<shs::RenderItem> projection2 = set.to_render_items();
        if (projection2.size() != 1 || projection2[0].mesh != 9) return false;
        return true;
    }
    // ------------------------------------------------------------------
    // Registry handle policy: null rule, append-only rebinding, generation.
    bool test_registry_handle_policy()
    {
        using shs::MeshData;
        using shs::ResourceRegistry;

        ResourceRegistry registry{};

        // Unbound handle 0 and out-of-range handles never resolve.
        if (registry.get_mesh(0) != nullptr) return false;
        if (registry.get_mesh(42) != nullptr) return false;
        if (registry.get_texture(0) != nullptr) return false;
        if (registry.get_material(0) != nullptr) return false;
        if (registry.generation() != 0) return false;

        MeshData mesh_a{};
        mesh_a.positions.resize(3);
        const shs::MeshAssetHandle h_a = registry.add_mesh(mesh_a, "mesh");
        if (h_a == 0 || registry.get_mesh(h_a) == nullptr) return false;
        if (registry.get_mesh(h_a)->positions.size() != 3) return false;
        if (registry.find_mesh("mesh") != h_a) return false;

        // Append-only: a duplicate key rebinds the lookup last-wins while
        // the previously returned handle keeps resolving to the OLD asset
        // (already-bound scene items stay intact).
        MeshData mesh_b{};
        mesh_b.positions.resize(8);
        const shs::MeshAssetHandle h_b = registry.add_mesh(mesh_b, "mesh");
        if (h_b != h_a + 1) return false;
        if (registry.find_mesh("mesh") != h_b) return false;
        if (registry.get_mesh(h_a) == nullptr) return false;
        if (registry.get_mesh(h_a)->positions.size() != 3) return false;
        if (registry.get_mesh(h_b)->positions.size() != 8) return false;
        if (registry.generation() != 0) return false; // adds never bump

        // clear() is the only reset: generation bumps and pre-clear handles
        // are void while the store is empty.
        registry.clear();
        if (registry.generation() != 1) return false;
        if (registry.get_mesh(h_a) != nullptr) return false;
        if (registry.find_mesh("mesh") != 0) return false;

        // Re-add after clear(): the new handle reuses the 1-based slot, so a
        // pre-clear handle would ALIAS the new asset. Policy: hosts must
        // re-derive handles via find_* after clear(); never reuse them.
        MeshData mesh_c{};
        mesh_c.positions.resize(55);
        const shs::MeshAssetHandle h_c = registry.add_mesh(mesh_c, "mesh");
        if (h_c != h_a) return false; // same index, new generation
        if (registry.get_mesh(h_a)->positions.size() != 55) return false;
        if (registry.find_mesh("mesh") != h_c) return false;
        return true;
    }

    // ------------------------------------------------------------------
    // SceneResourceView projection: per-call resolution, unbound -> nullptr.
    bool test_scene_resource_view_projection()
    {
        using shs::MaterialData;
        using shs::MeshData;
        using shs::ResourceRegistry;
        using shs::SceneResourceView;

        // No registry bound: every lookup resolves to nullptr by policy.
        SceneResourceView unbound{};
        const shs::RenderItem probe = shs::make_render_item(1, 1);
        if (unbound.mesh(probe) != nullptr) return false;
        if (unbound.material(probe) != nullptr) return false;

        ResourceRegistry registry{};
        MeshData mesh{};
        mesh.positions.resize(6);
        shs::MaterialData material{};
        const shs::MeshAssetHandle h_mesh = registry.add_mesh(mesh, "m");
        const shs::MaterialAssetHandle h_mat =
            registry.add_material(material, "mat");

        SceneResourceView view{&registry};
        const shs::RenderItem bound = shs::make_render_item(
            (shs::MeshHandle)h_mesh, (shs::MaterialHandle)h_mat, {}, {}, {}, 7);

        const MeshData* mesh_ptr = view.mesh(bound);
        const shs::MaterialData* mat_ptr = view.material(bound);
        if (mesh_ptr == nullptr || mesh_ptr->positions.size() != 6) return false;
        if (mat_ptr == nullptr) return false;

        // Per-call resolution rule: after a registry mutation the view
        // re-resolves correctly (it never caches registry pointers).
        MeshData mesh2{};
        mesh2.positions.resize(77);
        const shs::MeshAssetHandle h_mesh2 = registry.add_mesh(mesh2, "m2");
        (void)h_mesh2;
        const MeshData* mesh_ptr2 = view.mesh(bound);
        if (mesh_ptr2 == nullptr || mesh_ptr2->positions.size() != 6) return false;

        // Unbound handles (0) resolve to nullptr: "no asset bound".
        const shs::RenderItem unbound_item = shs::make_render_item(0, 0);
        if (view.mesh(unbound_item) != nullptr) return false;
        if (view.material(unbound_item) != nullptr) return false;
        return true;
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_scene_object_identity() && ok;
    ok = test_projection_copy_independence() && ok;
    ok = test_registry_handle_policy() && ok;
    ok = test_scene_resource_view_projection() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[scene-identity-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[scene-identity-tests] all tests passed\n");
    return 0;
}

