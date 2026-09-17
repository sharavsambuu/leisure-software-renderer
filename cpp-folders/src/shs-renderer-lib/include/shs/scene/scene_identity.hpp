#pragma once

/*
    SHS RENDERER SAN

    FILE: scene_identity.hpp
    MODULE: scene
    PURPOSE: Scene/resource identity policy (step 4.3,
             engine_domain_separation_migration.md). One place states and
             ENFORCES how identity behaves across scene objects, asset
             handles and renderer projections:

             OBJECT IDENTITY (SceneObject / RenderItem::object_id)
               - object_id 0 is RESERVED (means "no identity"); SceneObjectSet
               never stores it (name-derived FNV-1a fallback, never 0).
               - object_id is stable across frames and across
               deletion/recreation: the same name always maps to the same id.
               - object_ids must be UNIQUE within one renderer projection
               (Scene::items). Duplicates are a policy violation detected by
               audit_scene_identity, never silently merged or renumbered.

             NAME IDENTITY (SceneObjectSet)
               - find(name) returns the FIRST registration (first-wins).
               - Duplicate names are detectable, not silently rejected;
               remove(name) deletes the first match only.

             ASSET HANDLES (ResourceRegistry)
               - Handles are 1-based indices; 0 means "unbound" and every
               getter returns nullptr for it (and for out-of-range handles).
               - The registry is APPEND-ONLY: adding never invalidates an
               existing handle; a duplicate key deliberately REBINDS the
               key to the newest asset (last-wins) while the old handle
               still resolves to the old asset.
               - Per-asset deletion is NOT supported (index handles would
               invalidate siblings); the only reset is clear(), which bumps
               the generation epoch. Handles from an older generation are
               stale: after clear() they must be re-derived via find_*
               (a stale handle can ALIAS a re-added asset's slot — callers
               must compare generation, not rely on resolution success).

             RENDERER PROJECTIONS (SceneResourceView / to_render_items)
               - Pointers into registry storage are valid only until the
               next registry mutation; projections re-resolve per frame
               through SceneResourceView and never cache them.
               - to_render_items() copies: later set mutations never leak
               into an already-produced projection.
*/

#include <cstddef>
#include <span>

#include "shs/scene/scene_objects.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs
{
    // Identity audit over a renderer projection (Scene::items). First
    // occurrence of an object_id is the owner; every later occurrence is
    // counted as a duplicate. Zero ids are counted separately (reserved).
    struct SceneIdentityReport
    {
        size_t zero_object_ids = 0;
        size_t duplicate_object_ids = 0;

        bool valid() const { return zero_object_ids == 0 && duplicate_object_ids == 0; }
        bool operator==(const SceneIdentityReport&) const = default;
    };

    inline SceneIdentityReport audit_scene_identity(
        std::span<const RenderItem> items)
    {
        SceneIdentityReport report{};
        for (size_t i = 0; i < items.size(); ++i)
        {
            if (items[i].object_id == 0)
            {
                report.zero_object_ids += 1;
                continue;
            }
            for (size_t j = 0; j < i; ++j)
            {
                if (items[j].object_id == items[i].object_id)
                {
                    report.duplicate_object_ids += 1;
                    break;
                }
            }
        }
        return report;
    }
} // namespace shs
