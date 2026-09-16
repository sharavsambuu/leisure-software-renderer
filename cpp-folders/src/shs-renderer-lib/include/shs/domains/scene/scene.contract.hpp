#pragma once

/*
    SHS RENDERER SAN

    FILE: scene.contract.hpp
    MODULE: domains/scene
    PURPOSE: CORE 1. TYPES — the scene pod's discoverability seam (R5b P3.6).
             Item/object/transform values + the to_render_items projection are
             the pod spine. Store CLASSES (SceneObjectSet, SceneElementSet,
             World) and ISystem are edge candidates — method-mutated tables
             like the resource registries; they stay visible but migrate to an
             edge subfolder with the R5b convergence (same ruling).
*/

#include "shs/domains/scene/scene_bindings.hpp"
#include "shs/domains/scene/scene_elements.hpp"
#include "shs/domains/scene/scene_instance.hpp"
#include "shs/domains/scene/scene_objects.hpp"
#include "shs/domains/scene/scene_types.hpp"

namespace shs::scene
{
    // --- item/object values (pod spine) ---
    using shs::Transform;
    using shs::Camera;
    using shs::DirectionalLight;
    using shs::RenderItem;
    using shs::Scene;
    using shs::SceneObject;
    using shs::MeshHandle;
    using shs::MaterialHandle;
    using shs::make_render_item;

    // --- store classes (visible, migrating in R5b convergence) ---
    using shs::SceneObjectSet;
} // namespace shs::scene
