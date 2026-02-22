# Scripting & ECS: The Practical Guide

This isn't about rigid hierarchies; it's about shifting logic from hard-coded C++ classes into a data-driven pipeline where Lua and C++ share a common vocabulary: the ECS.

## The Core Rule: ECS is the Intermediary
Don't use Lua to "hold" game state. Use it to **mutate** it.
If a Guard has 100 HP, that number lives in a C++ `HealthComponent`.
- **Lua's Job**: Check the HP, decide to play a "hurt" animation, or trigger 
  a flee behavior by writing a new target to the `MovementComponent`.
- **C++'s Job**: Handle the heavy lifting like Jolt physics, Vulkan culling, 
  and batch-processing the components Lua just touched.

By keeping the state in ECS, we get "Save/Load" and "Networking" almost for free, because we only have to serialize the components, not the entire Lua VM state.

## Why skip Class-based OOP?
Inheritance is a trap for games. A `FlamingSword` that is also a `QuestItem` and a `Weapon` results in a messy class tree.
In ECS, you just slap on the tags:
- `Entity` = [Mesh, Physics, Damage, LightSource, QuestMarker]
- **The Win**: You can turn a static chair into a physical weapon at runtime just by adding the `RigidBody` component. No recompiling, no casting.

## Performance vs. Flexibility
We're using a **System-centric** approach to avoid the "Update() for every object" bottleneck.

1. **Hot Path (C++)**: Systems like Physics and Rendering iterate over 
   contiguous arrays of data. This is cash-friendly and fast.
2. **Logic Path (Lua)**: We don't call a Lua script per-entity. Instead, we 
   process entities in batches. The `ScriptingSystem` gathers all entities 
   needing logic and feeds them to Lua in a single pass.

## Dealing with Metadata (Blender Tags)
Instead of writing code for every new enemy type, we use **Tags**.
In Blender, you add a custom property: `lua_script: "patrol.lua"`.
When the JSON loader sees this, it:
1. Spawns an entity.
2. Attaches a `ScriptComponent`.
3. Points it at `patrol.lua`.

This allows you to build entire game levels and behaviors without ever touching a C++ compiler.

---

> [!TIP]
> **Keep Scripts Small**: Use Lua for "The Brain" (Decision making) and C++ for "The Muscle" (Moving thousands of particles or solving physics).
