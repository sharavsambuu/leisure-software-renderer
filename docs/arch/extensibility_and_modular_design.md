# Extensibility: The 3-Tier Model

The goal of this engine isn't just to be fast; it's to be **infinitely moddable**. We achieve this by offering three distinct "tiers" of extensibility depending on who is doing the work.

## Tier 1: Composition (The Artist's Path)
**No coding required.** Everything is a prefab.
- **The Concept**: To make a new "Enemy Type," an artist just takes an existing `Entity` and swaps a few components.
- **Blender Workflow**: Add a `Flaming` tag and a `Large` tag in Blender. The engine's **Factory** sees these and attaches the `ParticleComponent` and `ScaleComponent` automatically.
- **Composition**: A "Flaming Large Orc" isn't a new C++ class—it's just a different set of rows in the ECS table.

## Tier 2: Scripting (The Designer's Path)
**Dynamic logic without recompiling.**
- **The Concept**: Use Lua to define one-off behaviors, quest logic, or custom HUD elements.
- **Hooks**: You don't write a "Update" loop in Lua. You write **Handlers**.
    - `on_trigger_enter(other)`
    - `on_player_interact()`
- **Safety**: Lua can't crash the engine. If a script errors, the engine logs it and keeps running the physics and rendering.

## Tier 3: Engineering (The Programmer's Path)
**Raw performance and custom heavy-lifting.**
- **The Concept**: Add new **Systems** (Processors) in C++ to handle massive datasets.
- **Plugin Architecture**: Want a custom Fluid Simulation?
    1. Define a `FluidComponent` struct.
    2. Write a `FluidSystem` that inherits from our base `System` class.
    3. The **Orchestrator** picks it up and gives it a slice of the frame time.
- **Bindings**: If your new C++ system needs to be controlled by Lua, you add one line to the `BindingRegistry`.

## How they talk: The "Glue"
Extensibility works because every tier speaks **ECS**:
- **C++ Components** are reflected into **Lua** automatically.
- **Lua Events** can be caught by **C++ Systems**.
- **Blender Metadata** drives the creation of both.

---

> [!NOTE]
> **Extensibility Rule**: If a feature is used by one entity, use **Lua**. If it's used by 1,000 entities and needs performance, move it to **C++**. Because they share the same components, moving a feature between these tiers is trivial.
