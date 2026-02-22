# Roadmap: Scene Workflow

This roadmap outlines the steps to build the world-authoring pipeline, leveraging the **Core Engine Foundation**.

## Dependencies
- 🛣️ **[Core Engine Foundation](./core_engine_foundation_roadmap.md)** (Requires Phase 1: Serialization & Phase 2: ECS Factory)

## Phase 1: The Loader Bridge
Implement the engine-side logic to interpret the world manifest.
- **Action: JSON Scene Parser**
    - Leverage the **Foundation Serializer** to parse the `.shs_scene` JSON format.
- **Action: Entity Instantiator**
    - Use the **Foundation Component Factory** to dynamically create entities and attach components based on the manifest.

## Phase 2: Asset Resolution (The Bridge)
- **Action: Resource Path Resolver**
    - Integrate with the **Foundation Resource Registry** to map Blender's relative paths to engine-local high-quality assets.
- **Action: Prefab System**
    - Implement the logic to instantiate nested JSON prefabs via the **Foundation ECS Factory**.

## Phase 3: The Exporter (Python)
Build the bridge from the DCC tool to the engine.
- **Action: Blender Python Exporter**
    - Create the `.blend` to `.shs_scene` export script.
- **Action: Proxy/Placeholder Logic**
    - Implement the logic to export low-poly proxies as high-quality asset references in the JSON.

## Phase 4: Full Automation
- **Action: Hot-Reload Pipeline**
    - Implement a file watcher that triggers a scene refresh via the **Foundation Resource Registry** when the JSON manifest changes.
- **Action: Blender Engine Sync**
    - (Advanced) Implement a simple socket-based link to live-sync Blender transforms with the engine.

---

> [!TIP]
> **Foundation First**: By building on the **Component Factory**, your scene loader automatically supports new component types the moment they are registered, meaning you never have to "update" the scene loader code again.
