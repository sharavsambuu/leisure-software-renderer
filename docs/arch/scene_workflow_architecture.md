# Scene Pipeline: Blender to ECS

Building a level editor is a waste of time. We use Blender as our primary "World Authoring" tool and treat it as the single authority for world data.

## The Data Flow: Blender → JSON → ECS
1. **Authoring (Blender)**: You place meshes, lights, and empties. You add metadata via **Custom Properties** (e.g., `health: 100`, `is_interactable: true`).
2. **Export (Python)**: A custom script walks the scene tree and dumps it into a custom JSON format (`.shs_scene`). It handles the coordinate conversion (Blender's Z-up vs. Vulkan's Y-up).
3. **Boilerplate Loading (C++)**: The engine parses the JSON and hits the **Entity Factory**. It spawns entities and slaps on the components based on the tags it found.

## Proxies: The Secret to Large Scenes
Don't load 5-million polygon trees into Blender. It will lag your UI. 
- Use **Low-poly proxies** in Blender.
- Tag them with an `asset_id` (e.g., `high_res_tree.gltf`).
- The engine's **Resource Registry** swaps the proxy for the real high-res model at runtime.

## Prefabs (Nesting)
A "Dungeon Room" might be a collection of 50 torches and crates. Instead of placing them manually every time:
1. Define the room as a **Blender Collection**.
2. Instantiate that collection multiple times.
3. The exporter recognizes these as **Prefabs** and tells the engine to instantiate them in batches, saving disk space and memory.

## Why JSON?
Binary formats are hard to debug. JSON is human-readable and works perfectly with Git. If you need performance later, we can switch to a binary "blob" format, but for now, the flexibility of text is a massive win for debugging level data.

---

> [!NOTE]
> **The Golden Rule**: If you're tempted to code a feature in C++, ask yourself: *"Can I represent this as a Tag in Blender instead?"*
