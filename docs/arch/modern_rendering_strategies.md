# Rendering Strategies: Real-World Hybrid Pipeline

Modern rendering isn't about choosing one "Best" algorithm. It's about knowing when to use which tool based on the hardware in the user's hands.

## 1. The Low-Bandwidth King: Visibility Buffer
If you're targeting high-poly scenes (Nanite style), don't store a 32-bit G-Buffer. It's too slow.
- **The Concept**: Render just the `TriangleID` and `InstanceID`.
- **The Shading**: Use a compute shader to manually calculate barycentrics and fetch textures.
- **Why?**: It completely decouples your triangle count from your shading cost.

## 2. Clustered Forward: Scaling Lights
Handling 1,000 lights in a standard forward pass is impossible. Deferred is great but expensive for transparency.
- **Our Solution**: Clustered Forward. We bin lights into a 3D grid in the frustum.
- **Trade-off**: It adds some complexity to the "Culling" phase, but it lets us use the same lighting logic for both opaque and transparent objects.

## 3. Platform Tuning

### PC (Discrete GPUs)
PC GPUs have massive throughput but are often bottlenecked by the CPU's ability to "Feed" them.
- **Bindless is Mandatory**: Don't swap descriptor sets. Bind everything once.
- **GPU-Driven Culling**: Let the GPU cull its own objects using an Compute pass. The CPU just sends a single "Draw everything" command.

### Mobile (Tile-Based Rendering)
Mobile chips are bandwidth-starved. If you read/write from DRAM too much, the phone gets hot and throttles.
- **Stay On-Chip**: Use Vulkan subpasses to keep data in the high-speed tile memory.
- **Quantization**: Don't use 32-bit floats for normals if 16-bit octahedral encoding will do. Your memory bus will thank you.

---

> [!TIP]
> **Performance First**: If a technique looks beautiful but triples your frame time, it's not a feature; it's a bug. Always benchmark on target hardware before committing to an architecture.
