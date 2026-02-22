# Compact Rendering Strategies

Compact rendering refers to a suite of techniques aimed at minimizing **memory bandwidth pressure**, **G-Buffer footprint**, and **shading redundancy**.

## 1. Visibility Rendering (Visibility Buffer)
The most "compact" form of deferred rendering. Instead of storing large G-Buffers, you store a single reference to the original geometry.

### Implementation Logic
1.  **Visibility Pass**: Render all geometry to a `uint64` (or `2x uint32`) buffer.
    ```glsl
    // Fragment Shader
    layout(location = 0) out uvec2 out_Visibility;
    void main() {
        out_Visibility = uvec2(instanceIndex, gl_PrimitiveID);
    }
    ```
2.  **Shading Pass**: Fullscreen compute shader that performs "software" attribute interpolation.
    ```glsl
    // Shading Compute Shader
    uvec2 vis = texelFetch(visibilityBuffer, pixelCoord, 0).xy;
    uint instID = vis.x;
    uint primID = vis.y;

    // 1. Fetch vertex indices for primID
    // 2. Fetch vertex data (pos, uv, norm)
    // 3. Project 3 vertices to screen space
    // 4. Calculate Barycentrics for the current pixel
    // 5. Interpolate: finalUV = uv0*b.x + uv1*b.y + uv2*b.z;
    ```
- **Pros**: Zero G-Buffer bandwidth for geometry attributes. Shading is decoupled from geometric complexity.

## 2. Clustered Forward
Efficiently handles thousands of lights without the G-Buffer cost of Classic Deferred.

### Cluster Mapping Logic
The frustum is divided into a 3D grid. The Z-slices are typically **logarithmic** to provide more precision near the camera.
- **Slice calculation**:
  `uint slice = uint(max(0.0, log(viewPos.z / nearZ) * sliceCount / log(farZ / nearZ)));`
- **Light Binning**: A GPU compute pass iterates over all active lights and "stamps" them into overlapping clusters.
- **The Cluster Buffer**: A flat array or 3D texture containing `(Offset, Count)` into a global `LightIndexBuffer`.

## 3. Attribute Compression & Quantization
Deeply compacting individual data elements to save DRAM and GPU cache.

### Octahedral Normals (2x8-bit)
Storing `(nx, ny, nz)` is expensive. Mapping a sphere to a 2D square (Octahedron map) allows storing high-quality normals in 16 bits.
```glsl
vec2 packNormalOct(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 res = n.xy;
    if (n.z < 0.0) res = (1.0 - abs(res.yx)) * sign(res);
    return res * 0.5 + 0.5; // Final 0..1 range
}
```

### Vertex Position Quantization
Instead of `3x float32` (12 bytes), store positions as `3x uint16` (6 bytes) relative to the mesh's AABB.
- **CPU**: `u_pos = (world_pos - mesh_min) / (mesh_max - mesh_min) * 65535.0;`
- **GPU**: `pos = float3(u_pos) / 65535.0 * (mesh_max - mesh_min) + mesh_min;`

## Comparison Matrix

| Strategy | Footprint | Calculation Cost | Complexity | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Visibility Buffer** | ~8 bytes/px | High (Barycentrics) | Extreme | Next-gen PC (Nanite style) |
| **Clustered Forward** | ~4-8 bytes/px | Medium (Binning) | Moderate | Cross-platform (Mobile/PC) |
| **Standard Deferred** | ~16-32 bytes/px| Low | Low | Legacy PC |
| **Hybrid Deferred** | ~12 bytes/px | Medium | Moderate | Mainstream modern |
