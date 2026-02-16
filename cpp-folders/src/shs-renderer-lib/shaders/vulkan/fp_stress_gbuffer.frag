#version 450

layout(push_constant) uniform DrawPush
{
    mat4 model;
    vec4 base_color;
    vec4 material_params; // x: metallic, y: roughness, z: ao, w: unlit/debug
} pc;

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal_ws;
layout(location = 2) in vec3 v_base_color;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_material;
layout(location = 3) out vec4 out_world_pos;

void main()
{
    // Unlit debug geometry (light-volume wireframe overlays) should not contribute
    // to deferred surface lighting.
    if (pc.material_params.w > 0.5)
    {
        out_albedo = vec4(0.0);
        out_normal = vec4(0.0, 1.0, 0.0, 0.0);
        out_material = vec4(0.0);
        out_world_pos = vec4(0.0);
        return;
    }

    vec3 N = normalize(v_normal_ws);
    out_albedo = vec4(clamp(v_base_color, vec3(0.0), vec3(1.0)), 1.0);
    out_normal = vec4(N, 1.0);
    out_material = vec4(
        clamp(pc.material_params.x, 0.0, 1.0),
        clamp(pc.material_params.y, 0.04, 1.0),
        clamp(pc.material_params.z, 0.0, 1.0),
        1.0);
    out_world_pos = vec4(v_world_pos, 1.0);
}
