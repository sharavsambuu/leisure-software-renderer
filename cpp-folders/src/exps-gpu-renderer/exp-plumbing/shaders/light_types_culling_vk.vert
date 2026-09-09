#version 450

layout(set = 0, binding = 0) uniform CameraUBO
{
    mat4 view_proj;
    vec4 camera_pos;
    vec4 sun_dir_to_scene_ws;
} ubo;

layout(push_constant) uniform DrawPush
{
    mat4 model;
    vec4 base_color;
    uvec4 mode_pad; // x: lit mode (0=debug color, 1=local-lit), y: local light count
    uvec4 light_indices_01;
    uvec4 light_indices_23;
} pc;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal_ws;
layout(location = 2) out vec3 v_base_color;
layout(location = 3) flat out uint v_lit_mode;

void main()
{
    vec4 world = pc.model * vec4(in_pos, 1.0);
    gl_Position = ubo.view_proj * world;
    v_world_pos = world.xyz;
    v_normal_ws = mat3(pc.model) * in_normal;
    v_base_color = pc.base_color.rgb;
    v_lit_mode = pc.mode_pad.x;
}
