#version 450

layout(set = 0, binding = 0) uniform CameraUBO
{
    mat4 view_proj;
    vec4 camera_pos;
    vec4 light_dir_ws;
    mat4 light_view_proj;
    vec4 shadow_params; // x=strength, y=bias_const, z=bias_slope, w=pcf_step
    vec4 shadow_misc;   // x=pcf_radius
} ubo;

layout(push_constant) uniform DrawPush
{
    mat4 model;
    vec4 base_color;
    uvec4 mode_pad; // x: lit mode (0=debug line, 1=lit fill)
} pc;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal_ws;
layout(location = 2) out vec3 v_base_color;
layout(location = 3) flat out uint v_lit_mode;
layout(location = 4) out vec4 v_shadow_pos;

void main()
{
    vec4 world = pc.model * vec4(in_pos, 1.0);
    gl_Position = ubo.view_proj * world;
    v_world_pos = world.xyz;
    v_normal_ws = mat3(pc.model) * in_normal;
    v_base_color = pc.base_color.rgb;
    v_lit_mode = pc.mode_pad.x;
    v_shadow_pos = ubo.light_view_proj * world;
}
