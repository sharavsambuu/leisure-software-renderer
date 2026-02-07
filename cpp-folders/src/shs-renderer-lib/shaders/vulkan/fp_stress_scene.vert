#version 450

layout(set = 0, binding = 0) uniform CameraUBO
{
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos_time;
    vec4 sun_dir_intensity;
    uvec4 screen_tile_lightcount; // x: width, y: height, z: tiles_x, w: light_count
    uvec4 params;                 // x: tiles_y, y: max_per_tile, z: tile_size, w: culling_mode
    uvec4 culling_params;         // x: cluster_z_slices
    vec4 depth_params;            // x: near, y: far
    vec4 exposure_gamma;          // x: exposure, y: gamma
} ubo;

layout(push_constant) uniform DrawPush
{
    mat4 model;
    vec4 base_color;
    vec4 material_params;
} pc;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal_ws;
layout(location = 2) out vec3 v_base_color;

void main()
{
    vec4 world = pc.model * vec4(in_pos, 1.0);
    gl_Position = ubo.view_proj * world;
    v_world_pos = world.xyz;
    v_normal_ws = mat3(pc.model) * in_normal;
    v_base_color = pc.base_color.rgb;
}
