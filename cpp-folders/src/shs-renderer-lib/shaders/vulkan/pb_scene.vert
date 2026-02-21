#version 450

layout(set = 0, binding = 1) uniform ObjectUBO
{
    mat4 mvp;
    mat4 prev_mvp;
    mat4 model;
    mat4 light_mvp;
    vec4 base_color_metallic;
    vec4 roughness_ao_emissive_hastex;
    vec4 camera_pos_sun_intensity;
    vec4 sun_color_pad;
    vec4 sun_dir_ws_pad;
    vec4 shadow_params;
    uvec4 extra_indices;
} ubo;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal_ws;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec4 v_shadow_pos;
layout(location = 4) out vec4 v_curr_clip;
layout(location = 5) out vec4 v_prev_clip;

void main()
{
    vec4 world = ubo.model * vec4(in_pos, 1.0);
    gl_Position = ubo.mvp * vec4(in_pos, 1.0);

    v_world_pos = world.xyz;
    v_normal_ws = normalize(mat3(ubo.model) * in_normal);
    v_uv = in_uv;
    v_shadow_pos = ubo.light_mvp * vec4(in_pos, 1.0);
    v_curr_clip = gl_Position;
    v_prev_clip = ubo.prev_mvp * vec4(in_pos, 1.0);
}
