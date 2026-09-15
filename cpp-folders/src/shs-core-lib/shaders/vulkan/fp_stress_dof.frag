#version 450

layout(set = 0, binding = 0) uniform CameraUBO
{
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos_time;
    vec4 sun_dir_intensity;
    uvec4 screen_tile_lightcount;
    uvec4 params;
    uvec4 culling_params;
    vec4 depth_params;
    vec4 exposure_gamma;
    mat4 sun_shadow_view_proj;
    vec4 sun_shadow_params;
    vec4 sun_shadow_filter;
    vec4 temporal_params;
} ubo;

layout(set = 0, binding = 5) uniform sampler2D u_depth_tex;
layout(set = 1, binding = 6) uniform sampler2D u_post_input;

layout(location = 0) out vec4 out_color;

const uint SEMDBG_DOF_COC = 13u;
const uint SEMDBG_DOF_BLUR = 14u;
const uint SEMDBG_DOF_FACTOR = 15u;

float depth01_to_view(float d, float near_z, float far_z)
{
    float n = max(near_z, 0.001);
    float f = max(far_z, n + 0.01);
    float denom = max(f - d * (f - n), 1e-5);
    return (n * f) / denom;
}

void main()
{
    vec2 inv_size = vec2(
        1.0 / max(float(ubo.screen_tile_lightcount.x), 1.0),
        1.0 / max(float(ubo.screen_tile_lightcount.y), 1.0));
    vec2 uv = clamp(gl_FragCoord.xy * inv_size, vec2(0.0), vec2(1.0));

    vec3 base = texture(u_post_input, uv).rgb;
    float d01 = texture(u_depth_tex, uv).r;
    float near_z = max(ubo.depth_params.x, 0.001);
    float far_z = max(ubo.depth_params.y, near_z + 0.01);
    float view_z = (d01 < 0.9999) ? depth01_to_view(d01, near_z, far_z) : far_z;

    const float focus_distance = 30.0;
    const float focus_range = 56.0;
    float coc = clamp(abs(view_z - focus_distance) / focus_range, 0.0, 1.0);
    float radius = coc * 2.2;

    vec2 offsets[12] = vec2[](
        vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0),
        vec2(0.7, 0.7), vec2(-0.7, 0.7), vec2(0.7, -0.7), vec2(-0.7, -0.7),
        vec2(1.4, 0.2), vec2(-1.4, 0.2), vec2(0.2, 1.4), vec2(0.2, -1.4));

    vec3 blurred = base;
    if (radius >= 0.95)
    {
        vec3 sum = vec3(0.0);
        float wsum = 0.0;
        for (int i = 0; i < 12; ++i)
        {
            vec2 suv = clamp(uv + offsets[i] * inv_size * radius, vec2(0.0), vec2(1.0));
            float w = 1.0;
            sum += texture(u_post_input, suv).rgb * w;
            wsum += w;
        }
        blurred = (wsum > 0.0) ? (sum / wsum) : base;
    }
    float dof_factor = smoothstep(0.24, 1.0, coc) * 0.30;

    uint debug_mode = ubo.culling_params.z;
    if (debug_mode == SEMDBG_DOF_COC)
    {
        out_color = vec4(vec3(coc), 1.0);
        return;
    }
    if (debug_mode == SEMDBG_DOF_BLUR)
    {
        out_color = vec4(blurred, 1.0);
        return;
    }
    if (debug_mode == SEMDBG_DOF_FACTOR)
    {
        out_color = vec4(vec3(dof_factor), 1.0);
        return;
    }

    vec3 out_rgb = mix(base, blurred, dof_factor);
    out_color = vec4(out_rgb, 1.0);
}
