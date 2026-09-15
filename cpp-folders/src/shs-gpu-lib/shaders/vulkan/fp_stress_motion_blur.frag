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

layout(set = 1, binding = 4) uniform sampler2D u_history_color;
layout(set = 1, binding = 6) uniform sampler2D u_post_input;

layout(location = 0) out vec4 out_color;

const uint SEMDBG_MOTION = 12u;

vec3 heatmap_blue_cyan_yellow_red(float t)
{
    float x = clamp(t, 0.0, 1.0);
    if (x < 0.33) return mix(vec3(0.05, 0.12, 0.70), vec3(0.00, 0.95, 1.00), x / 0.33);
    if (x < 0.66) return mix(vec3(0.00, 0.95, 1.00), vec3(1.00, 0.95, 0.20), (x - 0.33) / 0.33);
    return mix(vec3(1.00, 0.95, 0.20), vec3(0.95, 0.20, 0.08), (x - 0.66) / 0.34);
}

void main()
{
    vec2 inv_size = vec2(
        1.0 / max(float(ubo.screen_tile_lightcount.x), 1.0),
        1.0 / max(float(ubo.screen_tile_lightcount.y), 1.0));
    vec2 uv = clamp(gl_FragCoord.xy * inv_size, vec2(0.0), vec2(1.0));
    vec3 base = texture(u_post_input, uv).rgb;

    if (ubo.culling_params.z == SEMDBG_MOTION)
    {
        if (ubo.temporal_params.y < 0.5)
        {
            out_color = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        vec3 hist = texture(u_history_color, uv).rgb;
        vec3 delta = abs(base - hist);
        float mag = clamp(length(delta) * 3.0, 0.0, 1.0);
        vec3 heat = heatmap_blue_cyan_yellow_red(mag);
        out_color = vec4(heat, 1.0);
        return;
    }

    float phase = fract(ubo.camera_pos_time.w * 0.19);
    vec2 dir = normalize(vec2(cos(phase * 6.2831853), sin(phase * 6.2831853)));
    vec2 step_uv = dir * inv_size * 2.1;

    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int i = -3; i <= 3; ++i)
    {
        float t = float(i) / 3.0;
        float w = 1.0 - abs(t);
        vec2 suv = clamp(uv + step_uv * float(i), vec2(0.0), vec2(1.0));
        sum += texture(u_post_input, suv).rgb * w;
        wsum += w;
    }

    vec3 blurred = (wsum > 0.0) ? (sum / wsum) : base;
    vec3 out_rgb = mix(base, blurred, 0.34);
    out_color = vec4(out_rgb, 1.0);
}
