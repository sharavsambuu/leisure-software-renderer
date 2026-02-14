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

layout(set = 1, binding = 0) uniform sampler2D u_shadow_map;

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal_ws;
layout(location = 2) in vec3 v_base_color;
layout(location = 3) flat in uint v_lit_mode;
layout(location = 4) in vec4 v_shadow_pos;

layout(location = 0) out vec4 out_color;

const float kAmbientBase = 0.22;
const float kAmbientHemi = 0.12;

float shadow_visibility(float ndotl)
{
    vec3 proj = v_shadow_pos.xyz / max(v_shadow_pos.w, 1e-6);
    vec2 uv = proj.xy * 0.5 + 0.5;
    float depth = proj.z;

    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0 || depth <= 0.0 || depth >= 1.0)
    {
        return 1.0;
    }

    float bias = max(ubo.shadow_params.y + ubo.shadow_params.z * (1.0 - ndotl), 0.00002);
    float z_test = depth - bias;

    int radius = int(clamp(floor(ubo.shadow_misc.x + 0.5), 0.0, 6.0));
    float pcf_step = max(ubo.shadow_params.w, 0.5);

    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    vec2 min_uv = vec2(1e-4);
    vec2 max_uv = vec2(1.0 - 1e-4);

    if (radius <= 0)
    {
        float z_ref = texture(u_shadow_map, clamp(uv, min_uv, max_uv)).r;
        float vis = (z_test <= z_ref) ? 1.0 : 0.0;
        return mix(1.0, vis, clamp(ubo.shadow_params.x, 0.0, 1.0));
    }

    float shadow_sum = 0.0;
    float weight_sum = 0.0;
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            vec2 o = vec2(float(x), float(y));
            float w = 1.0 / (1.0 + dot(o, o));
            vec2 pcf_uv = clamp(uv + o * texel * pcf_step, min_uv, max_uv);
            float z_ref = texture(u_shadow_map, pcf_uv).r;
            shadow_sum += (z_test > z_ref) ? w : 0.0;
            weight_sum += w;
        }
    }

    float shadow = shadow_sum / max(weight_sum, 1e-6);
    float vis = 1.0 - shadow;
    return mix(1.0, vis, clamp(ubo.shadow_params.x, 0.0, 1.0));
}

void main()
{
    vec3 base = clamp(v_base_color, vec3(0.0), vec3(1.0));
    if (v_lit_mode == 0u)
    {
        out_color = vec4(base, 1.0);
        return;
    }

    vec3 N = normalize(v_normal_ws);
    vec3 L = normalize(-ubo.light_dir_ws.xyz);
    vec3 V = normalize(ubo.camera_pos.xyz - v_world_pos);
    vec3 H = normalize(L + V);

    float ndotl = max(dot(N, L), 0.0);
    float ndoth = max(dot(N, H), 0.0);

    float hemi = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    float ambient = kAmbientBase + kAmbientHemi * hemi;
    float shadow_vis = shadow_visibility(ndotl);
    float diffuse = 0.72 * ndotl * shadow_vis;
    float specular = (ndotl > 0.0) ? (0.35 * pow(ndoth, 32.0) * shadow_vis) : 0.0;

    vec3 lit = clamp(base * (ambient + diffuse) + vec3(specular), vec3(0.0), vec3(1.0));
    out_color = vec4(lit, 1.0);
}
