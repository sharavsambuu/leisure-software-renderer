#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../shs-renderer-lib/shaders/vulkan/common/light_constants.glsl"
#include "../../shs-renderer-lib/shaders/vulkan/common/math.glsl"
#include "../../shs-renderer-lib/shaders/vulkan/common/light_math.glsl"

layout(set = 0, binding = 0) uniform CameraUBO
{
    mat4 view_proj;
    vec4 camera_pos;
    vec4 sun_dir_to_scene_ws;
} ubo;

struct GPULight
{
    vec4 position_range;
    vec4 color_intensity;
    vec4 direction_inner;
    vec4 axis_outer;
    vec4 up_shape_x;
    vec4 shape_attenuation;
    uvec4 type_shape_flags;
};

layout(set = 0, binding = 1) uniform LightUBO
{
    uvec4 counts; // x: valid light count
    GPULight lights[64];
} light_ubo;

layout(set = 0, binding = 2) uniform sampler2D u_ssao_tex;

layout(push_constant) uniform DrawPush
{
    mat4 model;
    vec4 base_color;
    uvec4 mode_pad; // x: lit mode, y: local light count
    uvec4 light_indices_01;
    uvec4 light_indices_23;
} pc;

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal_ws;
layout(location = 2) in vec3 v_base_color;
layout(location = 3) flat in uint v_lit_mode;

layout(location = 0) out vec4 out_color;

const float kAmbientBase = 0.22;
const float kAmbientHemi = 0.12;

uint push_light_index(uint i)
{
    if (i < 4u) return pc.light_indices_01[i];
    return pc.light_indices_23[i - 4u];
}

void main()
{
    vec3 base = clamp(v_base_color, vec3(0.0), vec3(1.0));
    if (v_lit_mode == 0u)
    {
        out_color = vec4(base, 1.0);
        return;
    }

    vec3 N = shs_normalize_or(v_normal_ws, vec3(0.0, 1.0, 0.0));
    vec3 V = shs_normalize_or(ubo.camera_pos.xyz - v_world_pos, vec3(0.0, 0.0, 1.0));

    float NdotV = max(dot(N, V), 0.001);

    // Fixed material properties for demo
    float roughness = 0.4;
    float metallic = 0.0;
    vec3 f0 = mix(vec3(0.04), base, metallic);

    // Ambient
    float hemi = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    float ao = texture(u_ssao_tex, gl_FragCoord.xy / textureSize(u_ssao_tex, 0)).r;
    vec3 ambient = base * (kAmbientBase + kAmbientHemi * hemi) * 0.5 * ao; 
    vec3 lit = ambient;

    uint valid_count = min(light_ubo.counts.x, 64u); // max GPULights

    for (uint light_idx = 0u; light_idx < valid_count; ++light_idx)
    {

        GPULight light = light_ubo.lights[light_idx];

        uint light_type = light.type_shape_flags.x;
        uint attenuation_model = light.type_shape_flags.w;

        vec3 light_pos = light.position_range.xyz;
        float range = light.position_range.w;
        vec3 radiance_color = max(light.color_intensity.rgb, vec3(0.0));
        float intensity = max(light.color_intensity.a, 0.0);

        vec3 L = vec3(0.0, 1.0, 0.0);
        float distance_to_light = 0.0;
        float shaping = 0.0;
        float spec_power = 32.0;
        float spec_scale = 0.25;

        if (light_type == SHS_LIGHT_TYPE_POINT)
        {
            vec3 to_light = light_pos - v_world_pos;
            distance_to_light = length(to_light);
            if (distance_to_light <= 1e-4 || distance_to_light > range) continue;
            L = to_light / distance_to_light;
            shaping = 1.0;
            spec_power = 36.0;
            spec_scale = 0.30;
        }
        else if (light_type == SHS_LIGHT_TYPE_SPOT)
        {
            vec3 to_light = light_pos - v_world_pos;
            distance_to_light = length(to_light);
            if (distance_to_light <= 1e-4 || distance_to_light > range) continue;
            L = to_light / distance_to_light;

            vec3 light_to_surface = -L;
            vec3 dir = shs_normalize_or(light.direction_inner.xyz, vec3(0.0, -1.0, 0.0));
            float inner_cos = clamp(light.direction_inner.w, -1.0, 1.0);
            float outer_cos = clamp(light.axis_outer.w, -1.0, inner_cos - 1e-4);
            float cos_theta = dot(light_to_surface, dir);
            if (cos_theta <= outer_cos) continue;

            float t = (cos_theta - outer_cos) / max(inner_cos - outer_cos, 1e-5);
            t = clamp(t, 0.0, 1.0);
            shaping = t * t * (3.0 - 2.0 * t);
            spec_power = 34.0;
            spec_scale = 0.32;
        }
        else if (light_type == SHS_LIGHT_TYPE_RECT_AREA)
        {
            vec3 right = shs_normalize_or(light.axis_outer.xyz, vec3(1.0, 0.0, 0.0));
            vec3 up = shs_normalize_or(light.up_shape_x.xyz, vec3(0.0, 1.0, 0.0));
            vec3 fwd = shs_normalize_or(light.direction_inner.xyz, vec3(0.0, -1.0, 0.0));
            float half_w = max(light.up_shape_x.w, 0.001);
            float half_h = max(light.shape_attenuation.x, 0.001);

            vec3 d = v_world_pos - light_pos;
            float ux = clamp(dot(d, right), -half_w, half_w);
            float uy = clamp(dot(d, up), -half_h, half_h);
            vec3 emit_pt = light_pos + right * ux + up * uy;

            vec3 to_light = emit_pt - v_world_pos;
            distance_to_light = length(to_light);
            if (distance_to_light <= 1e-4 || distance_to_light > range) continue;
            L = to_light / distance_to_light;

            vec3 light_to_surface = -L;
            float emission_facing = max(dot(fwd, light_to_surface), 0.0);
            if (emission_facing <= 0.0) continue;

            shaping = 0.65 + 0.55 * emission_facing;
            spec_power = 26.0;
            spec_scale = 0.26;
        }
        else if (light_type == SHS_LIGHT_TYPE_TUBE_AREA)
        {
            vec3 axis = shs_normalize_or(light.axis_outer.xyz, vec3(1.0, 0.0, 0.0));
            float half_len = max(light.up_shape_x.w, 0.001);
            vec3 a = light_pos - axis * half_len;
            vec3 b = light_pos + axis * half_len;

            vec3 ab = b - a;
            float denom = max(dot(ab, ab), 1e-6);
            float t = clamp(dot(v_world_pos - a, ab) / denom, 0.0, 1.0);
            vec3 emit_pt = a + ab * t;

            vec3 to_light = emit_pt - v_world_pos;
            distance_to_light = length(to_light);
            if (distance_to_light <= 1e-4 || distance_to_light > range) continue;
            L = to_light / distance_to_light;

            float radial_softening = clamp(1.0 - distance_to_light / max(range, 0.1), 0.0, 1.0);
            shaping = 0.75 + 0.35 * radial_softening;
            spec_power = 22.0;
            spec_scale = 0.20;
        }
        else
        {
            continue;
        }

        float attenuation = shs_eval_light_attenuation_smoothstep(
            distance_to_light,
            range,
            attenuation_model,
            light.shape_attenuation.y,
            light.shape_attenuation.z,
            light.shape_attenuation.w) * max(shaping, 0.0);
        if (attenuation <= 0.0)
        {
            continue;
        }

        float ndotl = max(dot(N, L), 0.0);
        if (ndotl <= 0.0)
        {
            continue;
        }

        vec3 radiance = radiance_color * intensity * attenuation;
        vec3 H = shs_normalize_or(L + V, L);
        float NdotH = max(dot(N, H), 0.0);
        float VdotH = max(dot(V, H), 0.0);

        // NDF (GGX)
        float alpha = roughness * roughness;
        float alphaSq = alpha * alpha;
        float denomNDF = (NdotH * NdotH * (alphaSq - 1.0) + 1.0);
        float D = alphaSq / (3.1415926535 * denomNDF * denomNDF);

        // Geometry (Schlick-GGX)
        float k = (roughness + 1.0);
        k = (k * k) / 8.0;
        float g1L = ndotl / (ndotl * (1.0 - k) + k);
        float g1V = NdotV / (NdotV * (1.0 - k) + k);
        float G = g1L * g1V;

        // Fresnel (Schlick)
        vec3 F = f0 + (1.0 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);

        vec3 specular = (D * G * F) / (4.0 * NdotV * ndotl + 0.0001);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        vec3 diffuse = kD * base / 3.1415926535;

        lit += (diffuse + specular) * radiance * ndotl;
    }

    out_color = vec4(clamp(lit, vec3(0.0), vec3(1.0)), 1.0);
}
