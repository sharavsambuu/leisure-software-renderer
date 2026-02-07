#version 450

layout(set = 0, binding = 0) uniform sampler2D u_base_color;

layout(set = 0, binding = 1) uniform ObjectUBO
{
    mat4 mvp;
    mat4 prev_mvp;
    mat4 model;
    mat4 light_mvp;
    vec4 base_color_metallic;
    vec4 roughness_ao_emissive_hastex;
    vec4 camera_pos_sun_intensity;
    vec4 sun_dir_ws_pad;
} ubo;

layout(set = 1, binding = 0) uniform sampler2D u_shadow_map;

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal_ws;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec4 v_shadow_pos;
layout(location = 4) in vec4 v_curr_clip;
layout(location = 5) in vec4 v_prev_clip;

layout(location = 0) out vec4 out_hdr;
layout(location = 1) out vec2 out_velocity;

const float PI = 3.14159265359;

float distribution_ggx(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denom * denom, 1e-6);
}

float geometry_schlick_ggx(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float denom = NdotV * (1.0 - k) + k;
    return NdotV / max(denom, 1e-6);
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometry_schlick_ggx(NdotV, roughness);
    float ggx1 = geometry_schlick_ggx(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnel_schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float soft_shadow(vec3 N, vec3 L)
{
    vec3 proj = v_shadow_pos.xyz / max(v_shadow_pos.w, 1e-6);
    proj = proj * 0.5 + 0.5;
    if (proj.x <= 0.0 || proj.x >= 1.0 || proj.y <= 0.0 || proj.y >= 1.0 || proj.z <= 0.0 || proj.z >= 1.0)
    {
        return 1.0;
    }

    float current_depth = proj.z;
    float bias = max(0.0008 * (1.0 - dot(N, L)), 0.0002);

    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    float base_radius = 1.25;
    float normal_factor = 1.0 - max(dot(N, L), 0.0);
    float radius = mix(base_radius, 4.0, normal_factor);

    float shadow_sum = 0.0;
    float weight_sum = 0.0;
    for (int y = -2; y <= 2; ++y)
    {
        for (int x = -2; x <= 2; ++x)
        {
            vec2 o = vec2(float(x), float(y));
            float w = 1.0 / (1.0 + dot(o, o));
            float pcf_depth = texture(u_shadow_map, proj.xy + o * texel * radius).r;
            shadow_sum += (current_depth - bias > pcf_depth) ? w : 0.0;
            weight_sum += w;
        }
    }
    float shadow = shadow_sum / max(weight_sum, 1e-6);
    return 1.0 - shadow;
}

void main()
{
    vec3 N = normalize(v_normal_ws);
    vec3 V = normalize(ubo.camera_pos_sun_intensity.xyz - v_world_pos);
    vec3 L = normalize(-ubo.sun_dir_ws_pad.xyz);
    vec3 H = normalize(V + L);

    float metallic = clamp(ubo.base_color_metallic.a, 0.0, 1.0);
    float roughness = clamp(ubo.roughness_ao_emissive_hastex.x, 0.04, 1.0);
    float ao = clamp(ubo.roughness_ao_emissive_hastex.y, 0.0, 1.0);
    float emissive_intensity = max(ubo.roughness_ao_emissive_hastex.z, 0.0);
    float has_tex = ubo.roughness_ao_emissive_hastex.w;

    vec3 tex_albedo = texture(u_base_color, v_uv).rgb;
    vec3 albedo = ubo.base_color_metallic.rgb * mix(vec3(1.0), tex_albedo, clamp(has_tex, 0.0, 1.0));

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    float NDF = distribution_ggx(N, H, roughness);
    float G = geometry_smith(N, V, L, roughness);

    vec3 numerator = NDF * G * F;
    float denominator = max(4.0 * NdotV * NdotL, 1e-6);
    vec3 specular = numerator / denominator;

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    float shadow = soft_shadow(N, L);

    vec3 sun_color = vec3(1.0);
    float sun_intensity = max(ubo.camera_pos_sun_intensity.w, 0.0);
    vec3 radiance = sun_color * sun_intensity;

    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL * shadow;
    vec3 ambient = 0.03 * albedo * ao;
    vec3 emissive = albedo * emissive_intensity;

    vec3 hdr = max(ambient + Lo + emissive, vec3(0.0));

    vec2 curr_ndc = v_curr_clip.xy / max(v_curr_clip.w, 1e-6);
    vec2 prev_ndc = v_prev_clip.xy / max(v_prev_clip.w, 1e-6);

    out_hdr = vec4(hdr, 1.0);
    out_velocity = curr_ndc - prev_ndc;
}
