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
    vec4 shadow_params;
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
    proj.xy = proj.xy * 0.5 + 0.5;
    if (proj.x <= 0.0 || proj.x >= 1.0 || proj.y <= 0.0 || proj.y >= 1.0 || proj.z <= 0.0 || proj.z >= 1.0)
    {
        return 1.0;
    }

    float current_depth = proj.z;
    float ndotl = max(dot(N, L), 0.0);
    float bias = max(ubo.shadow_params.y + ubo.shadow_params.z * (1.0 - ndotl), 0.00002);
    float pcf_step = max(ubo.shadow_params.w, 0.5);
    int pcf_radius = int(clamp(floor(ubo.sun_dir_ws_pad.w + 0.5), 0.0, 6.0));

    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    if (pcf_radius <= 0)
    {
        float d = texture(u_shadow_map, clamp(proj.xy, vec2(1e-4), vec2(1.0 - 1e-4))).r;
        float vis = (current_depth - bias > d) ? 0.0 : 1.0;
        return mix(1.0, vis, clamp(ubo.shadow_params.x, 0.0, 1.0));
    }

    float shadow_sum = 0.0;
    float weight_sum = 0.0;
    for (int y = -pcf_radius; y <= pcf_radius; ++y)
    {
        for (int x = -pcf_radius; x <= pcf_radius; ++x)
        {
            vec2 o = vec2(float(x), float(y));
            float w = 1.0 / (1.0 + dot(o, o));
            vec2 pcf_uv = clamp(proj.xy + o * texel * pcf_step, vec2(1e-4), vec2(1.0 - 1e-4));
            float pcf_depth = texture(u_shadow_map, pcf_uv).r;
            shadow_sum += (current_depth - bias > pcf_depth) ? w : 0.0;
            weight_sum += w;
        }
    }
    float shadow = shadow_sum / max(weight_sum, 1e-6);
    float vis = 1.0 - shadow;
    return mix(1.0, vis, clamp(ubo.shadow_params.x, 0.0, 1.0));
}

vec3 eval_fake_ibl(vec3 N, vec3 V, vec3 base_color, float metallic, float roughness, float ao)
{
    vec3 n = normalize(N);
    vec3 v = normalize(V);
    vec3 r = reflect(-v, n);

    vec3 sky_zenith = vec3(0.32, 0.46, 0.72);
    vec3 sky_horizon = vec3(0.62, 0.66, 0.72);
    vec3 ground_tint = vec3(0.16, 0.15, 0.14);

    float up_n = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    float up_r = clamp(r.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 env_n = mix(ground_tint, mix(sky_horizon, sky_zenith, up_n), up_n);
    vec3 env_r = mix(ground_tint, mix(sky_horizon, sky_zenith, up_r), up_r);

    float m = clamp(metallic, 0.0, 1.0);
    float rgh = clamp(roughness, 0.0, 1.0);
    vec3 F0 = mix(vec3(0.04), max(base_color, vec3(0.0)), m);
    float fres = pow(1.0 - max(0.0, dot(n, v)), 5.0);
    vec3 F = F0 + (vec3(1.0) - F0) * fres;

    vec3 kd = (vec3(1.0) - F) * (1.0 - m);
    vec3 diffuse_ibl = kd * base_color * env_n * 0.12;
    float spec_strength = 0.02 + (1.0 - rgh) * 0.18;
    vec3 spec_ibl = env_r * F * spec_strength;
    return (diffuse_ibl + spec_ibl) * clamp(ao, 0.0, 1.0);
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
    vec3 ambient = eval_fake_ibl(N, V, albedo, metallic, roughness, ao);
    vec3 emissive = albedo * emissive_intensity;

    vec3 hdr = max(ambient + Lo + emissive, vec3(0.0));

    vec2 velocity = vec2(0.0);
    if (v_curr_clip.w > 1e-4 && v_prev_clip.w > 1e-4)
    {
        vec2 curr_ndc = v_curr_clip.xy / v_curr_clip.w;
        vec2 prev_ndc = v_prev_clip.xy / v_prev_clip.w;
        velocity = curr_ndc - prev_ndc;
        velocity = clamp(velocity, vec2(-1.0), vec2(1.0));
        if (any(isnan(velocity)) || any(isinf(velocity)))
        {
            velocity = vec2(0.0);
        }
    }

    out_hdr = vec4(hdr, 1.0);
    out_velocity = velocity;
}
