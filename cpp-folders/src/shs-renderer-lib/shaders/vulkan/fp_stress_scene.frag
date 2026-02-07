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

layout(set = 0, binding = 1, std430) readonly buffer LightPosRadius
{
    vec4 pos_radius[];
} light_pos_radius;

layout(set = 0, binding = 2, std430) readonly buffer LightColorIntensity
{
    vec4 color_intensity[];
} light_color_intensity;

layout(set = 0, binding = 3, std430) readonly buffer TileCounts
{
    uint tile_counts[];
};

layout(set = 0, binding = 4, std430) readonly buffer TileIndices
{
    uint tile_indices[];
};

layout(push_constant) uniform DrawPush
{
    mat4 model;
    vec4 base_color;
    vec4 material_params; // x: metallic, y: roughness, z: ao
} pc;

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal_ws;
layout(location = 2) in vec3 v_base_color;

layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;

float distribution_ggx(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
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
    float ggx1 = geometry_schlick_ggx(NdotV, roughness);
    float ggx2 = geometry_schlick_ggx(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnel_schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 eval_pbr_light(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, float metallic, float roughness)
{
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    float NDF = distribution_ggx(N, H, roughness);
    float G = geometry_smith(N, V, L, roughness);

    vec3 numerator = NDF * G * F;
    float denom = max(4.0 * max(dot(N, V), 0.0) * NdotL, 1e-6);
    vec3 specular = numerator / denom;

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    return (kD * albedo / PI + specular) * radiance * NdotL;
}

vec3 eval_point_light(uint idx, vec3 N, vec3 V, vec3 albedo, float metallic, float roughness)
{
    vec4 pr = light_pos_radius.pos_radius[idx];
    vec4 ci = light_color_intensity.color_intensity[idx];

    vec3 to_l = pr.xyz - v_world_pos;
    float dist = length(to_l);
    float radius = max(pr.w, 0.001);
    if (dist >= radius) return vec3(0.0);

    vec3 L = to_l / max(dist, 1e-5);
    float atten = 1.0 - dist / radius;
    atten *= atten;
    vec3 radiance = ci.rgb * ci.a * atten;
    return eval_pbr_light(N, V, L, radiance, albedo, metallic, roughness);
}

uint cluster_slice_from_view_depth(float view_depth, uint z_slices)
{
    float near_z = max(ubo.depth_params.x, 0.001);
    float far_z = max(ubo.depth_params.y, near_z + 0.01);
    float d = clamp(view_depth, near_z, far_z);
    float t = log(d / near_z) / max(log(far_z / near_z), 1e-6);
    uint zi = uint(clamp(floor(t * float(z_slices)), 0.0, float(z_slices - 1u)));
    return zi;
}

void main()
{
    vec3 albedo = v_base_color;
    float metallic = clamp(pc.material_params.x, 0.0, 1.0);
    float roughness = clamp(pc.material_params.y, 0.04, 1.0);
    float ao = clamp(pc.material_params.z, 0.0, 1.0);

    vec3 N = normalize(v_normal_ws);
    vec3 V = normalize(ubo.camera_pos_time.xyz - v_world_pos);

    vec3 color = albedo * 0.035 * ao;

    vec3 Ld = normalize(-ubo.sun_dir_intensity.xyz);
    vec3 sun_radiance = vec3(1.0, 0.97, 0.92) * max(ubo.sun_dir_intensity.w, 0.0);
    color += eval_pbr_light(N, V, Ld, sun_radiance, albedo, metallic, roughness);

    uint light_count = ubo.screen_tile_lightcount.w;
    uint tiles_x = max(ubo.screen_tile_lightcount.z, 1u);
    uint tiles_y = max(ubo.params.x, 1u);
    uint max_per_tile = max(ubo.params.y, 1u);
    uint tile_size = max(ubo.params.z, 1u);
    uint culling_mode = ubo.params.w;

    if (culling_mode != 0u)
    {
        uvec2 tile = uvec2(gl_FragCoord.xy) / tile_size;
        tile.x = min(tile.x, tiles_x - 1u);
        tile.y = min(tile.y, tiles_y - 1u);
        uint list_id = tile.y * tiles_x + tile.x;
        if (culling_mode == 3u)
        {
            uint z_slices = max(ubo.culling_params.x, 1u);
            float view_depth = max(0.001, -(ubo.view * vec4(v_world_pos, 1.0)).z);
            uint zi = cluster_slice_from_view_depth(view_depth, z_slices);
            list_id = (zi * tiles_y + tile.y) * tiles_x + tile.x;
        }
        uint count = min(tile_counts[list_id], max_per_tile);
        uint base = list_id * max_per_tile;
        for (uint i = 0u; i < count; ++i)
        {
            uint idx = tile_indices[base + i];
            if (idx >= light_count) continue;
            color += eval_point_light(idx, N, V, albedo, metallic, roughness);
        }
    }
    else
    {
        for (uint i = 0u; i < light_count; ++i)
        {
            color += eval_point_light(i, N, V, albedo, metallic, roughness);
        }
    }

    float exposure = max(ubo.exposure_gamma.x, 0.0001);
    float inv_gamma = 1.0 / max(ubo.exposure_gamma.y, 0.001);
    vec3 mapped = (color * exposure) / (vec3(1.0) + color * exposure);
    mapped = pow(mapped, vec3(inv_gamma));
    out_color = vec4(mapped, 1.0);
}
