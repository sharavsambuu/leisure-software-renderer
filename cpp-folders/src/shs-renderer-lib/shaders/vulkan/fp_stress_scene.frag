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
    mat4 sun_shadow_view_proj;
    vec4 sun_shadow_params;       // x: strength, y: bias_const, z: bias_slope, w: pcf_radius
    vec4 sun_shadow_filter;       // x: pcf_step, y: enabled
} ubo;

struct CullingLightGPU
{
    vec4 position_range;
    vec4 color_intensity;
    vec4 direction_spot;
    vec4 axis_spot_outer;
    vec4 up_shape_x;
    vec4 shape_attenuation;
    uvec4 type_shape_flags;
};

struct ShadowLightGPU
{
    mat4 light_view_proj;
    vec4 position_range; // xyz: light pos, w: range/far
    vec4 shadow_params;  // x: strength, y: bias_const, z: bias_slope, w: pcf_radius
    uvec4 meta;          // x: shadow technique, y: layer base, z: reserved, w: enabled
};

layout(set = 0, binding = 1, std430) readonly buffer LightBuffer
{
    CullingLightGPU lights[];
} light_buffer;

layout(set = 0, binding = 2, std430) readonly buffer TileCounts
{
    uint tile_counts[];
};

layout(set = 0, binding = 3, std430) readonly buffer TileIndices
{
    uint tile_indices[];
};

layout(set = 0, binding = 6) uniform sampler2D u_sun_shadow_map;
layout(set = 0, binding = 7) uniform sampler2DArray u_local_shadow_map;
layout(set = 0, binding = 8) uniform sampler2DArray u_point_shadow_faces;

layout(set = 0, binding = 9, std430) readonly buffer ShadowBuffer
{
    ShadowLightGPU shadow_lights[];
} shadow_buffer;

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
const uint LIGHT_TYPE_POINT = 1u;
const uint LIGHT_TYPE_SPOT = 2u;
const uint LIGHT_TYPE_RECT_AREA = 3u;
const uint LIGHT_TYPE_TUBE_AREA = 4u;
const uint LIGHT_FLAG_ENABLED = 1u;
const uint LIGHT_FLAG_AFFECTS_SHADOWS = 8u;
const uint LIGHT_ATTEN_LINEAR = 0u;
const uint LIGHT_ATTEN_SMOOTH = 1u;
const uint LIGHT_ATTEN_INVERSE_SQUARE = 2u;
const uint SHADOW_TECH_NONE = 0u;
const uint SHADOW_TECH_DIRECTIONAL = 1u;
const uint SHADOW_TECH_SPOT = 2u;
const uint SHADOW_TECH_POINT = 3u;
const uint SHADOW_TECH_AREA_PROXY = 4u;

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

float eval_light_attenuation(
    float dist,
    float range,
    uint att_model,
    float att_power,
    float att_bias,
    float att_cutoff)
{
    float r = max(range, 1e-4);
    float t = clamp(dist / r, 0.0, 1.0);
    float edge = 1.0 - t;
    float a = 0.0;

    if (att_model == LIGHT_ATTEN_LINEAR)
    {
        a = edge;
    }
    else if (att_model == LIGHT_ATTEN_INVERSE_SQUARE)
    {
        float d2 = max(dist * dist, max(att_bias, 1e-5));
        float inv = (r * r) / d2;
        a = inv * edge * edge;
    }
    else
    {
        a = edge * edge;
    }

    a = pow(max(a, 0.0), max(att_power, 0.001));
    if (a <= max(att_cutoff, 0.0)) return 0.0;
    return a;
}

float shadow_bias(float ndotl, float bias_const, float bias_slope)
{
    float slope = 1.0 - clamp(ndotl, 0.0, 1.0);
    return bias_const + bias_slope * slope;
}

bool project_shadow_uvz(mat4 light_vp, vec3 pos_ws, out vec2 uv, out float z01)
{
    vec4 p = light_vp * vec4(pos_ws, 1.0);
    if (abs(p.w) < 1e-8) return false;
    vec3 ndc = p.xyz / p.w;
    uv = ndc.xy * 0.5 + 0.5;
    z01 = ndc.z * 0.5 + 0.5;
    return true;
}

float depth01_to_view(float d, float near_z, float far_z)
{
    float n = max(near_z, 0.001);
    float f = max(far_z, n + 0.01);
    float denom = max(f - d * (f - n), 1e-5);
    return (n * f) / denom;
}

float sample_shadow_pcf_2d(
    sampler2D sm,
    vec2 uv,
    float z_test,
    int radius,
    float pcf_step)
{
    vec2 texel = 1.0 / vec2(textureSize(sm, 0));
    float lit = 0.0;
    float cnt = 0.0;
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            vec2 o = vec2(float(x), float(y)) * texel * pcf_step;
            float ref_depth = texture(sm, uv + o).r;
            lit += (z_test <= ref_depth) ? 1.0 : 0.0;
            cnt += 1.0;
        }
    }
    return (cnt > 0.0) ? (lit / cnt) : 1.0;
}

float sample_shadow_pcf_2d_array(
    sampler2DArray sm,
    vec2 uv,
    float layer,
    float z_test,
    int radius,
    float pcf_step)
{
    vec2 texel = 1.0 / vec2(textureSize(sm, 0).xy);
    float lit = 0.0;
    float cnt = 0.0;
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            vec2 o = vec2(float(x), float(y)) * texel * pcf_step;
            float ref_depth = texture(sm, vec3(uv + o, layer)).r;
            lit += (z_test <= ref_depth) ? 1.0 : 0.0;
            cnt += 1.0;
        }
    }
    return (cnt > 0.0) ? (lit / cnt) : 1.0;
}

float eval_directional_shadow(vec3 N, vec3 L)
{
    if (ubo.sun_shadow_filter.y < 0.5) return 1.0;

    vec2 uv = vec2(0.0);
    float z01 = 0.0;
    if (!project_shadow_uvz(ubo.sun_shadow_view_proj, v_world_pos, uv, z01)) return 1.0;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || z01 <= 0.0 || z01 >= 1.0) return 1.0;

    int radius = int(max(0.0, ubo.sun_shadow_params.w));
    float pcf_step = max(0.5, ubo.sun_shadow_filter.x);
    float bias = shadow_bias(max(dot(N, L), 0.0), ubo.sun_shadow_params.y, ubo.sun_shadow_params.z);
    float vis = sample_shadow_pcf_2d(u_sun_shadow_map, uv, z01 - bias, radius, pcf_step);
    float strength = clamp(ubo.sun_shadow_params.x, 0.0, 1.0);
    return mix(1.0, vis, strength);
}

void point_shadow_face_uv(vec3 dir, out uint face, out vec2 uv, out float view_depth_major)
{
    vec3 ad = abs(dir);
    face = 0u;
    uv = vec2(0.5);
    view_depth_major = 0.0;

    if (ad.x >= ad.y && ad.x >= ad.z)
    {
        view_depth_major = ad.x;
        if (dir.x >= 0.0)
        {
            face = 0u; // +X
            uv = vec2(-dir.z, -dir.y) / ad.x;
        }
        else
        {
            face = 1u; // -X
            uv = vec2(dir.z, -dir.y) / ad.x;
        }
    }
    else if (ad.y >= ad.x && ad.y >= ad.z)
    {
        view_depth_major = ad.y;
        if (dir.y >= 0.0)
        {
            face = 2u; // +Y
            uv = vec2(dir.x, dir.z) / ad.y;
        }
        else
        {
            face = 3u; // -Y
            uv = vec2(dir.x, -dir.z) / ad.y;
        }
    }
    else
    {
        view_depth_major = ad.z;
        if (dir.z >= 0.0)
        {
            face = 4u; // +Z
            uv = vec2(dir.x, -dir.y) / ad.z;
        }
        else
        {
            face = 5u; // -Z
            uv = vec2(-dir.x, -dir.y) / ad.z;
        }
    }

    uv = uv * 0.5 + 0.5;
}

float eval_local_shadow(uint idx, vec3 N, vec3 L)
{
    CullingLightGPU light = light_buffer.lights[idx];
    if ((light.type_shape_flags.z & LIGHT_FLAG_AFFECTS_SHADOWS) == 0u) return 1.0;

    ShadowLightGPU s = shadow_buffer.shadow_lights[idx];
    if (s.meta.w == 0u) return 1.0;

    uint tech = s.meta.x;
    float strength = clamp(s.shadow_params.x, 0.0, 1.0);
    if (strength <= 0.0) return 1.0;

    int radius = int(max(0.0, s.shadow_params.w));
    float pcf_step = max(0.5, ubo.sun_shadow_filter.x);
    float bias = shadow_bias(max(dot(N, L), 0.0), s.shadow_params.y, s.shadow_params.z);

    if (tech == SHADOW_TECH_SPOT || tech == SHADOW_TECH_AREA_PROXY)
    {
        vec2 uv = vec2(0.0);
        float z01 = 0.0;
        if (!project_shadow_uvz(s.light_view_proj, v_world_pos, uv, z01)) return 1.0;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || z01 <= 0.0 || z01 >= 1.0) return 1.0;
        float layer = float(s.meta.y);
        float vis = sample_shadow_pcf_2d_array(u_local_shadow_map, uv, layer, z01 - bias, radius, pcf_step);
        return mix(1.0, vis, strength);
    }

    if (tech == SHADOW_TECH_POINT)
    {
        vec3 rel = v_world_pos - s.position_range.xyz;
        float rel_len = length(rel);
        float range = max(s.position_range.w, 0.1);
        if (rel_len <= 1e-4 || rel_len >= range) return 1.0;

        vec3 dir = rel / rel_len;
        uint face = 0u;
        vec2 uv = vec2(0.5);
        float view_depth_major = 0.0;
        point_shadow_face_uv(dir, face, uv, view_depth_major);

        uint layer = s.meta.y + min(face, 5u);
        vec2 texel = 1.0 / vec2(textureSize(u_point_shadow_faces, 0).xy);
        float near_z = 0.05;
        float z_test = max(0.0, view_depth_major - bias);

        float lit = 0.0;
        float cnt = 0.0;
        for (int y = -radius; y <= radius; ++y)
        {
            for (int x = -radius; x <= radius; ++x)
            {
                vec2 o = vec2(float(x), float(y)) * texel * pcf_step;
                float d01 = texture(u_point_shadow_faces, vec3(uv + o, float(layer))).r;
                float ref_depth = depth01_to_view(d01, near_z, range);
                lit += (z_test <= ref_depth) ? 1.0 : 0.0;
                cnt += 1.0;
            }
        }

        float vis = (cnt > 0.0) ? (lit / cnt) : 1.0;
        return mix(1.0, vis, strength);
    }

    return 1.0;
}

vec3 eval_local_light(uint idx, vec3 N, vec3 V, vec3 albedo, float metallic, float roughness)
{
    CullingLightGPU light = light_buffer.lights[idx];
    uint light_type = light.type_shape_flags.x;
    uint flags = light.type_shape_flags.z;
    uint att_model = light.type_shape_flags.w;
    if ((flags & LIGHT_FLAG_ENABLED) == 0u) return vec3(0.0);
    if (light_type != LIGHT_TYPE_POINT &&
        light_type != LIGHT_TYPE_SPOT &&
        light_type != LIGHT_TYPE_RECT_AREA &&
        light_type != LIGHT_TYPE_TUBE_AREA)
    {
        return vec3(0.0);
    }

    vec3 light_sample_pos = light.position_range.xyz;
    float range = max(light.position_range.w, 0.001);
    vec3 local_to_light = light_sample_pos - v_world_pos;
    float dist = length(local_to_light);

    if (light_type == LIGHT_TYPE_RECT_AREA)
    {
        vec3 right = normalize(light.axis_spot_outer.xyz);
        vec3 up = normalize(light.up_shape_x.xyz);
        float hx = max(light.up_shape_x.w, 1e-4);
        float hy = max(light.shape_attenuation.x, 1e-4);
        vec3 rel = v_world_pos - light.position_range.xyz;
        float x = clamp(dot(rel, right), -hx, hx);
        float y = clamp(dot(rel, up), -hy, hy);
        light_sample_pos = light.position_range.xyz + right * x + up * y;
        local_to_light = light_sample_pos - v_world_pos;
        dist = length(local_to_light);
    }
    else if (light_type == LIGHT_TYPE_TUBE_AREA)
    {
        vec3 axis = normalize(light.axis_spot_outer.xyz);
        float half_len = max(light.up_shape_x.w, 1e-4);
        vec3 p0 = light.position_range.xyz - axis * half_len;
        vec3 p1 = light.position_range.xyz + axis * half_len;
        vec3 seg = p1 - p0;
        float seg_len2 = max(dot(seg, seg), 1e-6);
        float u = clamp(dot(v_world_pos - p0, seg) / seg_len2, 0.0, 1.0);
        light_sample_pos = p0 + seg * u;
        local_to_light = light_sample_pos - v_world_pos;
        dist = length(local_to_light);
    }

    if (dist >= range) return vec3(0.0);
    vec3 L = local_to_light / max(dist, 1e-5);
    float atten = eval_light_attenuation(
        dist,
        range,
        att_model,
        light.shape_attenuation.y,
        light.shape_attenuation.z,
        light.shape_attenuation.w);
    if (atten <= 0.0) return vec3(0.0);

    if (light_type == LIGHT_TYPE_SPOT)
    {
        vec3 spot_dir = normalize(light.direction_spot.xyz);
        float inner_cos = clamp(light.direction_spot.w, -1.0, 1.0);
        float outer_cos = clamp(light.axis_spot_outer.w, -1.0, inner_cos);
        float cone_cos = dot(spot_dir, -L);
        float spot = smoothstep(outer_cos, inner_cos, cone_cos);
        if (spot <= 0.0) return vec3(0.0);
        atten *= spot;
    }
    else if (light_type == LIGHT_TYPE_RECT_AREA)
    {
        vec3 emit_dir = normalize(light.direction_spot.xyz);
        float one_sided = max(dot(emit_dir, v_world_pos - light_sample_pos), 0.0);
        if (one_sided <= 0.0) return vec3(0.0);
        atten *= one_sided;
    }
    else if (light_type == LIGHT_TYPE_TUBE_AREA)
    {
        float tube_radius = max(light.shape_attenuation.x, 1e-4);
        float edge_soften = clamp(tube_radius / max(range, 1e-4), 0.05, 1.0);
        atten *= mix(0.65, 1.0, edge_soften);
    }

    float shadow_vis = eval_local_shadow(idx, N, L);
    vec3 radiance = light.color_intensity.rgb * light.color_intensity.a * atten * shadow_vis;
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
    float sun_vis = eval_directional_shadow(N, Ld);
    color += eval_pbr_light(N, V, Ld, sun_radiance * sun_vis, albedo, metallic, roughness);

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
            color += eval_local_light(idx, N, V, albedo, metallic, roughness);
        }
    }
    else
    {
        for (uint i = 0u; i < light_count; ++i)
        {
            color += eval_local_light(i, N, V, albedo, metallic, roughness);
        }
    }

    float exposure = max(ubo.exposure_gamma.x, 0.0001);
    float inv_gamma = 1.0 / max(ubo.exposure_gamma.y, 0.001);
    vec3 mapped = (color * exposure) / (vec3(1.0) + color * exposure);
    mapped = pow(mapped, vec3(inv_gamma));
    out_color = vec4(mapped, 1.0);
}

