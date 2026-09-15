#version 450

layout(set = 1, binding = 1) uniform sampler2D u_gbuffer_normal;
layout(set = 1, binding = 3) uniform sampler2D u_gbuffer_world_pos;

layout(location = 0) out vec4 out_ao;

void main()
{
    vec2 inv_size = 1.0 / vec2(textureSize(u_gbuffer_world_pos, 0));
    vec2 uv = gl_FragCoord.xy * inv_size;

    vec4 center_world = texture(u_gbuffer_world_pos, uv);
    if (center_world.w < 0.5)
    {
        out_ao = vec4(1.0);
        return;
    }

    vec3 N = normalize(texture(u_gbuffer_normal, uv).xyz);
    float occ = 0.0;
    float samples = 0.0;
    const float radius = 3.2;
    const vec2 taps[8] = vec2[](
        vec2(1.0, 0.0),
        vec2(-1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(0.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, -1.0));

    for (int i = 0; i < 8; ++i)
    {
        vec2 suv = clamp(uv + taps[i] * inv_size * radius, vec2(0.0), vec2(1.0));
        vec4 sw = texture(u_gbuffer_world_pos, suv);
        if (sw.w < 0.5) continue;

        vec3 d = sw.xyz - center_world.xyz;
        float dist = length(d);
        if (dist <= 1e-4 || dist > 4.2) continue;

        vec3 dir = d / dist;
        float hemi = dot(N, dir);
        float near_w = clamp(1.0 - (dist / 4.2), 0.0, 1.0);
        occ += smoothstep(-0.24, -0.02, hemi) * near_w;
        samples += 1.0;
    }

    float occ_norm = (samples > 0.0) ? (occ / samples) : 0.0;
    float ao = 1.0 - occ_norm * 1.55;
    ao *= mix(0.78, 1.0, clamp(N.y * 0.5 + 0.5, 0.0, 1.0));
    ao = clamp(ao, 0.22, 1.0);

    out_ao = vec4(ao, ao, ao, 1.0);
}
