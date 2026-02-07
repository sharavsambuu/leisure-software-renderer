#version 450

layout(set = 0, binding = 0) uniform sampler2D u_bright;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform FlarePush
{
    vec2 sun_uv;
    float intensity;
    float halo_intensity;
    float chroma_shift;
    int ghosts;
} pc;

vec3 sample_chromatic(vec2 uv, vec2 dir, float shift)
{
    vec2 d = dir * shift;
    float r = texture(u_bright, uv + d).r;
    float g = texture(u_bright, uv).g;
    float b = texture(u_bright, uv - d).b;
    return vec3(r, g, b);
}

void main()
{
    vec2 uv = v_uv;
    vec2 center = vec2(0.5, 0.5);
    vec2 ghost_vec = (center - uv) * 0.55;

    vec3 flare = vec3(0.0);
    int ghost_count = max(pc.ghosts, 1);

    for (int i = 0; i < ghost_count; ++i)
    {
        vec2 suv = fract(uv + ghost_vec * float(i + 1));
        float dist_center = distance(suv, center) / 0.7071;
        float weight = pow(max(1.0 - dist_center, 0.0), 2.5);
        vec2 dir = normalize(suv - center + vec2(1e-5));
        flare += sample_chromatic(suv, dir / 1024.0, pc.chroma_shift) * weight;
    }

    vec2 to_sun = normalize(pc.sun_uv - center + vec2(1e-5));
    vec2 halo_uv = center - to_sun * 0.32;
    vec3 halo = texture(u_bright, halo_uv).rgb;
    float halo_mask = pow(max(1.0 - distance(uv, center) / 0.7071, 0.0), 1.5);

    vec3 outc = flare * pc.intensity + halo * pc.halo_intensity * halo_mask;
    out_color = vec4(max(outc, vec3(0.0)), 1.0);
}
