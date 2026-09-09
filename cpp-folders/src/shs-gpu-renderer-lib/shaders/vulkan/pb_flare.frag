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

float luma(vec3 c)
{
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float sun_visibility_fade(vec2 sun_uv)
{
    vec2 in_min = smoothstep(vec2(-0.05), vec2(0.05), sun_uv);
    vec2 in_max = 1.0 - smoothstep(vec2(0.95), vec2(1.05), sun_uv);
    return clamp(in_min.x * in_min.y * in_max.x * in_max.y, 0.0, 1.0);
}

float in_bounds(vec2 uv)
{
    vec2 m0 = step(vec2(0.0), uv);
    vec2 m1 = step(uv, vec2(1.0));
    return m0.x * m0.y * m1.x * m1.y;
}

float edge_fade(vec2 uv)
{
    vec2 fade_in = smoothstep(vec2(0.01), vec2(0.08), uv);
    vec2 fade_out = 1.0 - smoothstep(vec2(0.92), vec2(0.99), uv);
    return fade_in.x * fade_in.y * fade_out.x * fade_out.y;
}

float sample_sun_luma(vec2 sun_uv, vec2 texel)
{
    vec2 suv = clamp(sun_uv, vec2(0.001), vec2(0.999));
    vec3 accum = texture(u_bright, suv).rgb;
    accum += texture(u_bright, clamp(suv + vec2(texel.x, 0.0), vec2(0.001), vec2(0.999))).rgb;
    accum += texture(u_bright, clamp(suv - vec2(texel.x, 0.0), vec2(0.001), vec2(0.999))).rgb;
    accum += texture(u_bright, clamp(suv + vec2(0.0, texel.y), vec2(0.001), vec2(0.999))).rgb;
    accum += texture(u_bright, clamp(suv - vec2(0.0, texel.y), vec2(0.001), vec2(0.999))).rgb;
    return luma(accum * 0.2);
}

vec3 sample_chromatic(vec2 uv, vec2 dir_texel, float shift)
{
    vec2 d = dir_texel * shift;
    float r = texture(u_bright, uv + d).r;
    float g = texture(u_bright, uv).g;
    float b = texture(u_bright, uv - d).b;
    return vec3(r, g, b);
}

void main()
{
    vec2 uv = v_uv;
    vec2 center = vec2(0.5, 0.5);
    vec2 texel = 1.0 / vec2(textureSize(u_bright, 0));
    float sun_fade = sun_visibility_fade(pc.sun_uv);
    float sun_luma = sample_sun_luma(pc.sun_uv, texel * 4.0);
    float sensitivity = smoothstep(0.010, 0.120, sun_luma);
    float gain = pc.intensity * sun_fade * sensitivity;

    if (gain <= 1e-6)
    {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec2 ghost_vec = (center - uv) * 0.42;

    vec3 flare = vec3(0.0);
    int ghost_count = max(pc.ghosts, 1);
    float ghost_denom = float(max(ghost_count - 1, 1));

    for (int i = 0; i < ghost_count; ++i)
    {
        float fi = float(i);
        vec2 raw_uv = uv + ghost_vec * (fi + 1.0);
        float bounds = in_bounds(raw_uv);
        vec2 suv = clamp(raw_uv, vec2(0.001), vec2(0.999));
        float dist_center = distance(suv, center) / 0.7071;
        float radial_weight = pow(max(1.0 - dist_center, 0.0), 2.2);
        float ghost_falloff = mix(1.0, 0.55, fi / ghost_denom);
        vec2 dir = normalize(suv - center + vec2(1e-5));
        flare += sample_chromatic(suv, dir * texel, pc.chroma_shift) * radial_weight * ghost_falloff * bounds;
    }

    vec2 to_sun = normalize(pc.sun_uv - center + vec2(1e-5));
    vec2 halo_center = center - to_sun * 0.28;
    float ring = 1.0 - abs(distance(uv, halo_center) - 0.22) / 0.22;
    float halo_mask = pow(clamp(ring, 0.0, 1.0), 2.0);
    vec2 halo_uv = clamp(uv + to_sun * 0.06, vec2(0.001), vec2(0.999));
    vec3 halo = sample_chromatic(halo_uv, to_sun * texel, pc.chroma_shift * 0.65);
    float halo_gain = pc.halo_intensity * sun_fade * sensitivity;

    vec3 outc = (flare * gain + halo * halo_gain * halo_mask) * edge_fade(uv);
    out_color = vec4(max(outc, vec3(0.0)), 1.0);
}
