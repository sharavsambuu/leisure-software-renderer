#version 450

layout(set = 0, binding = 0) uniform sampler2D u_bright;
layout(set = 0, binding = 1) uniform sampler2D u_depth;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform ShaftsPush
{
    vec2 sun_uv;
    float intensity;
    float density;
    float decay;
    float weight;
    float exposure;
    int steps;
} pc;

float sun_visibility_fade(vec2 sun_uv)
{
    vec2 in_min = smoothstep(vec2(-0.05), vec2(0.05), sun_uv);
    vec2 in_max = 1.0 - smoothstep(vec2(0.95), vec2(1.05), sun_uv);
    return clamp(in_min.x * in_min.y * in_max.x * in_max.y, 0.0, 1.0);
}

void main()
{
    float sun_fade = sun_visibility_fade(pc.sun_uv);
    if (sun_fade <= 1e-4 || pc.intensity <= 1e-6 || pc.steps <= 0)
    {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec2 uv = v_uv;
    vec2 delta = (pc.sun_uv - uv) * (pc.density / float(max(pc.steps, 1)));
    // Limit per-step delta so off-screen sun does not create unstable streaks.
    delta = clamp(delta, vec2(-0.02), vec2(0.02));

    vec3 accum = vec3(0.0);
    float illum_decay = 1.0;
    vec2 coord = uv;

    for (int i = 0; i < pc.steps; ++i)
    {
        coord += delta;
        vec2 suv = clamp(coord, vec2(0.001), vec2(0.999));
        vec3 sample_c = texture(u_bright, suv).rgb;
        float d = texture(u_depth, suv).r;
        // Smooth sky/geometry transition to reduce edge flicker.
        float occ = smoothstep(0.995, 0.9997, d);
        accum += sample_c * illum_decay * pc.weight * occ;
        illum_decay *= pc.decay;
    }

    vec3 shafts = accum * pc.exposure * pc.intensity * sun_fade;
    out_color = vec4(max(shafts, vec3(0.0)), 1.0);
}
