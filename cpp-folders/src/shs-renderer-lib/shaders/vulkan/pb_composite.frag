#version 450

layout(set = 0, binding = 0) uniform sampler2D u_scene_hdr;
layout(set = 0, binding = 1) uniform sampler2D u_velocity;
layout(set = 0, binding = 2) uniform sampler2D u_shafts;
layout(set = 0, binding = 3) uniform sampler2D u_flare;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform CompositePush
{
    vec2 inv_size;
    float mb_strength;
    float shafts_strength;
    float flare_strength;
    int mb_samples;
    float exposure;
    float gamma;
} pc;

vec3 sample_motion_blur(vec2 uv)
{
    vec2 vel_ndc = texture(u_velocity, uv).xy;
    vec2 vel_uv = vel_ndc * 0.5 * pc.mb_strength;

    int samples = max(pc.mb_samples, 1);
    if (samples <= 1)
    {
        return texture(u_scene_hdr, uv).rgb;
    }

    vec3 accum = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < samples; ++i)
    {
        float t = (float(i) / float(samples - 1)) * 2.0 - 1.0;
        float w = 1.0 - abs(t);
        vec2 suv = uv + vel_uv * t;
        accum += texture(u_scene_hdr, suv).rgb * w;
        wsum += w;
    }
    return accum / max(wsum, 1e-6);
}

void main()
{
    vec3 scene_blur = sample_motion_blur(v_uv);
    vec3 shafts = texture(u_shafts, v_uv).rgb * pc.shafts_strength;
    vec3 flare = texture(u_flare, v_uv).rgb * pc.flare_strength;

    vec3 hdr = max(scene_blur + shafts + flare, vec3(0.0));

    vec3 mapped = (hdr * pc.exposure) / (vec3(1.0) + hdr * pc.exposure);
    mapped = pow(mapped, vec3(1.0 / max(pc.gamma, 1e-3)));

    out_color = vec4(mapped, 1.0);
}
