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

void main()
{
    vec2 uv = v_uv;
    vec2 delta = (pc.sun_uv - uv) * (pc.density / float(max(pc.steps, 1)));

    vec3 accum = vec3(0.0);
    float illum_decay = 1.0;
    vec2 coord = uv;

    for (int i = 0; i < pc.steps; ++i)
    {
        coord += delta;
        vec3 sample_c = texture(u_bright, coord).rgb;
        float d = texture(u_depth, coord).r;
        float occ = (d > 0.9995) ? 1.0 : 0.15;
        accum += sample_c * illum_decay * pc.weight * occ;
        illum_decay *= pc.decay;
    }

    vec3 shafts = accum * pc.exposure * pc.intensity;
    out_color = vec4(max(shafts, vec3(0.0)), 1.0);
}
