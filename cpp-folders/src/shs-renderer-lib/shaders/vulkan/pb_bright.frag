#version 450

layout(set = 0, binding = 0) uniform sampler2D u_scene_hdr;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform BrightPush
{
    float threshold;
    float intensity;
    float knee;
    float pad;
} pc;

float luma(vec3 c)
{
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

void main()
{
    vec3 c = texture(u_scene_hdr, v_uv).rgb;
    float y = luma(c);
    float soft = max(y - pc.threshold + pc.knee, 0.0);
    soft = (soft * soft) / max(4.0 * pc.knee + 1e-5, 1e-5);
    float w = max(y - pc.threshold, soft) / max(y, 1e-5);
    out_color = vec4(c * w * pc.intensity, 1.0);
}
