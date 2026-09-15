#version 450

layout(set = 0, binding = 0) uniform sampler2D u_sky;

layout(push_constant) uniform SkyPush
{
    mat4 inv_viewproj;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_velocity;

const float PI = 3.14159265359;
const float INV_TWO_PI = 1.0 / (2.0 * PI);
const float SKY_EXPOSURE = 1.0;

void main()
{
    vec2 ndc = v_uv * 2.0 - 1.0;

    // inv_viewproj uses a translation-free view matrix, so unprojected point
    // direction is stable and independent from camera world position.
    vec4 p = pc.inv_viewproj * vec4(ndc, 1.0, 1.0);
    vec3 dir = normalize(p.xyz / max(p.w, 1e-6));

    float lon = atan(dir.z, dir.x);
    float lat = asin(clamp(dir.y, -1.0, 1.0));
    float u = lon * INV_TWO_PI + 0.5;
    float v = 0.5 - (lat / PI);

    vec3 sky = texture(u_sky, vec2(u, v)).rgb * SKY_EXPOSURE;
    out_color = vec4(sky, 1.0);
    out_velocity = vec2(0.0);
}
