#version 450

layout(set = 0, binding = 0) uniform sampler2D u_input;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform FxaaPush
{
    vec2 inv_size;
} pc;

float luma(vec3 c)
{
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main()
{
    vec2 px = pc.inv_size;

    vec3 rgb_m = texture(u_input, v_uv).rgb;
    vec3 rgb_nw = texture(u_input, v_uv + vec2(-px.x, -px.y)).rgb;
    vec3 rgb_ne = texture(u_input, v_uv + vec2( px.x, -px.y)).rgb;
    vec3 rgb_sw = texture(u_input, v_uv + vec2(-px.x,  px.y)).rgb;
    vec3 rgb_se = texture(u_input, v_uv + vec2( px.x,  px.y)).rgb;

    float l_m = luma(rgb_m);
    float l_nw = luma(rgb_nw);
    float l_ne = luma(rgb_ne);
    float l_sw = luma(rgb_sw);
    float l_se = luma(rgb_se);

    float l_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    float l_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));

    vec2 dir;
    dir.x = -((l_nw + l_ne) - (l_sw + l_se));
    dir.y =  ((l_nw + l_sw) - (l_ne + l_se));

    float dir_reduce = max((l_nw + l_ne + l_sw + l_se) * (0.25 * (1.0 / 8.0)), 1.0 / 128.0);
    float rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);
    dir = clamp(dir * rcp_dir_min, vec2(-8.0), vec2(8.0)) * px;

    vec3 rgb_a = 0.5 * (
        texture(u_input, v_uv + dir * (1.0 / 3.0 - 0.5)).rgb +
        texture(u_input, v_uv + dir * (2.0 / 3.0 - 0.5)).rgb
    );

    vec3 rgb_b = rgb_a * 0.5 + 0.25 * (
        texture(u_input, v_uv + dir * -0.5).rgb +
        texture(u_input, v_uv + dir * 0.5).rgb
    );

    float l_b = luma(rgb_b);
    vec3 outc = (l_b < l_min || l_b > l_max) ? rgb_a : rgb_b;
    out_color = vec4(outc, 1.0);
}
