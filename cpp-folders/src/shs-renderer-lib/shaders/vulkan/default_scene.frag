#version 450

layout(set = 0, binding = 0) uniform sampler2D u_base_color;

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

void main()
{
    vec3 L = normalize(vec3(-0.35, -1.0, -0.25));
    float lambert = max(dot(normalize(v_normal), -L), 0.0);
    vec3 tex_color = texture(u_base_color, v_uv).rgb;
    vec3 albedo = v_color * tex_color;
    vec3 lit = albedo * (0.20 + 0.80 * lambert);
    out_color = vec4(lit, 1.0);
}
