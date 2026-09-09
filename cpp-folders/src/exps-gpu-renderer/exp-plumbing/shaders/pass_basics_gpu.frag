#version 450

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;
layout(location = 0) out vec4 out_color;

void main()
{
    vec3 L = normalize(vec3(-0.35, -1.0, -0.25));
    float lambert = max(dot(normalize(v_normal), -L), 0.0);
    vec3 lit = v_color * (0.20 + 0.80 * lambert);
    out_color = vec4(lit, 1.0);
}
