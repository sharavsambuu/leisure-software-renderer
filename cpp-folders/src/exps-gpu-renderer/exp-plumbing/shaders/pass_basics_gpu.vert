#version 450

layout(push_constant) uniform PushConsts
{
    mat4 mvp;
    vec4 base_color;
} pc;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_normal;

void main()
{
    gl_Position = pc.mvp * vec4(in_pos, 1.0);
    v_color = pc.base_color.rgb;
    v_normal = normalize(in_normal);
}
