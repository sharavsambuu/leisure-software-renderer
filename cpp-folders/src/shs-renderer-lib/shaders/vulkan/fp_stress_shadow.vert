#version 450

layout(push_constant) uniform ShadowPush
{
    mat4 light_view_proj;
    mat4 model;
} pc;

layout(location = 0) in vec3 in_pos;

void main()
{
    gl_Position = pc.light_view_proj * pc.model * vec4(in_pos, 1.0);
}

