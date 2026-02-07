#version 450

layout(push_constant) uniform ShadowPush
{
    mat4 light_mvp;
} pc;

layout(location = 0) in vec3 in_pos;

void main()
{
    gl_Position = pc.light_mvp * vec4(in_pos, 1.0);
}
