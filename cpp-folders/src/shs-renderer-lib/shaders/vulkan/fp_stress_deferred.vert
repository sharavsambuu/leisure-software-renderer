#version 450

void main()
{
    vec2 pos = vec2(-1.0, -1.0);
    if (gl_VertexIndex == 1) pos = vec2(3.0, -1.0);
    if (gl_VertexIndex == 2) pos = vec2(-1.0, 3.0);
    gl_Position = vec4(pos, 0.0, 1.0);
}
