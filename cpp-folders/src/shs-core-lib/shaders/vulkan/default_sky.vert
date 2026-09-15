#version 450

layout(location = 0) out vec2 v_ndc;

void main()
{
    const vec2 ndc[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 p = ndc[gl_VertexIndex];
    gl_Position = vec4(p, 0.0, 1.0);
    v_ndc = p;
}
