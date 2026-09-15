// tier0 demo 01 — triangle rasterization with barycentric varying interpolation.
// Slang realization of the GLSL-frozen reference semantics: vertex stage emits
// clip-space position + per-vertex color; the rasterizer interpolates color
// across the face (barycentric weights computed on GPU — the _sw twin of this
// demo implements the same weights by hand).

struct VSInput
{
    float3 position : POSITION;
    float4 color    : COLOR0;
    float2 uv       : TEXCOORD0;
};

struct VSOutput
{
    float4 position : SV_Position;
    float4 color    : COLOR0;
    float2 uv       : TEXCOORD0;
};

struct PushConstants
{
    float4x4 mvp;    // column-major (see -matrix-layout-column-major pin)
    float4   params; // spare uniforms, kept identical across tier0 demos
};

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
VSOutput vs_main(VSInput input)
{
    VSOutput out_;
    out_.position = mul(pc.mvp, float4(input.position, 1.0));
    out_.color    = input.color;
    out_.uv       = input.uv;
    return out_;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target
{
    return input.color;
}
