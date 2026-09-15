// tier0 demo 04 — texture sampling: nearest vs bilinear, plus scissor test.
// The CPU generates a checkerboard RGBA8 texture; the _vk path samples it via
// a combined image sampler (set 0, binding 0) whose filtering differs per
// descriptor set. The _sw twin implements both filters by hand over the same
// texel data.

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
    float4   params; // params.x unused here, kept for cross-demo symmetry
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] Sampler2D tex0;

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
    return tex0.Sample(input.uv);
}
