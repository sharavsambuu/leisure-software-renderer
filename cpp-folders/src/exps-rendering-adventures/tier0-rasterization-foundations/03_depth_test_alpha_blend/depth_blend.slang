// tier0 demo 03 — depth test + alpha blending + draw-order sensitivity.
// Alpha arrives in vertex color; the _vk pipeline configures SRC_ALPHA /
// ONE_MINUS_SRC_ALPHA blending, the _sw twin blends with the same factors
// on its own z-buffer.

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
    float4   params;
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
