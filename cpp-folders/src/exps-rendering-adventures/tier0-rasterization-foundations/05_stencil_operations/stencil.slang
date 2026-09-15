// tier0 demo 05 — stencil buffer operations.
// Same shader for both stencil configurations: pipeline A writes stencil ref
// 1 where the triangle is drawn, pipeline B draws the quad only where the
// stencil test passes (front == 1). The _sw twin runs the same two-state
// machine on a CPU-side stencil buffer.

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
    // row-vector form: mul(float4(pos,1), pc.mvp) == M*v with the transposed
    // (RowMajor-decoded) push bytes uploaded by make_push_row_major; slangc
    // canonicalizes this to OpMatrixTimesVector (verified via spirv-dis).
    out_.position = mul(float4(input.position, 1.0), pc.mvp);
    out_.color    = input.color;
    out_.uv       = input.uv;
    return out_;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target
{
    return input.color;
}
