// tier0 demo 02 — perspective vs orthographic projection.
// One shader, two draws: each draw receives its own MVP via push constants
// (left cube: glm::perspective, right cube: glm::ortho). The math difference
// is entirely in the matrices — the _sw twin recomputes the same MVPs with GLM.

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
    // row-vector form: the compiler emits OpVectorTimesMatrix(v, M); combined
    // with the RowMajor std430 push-constant layout, the host must upload M^T
    // so the bytes decode as the matrix the GLM _sw twin used (see
    // make_push_row_major).
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
