// tier1 demo 08 — normal mapping shader.
// Geometric normal arrives in COLOR0.xyz (T0Vertex reuse); the tangent frame
// is constructed per pixel exactly like the _sw twin (T = normalize(cross(
// up, N))). params.x selects flat vs mapped; params.yzw is the pre-
// normalized light dir. Albedo/ambient/bump-strength literals mirror the _sw.

struct VSInput
{
    float3 position : POSITION;
    float4 color    : COLOR0; // xyz = geometric normal
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
    float4   params; // x: use_nmap, yzw: light dir (pre-normalized)
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] Sampler2D bump0;

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
    const float3 albedo = float3(0.85, 0.87, 0.90);
    const float  ambient = 0.08;
    const float3 up = float3(0.0, 1.0, 0.0);

    float3 n = normalize(input.color.xyz);
    float3 shade_n = n;
    if (pc.params.x > 0.5)
    {
        float3 t = normalize(cross(up, n));
        float3 b = cross(n, t);
        float3 n_t = bump0.Sample(input.uv).xyz * 2.0 - float3(1.0, 1.0, 1.0);
        shade_n = normalize(t * n_t.x + b * n_t.y + n * n_t.z);
    }

    const float3 light = pc.params.yzw;
    const float  diff = max(dot(shade_n, light), 0.0);
    return float4(albedo * diff + float3(ambient, ambient, ambient), 1.0);
}
