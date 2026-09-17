// G2 fixed ABI: no vertex attributes, descriptors, or push constants.
[shader("vertex")]
float4 vs_main(uint vertex : SV_VulkanVertexID) : SV_Position
{
    float2 positions[3] = { float2(-0.5, -0.5), float2(0.5, -0.5), float2(0.0, 0.5) };
    return float4(positions[vertex % 3], 0.0, 1.0);
}

[shader("fragment")]
float4 fs_main() : SV_Target0
{
    return float4(1.0, 0.25, 0.0, 1.0);
}
