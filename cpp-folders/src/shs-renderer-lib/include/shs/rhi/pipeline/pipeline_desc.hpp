#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pipeline_desc.hpp
    МОДУЛЬ: rhi/pipeline
    ЗОРИЛГО: Graphics/Compute pipeline state descriptor-ууд.
            Vulkan дээр pipeline layout болон render state байгуулах contract.
*/


#include <cstdint>
#include "shs/rhi/resource/resource_desc.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace rhi
    {
    enum class RHIShaderStage : uint8_t
    {
        Vertex = 0,
        Fragment = 1,
        Compute = 2
    };

    struct RHIShaderModuleDesc
    {
        RHIShaderStage stage = RHIShaderStage::Vertex;
        const void* bytecode = nullptr;
        uint64_t bytecode_size = 0;
        const char* entry = "main";
    };

    enum class RHICullMode : uint8_t
    {
        None = 0,
        Back = 1,
        Front = 2
    };

    enum class RHIFrontFace : uint8_t
    {
        CCW = 0,
        CW = 1
    };

    struct RHIRasterStateDesc
    {
        RHICullMode cull = RHICullMode::Back;
        RHIFrontFace front_face = RHIFrontFace::CCW;
        bool depth_clamp = false;
    };

    struct RHIDepthStateDesc
    {
        bool enable_test = true;
        bool enable_write = true;
    };

    struct RHIBlendStateDesc
    {
        bool enable = false;
    };

    struct RHIRenderTargetLayoutDesc
    {
        RHIFormat color_format = RHIFormat::RGBA8_UNorm;
        RHIFormat depth_format = RHIFormat::D32F;
        bool has_depth = true;
    };

    // Minimal explicit vertex ABI: tightly packed float2 at location/binding 0.
    enum class RHIVertexLayout : uint8_t { Procedural, Position2F };

    struct RHIGraphicsPipelineDesc
    {
        RHIShaderModuleDesc vs{};
        RHIShaderModuleDesc fs{};
        RHIRasterStateDesc raster{};
        RHIDepthStateDesc depth{};
        RHIBlendStateDesc blend{};
        RHIRenderTargetLayoutDesc rt{};
        RHIVertexLayout vertex_layout = RHIVertexLayout::Procedural;
    };

    struct RHIComputePipelineDesc
    {
        RHIShaderModuleDesc cs{};
    };

    } // inline namespace rhi
}

