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

    struct RHIGraphicsPipelineDesc
    {
        RHIShaderModuleDesc vs{};
        RHIShaderModuleDesc fs{};
        RHIRasterStateDesc raster{};
        RHIDepthStateDesc depth{};
        RHIBlendStateDesc blend{};
        RHIRenderTargetLayoutDesc rt{};
    };

    struct RHIComputePipelineDesc
    {
        RHIShaderModuleDesc cs{};
    };
}

