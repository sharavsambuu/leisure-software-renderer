#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pipeline_desc.hpp
    МОДУЛЬ: rhi/pipeline
    ЗОРИЛГО: Graphics/Compute pipeline state descriptor-ууд.
            Vulkan дээр pipeline layout болон render state байгуулах contract.
*/


#include <cstdint>
#include <cstddef>
#include <cstring>
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

    // Vendor-free descriptor gate for the fixed offscreen ABI: triangle list,
    // no vertex attributes, one RGBA8 color output, no depth or blending.
    // Every backend's offscreen realization shares this gate so acceptance and
    // rejection cannot drift between realizations (a GPU pipeline and a named CPU
    // realization must agree on which descriptors are in contract). It checks the
    // descriptor envelope and the SPIR-V module header only — it is NOT a SPIR-V
    // validator and NOT a reflection pass. Each backend additionally binds the
    // module to its own execution mechanism.
    [[nodiscard]] inline bool rhi_shader_module_supported(
        const RHIShaderModuleDesc& d, RHIShaderStage stage)
    {
        if (d.stage != stage || !d.bytecode || d.bytecode_size < 20 || d.bytecode_size % 4 ||
            reinterpret_cast<uintptr_t>(d.bytecode) % alignof(uint32_t) || !d.entry || !*d.entry)
            return false;
        uint32_t header[5]{};
        std::memcpy(header, d.bytecode, sizeof(header));
        // Vulkan 1.1 core supports SPIR-V through 1.3 (no new extensions).
        return header[0] == 0x07230203 && header[1] >= 0x00010000 &&
            header[1] <= 0x00010300 && header[3] != 0 && header[4] == 0;
    }

    [[nodiscard]] inline bool rhi_graphics_pipeline_desc_supported(
        const RHIGraphicsPipelineDesc& d)
    {
        return rhi_shader_module_supported(d.vs, RHIShaderStage::Vertex) &&
            rhi_shader_module_supported(d.fs, RHIShaderStage::Fragment) &&
            d.rt.color_format == RHIFormat::RGBA8_UNorm && !d.rt.has_depth &&
            !d.depth.enable_test && !d.depth.enable_write && !d.blend.enable &&
            (d.vertex_layout == RHIVertexLayout::Procedural ||
             d.vertex_layout == RHIVertexLayout::Position2F) &&
            !d.raster.depth_clamp &&
            (d.raster.cull == RHICullMode::None || d.raster.cull == RHICullMode::Back ||
             d.raster.cull == RHICullMode::Front) &&
            (d.raster.front_face == RHIFrontFace::CCW || d.raster.front_face == RHIFrontFace::CW);
    }

    struct RHIComputePipelineDesc
    {
        RHIShaderModuleDesc cs{};
    };

    } // inline namespace rhi
}

