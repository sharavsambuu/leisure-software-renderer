#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: resource_desc.hpp
    МОДУЛЬ: rhi/resource
    ЗОРИЛГО: Buffer/Image/Sampler resource descriptor-уудыг backend-neutral хэлбэрт оруулна.
            Vulkan руу порт хийх үед VkBufferCreateInfo/VkImageCreateInfo руу хөрвүүлэх үндэс.
*/


#include <cstdint>

namespace shs
{
    enum class RHIFormat : uint16_t
    {
        Unknown = 0,
        RGBA8_UNorm = 1,
        BGRA8_UNorm = 2,
        RGBA16F = 3,
        RGBA32F = 4,
        D24S8 = 10,
        D32F = 11
    };

    enum class RHIImageType : uint8_t
    {
        Tex2D = 0,
        TexCube = 1
    };

    enum class RHIMemoryClass : uint8_t
    {
        Auto = 0,
        CPUVisible = 1,
        GPUOnly = 2,
        Readback = 3
    };

    enum RHIBufferUsageBits : uint32_t
    {
        RHIBufferUsage_None = 0,
        RHIBufferUsage_Vertex = 1u << 0u,
        RHIBufferUsage_Index = 1u << 1u,
        RHIBufferUsage_Uniform = 1u << 2u,
        RHIBufferUsage_Storage = 1u << 3u,
        RHIBufferUsage_TransferSrc = 1u << 4u,
        RHIBufferUsage_TransferDst = 1u << 5u
    };

    enum RHIImageUsageBits : uint32_t
    {
        RHIImageUsage_None = 0,
        RHIImageUsage_Sampled = 1u << 0u,
        RHIImageUsage_ColorAttachment = 1u << 1u,
        RHIImageUsage_DepthStencilAttachment = 1u << 2u,
        RHIImageUsage_Storage = 1u << 3u,
        RHIImageUsage_TransferSrc = 1u << 4u,
        RHIImageUsage_TransferDst = 1u << 5u
    };

    struct RHIBufferDesc
    {
        uint64_t size_bytes = 0;
        uint32_t usage = RHIBufferUsage_None;
        RHIMemoryClass memory = RHIMemoryClass::Auto;
    };

    struct RHIImageDesc
    {
        RHIImageType type = RHIImageType::Tex2D;
        int width = 0;
        int height = 0;
        int mip_levels = 1;
        int layers = 1;
        RHIFormat format = RHIFormat::Unknown;
        uint32_t usage = RHIImageUsage_None;
        RHIMemoryClass memory = RHIMemoryClass::Auto;
    };

    enum class RHIFilter : uint8_t
    {
        Nearest = 0,
        Linear = 1
    };

    enum class RHIAddressMode : uint8_t
    {
        ClampToEdge = 0,
        Repeat = 1,
        MirrorRepeat = 2
    };

    struct RHISamplerDesc
    {
        RHIFilter min_filter = RHIFilter::Linear;
        RHIFilter mag_filter = RHIFilter::Linear;
        RHIAddressMode address_u = RHIAddressMode::ClampToEdge;
        RHIAddressMode address_v = RHIAddressMode::ClampToEdge;
        RHIAddressMode address_w = RHIAddressMode::ClampToEdge;
        bool enable_anisotropy = false;
        float max_anisotropy = 1.0f;
    };
}

