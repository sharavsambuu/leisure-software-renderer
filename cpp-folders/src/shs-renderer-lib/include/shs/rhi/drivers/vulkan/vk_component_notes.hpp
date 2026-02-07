#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: vk_component_notes.hpp
    МОДУЛЬ: rhi/drivers/vulkan
    ЗОРИЛГО: Vulkan-ийн үндсэн бүрэлдэхүүн хэсгүүдийг shs-renderer-lib архитектуртай
            хэрхэн уялдуулах тухай тэмдэглэл (header-level notes).
*/


#include <array>

namespace shs
{
    enum class VulkanCoreComponentId : unsigned char
    {
        InstanceAndValidation = 0,
        PhysicalDevice,
        LogicalDeviceAndQueues,
        SurfaceAndSwapchain,
        RenderPassAndFramebuffer,
        PipelineSystem,
        ShaderModules,
        DescriptorSystem,
        CommandSystem,
        Synchronization,
        MemoryAndResources,
        ImageLayoutsAndViews,
        DepthStencilAndMsaa,
        QueryAndDebug
    };

    struct VulkanCoreComponentNote
    {
        VulkanCoreComponentId id{};
        const char* component = "";
        const char* role = "";
        const char* shs_arch_mapping = "";
    };

    inline constexpr std::array<VulkanCoreComponentNote, 14> kVulkanCoreNotes{{
        {VulkanCoreComponentId::InstanceAndValidation, "Instance + Validation", "Global API and debug layers", "backends/vulkan + rhi/capabilities.hpp"},
        {VulkanCoreComponentId::PhysicalDevice, "PhysicalDevice", "GPU capability/queue family selection", "rhi/capabilities.hpp"},
        {VulkanCoreComponentId::LogicalDeviceAndQueues, "LogicalDevice + Queues", "Queue creation and submission lanes", "rhi/command_desc.hpp + rhi/sync_desc.hpp"},
        {VulkanCoreComponentId::SurfaceAndSwapchain, "Surface + Swapchain", "Presentation image lifecycle", "rhi/backend.hpp (begin/end frame contract)"},
        {VulkanCoreComponentId::RenderPassAndFramebuffer, "RenderPass/Framebuffer", "Attachment topology", "rhi/pipeline_desc.hpp + pipeline/render_pass.hpp"},
        {VulkanCoreComponentId::PipelineSystem, "PipelineLayout + Pipelines", "Graphics/Compute pipeline state", "rhi/pipeline_desc.hpp"},
        {VulkanCoreComponentId::ShaderModules, "Shader Modules (SPIR-V)", "Stage bytecode and entry points", "rhi/pipeline_desc.hpp"},
        {VulkanCoreComponentId::DescriptorSystem, "Descriptor Sets", "Resource binding model", "rhi/resource_desc.hpp + future descriptor contracts"},
        {VulkanCoreComponentId::CommandSystem, "Command Pools/Buffers", "Record/submit draw and compute work", "rhi/command_desc.hpp"},
        {VulkanCoreComponentId::Synchronization, "Fences/Semaphores/Barriers", "Frame and hazard ordering", "rhi/sync_desc.hpp"},
        {VulkanCoreComponentId::MemoryAndResources, "Memory + Buffer/Image", "Allocation/upload/readback paths", "rhi/resource_desc.hpp"},
        {VulkanCoreComponentId::ImageLayoutsAndViews, "Image Layouts + Views", "Usage transitions and subresource views", "rhi/resource_desc.hpp + rhi/sync_desc.hpp"},
        {VulkanCoreComponentId::DepthStencilAndMsaa, "Depth/Stencil + MSAA", "Depth attachment and anti-aliasing", "rhi/pipeline_desc.hpp"},
        {VulkanCoreComponentId::QueryAndDebug, "Query + Debug Utils", "Timing/profiling/object labels", "core/context.hpp debug stats + backend debug extensions"}
    }};
}

