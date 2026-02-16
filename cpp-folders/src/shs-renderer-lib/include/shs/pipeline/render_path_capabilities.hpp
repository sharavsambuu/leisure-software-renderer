#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_capabilities.hpp
    MODULE: pipeline
    PURPOSE: Capability snapshot and compatibility rule inputs for render path recipes.
*/


#include "shs/core/context.hpp"
#include "shs/rhi/drivers/vulkan/vk_backend.hpp"

namespace shs
{
    struct RenderPathCapabilitySet
    {
        RenderBackendType backend = RenderBackendType::Software;
        bool has_backend = false;

        bool supports_present = false;
        bool supports_offscreen = true;

        bool depth_attachment_known = false;
        bool supports_depth_attachment = true;

        bool supports_occlusion_query = false;
        bool supports_secondary_command_recording = false;
        bool supports_async_compute = false;

        BackendCapabilities backend_caps{};
    };

    inline RenderPathCapabilitySet make_render_path_capability_set(RenderBackendType backend, const BackendCapabilities& caps)
    {
        RenderPathCapabilitySet out{};
        out.backend = backend;
        out.has_backend = true;
        out.backend_caps = caps;
        out.supports_present = caps.supports_present;
        out.supports_offscreen = caps.supports_offscreen;
        out.supports_secondary_command_recording = caps.features.multithread_command_recording;
        out.supports_async_compute = caps.features.async_compute;
        out.supports_depth_attachment = true;
        out.depth_attachment_known = false;
        // Treat this as "occlusion culling support" (hardware query or software depth-cull path).
        out.supports_occlusion_query = true;
        return out;
    }

    inline RenderPathCapabilitySet make_render_path_capability_set(const Context& ctx, RenderBackendType backend)
    {
        RenderPathCapabilitySet out{};
        out.backend = backend;

        const IRenderBackend* rb = ctx.backend(backend);
        if (!rb) return out;

        out.has_backend = true;
        out.backend_caps = rb->capabilities();
        out.supports_present = out.backend_caps.supports_present;
        out.supports_offscreen = out.backend_caps.supports_offscreen;
        out.supports_secondary_command_recording = out.backend_caps.features.multithread_command_recording;
        out.supports_async_compute = out.backend_caps.features.async_compute;

        // Baseline assumption during early compile time: depth is available unless
        // a backend can explicitly report the opposite.
        out.supports_depth_attachment = true;
        out.depth_attachment_known = false;
        // Treat this as "occlusion culling support" (hardware query or software depth-cull path).
        out.supports_occlusion_query = true;

        if (backend == RenderBackendType::Vulkan)
        {
#ifdef SHS_HAS_VULKAN
            const auto* vk_backend = dynamic_cast<const VulkanRenderBackend*>(rb);
            if (vk_backend)
            {
                out.depth_attachment_known = true;
                out.supports_depth_attachment = vk_backend->has_depth_attachment();
            }
#endif
        }

        return out;
    }
}
