#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: capabilities.hpp
    МОДУЛЬ: rhi/core
    ЗОРИЛГО: Backend-ийн queue/feature/limit мэдээллийг нэгэн жигд contract-оор илэрхийлнэ.
            Vulkan/OpenGL/Software-ийн боломжуудыг pass болон pipeline түвшинд харьцуулах суурь.
*/


#include <cstdint>

namespace shs
{
    struct BackendQueueCaps
    {
        uint32_t graphics_count = 0;
        uint32_t compute_count = 0;
        uint32_t transfer_count = 0;
        uint32_t present_count = 0;
    };

    struct BackendFeatureCaps
    {
        bool validation_layers = false;
        bool timeline_semaphore = false;
        bool descriptor_indexing = false;
        bool dynamic_rendering = false;
        bool push_constants = false;
        bool multithread_command_recording = false;
        bool async_compute = false;
        bool ray_query = false;
        bool mesh_shader = false;
    };

    struct BackendLimitCaps
    {
        uint32_t max_frames_in_flight = 2;
        uint32_t max_color_attachments = 1;
        uint32_t max_descriptor_sets_per_pipeline = 1;
        uint32_t max_push_constant_bytes = 0;
        uint32_t min_uniform_buffer_offset_alignment = 1;

        // Mesh Shader Limits
        uint32_t max_mesh_workgroup_size[3] = {1, 1, 1};
        uint32_t max_mesh_workgroup_total_count = 1;
        uint32_t max_mesh_output_vertices = 0;
        uint32_t max_mesh_output_primitives = 0;
        uint32_t max_task_workgroup_size[3] = {1, 1, 1};
        uint32_t max_task_workgroup_total_count = 1;
    };

    struct BackendCapabilities
    {
        BackendQueueCaps queues{};
        BackendFeatureCaps features{};
        BackendLimitCaps limits{};
        bool supports_present = false;
        bool supports_offscreen = true;
        bool depth_attachment_known = false;
        bool supports_depth_attachment = true;
    };
}
