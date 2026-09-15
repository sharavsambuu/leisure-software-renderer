#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: command_desc.hpp
    МОДУЛЬ: rhi/command
    ЗОРИЛГО: Command recording contract.
            Vulkan command buffer моделийг backend-neutral байдлаар төлөөлнө.
*/


#include <cstdint>
#include "shs/rhi/sync/sync_desc.hpp"

namespace shs
{
    enum class RHIQueueClass : uint8_t
    {
        Graphics = 0,
        Compute = 1,
        Transfer = 2,
        Present = 3
    };

    struct RHICmdBeginPassDesc
    {
        uint64_t color_target = 0;
        uint64_t depth_target = 0;
        bool clear_color = false;
        bool clear_depth = false;
    };

    struct RHICmdBindPipelineDesc
    {
        uint64_t pipeline = 0;
    };

    struct RHICmdBindVertexBufferDesc
    {
        uint64_t buffer = 0;
        uint64_t offset = 0;
    };

    struct RHICmdBindIndexBufferDesc
    {
        uint64_t buffer = 0;
        uint64_t offset = 0;
        bool index_u32 = false;
    };

    struct RHICmdDrawIndexedDesc
    {
        uint32_t index_count = 0;
        uint32_t instance_count = 1;
        uint32_t first_index = 0;
        int32_t vertex_offset = 0;
        uint32_t first_instance = 0;
    };

    struct RHICmdDispatchDesc
    {
        uint32_t group_x = 1;
        uint32_t group_y = 1;
        uint32_t group_z = 1;
    };

    struct RHICmdBarrierDesc
    {
        RHIMemoryBarrierDesc memory{};
    };
}

