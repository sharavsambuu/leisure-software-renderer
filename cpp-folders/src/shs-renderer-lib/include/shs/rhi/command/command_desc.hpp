#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: command_desc.hpp
    МОДУЛЬ: rhi/command
    ЗОРИЛГО: Command recording contract.
            Vulkan command buffer моделийг backend-neutral байдлаар төлөөлнө.
*/


#include <cstdint>
#include <variant>
#include "shs/rhi/sync/sync_desc.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace rhi
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

    struct RHICmdDrawDesc
    {
        uint32_t vertex_count = 0;
        uint32_t instance_count = 1;
        uint32_t first_vertex = 0;
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

    // P2 driver contract: passes emit a closed value command stream (an arena
    // span of RHICmd); the driver translates it. No Vk* types appear here —
    // stable 64-bit resource IDs only (arch doc §4 rule 1).
    struct RHICmdEndPassDesc
    {
        uint8_t reserved = 0;
    };

    using RHICmdPayload = std::variant<
        RHICmdBeginPassDesc,
        RHICmdBindPipelineDesc,
        RHICmdBindVertexBufferDesc,
        RHICmdBindIndexBufferDesc,
        RHICmdDrawIndexedDesc,
        RHICmdDispatchDesc,
        RHICmdBarrierDesc,
        RHICmdEndPassDesc,
        RHICmdDrawDesc
    >;

    struct RHICmd
    {
        RHICmdPayload payload{};
    };

    inline RHICmd rhi_cmd_begin_pass(const RHICmdBeginPassDesc& d) { return RHICmd{d}; }
    inline RHICmd rhi_cmd_bind_pipeline(uint64_t pipeline) { return RHICmd{RHICmdBindPipelineDesc{pipeline}}; }
    inline RHICmd rhi_cmd_bind_vertex_buffer(uint64_t buffer, uint64_t offset) { return RHICmd{RHICmdBindVertexBufferDesc{buffer, offset}}; }
    inline RHICmd rhi_cmd_bind_index_buffer(uint64_t buffer, uint64_t offset, bool index_u32) { return RHICmd{RHICmdBindIndexBufferDesc{buffer, offset, index_u32}}; }
    inline RHICmd rhi_cmd_draw(const RHICmdDrawDesc& d) { return RHICmd{d}; }
    inline RHICmd rhi_cmd_draw_indexed(const RHICmdDrawIndexedDesc& d) { return RHICmd{d}; }
    inline RHICmd rhi_cmd_dispatch(uint32_t x, uint32_t y, uint32_t z) { return RHICmd{RHICmdDispatchDesc{x, y, z}}; }
    inline RHICmd rhi_cmd_barrier(const RHIMemoryBarrierDesc& b) { return RHICmd{RHICmdBarrierDesc{b}}; }
    inline RHICmd rhi_cmd_end_pass() { return RHICmd{RHICmdEndPassDesc{}}; }

    } // inline namespace rhi
}

