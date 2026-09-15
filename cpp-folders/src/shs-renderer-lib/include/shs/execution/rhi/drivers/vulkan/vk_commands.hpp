#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_commands.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Vulkan driver pod — CommandDesc stream → recording.
            Passes emit an arena span of RHICmd (closed value stream);
            record_commands() translates it into Sink calls in order.
            The sink concept is fulfilled by the backend's Vulkan command
            recorder (device-bound) and by test spies (GPU-free), keeping the
            translation layer replayable and testable without a device.
*/

#include <cstdint>
#include <span>
#include <variant>

#include "shs/execution/rhi/command/command_desc.hpp"
#include "shs/execution/rhi/sync/sync_desc.hpp"

namespace shs
{
    // Sink concept (compile-time duck typing):
    //   void begin_pass(const RHICmdBeginPassDesc&);
    //   void end_pass(const RHICmdEndPassDesc&);
    //   void bind_pipeline(const RHICmdBindPipelineDesc&);
    //   void bind_vertex_buffer(const RHICmdBindVertexBufferDesc&);
    //   void bind_index_buffer(const RHICmdBindIndexBufferDesc&);
    //   void draw_indexed(const RHICmdDrawIndexedDesc&);
    //   void dispatch(const RHICmdDispatchDesc&);
    //   void barrier(const RHICmdBarrierDesc&);
    //
    // The sink receives stable 64-bit IDs only — resolving them to Vk* handles
    // happens below the driver boundary.

    template <typename Sink>
    void record_commands(std::span<const RHICmd> stream, Sink& sink)
    {
        for (const RHICmd& cmd : stream)
        {
            if (const auto* d = std::get_if<RHICmdBeginPassDesc>(&cmd.payload)) sink.begin_pass(*d);
            else if (const auto* d = std::get_if<RHICmdEndPassDesc>(&cmd.payload)) sink.end_pass(*d);
            else if (const auto* d = std::get_if<RHICmdBindPipelineDesc>(&cmd.payload)) sink.bind_pipeline(*d);
            else if (const auto* d = std::get_if<RHICmdBindVertexBufferDesc>(&cmd.payload)) sink.bind_vertex_buffer(*d);
            else if (const auto* d = std::get_if<RHICmdBindIndexBufferDesc>(&cmd.payload)) sink.bind_index_buffer(*d);
            else if (const auto* d = std::get_if<RHICmdDrawIndexedDesc>(&cmd.payload)) sink.draw_indexed(*d);
            else if (const auto* d = std::get_if<RHICmdDispatchDesc>(&cmd.payload)) sink.dispatch(*d);
            else if (const auto* d = std::get_if<RHICmdBarrierDesc>(&cmd.payload)) sink.barrier(*d);
        }
    }
}
