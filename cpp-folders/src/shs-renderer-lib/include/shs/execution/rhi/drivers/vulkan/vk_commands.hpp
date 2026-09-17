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

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <type_traits>
#include <variant>

#include "shs/execution/rhi/command/command_desc.hpp"
#include "shs/execution/rhi/sync/sync_desc.hpp"

namespace shs
{
    enum class VulkanRecordingError : uint8_t
    {
        DeviceUnavailable,
        CommandBufferUnavailable,
        UnsupportedCommand,
        MissingBuffer,
        InvalidRecordingOrder
    };

    struct VulkanRecordingFailure
    {
        VulkanRecordingError code;
        // SIZE_MAX denotes a recording prerequisite, not a stream command.
        std::size_t command_index = SIZE_MAX;
    };

    // Sink methods return expected<void, VulkanRecordingError>.
    // Infallible void-returning translation spies are also supported.
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

    // Fail-fast translation: earlier calls are not rolled back on rejection.
    template <typename Sink>
    [[nodiscard]] std::expected<void, VulkanRecordingFailure> record_commands(
        std::span<const RHICmd> stream, Sink& sink)
    {
        // Preflight nested passes before issuing any sink calls. Each stream
        // starts outside a pass. Other ordering/resource checks remain separate.
        bool inside_pass = false;
        for (std::size_t i = 0; i < stream.size(); ++i)
        {
            if (std::holds_alternative<RHICmdBeginPassDesc>(stream[i].payload))
            {
                if (inside_pass)
                    return std::unexpected(VulkanRecordingFailure{
                        VulkanRecordingError::InvalidRecordingOrder, i});
                inside_pass = true;
            }
            else if (std::holds_alternative<RHICmdEndPassDesc>(stream[i].payload))
                inside_pass = false;
        }

        const auto invoke = [](auto&& call) -> std::expected<void, VulkanRecordingError> {
            if constexpr (std::is_void_v<decltype(call())>)
            {
                call();
                return {};
            }
            else return call();
        };
        for (std::size_t i = 0; i < stream.size(); ++i)
        {
            const RHICmd& cmd = stream[i];
            std::expected<void, VulkanRecordingError> result;
            if (const auto* d = std::get_if<RHICmdBeginPassDesc>(&cmd.payload))
                result = invoke([&] { return sink.begin_pass(*d); });
            else if (const auto* d = std::get_if<RHICmdEndPassDesc>(&cmd.payload))
                result = invoke([&] { return sink.end_pass(*d); });
            else if (const auto* d = std::get_if<RHICmdBindPipelineDesc>(&cmd.payload))
                result = invoke([&] { return sink.bind_pipeline(*d); });
            else if (const auto* d = std::get_if<RHICmdBindVertexBufferDesc>(&cmd.payload))
                result = invoke([&] { return sink.bind_vertex_buffer(*d); });
            else if (const auto* d = std::get_if<RHICmdBindIndexBufferDesc>(&cmd.payload))
                result = invoke([&] { return sink.bind_index_buffer(*d); });
            else if (const auto* d = std::get_if<RHICmdDrawIndexedDesc>(&cmd.payload))
                result = invoke([&] { return sink.draw_indexed(*d); });
            else if (const auto* d = std::get_if<RHICmdDispatchDesc>(&cmd.payload))
                result = invoke([&] { return sink.dispatch(*d); });
            else if (const auto* d = std::get_if<RHICmdBarrierDesc>(&cmd.payload))
                result = invoke([&] { return sink.barrier(*d); });
            else result = std::unexpected(VulkanRecordingError::UnsupportedCommand);
            if (!result) return std::unexpected(VulkanRecordingFailure{result.error(), i});
        }
        return {};
    }
}
