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
        InvalidRecordingOrder,
        MissingPipeline,
        MissingImage,
        MissingBinding,
        InvalidCommand
    };

    enum class VulkanRecordingStage : uint8_t { Prerequisite, Validation, Recording };
    enum class VulkanCommandKind : uint8_t
    {
        None, BeginPass, EndPass, BindPipeline, BindVertexBuffer, BindIndexBuffer,
        DrawIndexed, Dispatch, Barrier
    };

    inline VulkanCommandKind vulkan_command_kind(const RHICmd& cmd)
    {
        if (std::holds_alternative<RHICmdBeginPassDesc>(cmd.payload)) return VulkanCommandKind::BeginPass;
        if (std::holds_alternative<RHICmdEndPassDesc>(cmd.payload)) return VulkanCommandKind::EndPass;
        if (std::holds_alternative<RHICmdBindPipelineDesc>(cmd.payload)) return VulkanCommandKind::BindPipeline;
        if (std::holds_alternative<RHICmdBindVertexBufferDesc>(cmd.payload)) return VulkanCommandKind::BindVertexBuffer;
        if (std::holds_alternative<RHICmdBindIndexBufferDesc>(cmd.payload)) return VulkanCommandKind::BindIndexBuffer;
        if (std::holds_alternative<RHICmdDrawIndexedDesc>(cmd.payload)) return VulkanCommandKind::DrawIndexed;
        if (std::holds_alternative<RHICmdDispatchDesc>(cmd.payload)) return VulkanCommandKind::Dispatch;
        if (std::holds_alternative<RHICmdBarrierDesc>(cmd.payload)) return VulkanCommandKind::Barrier;
        return VulkanCommandKind::None;
    }

    struct VulkanRecordingFailure
    {
        VulkanRecordingError code;
        // SIZE_MAX denotes a recording prerequisite, not a stream command.
        std::size_t command_index = SIZE_MAX;
        VulkanRecordingStage stage = VulkanRecordingStage::Prerequisite;
        VulkanCommandKind command = VulkanCommandKind::None;
        uint64_t resource_id = 0;
    };

    // Complete, self-contained streams only; bindings do not leak across calls
    // or pass boundaries. Vertex buffers are optional (procedural vertices).
    [[nodiscard]] inline std::expected<void, VulkanRecordingFailure> validate_command_order(
        std::span<const RHICmd> stream)
    {
        bool inside = false, pipeline = false, index = false;
        std::size_t begin = 0;
        const auto fail = [&](VulkanRecordingError code, std::size_t i, uint64_t id = 0) {
            return std::unexpected(VulkanRecordingFailure{code, i,
                VulkanRecordingStage::Validation, vulkan_command_kind(stream[i]), id});
        };
        for (std::size_t i = 0; i < stream.size(); ++i)
        {
            const auto& p = stream[i].payload;
            if (std::holds_alternative<RHICmdBeginPassDesc>(p))
            {
                if (inside) return fail(VulkanRecordingError::InvalidRecordingOrder, i);
                inside = true; pipeline = false; index = false; begin = i;
            }
            else if (std::holds_alternative<RHICmdEndPassDesc>(p))
            {
                if (!inside) return fail(VulkanRecordingError::InvalidRecordingOrder, i);
                inside = false; pipeline = false; index = false;
            }
            else if (const auto* d = std::get_if<RHICmdBindPipelineDesc>(&p))
            {
                if (!d->pipeline) return fail(VulkanRecordingError::MissingPipeline, i);
                pipeline = true;
            }
            else if (const auto* d = std::get_if<RHICmdBindVertexBufferDesc>(&p))
            {
                if (!d->buffer) return fail(VulkanRecordingError::MissingBuffer, i);
            }
            else if (const auto* d = std::get_if<RHICmdBindIndexBufferDesc>(&p))
            {
                if (!d->buffer) return fail(VulkanRecordingError::MissingBuffer, i);
                if (d->offset % (d->index_u32 ? 4 : 2))
                    return fail(VulkanRecordingError::InvalidCommand, i, d->buffer);
                index = true;
            }
            else if (std::holds_alternative<RHICmdDrawIndexedDesc>(p))
            {
                if (!inside) return fail(VulkanRecordingError::InvalidRecordingOrder, i);
                if (!pipeline || !index) return fail(VulkanRecordingError::MissingBinding, i);
            }
            else if (std::holds_alternative<RHICmdDispatchDesc>(p))
            {
                if (inside) return fail(VulkanRecordingError::InvalidRecordingOrder, i);
                if (!pipeline) return fail(VulkanRecordingError::MissingBinding, i);
            }
            else if (const auto* d = std::get_if<RHICmdBarrierDesc>(&p))
            {
                // In-pass barriers require render-pass dependency metadata not
                // represented by this command contract yet.
                if (inside) return fail(VulkanRecordingError::UnsupportedCommand, i);
                if (d->memory.src_stage > RHIPipelineStage::Bottom ||
                    d->memory.dst_stage > RHIPipelineStage::Bottom ||
                    d->memory.src_access > RHIAccess::ReadWrite ||
                    d->memory.dst_access > RHIAccess::ReadWrite)
                    return fail(VulkanRecordingError::InvalidCommand, i);
            }
            else return fail(VulkanRecordingError::UnsupportedCommand, i);
        }
        // An unterminated pass is attributed to its opening command, not EOF.
        if (inside) return fail(VulkanRecordingError::InvalidRecordingOrder, begin);
        return {};
    }

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
    //
    // Three stages, attributed in every failure:
    //   Prerequisite — device/command-buffer handles checked before the stream;
    //   Validation   — whole-stream preflight (no sink calls issued);
    //   Recording    — sink calls in order, fail-fast with no rollback.

    // Fail-fast translation: earlier calls are not rolled back on rejection.
    template <typename Sink>
    [[nodiscard]] std::expected<void, VulkanRecordingFailure> record_commands(
        std::span<const RHICmd> stream, Sink& sink)
    {
        // Optional device/resource hooks are pure checks: never issue vkCmd*.
        // Prerequisites take precedence, then stream order, then sink validation.
        if constexpr (requires { sink.recording_ready(); })
        {
            if (auto ready = sink.recording_ready(); !ready)
                return std::unexpected(VulkanRecordingFailure{ready.error()});
        }
        if (auto valid = validate_command_order(stream); !valid) return valid;
        if constexpr (requires { sink.validate_command(stream[0], false); })
        {
            bool inside_pass = false;
            for (std::size_t i = 0; i < stream.size(); ++i)
            {
                if (auto valid = sink.validate_command(stream[i], inside_pass); !valid)
                {
                    auto failure = valid.error();
                    failure.command_index = i;
                    failure.stage = VulkanRecordingStage::Validation;
                    failure.command = vulkan_command_kind(stream[i]);
                    return std::unexpected(failure);
                }
                if (std::holds_alternative<RHICmdBeginPassDesc>(stream[i].payload)) inside_pass = true;
                if (std::holds_alternative<RHICmdEndPassDesc>(stream[i].payload)) inside_pass = false;
            }
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
            if (!result) return std::unexpected(VulkanRecordingFailure{
                result.error(), i, VulkanRecordingStage::Recording, vulkan_command_kind(cmd)});
        }
        return {};
    }
}
