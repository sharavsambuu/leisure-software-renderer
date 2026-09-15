#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_sync.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Vulkan driver pod — frame-in-flight synchronization.
            Uses rhi/sync/vk_runtime.hpp (the backend-neutral submission /
            fence / timeline model) + sync_desc.hpp value descs. The frame
            slot bookkeeping (which frames are in flight, fence reuse,
            semaphore timeline values) is pure and GPU-free testable; the
            real VkFence/VkSemaphore pooling hangs off the same slot logic
            in the device-bound backend.
*/

#include <cstdint>
#include <memory_resource>
#include <vector>

#include "shs/execution/rhi/sync/sync_desc.hpp"
#include "shs/execution/rhi/sync/vk_runtime.hpp"

namespace shs
{
    struct VulkanFrameSyncStats
    {
        uint64_t begin_frames = 0;
        uint64_t end_frames = 0;
        uint64_t waits_issued = 0;
        uint64_t signals_issued = 0;
        uint64_t submissions = 0;
    };

    // Frame-slot rotation mirroring VkFence per-frame-in-flight semantics.
    class VulkanFrameSync
    {
    public:
        VulkanFrameSync() { configure(slot_count_); }
        void configure(uint32_t frames_in_flight)
        {
            slot_count_ = frames_in_flight == 0 ? 1 : frames_in_flight;
            slots_.assign(slot_count_, Slot{});
            VulkanLikeRuntimeConfig cfg{};
            cfg.frames_in_flight = slot_count_;
            cfg.allow_parallel_tasks = false; // driver-level: submissions stay ordered
            runtime_.configure(cfg);
        }

        // Acquire the frame slot: when the slot is still in flight this models
        // vkWaitForFences (the wait is accounted, then the slot is reused).
        uint64_t begin_frame(uint64_t frame_index)
        {
            Slot& slot = slots_[(size_t)(frame_index % slot_count_)];
            if (slot.in_flight) stats_.waits_issued++;
            slot.in_flight = false;
            slot.last_frame = frame_index;
            stats_.begin_frames++;
            return slot_fence_id(frame_index);
        }

        // Submit the frame: signals the graphics-queue timeline (value = frame+1)
        // and marks the slot in flight, mirroring the per-frame VkFence signal.
        void end_frame(uint64_t frame_index)
        {
            VulkanLikeSubmission sub{};
            sub.queue = RHIQueueClass::Graphics;
            sub.fence_id = slot_fence_id(frame_index);
            RHISemaphoreSignalDesc sig{};
            sig.semaphore_id = runtime_.queue_timeline_semaphore(RHIQueueClass::Graphics);
            sig.value = frame_index + 1;
            sig.stage = RHIPipelineStage::ColorOutput;
            sub.signals.push_back(sig);
            runtime_.submit(std::move(sub));
            runtime_.execute_all(); // driver frame submission is synchronous
            slots_[(size_t)(frame_index % slot_count_)].in_flight = true;
            stats_.end_frames++;
            stats_.signals_issued++;
        }

        [[nodiscard]] bool frame_in_flight(uint64_t frame_index) const
        {
            return slots_[(size_t)(frame_index % slot_count_)].in_flight;
        }

        [[nodiscard]] uint64_t timeline_value(uint64_t semaphore_id) const { return runtime_.timeline_value(semaphore_id); }
        [[nodiscard]] uint64_t graphics_timeline_id() { return runtime_.queue_timeline_semaphore(RHIQueueClass::Graphics); }
        [[nodiscard]] const VulkanFrameSyncStats& stats() const { return stats_; }
        [[nodiscard]] const VulkanLikeRuntimeStats& runtime_stats() const { return runtime_.stats(); }

    private:
        struct Slot
        {
            uint64_t last_frame = UINT64_MAX;
            bool in_flight = false;
        };

        [[nodiscard]] uint64_t slot_fence_id(uint64_t frame_index) const
        {
            return 1 + (frame_index % slot_count_); // stable per-slot fence id
        }

        VulkanLikeRuntime runtime_{};
        std::pmr::vector<Slot> slots_{};
        uint32_t slot_count_ = 2;
        VulkanFrameSyncStats stats_{};
    };
}
