#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: vk_runtime.hpp
    МОДУЛЬ: rhi/sync
    ЗОРИЛГО: Vulkan маягийн frame/queue/submission/sync загварыг software орчинд дууриах runtime.
            Энэ нь жинхэнэ GPU async биш боловч архитектур, dependency, scheduling урсгалыг
            backend-agnostic байдлаар турших боломж өгнө.
*/


#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shs/job/job_system.hpp"
#include "shs/job/wait_group.hpp"
#include "shs/rhi/command/command_desc.hpp"
#include "shs/rhi/sync/sync_desc.hpp"

namespace shs
{
    struct VulkanLikeRuntimeConfig
    {
        uint32_t frames_in_flight = 2;
        bool allow_parallel_tasks = true;
    };

    struct VulkanLikeRuntimeStats
    {
        uint64_t submissions = 0;
        uint64_t submissions_executed = 0;
        uint64_t stalled_submissions = 0;
        uint64_t tasks_executed = 0;
        uint64_t tasks_parallel = 0;

        void reset_frame()
        {
            submissions = 0;
            submissions_executed = 0;
            stalled_submissions = 0;
            tasks_executed = 0;
            tasks_parallel = 0;
        }
    };

    struct VulkanLikeTask
    {
        std::string label{};
        std::function<void()> fn{};
    };

    struct VulkanLikeSubmission
    {
        RHIQueueClass queue = RHIQueueClass::Graphics;
        std::vector<RHISemaphoreWaitDesc> waits{};
        std::vector<RHISemaphoreSignalDesc> signals{};
        uint64_t fence_id = 0;
        bool allow_parallel_tasks = false;
        std::vector<VulkanLikeTask> tasks{};
        std::string label{};
    };

    class VulkanLikeRuntime
    {
    public:
        void configure(const VulkanLikeRuntimeConfig& cfg)
        {
            cfg_ = cfg;
            if (cfg_.frames_in_flight == 0) cfg_.frames_in_flight = 1;
            frame_slots_.resize((size_t)cfg_.frames_in_flight);
        }

        void set_job_system(IJobSystem* js)
        {
            js_ = js;
        }

        void begin_frame(uint64_t frame_index)
        {
            if (frame_slots_.empty()) frame_slots_.resize((size_t)cfg_.frames_in_flight);
            current_frame_index_ = frame_index;
            stats_.reset_frame();
            clear_pending();

            const size_t slot_idx = (size_t)(frame_index % (uint64_t)frame_slots_.size());
            FrameSlot& slot = frame_slots_[slot_idx];
            if (slot.in_flight && slot.fence_id != 0 && !fence_signaled(slot.fence_id))
            {
                // Vulkan-ийн frame-in-flight fence wait-ийг дууриаж idle хүртэл хүлээнэ.
                if (js_) js_->wait_idle();
                signal_fence(slot.fence_id);
            }
            slot.in_flight = false;
            slot.frame_index = frame_index;
            slot.fence_id = 0;
        }

        uint64_t queue_timeline_semaphore(RHIQueueClass queue)
        {
            const size_t qi = queue_index(queue);
            if (queue_timeline_ids_[qi] == 0)
            {
                queue_timeline_ids_[qi] = new_semaphore();
            }
            return queue_timeline_ids_[qi];
        }

        uint64_t timeline_value(uint64_t semaphore_id) const
        {
            const auto it = timeline_values_.find(semaphore_id);
            return (it == timeline_values_.end()) ? 0 : it->second;
        }

        uint64_t new_semaphore()
        {
            return ++next_semaphore_id_;
        }

        uint64_t new_fence(bool signaled = false)
        {
            const uint64_t id = ++next_fence_id_;
            fences_[id] = signaled;
            return id;
        }

        void submit(VulkanLikeSubmission submission)
        {
            const size_t qi = queue_index(submission.queue);
            pending_[qi].push_back(std::move(submission));
            stats_.submissions++;
        }

        void execute_all()
        {
            // Wait dependency хангагдсан submission-уудыг queue-үүдээс боловсруулна.
            while (true)
            {
                bool progressed = false;
                for (size_t qi = 0; qi < pending_.size(); ++qi)
                {
                    auto& q = pending_[qi];
                    size_t i = 0;
                    while (i < q.size())
                    {
                        if (!waits_satisfied(q[i]))
                        {
                            ++i;
                            continue;
                        }
                        execute_submission(q[i]);
                        q.erase(q.begin() + (ptrdiff_t)i);
                        progressed = true;
                    }
                }

                if (all_queues_empty()) break;
                if (!progressed)
                {
                    // Deadlock маягийн хүлээлт илэрвэл эхний submission-ийг force-run хийнэ.
                    for (size_t qi = 0; qi < pending_.size(); ++qi)
                    {
                        auto& q = pending_[qi];
                        if (q.empty()) continue;
                        stats_.stalled_submissions++;
                        execute_submission(q.front());
                        q.erase(q.begin());
                        break;
                    }
                }
            }
        }

        void end_frame()
        {
            if (frame_slots_.empty()) return;
            const size_t slot_idx = (size_t)(current_frame_index_ % (uint64_t)frame_slots_.size());
            FrameSlot& slot = frame_slots_[slot_idx];
            slot.in_flight = true;
            if (slot.fence_id == 0) slot.fence_id = new_fence(true);
            else signal_fence(slot.fence_id);
        }

        const VulkanLikeRuntimeStats& stats() const
        {
            return stats_;
        }

    private:
        struct FrameSlot
        {
            uint64_t frame_index = 0;
            uint64_t fence_id = 0;
            bool in_flight = false;
        };

        static constexpr size_t queue_index(RHIQueueClass q)
        {
            return (size_t)q;
        }

        bool all_queues_empty() const
        {
            for (const auto& q : pending_)
            {
                if (!q.empty()) return false;
            }
            return true;
        }

        void clear_pending()
        {
            for (auto& q : pending_) q.clear();
        }

        bool waits_satisfied(const VulkanLikeSubmission& sub) const
        {
            for (const auto& w : sub.waits)
            {
                if (timeline_value(w.semaphore_id) < w.value) return false;
            }
            return true;
        }

        bool fence_signaled(uint64_t fence_id) const
        {
            const auto it = fences_.find(fence_id);
            return it == fences_.end() ? true : it->second;
        }

        void signal_fence(uint64_t fence_id)
        {
            if (fence_id == 0) return;
            fences_[fence_id] = true;
        }

        void execute_submission(const VulkanLikeSubmission& sub)
        {
            if (sub.allow_parallel_tasks && cfg_.allow_parallel_tasks && js_ && sub.tasks.size() > 1)
            {
                WaitGroup wg{};
                for (const auto& t : sub.tasks)
                {
                    if (!t.fn) continue;
                    wg.add(1);
                    js_->enqueue([fn = t.fn, &wg]() {
                        fn();
                        wg.done();
                    });
                    stats_.tasks_parallel++;
                }
                wg.wait();
            }
            else
            {
                for (const auto& t : sub.tasks)
                {
                    if (!t.fn) continue;
                    t.fn();
                }
            }

            for (const auto& s : sub.signals)
            {
                const uint64_t& cur = timeline_values_[s.semaphore_id];
                (void)cur;
                timeline_values_[s.semaphore_id] = std::max(timeline_values_[s.semaphore_id], s.value);
            }
            if (sub.fence_id != 0) signal_fence(sub.fence_id);

            stats_.submissions_executed++;
            stats_.tasks_executed += (uint64_t)sub.tasks.size();
        }

        VulkanLikeRuntimeConfig cfg_{};
        IJobSystem* js_ = nullptr;
        uint64_t current_frame_index_ = 0;
        uint64_t next_semaphore_id_ = 100;
        uint64_t next_fence_id_ = 10;
        std::unordered_map<uint64_t, uint64_t> timeline_values_{};
        std::unordered_map<uint64_t, bool> fences_{};
        std::array<std::vector<VulkanLikeSubmission>, 4> pending_{};
        std::array<uint64_t, 4> queue_timeline_ids_{0, 0, 0, 0};
        std::vector<FrameSlot> frame_slots_{};
        VulkanLikeRuntimeStats stats_{};
    };
}
