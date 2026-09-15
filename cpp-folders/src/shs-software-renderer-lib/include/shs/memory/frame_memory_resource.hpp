#pragma once

/*
    FrameMemoryResource — the transient frame arena (spec §3 Rule 5.1).

    Linear bump allocator over one preallocated buffer; O(1) reset at frame
    boundaries. Backs per-frame command streams, batch spans, event logs, and
    other transient tier allocations ONLY. Persistent state (world snapshots,
    entity tables) must use std::pmr::get_default_resource() — assigning
    persistent objects from this arena is a tiering violation.

    P1.5 (roadmap): promoted from the demo-private copies (tetris/snake/fps),
    per §7.2 rule 6 (library ownership). Overflow is strict: bad_alloc, never
    a silent fallback to the persistent tier (the old snake copy silently
    spilled into get_default_resource(), which broke tier separation).
*/

#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <new>

namespace shs::memory
{
    class FrameMemoryResource final : public std::pmr::memory_resource
    {
    public:
        static constexpr std::size_t kDefaultCapacity = 8ull * 1024ull * 1024ull;

        explicit FrameMemoryResource(std::size_t capacity_bytes = kDefaultCapacity)
            : capacity_(capacity_bytes), buffer_(std::make_unique<std::byte[]>(capacity_bytes))
        {
        }

        FrameMemoryResource(const FrameMemoryResource&) = delete;
        FrameMemoryResource& operator=(const FrameMemoryResource&) = delete;

        // Frame boundary: O(1) reset. Everything allocated from the arena is
        // dead after this call — no destructors run (frame tier holds only
        // trivially-destructible or externally-cleaned data by law).
        void reset() noexcept { offset_ = 0; }

        std::pmr::memory_resource* get() noexcept { return this; }

        std::size_t capacity() const noexcept { return capacity_; }
        std::size_t used() const noexcept { return offset_; }
        std::size_t high_water_mark() const noexcept { return high_water_; }

    protected:
        void* do_allocate(std::size_t bytes, std::size_t alignment) override
        {
            // Align against the REAL buffer address (the buffer itself is only
            // max_align_t-aligned, so offset-relative rounding is not enough).
            const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(buffer_.get());
            const std::uintptr_t current = base + offset_;
            const std::uintptr_t aligned_addr =
                (current + (alignment - 1)) & ~(static_cast<std::uintptr_t>(alignment) - 1);
            const std::size_t end_offset = static_cast<std::size_t>(aligned_addr - base) + bytes;
            if (end_offset > capacity_) throw std::bad_alloc();
            offset_ = end_offset;
            if (offset_ > high_water_) high_water_ = offset_;
            return reinterpret_cast<void*>(aligned_addr);
        }

        void do_deallocate(void*, std::size_t, std::size_t) noexcept override {}

        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
        {
            return this == &other;
        }

    private:
        std::size_t capacity_;
        std::unique_ptr<std::byte[]> buffer_;
        std::size_t offset_ = 0;
        std::size_t high_water_ = 0;
    };

} // namespace shs::memory
