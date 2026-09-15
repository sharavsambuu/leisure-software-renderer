#pragma once

/*
    SoaTable<Ts...> — contiguous generational column table (spec §7.2).

    §7.2-compliant hot-state backing store for Domain Pod columns:
      - one contiguous pmr allocation per column, base 64-byte aligned (rule 4)
      - upfront reserve(); growth is geometric and COLD — bumps
        compaction_count() so callers can assert it never happens per-frame
        (rule 1)
      - stability by generational uint32 handle, never by pointer (rule 2,
        Rule 6.1); stale handles are rejected, never UB
      - order-independent density via swap-and-pop removal (rule 3)
      - zero node-based storage anywhere (rule 5)

    Dense layout: logical element i lives at column offset i for every
    column simultaneously. Cross-frame references use Handle{slot, gen};
    within-frame kernels walk the dense span directly (cache-streaming target
    for §7.1 prefetch/store kernels).

    P1.5 (roadmap): shared lib utility promoted per §7.2 rule 6 — demos must
    not define private copies.
*/

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory_resource>
#include <new>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace shs::containers
{
    // Generational handle: stable across growth AND swap-and-pop removal.
    struct SoaHandle
    {
        std::uint32_t slot = 0;
        std::uint32_t generation = 0;

        bool operator==(const SoaHandle&) const = default;
    };

    // §7.2 rule 4: every column base is 64-byte aligned (one cache line).
    inline constexpr std::size_t kSoaColumnAlignment = 64;

    inline constexpr std::uint32_t kSoaFreeSlot = 0xFFFFFFFFu;

    template <typename... Ts>
    class SoaTable
    {
        static_assert(sizeof...(Ts) > 0, "SoaTable needs at least one column");

    public:
        using Handle = SoaHandle;

        template <std::size_t I>
        using column_element_t = std::tuple_element_t<I, std::tuple<Ts...>>;

        static constexpr std::size_t column_count = sizeof...(Ts);

        explicit SoaTable(std::pmr::memory_resource* resource, std::size_t initial_capacity = 64)
            : resource_(resource), free_list_(resource), capacity_(0)
        {
            reserve(initial_capacity);
        }

        SoaTable(const SoaTable&) = delete;
        SoaTable& operator=(const SoaTable&) = delete;

        SoaTable(SoaTable&& other) noexcept { move_from(other); }

        SoaTable& operator=(SoaTable&& other) noexcept
        {
            if (this != &other)
            {
                release();
                move_from(other);
            }
            return *this;
        }

        ~SoaTable() { release(); }

        // ---- capacity ------------------------------------------------------

        // Upfront capacity reservation (§7.2 rule 1: reserve up front from a
        // capacity estimate). Growth beyond capacity is a cold-path compaction
        // event — assert compaction_count() stability in per-frame hot loops.
        void reserve(std::size_t n)
        {
            if (n > capacity_) grow(geometric_capacity(n));
        }

        std::size_t size() const noexcept { return size_; }
        std::size_t capacity() const noexcept { return capacity_; }
        bool empty() const noexcept { return size_ == 0; }

        // Number of cold-path column re-allocations since construction.
        std::size_t compaction_count() const noexcept { return compactions_; }


        // ---- insertion / removal ------------------------------------------

        template <typename... Args>
        Handle insert(Args&&... values)
        {
            static_assert(sizeof...(Args) == sizeof...(Ts), "one value per column");

            if (size_ == capacity_) grow(capacity_ == 0 ? kMinCapacity : capacity_ * 2);

            std::uint32_t slot;
            if (!free_list_.empty())
            {
                slot = free_list_.back();
                free_list_.pop_back();
            }
            else
            {
                slot = slot_count_++;
            }

            const std::uint32_t dense = static_cast<std::uint32_t>(size_);
            emplace_row(dense, std::forward<Args>(values)...);

            slots_[slot].dense = dense;
            dense_to_slot_[dense] = slot;
            ++size_;

            return Handle{slot, slots_[slot].generation};
        }

        // Generational erase: swap-and-pop; keeps live elements dense (rule 3).
        // Returns false for stale/dead handles — never UB.
        bool erase(Handle h)
        {
            if (!alive(h)) return false;
            erase_dense(slots_[h.slot].dense);
            return true;
        }

        // Unconditional swap-and-pop at a dense position (for kernels walking
        // columns and killing in place — see §7.2 rule 3).
        void erase_dense(std::size_t dense)
        {
            const std::uint32_t last = static_cast<std::uint32_t>(size_) - 1;
            const std::uint32_t d = static_cast<std::uint32_t>(dense);

            if (d != last) move_row(last, d); // swap-and-pop: last row -> hole

            destroy_row(last);

            const std::uint32_t moved_slot = dense_to_slot_[last];
            const std::uint32_t freed_slot = dense_to_slot_[d];

            if (d != last)
            {
                dense_to_slot_[d] = moved_slot;
                slots_[moved_slot].dense = d;
            }

            // Retire the freed slot: bump generation so outstanding handles
            // to it become stale (rule 2).
            ++slots_[freed_slot].generation;
            slots_[freed_slot].dense = kSoaFreeSlot;
            free_list_.push_back(freed_slot);

            --size_;
        }

        // ---- handle queries ------------------------------------------------

        bool alive(Handle h) const noexcept
        {
            return h.slot < slot_count_ && slots_[h.slot].dense != kSoaFreeSlot &&
                   slots_[h.slot].generation == h.generation;
        }

        // Dense index of a live handle (alive() must hold).
        std::size_t dense_index(Handle h) const noexcept { return slots_[h.slot].dense; }

        // Slot backing dense element i (for handle hand-out during walks).
        Handle handle_at(std::size_t dense) const noexcept
        {
            const std::uint32_t slot = dense_to_slot_[dense];
            return Handle{slot, slots_[slot].generation};
        }

        // ---- column access ---------------------------------------------------

        // Contiguous view of column I — the cache-streaming target for §7.1
        // kernels. Bases are 64-byte aligned (rule 4).
        template <std::size_t I>
        std::span<column_element_t<I>> column() noexcept
        {
            return std::span<column_element_t<I>>(
                static_cast<column_element_t<I>*>(columns_[I]), size_);
        }

        template <std::size_t I>
        std::span<const column_element_t<I>> column() const noexcept
        {
            return std::span<const column_element_t<I>>(
                static_cast<const column_element_t<I>*>(columns_[I]), size_);
        }

    private:
        struct Slot
        {
            std::uint32_t generation = 0;
            std::uint32_t dense = kSoaFreeSlot;
        };

        static constexpr std::size_t kMinCapacity = 16;

        template <std::size_t I = 0>
        void allocate_columns(std::size_t cap)
        {
            if constexpr (I < sizeof...(Ts))
            {
                using E = column_element_t<I>;
                columns_[I] = resource_->allocate(cap * sizeof(E), kSoaColumnAlignment);
                allocate_columns<I + 1>(cap);
            }
        }

        template <std::size_t I = 0>
        void deallocate_columns(std::size_t cap) noexcept
        {
            if constexpr (I < sizeof...(Ts))
            {
                using E = column_element_t<I>;
                resource_->deallocate(columns_[I], cap * sizeof(E), kSoaColumnAlignment);
                deallocate_columns<I + 1>(cap);
            }
        }

        // Placement-construct a full row at dense position d.
        template <std::size_t I = 0, typename... Args>
        void emplace_row(std::uint32_t d, Args&&... values)
        {
            if constexpr (I < sizeof...(Ts))
            {
                using E = column_element_t<I>;
                void* dst = static_cast<E*>(columns_[I]) + d;
                ::new (dst) E(std::get<I>(std::forward_as_tuple(std::forward<Args>(values)...)));
                emplace_row<I + 1>(d, std::forward<Args>(values)...);
            }
        }

        // Move row src into hole dst, then destroy src (swap-and-pop step).
        template <std::size_t I = 0>
        void move_row(std::uint32_t src, std::uint32_t dst)
        {
            if constexpr (I < sizeof...(Ts))
            {
                using E = column_element_t<I>;
                E* s = static_cast<E*>(columns_[I]) + src;
                E* t = static_cast<E*>(columns_[I]) + dst;
                if constexpr (std::is_trivially_copyable_v<E>)
                {
                    std::memcpy(static_cast<void*>(t), static_cast<const void*>(s), sizeof(E));
                }
                else
                {
                    ::new (static_cast<void*>(t)) E(std::move(*s));
                    s->~E();
                }
                move_row<I + 1>(src, dst);
            }
        }

        template <std::size_t I = 0>
        void destroy_row(std::uint32_t d)
        {
            if constexpr (I < sizeof...(Ts))
            {
                using E = column_element_t<I>;
                (static_cast<E*>(columns_[I]) + d)->~E();
                destroy_row<I + 1>(d);
            }
        }


        static std::size_t geometric_capacity(std::size_t requested) noexcept
        {
            // Geometric (power-of-two) growth; §7.2 rule 1 cold-path compaction.
            return std::bit_ceil(requested < kMinCapacity ? std::size_t{kMinCapacity} : requested);
        }

        void grow(std::size_t requested)
        {
            const std::size_t new_capacity = geometric_capacity(requested);
            if (new_capacity <= capacity_) return;

            const std::size_t old_capacity = capacity_;

            // Columns: new buffer, copy live prefix, retire old (cold path).
            realloc_columns<0>(new_capacity);

            // Slot table and dense->slot map grow with the columns.
            auto* new_slots = static_cast<Slot*>(
                resource_->allocate(new_capacity * sizeof(Slot), alignof(Slot)));
            auto* new_dts = static_cast<std::uint32_t*>(resource_->allocate(
                new_capacity * sizeof(std::uint32_t), alignof(std::uint32_t)));

            if (old_capacity > 0)
            {
                std::memcpy(static_cast<void*>(new_slots), static_cast<const void*>(slots_),
                            slot_count_ * sizeof(Slot));
                std::memcpy(static_cast<void*>(new_dts), static_cast<const void*>(dense_to_slot_),
                            size_ * sizeof(std::uint32_t));
                resource_->deallocate(slots_, old_capacity * sizeof(Slot), alignof(Slot));
                resource_->deallocate(dense_to_slot_, old_capacity * sizeof(std::uint32_t),
                                      alignof(std::uint32_t));
            }

            slots_ = new_slots;
            dense_to_slot_ = new_dts;
            capacity_ = new_capacity;
            ++compactions_;
        }

        template <std::size_t I = 0>
        void realloc_columns(std::size_t new_capacity)
        {
            if constexpr (I < sizeof...(Ts))
            {
                using E = column_element_t<I>;
                void* fresh = resource_->allocate(new_capacity * sizeof(E), kSoaColumnAlignment);
                if (size_ > 0)
                {
                    std::memcpy(fresh, columns_[I], size_ * sizeof(E));
                }
                resource_->deallocate(columns_[I], capacity_ * sizeof(E), kSoaColumnAlignment);
                columns_[I] = fresh;
                realloc_columns<I + 1>(new_capacity);
            }
        }

        void move_from(SoaTable& other) noexcept
        {
            resource_ = other.resource_;
            columns_ = other.columns_;
            slots_ = other.slots_;
            dense_to_slot_ = other.dense_to_slot_;
            free_list_ = std::move(other.free_list_);
            capacity_ = std::exchange(other.capacity_, std::size_t{0});
            size_ = std::exchange(other.size_, std::size_t{0});
            slot_count_ = std::exchange(other.slot_count_, std::uint32_t{0});
            compactions_ = std::exchange(other.compactions_, std::size_t{0});
            other.columns_.fill(nullptr);
            other.slots_ = nullptr;
            other.dense_to_slot_ = nullptr;
        }

        void release() noexcept
        {
            if (capacity_ == 0) return;
            deallocate_columns<>(capacity_);
            resource_->deallocate(slots_, capacity_ * sizeof(Slot), alignof(Slot));
            resource_->deallocate(dense_to_slot_, capacity_ * sizeof(std::uint32_t),
                                  alignof(std::uint32_t));
            capacity_ = 0;
        }

        std::pmr::memory_resource* resource_ = nullptr;
        std::array<void*, sizeof...(Ts)> columns_{};
        Slot* slots_ = nullptr;                   // per-slot generation + dense index
        std::uint32_t* dense_to_slot_ = nullptr;  // dense -> slot back-map
        std::pmr::vector<std::uint32_t> free_list_{}; // retired slots (cold structure)
        std::size_t capacity_ = 0;
        std::size_t size_ = 0;
        std::uint32_t slot_count_ = 0;
        std::size_t compactions_ = 0;
    };

} // namespace shs::containers

