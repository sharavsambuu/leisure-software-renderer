#pragma once

/*
    FlatMap<K, V> — open-addressing pmr keyed map (spec §7.2 rule 5).

    Node-free keyed lookup over contiguous storage: SoA key/value arrays with
    64-byte-aligned bases, separate control-byte array, linear probing,
    power-of-two capacity, tombstone deletion with lazy rehash. All storage is
    pmr-allocated; no std::map/unordered_map anywhere in hot state.

    P1.5 (roadmap): shared lib utility promoted per §7.2 rule 6.
*/

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory_resource>
#include <new>
#include <utility>

namespace shs::containers
{
    template <typename K, typename V, typename Hash = std::hash<K>,
              typename KeyEqual = std::equal_to<K>>
    class FlatMap
    {
    public:
        static constexpr float kMaxLoadFactor = 0.7f;
        static constexpr std::size_t kMinCapacity = 16;

        enum class Slot : std::uint8_t
        {
            kEmpty = 0,
            kFull = 1,
            kTombstone = 2,
        };

        explicit FlatMap(std::pmr::memory_resource* resource, std::size_t initial_capacity = kMinCapacity,
                         Hash hash = Hash{}, KeyEqual equal = KeyEqual{})
            : resource_(resource), hash_(hash), equal_(equal)
        {
            reserve(initial_capacity);
        }

        FlatMap(const FlatMap&) = delete;
        FlatMap& operator=(const FlatMap&) = delete;

        FlatMap(FlatMap&& other) noexcept { move_from(other); }

        FlatMap& operator=(FlatMap&& other) noexcept
        {
            if (this != &other)
            {
                release();
                move_from(other);
            }
            return *this;
        }

        ~FlatMap() { release(); }

        // ---- capacity ------------------------------------------------------

        void reserve(std::size_t n)
        {
            const std::size_t needed = static_cast<std::size_t>(
                static_cast<double>(n) / kMaxLoadFactor) + 1;
            if (needed > capacity_) rehash(next_pow2(needed));
        }

        std::size_t size() const noexcept { return size_; }
        std::size_t capacity() const noexcept { return capacity_; }
        bool empty() const noexcept { return size_ == 0; }

        // ---- lookup ----------------------------------------------------------

        V* find(const K& key) noexcept
        {
            return const_cast<V*>(static_cast<const FlatMap*>(this)->find(key));
        }

        const V* find(const K& key) const noexcept
        {
            if (capacity_ == 0) return nullptr;
            std::size_t i = probe_start(key);
            while (true)
            {
                const Slot s = states_[i];
                if (s == Slot::kEmpty) return nullptr;
                if (s == Slot::kFull && equal_(keys_[i], key)) return &values_[i];
                i = (i + 1) & mask_;
            }
        }

        bool contains(const K& key) const noexcept { return find(key) != nullptr; }

        // ---- modification ----------------------------------------------------

        // Insert or overwrite; returns pointer to the stored value.
        V* insert_or_assign(const K& key, V value)
        {
            if (dirty_ * 4 >= capacity_ * 3 || size_ + 1 > capacity_ * kMaxLoadFactor)
            {
                rehash(capacity_ * 2);
            }
            const std::size_t idx = insert_probe(key, std::move(value));
            return &values_[idx];
        }

        // Erase via tombstone (linear probing stays intact). Returns false if
        // the key was absent.
        bool erase(const K& key)
        {
            if (capacity_ == 0) return false;
            std::size_t i = probe_start(key);
            while (true)
            {
                const Slot s = states_[i];
                if (s == Slot::kEmpty) return false;
                if (s == Slot::kFull && equal_(keys_[i], key))
                {
                    states_[i] = Slot::kTombstone;
                    keys_[i].~K();
                    values_[i].~V();
                    --size_;
                    return true;
                }
                i = (i + 1) & mask_;
            }
        }

        void clear()
        {
            for (std::size_t i = 0; i < capacity_; ++i)
            {
                if (states_[i] == Slot::kFull)
                {
                    keys_[i].~K();
                    values_[i].~V();
                }
                states_[i] = Slot::kEmpty;
            }
            size_ = 0;
            dirty_ = 0;
        }

        // Iteration over live entries.
        template <typename Fn>
        void for_each(Fn&& fn)
        {
            for (std::size_t i = 0; i < capacity_; ++i)
            {
                if (states_[i] == Slot::kFull) fn(keys_[i], values_[i]);
            }
        }

    private:
        static std::size_t next_pow2(std::size_t v) noexcept
        {
            std::size_t p = kMinCapacity;
            while (p < v) p <<= 1;
            return p;
        }

        std::size_t probe_start(const K& key) const noexcept { return hash_(key) & mask_; }

        // Insert without capacity checks; returns index of the stored slot.
        std::size_t insert_probe(const K& key, V value)
        {
            std::size_t i = probe_start(key);
            std::size_t first_tombstone = kNoIndex;
            while (true)
            {
                const Slot s = states_[i];
                if (s == Slot::kFull)
                {
                    if (equal_(keys_[i], key))
                    {
                        values_[i] = std::move(value);
                        return i; // overwrite in place
                    }
                }
                else
                {
                    if (s == Slot::kTombstone && first_tombstone == kNoIndex)
                    {
                        first_tombstone = i;
                    }
                    if (s == Slot::kEmpty) break;
                }
                i = (i + 1) & mask_;
            }
            const std::size_t target = (first_tombstone != kNoIndex) ? first_tombstone : i;
            if (states_[target] == Slot::kEmpty) ++dirty_;
            ::new (static_cast<void*>(&keys_[target])) K(key);
            ::new (static_cast<void*>(&values_[target])) V(std::move(value));
            states_[target] = Slot::kFull;
            ++size_;
            return target;
        }


        void rehash(std::size_t new_capacity)
        {
            const std::size_t old_capacity = capacity_;
            K* old_keys = keys_;
            V* old_values = values_;
            Slot* old_states = states_;

            capacity_ = next_pow2(new_capacity);
            mask_ = capacity_ - 1;
            keys_ = static_cast<K*>(resource_->allocate(capacity_ * sizeof(K), kMapAlignment));
            values_ = static_cast<V*>(resource_->allocate(capacity_ * sizeof(V), kMapAlignment));
            states_ = static_cast<Slot*>(
                resource_->allocate(capacity_ * sizeof(Slot), alignof(Slot)));
            std::memset(static_cast<void*>(states_), 0, capacity_ * sizeof(Slot)); // all kEmpty

            size_ = 0;
            dirty_ = 0;
            if (old_capacity == 0) return;

            for (std::size_t i = 0; i < old_capacity; ++i)
            {
                if (old_states[i] == Slot::kFull)
                {
                    insert_probe(std::move_if_noexcept(old_keys[i]),
                                 std::move_if_noexcept(old_values[i]));
                    old_keys[i].~K();
                    old_values[i].~V();
                }
            }
            resource_->deallocate(old_keys, old_capacity * sizeof(K), kMapAlignment);
            resource_->deallocate(old_values, old_capacity * sizeof(V), kMapAlignment);
            resource_->deallocate(old_states, old_capacity * sizeof(Slot), alignof(Slot));
        }

        void release() noexcept
        {
            if (capacity_ == 0) return;
            for (std::size_t i = 0; i < capacity_; ++i)
            {
                if (states_[i] == Slot::kFull)
                {
                    keys_[i].~K();
                    values_[i].~V();
                }
            }
            resource_->deallocate(keys_, capacity_ * sizeof(K), kMapAlignment);
            resource_->deallocate(values_, capacity_ * sizeof(V), kMapAlignment);
            resource_->deallocate(states_, capacity_ * sizeof(Slot), alignof(Slot));
            capacity_ = 0;
        }

        void move_from(FlatMap& other) noexcept
        {
            resource_ = other.resource_;
            keys_ = std::exchange(other.keys_, nullptr);
            values_ = std::exchange(other.values_, nullptr);
            states_ = std::exchange(other.states_, nullptr);
            capacity_ = std::exchange(other.capacity_, std::size_t{0});
            mask_ = std::exchange(other.mask_, std::size_t{0});
            size_ = std::exchange(other.size_, std::size_t{0});
            dirty_ = std::exchange(other.dirty_, std::size_t{0});
            hash_ = std::move(other.hash_);
            equal_ = std::move(other.equal_);
        }

        static constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);
        static constexpr std::size_t kMapAlignment = 64; // §7.2 rule 4: column bases

        std::pmr::memory_resource* resource_ = nullptr;
        K* keys_ = nullptr;
        V* values_ = nullptr;
        Slot* states_ = nullptr;
        std::size_t capacity_ = 0;
        std::size_t mask_ = 0;
        std::size_t size_ = 0;   // live entries
        std::size_t dirty_ = 0;  // full + tombstones (rehash trigger)
        Hash hash_{};
        KeyEqual equal_{};
    };

} // namespace shs::containers

