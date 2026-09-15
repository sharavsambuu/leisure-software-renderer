#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory_resource>
#include <new>
#include <vector>

#include "shs/containers/flat_map.hpp"
#include "shs/containers/soa_table.hpp"
#include "shs/memory/frame_memory_resource.hpp"

// Pure value tests for the §7.2 contiguous backing-store utilities (roadmap
// P1.5 ctest gate: shs_renderer_vop_containers_*). Headless: the binary links
// only the header-only shs::renderer-values INTERFACE target — no SDL, no
// Vulkan, no assimp, no Context.
namespace
{
    // --- FrameMemoryResource ------------------------------------------------

    // O(1) reset, alignment honored, bump region reuse, diagnostics sane.
    bool test_arena_basics()
    {
        shs::memory::FrameMemoryResource arena{1024 * 1024};

        void* a = arena.allocate(64, 1);           // unaligned request
        auto* aligned = static_cast<std::uint8_t*>(arena.allocate(3, 64));
        if (reinterpret_cast<std::uintptr_t>(aligned) % 64 != 0) return false;
        if (reinterpret_cast<std::uintptr_t>(a) % alignof(std::max_align_t) != 0) return false;

        const std::size_t before_reset = arena.used();
        if (before_reset < 67) return false;       // accounting includes padding

        arena.reset();                              // O(1) frame boundary
        if (arena.used() != 0) return false;
        if (arena.high_water_mark() < before_reset) return false;

        // Re-allocation after reset reuses the same region (bump behavior).
        void* b = arena.allocate(64, 1);
        if (b != a) return false;

        arena.deallocate(b, 64, 1);                 // no-op by frame-tier contract
        return true;
    }

    // Overflow is strict bad_alloc — never a silent fallback to the
    // persistent tier (the §3 Rule 5.1 violation the old snake copy had).
    bool test_arena_overflow_strict()
    {
        shs::memory::FrameMemoryResource arena{4096};
        (void)arena.allocate(4096, 1);
        bool threw = false;
        try
        {
            (void)arena.allocate(1, 1);
        }
        catch (const std::bad_alloc&)
        {
            threw = true;
        }
        return threw;
    }

    // --- SoaTable -------------------------------------------------------------

    // Insert/erase round trip; generational handle stability across swap-and-pop.
    bool test_soa_insert_erase_handles()
    {
        std::pmr::monotonic_buffer_resource arena{1 << 20};
        shs::containers::SoaTable<std::uint32_t, float> table{&arena, 8};

        auto h0 = table.insert(100u, 1.0f);
        auto h1 = table.insert(200u, 2.0f);
        auto h2 = table.insert(300u, 3.0f);
        if (table.size() != 3) return false;
        if (!table.alive(h0) || !table.alive(h1) || !table.alive(h2)) return false;

        // Dense reads through the column spans.
        if (table.column<0>()[0] != 100u || table.column<0>()[1] != 200u ||
            table.column<0>()[2] != 300u)
            return false;
        if (table.column<1>()[1] != 2.0f) return false;

        // Erase the middle: swap-and-pop moves dense row 2 into dense 1. h2
        // must still resolve (to its new dense index); h1 must be dead.
        if (!table.erase(h1)) return false;
        if (table.size() != 2) return false;
        if (table.alive(h1)) return false;                       // stale handle
        if (!table.alive(h2)) return false;
        if (table.dense_index(h2) != 1) return false;
        if (table.column<0>()[1] != 300u || table.column<1>()[1] != 3.0f) return false;
        if (!table.alive(h0) || table.dense_index(h0) != 0) return false;
        return true;
    }

    // Growth is geometric + cold; handles stay valid across compactions
    // (§7.2 rule 1 + rule 2).
    bool test_soa_growth_handle_stability()
    {
        std::pmr::monotonic_buffer_resource arena{1 << 20};
        shs::containers::SoaTable<std::uint64_t, std::uint64_t> table{&arena, 4};

        constexpr std::size_t kCount = 1000;
        std::vector<shs::containers::SoaHandle> handles;
        handles.reserve(kCount);

        for (std::size_t i = 0; i < kCount; ++i)
        {
            handles.push_back(table.insert(static_cast<std::uint64_t>(i * 7),
                                           static_cast<std::uint64_t>(i * 13)));
        }
        if (table.size() != kCount) return false;
        if (table.compaction_count() == 0) return false;   // grew several times
        if (table.capacity() < kCount) return false;

        // Every handle still resolves to the exact same row values.
        for (std::size_t i = 0; i < kCount; ++i)
        {
            if (!table.alive(handles[i])) return false;
            const std::size_t d = table.dense_index(handles[i]);
            if (table.column<0>()[d] != i * 7) return false;
            if (table.column<1>()[d] != i * 13) return false;
        }
        return true;
    }

    // Dense-walk kill pattern: swap-and-pop keeps the table dense (§7.2
    // rule 3) — the exact pattern the snake particle kernel needs.
    bool test_soa_density_preservation()
    {
        std::pmr::monotonic_buffer_resource arena{1 << 20};
        shs::containers::SoaTable<std::uint32_t, float> table{&arena, 16};

        for (std::uint32_t i = 0; i < 100; ++i) (void)table.insert(i, static_cast<float>(i));
        if (table.size() != 100) return false;

        // Kill every even-keyed row in place (walk-and-kill kernel).
        for (std::size_t i = 0; i < table.size();)
        {
            if (table.column<0>()[i] % 2 == 0)
            {
                table.erase_dense(i);   // no ++i: swapped row re-examined
            }
            else
            {
                ++i;
            }
        }

        if (table.size() != 50) return false;
        for (std::size_t i = 0; i < table.size(); ++i)
        {
            const auto key = table.column<0>()[i];
            if (key % 2 == 0) return false;
            if (static_cast<float>(key) != table.column<1>()[i]) return false;
        }
        return true;
    }

    // Column bases are 64-byte aligned (§7.2 rule 4).
    bool test_soa_column_alignment()
    {
        std::pmr::monotonic_buffer_resource arena{1 << 20};
        shs::containers::SoaTable<std::uint8_t, float, std::uint32_t> table{&arena, 4};
        (void)table.insert(1u, 2.0f, 3u);

        if (reinterpret_cast<std::uintptr_t>(table.column<0>().data()) % 64 != 0) return false;
        if (reinterpret_cast<std::uintptr_t>(table.column<1>().data()) % 64 != 0) return false;
        if (reinterpret_cast<std::uintptr_t>(table.column<2>().data()) % 64 != 0) return false;
        return true;
    }


    // Linear walk stays cache-resident: 1M rows streamed through two columns
    // in a generous fixed budget (headless §7.1 smoke; no GPU/OS deps).
    bool test_soa_linear_walk_cache_resident()
    {
        std::pmr::monotonic_buffer_resource arena{1 << 21};
        shs::containers::SoaTable<float, float> table{&arena, 1u << 20};

        constexpr std::size_t kCount = 1u << 20;
        for (std::size_t i = 0; i < kCount; ++i) (void)table.insert(1.0f, 0.0f);
        // Upfront reserve honored: at most one boundary grow when the last
        // row hits full capacity — never per-insert reallocation.
        if (table.compaction_count() > 1) return false;

        auto pos = table.column<0>();
        auto vel = table.column<1>();

        const auto t0 = std::chrono::steady_clock::now();
        float acc = 0.0f;
        for (std::size_t i = 0; i < kCount; ++i)
        {
            pos[i] += vel[i];
            acc += pos[i];
        }
        const auto t1 = std::chrono::steady_clock::now();

        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (acc <= 0.0f) return false;   // fold to defeat dead-code elimination
        // Contiguous walk of 1M rows = 2 columns x 4 MB. Even cold DRAM
        // streaming is >>10 GB/s; a node-chasing layout cannot pass this.
        return ms < 100;
    }

    // --- FlatMap ---------------------------------------------------------------

    // insert/find/update/erase round trip including collision probing.
    bool test_flat_map_basics()
    {
        std::pmr::monotonic_buffer_resource arena{1 << 20};
        shs::containers::FlatMap<std::uint32_t, std::uint32_t> map{&arena, 16};

        for (std::uint32_t k = 0; k < 100; ++k)
        {
            (void)map.insert_or_assign(k * 16, k + 1);   // stride forces collisions
        }
        if (map.size() != 100) return false;
        for (std::uint32_t k = 0; k < 100; ++k)
        {
            const std::uint32_t* v = map.find(k * 16);
            if (v == nullptr || *v != k + 1) return false;
        }
        if (map.find(999) != nullptr) return false;

        // Overwrite in place.
        (void)map.insert_or_assign(0, 42);
        if (*map.find(0) != 42) return false;
        if (map.size() != 100) return false;

        // Erase + tombstone probing: erased key gone, collision chain behind
        // the tombstone still resolvable.
        if (!map.erase(0)) return false;
        if (map.contains(0)) return false;
        if (map.size() != 99) return false;
        if (map.erase(0)) return false;
        for (std::uint32_t k = 1; k < 100; ++k)
        {
            const std::uint32_t* v = map.find(k * 16);
            if (v == nullptr || *v != k + 1) return false;
        }
        return true;
    }


    // Rehash on growth preserves all entries (map is handle-free: re-look-up).
    bool test_flat_map_growth()
    {
        std::pmr::monotonic_buffer_resource arena{1 << 20};
        shs::containers::FlatMap<std::uint32_t, float> map{&arena, 8};

        constexpr std::uint32_t kCount = 5000;
        for (std::uint32_t k = 0; k < kCount; ++k)
        {
            (void)map.insert_or_assign(k, static_cast<float>(k));
        }
        if (map.size() != kCount) return false;
        if (map.capacity() < kCount) return false;

        for (std::uint32_t k = 0; k < kCount; ++k)
        {
            const float* v = map.find(k);
            if (v == nullptr || *v != static_cast<float>(k)) return false;
        }
        return true;
    }
} // namespace

int main()
{
    struct Case { const char* name; bool (*fn)(); };
    const Case cases[] = {
        {"arena_basics", test_arena_basics},
        {"arena_overflow_strict", test_arena_overflow_strict},
        {"soa_insert_erase_handles", test_soa_insert_erase_handles},
        {"soa_growth_handle_stability", test_soa_growth_handle_stability},
        {"soa_density_preservation", test_soa_density_preservation},
        {"soa_column_alignment", test_soa_column_alignment},
        {"soa_linear_walk_cache_resident", test_soa_linear_walk_cache_resident},
        {"flat_map_basics", test_flat_map_basics},
        {"flat_map_growth", test_flat_map_growth},
    };

    bool ok = true;
    for (const Case& c : cases)
    {
        const bool passed = c.fn();
        std::fprintf(stderr, "[containers-tests] %-34s %s\n", c.name,
                     passed ? "ok" : "FAIL");
        ok = passed && ok;
    }

    if (!ok)
    {
        std::fprintf(stderr, "[containers-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[containers-tests] all tests passed\n");
    return 0;
}

