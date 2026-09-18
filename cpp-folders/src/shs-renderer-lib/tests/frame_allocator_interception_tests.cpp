#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory_resource>
#include <new>
#include <vector>

#include <glm/glm.hpp>

#include "shs/app/vertical_slice_host.hpp"
#include "shs/scene/scene_objects.hpp"

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; } } while (false)

// Governance review 2026-09-18 todo G1.1 (Tension 2 / P2): turn the
// zero-heap-allocation frame law into a MECHANICAL gate. Global
// operator new/delete are overridden with counting replacements; the test
// arms them around VerticalSliceHost::run_frame and fails if any heap
// allocation escapes the caller's designated frame arena. A negative
// self-check proves the interception is real (a deliberate heap allocation
// counts), so the gate can never pass vacuously.
namespace
{
    std::atomic<bool> g_armed{false};
    std::atomic<uint64_t> g_alloc_count{0};
    std::atomic<size_t> g_alloc_bytes{0};

    inline void record_alloc(std::size_t n) noexcept
    {
        if (g_armed.load(std::memory_order_relaxed))
        {
            g_alloc_count.fetch_add(1, std::memory_order_relaxed);
            g_alloc_bytes.fetch_add(n, std::memory_order_relaxed);
        }
    }

    // RAII arm/disarm window. All allocation inside the scope is counted.
    struct ArmedWindow
    {
        ArmedWindow() { g_alloc_count.store(0); g_alloc_bytes.store(0); g_armed.store(true); }
        ~ArmedWindow() { g_armed.store(false); }
        ArmedWindow(const ArmedWindow&) = delete;
        ArmedWindow& operator=(const ArmedWindow&) = delete;
    };
}

// ---- global counting allocator replacements ------------------------------
void* operator new(std::size_t n)
{
    record_alloc(n);
    void* p = std::malloc(n);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n)
{
    return ::operator new(n);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

// Aligned variants: RasterVertex embeds 16-byte-aligned glm::vec4 members,
// so std::vector<RasterVertex> allocates through these — they MUST be
// policed or the rasterizer clip-buffer escape would be invisible.
void* operator new(std::size_t n, std::align_val_t al)
{
    const std::size_t a = static_cast<std::size_t>(al);
    const std::size_t rounded = (n + a - 1) / a * a;
    record_alloc(rounded == 0 ? a : rounded);
    void* p = std::aligned_alloc(a, rounded == 0 ? a : rounded);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n, std::align_val_t al)
{
    return ::operator new(n, al);
}
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
// ---------------------------------------------------------------------------


namespace
{
    shs::resources::MeshData make_triangle()
    {
        shs::resources::MeshData mesh{};
        mesh.positions = {
            glm::vec3(-0.5f, -0.5f, 0.0f),
            glm::vec3(0.5f, -0.5f, 0.0f),
            glm::vec3(0.0f, 0.5f, 0.0f),
        };
        mesh.indices = {0u, 1u, 2u};
        return mesh;
    }

    shs::resources::MaterialData make_material(const glm::vec3& color, const char* key)
    {
        shs::resources::MaterialData material{};
        material.name = key;
        material.base_color = color;
        return material;
    }
}

int main()
{
    // --- 1) negative self-check: the interception is real -------------------
    // Without this pin a broken override (armed flag never observed) would
    // make the frame gate pass vacuously.
    {
        {
            ArmedWindow window;
            (void)window;
            std::vector<int> deliberate(64, 7);
            deliberate.push_back(1);
            CHECK(g_alloc_count.load() >= 1);
            CHECK(g_alloc_bytes.load() >= sizeof(int) * 65);
        }
        // Disarmed: normal allocation must NOT count.
        g_alloc_count.store(0);
        std::vector<int> uncounted(32, 5);
        CHECK(g_alloc_count.load() == 0);
    }

    // --- 2) the law: steady-state run_frame is heap-allocation-free ---------
    // Setup + two WARM-UP frames run un-armed: host construction, the static
    // shader program, first-touch capacity (scene item projection, pmr event
    // vectors) are one-time caller-owned costs, not per-frame escapes. Every
    // frame AFTER warm-up must allocate ZERO from the global heap: per-frame
    // storage comes from the designated frame arena (a stack-buffer monotonic
    // resource here) or reuses warm capacity.
    {
        shs::resources::ResourceRegistry registry{};
        const auto mesh_handle = registry.add_mesh(make_triangle(), "tri");
        const auto material_handle =
            registry.add_material(make_material({1.0f, 0.25f, 0.0f}, "red"), "red");
        shs::app::VerticalSliceHost host{{48, 48}, registry};
        (void)host.objects().add(shs::scene::SceneObject{"tri",
            (shs::scene::MeshHandle)mesh_handle,
            (shs::scene::MaterialHandle)material_handle, {}});

        std::vector<shs::app::VerticalSliceFrameReport> warm{};
        warm.reserve(2);
        for (int i = 0; i < 2; ++i)
        {
            unsigned char arena_bytes[1u << 16];
            std::pmr::monotonic_buffer_resource arena{arena_bytes, sizeof arena_bytes};
            warm.push_back(host.run_frame({}, {}, {}, 1.0f / 60.0f, arena));
        }
        CHECK(warm.size() == 2);
        CHECK(warm[0] == warm[1]); // deterministic before the gate even runs

        // Armed steady-state frames: the gate. (Capacity reserved up front —
        // the reports are collected outside the armed window.)
        std::vector<shs::app::VerticalSliceFrameReport> steady{};
        steady.reserve(3);
        for (int frame = 0; frame < 3; ++frame)
        {
            unsigned char arena_bytes[1u << 16];
            std::pmr::monotonic_buffer_resource arena{arena_bytes, sizeof arena_bytes};
            shs::app::VerticalSliceFrameReport report{};
            {
                ArmedWindow window;
                report = host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
            }
            steady.push_back(std::move(report));
        }
        const uint64_t escaped = g_alloc_count.load();
        if (escaped != 0)
        {
            std::fprintf(stderr,
                "FAIL line %d: %llu heap allocation(s) (~%zu bytes) escaped the "
                "frame arena during run_frame\n",
                __LINE__, (unsigned long long)escaped, g_alloc_bytes.load());
            return 1;
        }
        CHECK(steady.size() == 3);
        for (const auto& report : steady)
        {
            CHECK(report.items_drawn == 1); // the triangle really rastered
            CHECK(report.pixel_digest == steady[0].pixel_digest);
        }
        CHECK(steady[0] == warm[1]); // warm-up frame identical
    }

    std::printf("frame allocator interception tests: all passed\n");
    return 0;
}
