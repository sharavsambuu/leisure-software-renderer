#include <cstdio>
#include <cstring>

#include <glm/glm.hpp>

#include "shs/render/software/rasterizer.hpp"
#include "shs/task/thread_pool_job_system.hpp"

using namespace shs::render;

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; } } while (false)

// R3 (renderer-lib review 2026-09-18): tile-path parity gate. The same mesh
// (overlapping + clipped triangles spanning multiple tiles) must rasterize
// BYTE-IDENTICALLY through the streaming path (no job system) and the R3
// tile path (one wait-free job per tile, ThreadPoolJobSystem), for several
// worker counts and tile sizes. Depth, motion, and color buffers are all
// compared; stats must match exactly.
namespace
{
    shs::resources::MeshData make_scene()
    {
        shs::resources::MeshData mesh{};
        // Big quad straddling several 32px tiles (in NDC; screen = (n+1)/2*(W-1)).
        // W=H=96: screen x = (n+1)*47.5
        mesh.positions = {
            {-1.0f, -1.0f, 0.5f}, {0.9f, -1.0f, 0.5f}, {-1.0f, 0.9f, 0.5f},   // tri 1
            {0.9f, 0.9f, 0.2f}, {0.9f, -1.0f, 0.2f}, {-1.0f, 0.9f, 0.2f},     // tri 2 overlaps tri 1
            {-0.4f, -0.4f, 0.8f}, {0.4f, -0.4f, 0.8f}, {0.0f, 0.4f, 0.8f},    // tri 3 small, on top
            {-1.5f, -0.5f, 0.6f}, {1.5f, -0.5f, 0.6f}, {0.0f, 1.5f, 0.6f},    // tri 4 clipped by frustum
        };
        mesh.indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        return mesh;
    }

    const auto& scene_program()
    {
        using namespace shs::render;
        static const auto program = ShaderProgramFn{
            [](const ShaderVertex& v, const ShaderUniforms& u) -> VertexOut {
                VertexOut o{};
                const glm::vec4 world = u.model * glm::vec4(v.position, 1.0f);
                o.clip = glm::vec4(v.position, 1.0f); // positions are already NDC-ish
                o.world_pos = glm::vec3(world);
                o.normal_ws = glm::vec3(0.0f, 0.0f, 1.0f);
                o.uv = v.uv;
                set_varying(o, VaryingSemantic::WorldPos, glm::vec4(o.world_pos, 1.0f));
                set_varying(o, VaryingSemantic::NormalWS, glm::vec4(o.normal_ws, 0.0f));
                set_varying(o, VaryingSemantic::UV0, glm::vec4(v.uv, 0.0f, 0.0f));
                return o;
            },
            [](const FragmentIn& fin, const ShaderUniforms& u) -> FragmentOut {
                FragmentOut o{};
                const float v = (float)(fin.px & 7) / 8.0f + u.base_color.r;
                o.color = ColorF{v * fin.depth01, fin.uv.x + fin.world_pos.y * 0.0f + 0.5f, 0.5f, 1.0f};
                if (v > 1.6f) o.discard = true;
                return o;
            }};
        return program;
    }

    struct Buffers
    {
        RT_ColorHDR hdr;
        RT_ColorDepthMotion depth_motion;
        RasterizerStats stats{};

        explicit Buffers(int w, int h)
            : hdr(w, h, ColorF{0.0f, 0.0f, 0.0f, 0.0f}),
              depth_motion(w, h, 0.1f, 10.0f) {}
    };

    void render(Buffers& buffers, const shs::resources::MeshData& mesh,
        shs::task::IJobSystem* jobs, int tile_size)
    {
        buffers.hdr.clear(ColorF{0.0f, 0.0f, 0.0f, 0.0f});
        buffers.depth_motion.clear_all();
        ShaderUniforms uniforms{};
        uniforms.model = glm::mat4(1.0f);
        uniforms.viewproj = glm::mat4(1.0f);
        uniforms.prev_model = glm::mat4(1.0f);
        uniforms.prev_viewproj = glm::mat4(1.0f);
        uniforms.base_color = glm::vec3(0.25f, 0.0f, 0.0f);
        uniforms.enable_motion_vectors = true;
        RasterizerConfig cfg{};
        cfg.cull_mode = RasterizerCullMode::None;
        cfg.job_system = jobs;
        cfg.tile_size = tile_size;
        buffers.stats = rasterize_mesh(mesh, scene_program(), uniforms,
            RasterizerTarget{&buffers.hdr, &buffers.depth_motion}, cfg);
    }
}

int main()
{
    const shs::resources::MeshData mesh = make_scene();

    // 1) Streaming reference (no job system).
    Buffers reference(96, 96);
    render(reference, mesh, nullptr, 32);
    CHECK(reference.stats.tri_input == 4);
    CHECK(reference.stats.tri_raster >= 3); // clipped tri still rasterizes its fan
    CHECK(reference.stats.tri_raster >= 3);

    // Sanity: something actually rendered, background survived.
    int covered = 0;
    for (int y = 0; y < 96; ++y)
        for (int x = 0; x < 96; ++x)
            if (reference.hdr.color.at(x, y).a == 1.0f) ++covered;
    CHECK(covered > 512);
    CHECK(covered < 96 * 96);

    // 2) Tile path must be byte-identical for several worker/tile combos.
    const size_t worker_counts[] = {1, 2, 4, 8};
    const int tile_sizes[] = {8, 16, 32, 64, 96, 128};
    for (size_t workers : worker_counts)
    {
        for (int tile : tile_sizes)
        {
            shs::task::ThreadPoolJobSystem jobs{workers};
            Buffers tiled(96, 96);
            render(tiled, mesh, &jobs, tile);
            CHECK(tiled.stats.tri_input == reference.stats.tri_input);
            CHECK(tiled.stats.tri_after_clip == reference.stats.tri_after_clip);
            CHECK(tiled.stats.tri_raster == reference.stats.tri_raster);
            CHECK(std::memcmp(tiled.hdr.color.data.data(), reference.hdr.color.data.data(),
                sizeof(ColorF) * 96 * 96) == 0);
            CHECK(std::memcmp(tiled.depth_motion.depth.data.data(), reference.depth_motion.depth.data.data(),
                sizeof(float) * 96 * 96) == 0);
            CHECK(std::memcmp(tiled.depth_motion.motion.data.data(), reference.depth_motion.motion.data.data(),
                sizeof(Motion2f) * 96 * 96) == 0);
        }
    }

    std::fprintf(stderr, "tile parity: streaming == tiled for all worker/tile combos (%d covered pixels) — OK\n", covered);
    return 0;
}
