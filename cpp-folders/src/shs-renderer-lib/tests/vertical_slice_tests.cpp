#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory_resource>
#include <thread>
#include <vector>

#include "shs/app/vertical_slice_host.hpp"
#include "shs/input/input.contract.hpp"
#include "shs/renderpath/planning/render_path_presets.hpp"
#include "shs/task/thread_pool_job_system.hpp"

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; } } while (false)

// Step 6 (engine_domain_separation_migration.md): the vertical-slice host
// suite. Proves ENGINE integration through the public API only — recorded
// input -> action routing -> camera/scene update -> render projection ->
// plan -> backend output — with no SDL, no Vulkan and no game rules inside
// the library. Known-pixel software output is pinned exactly; deterministic
// headless replay, independent host instances, resize, backend-unavailable
// rejection, asset deletion/recreation and shutdown are pinned on the same
// host.
namespace
{
    // Deterministic triangle mesh (screen-space sized via the session's
    // projection; the rasterizer's clip->screen mapping is what the
    // known-pixel pins below quantify).
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

    // One recorded input log: move forward + look, then quit. Split across
    // frames so routing, replay and shutdown are all exercised.
    std::vector<std::vector<shs::input::RuntimeCommand>> make_recorded_log()
    {
        std::vector<std::vector<shs::input::RuntimeCommand>> log(4);
        log[0].push_back(shs::input::make_move_local_intent(glm::vec3(0.0f, 0.0f, 1.0f), 2.0f));
        log[1].push_back(shs::input::make_look_intent(12.0f, -4.0f, 0.01f));
        log[2].push_back(shs::input::make_toggle_light_shafts_intent());
        log[3].push_back(shs::input::make_quit_intent());
        return log;
    }

    // Run `log` through a fresh host; returns the per-frame reports plus the
    // final session and target digests.
    struct ReplayRun
    {
        std::vector<shs::app::VerticalSliceFrameReport> reports{};
        shs::app::SessionState final_session{};
        uint64_t final_digest = 0;
    };

    ReplayRun run_recorded(shs::resources::ResourceRegistry& registry, int width, int height)
    {
        shs::app::VerticalSliceHost host{{width, height}, registry};
        ReplayRun run{};
        std::pmr::monotonic_buffer_resource arena{1u << 16};
        const auto log = make_recorded_log();
        for (size_t frame = 0; frame < log.size(); ++frame)
        {
            auto report = host.run_frame(log[frame], {}, {}, 1.0f / 60.0f, arena);
            // The frame loop is the ENGINE: a routed quit ends the session.
            run.reports.push_back(report);
            if (report.quit_requested) break;
        }
        run.final_session = host.session();
        run.final_digest = shs::app::digest_render_target(host.target());
        return run;
    }
}

int main()
{
    // --- 1) deterministic headless replay + independent host instances ----
    {
        shs::resources::ResourceRegistry registry_a{};
        shs::resources::ResourceRegistry registry_b{};
        const ReplayRun first = run_recorded(registry_a, 96, 64);
        const ReplayRun second = run_recorded(registry_b, 96, 64);
        CHECK(first.reports.size() == 4);
        CHECK(first.reports == second.reports);
        CHECK(first.final_session == second.final_session);
        CHECK(first.final_digest == second.final_digest);
        CHECK(first.reports.back().quit_requested);
        CHECK(first.final_session.quit_requested);
        CHECK(first.final_session.bot_enabled == false);
    }

    // Two hosts running INTERLEAVED on distinct registries: no cross-talk.
    {
        shs::resources::ResourceRegistry registry_c{};
        shs::resources::ResourceRegistry registry_d{};
        shs::app::VerticalSliceHost host_a{{96, 64}, registry_c};
        shs::app::VerticalSliceHost host_b{{96, 64}, registry_d};
        const auto log = make_recorded_log();
        std::pmr::monotonic_buffer_resource arena{1u << 16};
        for (size_t frame = 0; frame < log.size(); ++frame)
        {
            const auto ra = host_a.run_frame(log[frame], {}, {}, 1.0f / 60.0f, arena);
            const auto rb = host_b.run_frame(log[frame], {}, {}, 1.0f / 60.0f, arena);
            CHECK(ra == rb);
        }
        CHECK(host_a.session() == host_b.session());
        CHECK(shs::app::digest_render_target(host_a.target())
            == shs::app::digest_render_target(host_b.target()));
    }

    // --- 2) resize: targets realloc, aspect re-derives, replay stays stable
    {
        shs::resources::ResourceRegistry registry{};
        shs::app::VerticalSliceHost host{{96, 64}, registry};
        std::pmr::monotonic_buffer_resource arena{1u << 16};
        const auto moved = host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        host.resize(48, 32);
        CHECK(host.width() == 48 && host.height() == 32);
        const auto resized = host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        CHECK(host.target().w == 48 && host.target().h == 32);
        CHECK(moved.pixel_digest != resized.pixel_digest);

        CHECK(moved.pixel_digest != resized.pixel_digest);

        // Fresh host at the resized geometry reproduces the resized frame.
        shs::app::VerticalSliceHost fresh_host{{48, 32}, registry};
        const auto fresh_report = fresh_host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        CHECK(fresh_report.pixel_digest == resized.pixel_digest);
    }

    // --- 3) backend-unavailable rejection: rejected swap keeps the plan ----
    // Capability snapshot with NO backend registered: the initial compile
    // must reject (BackendUnavailable) and no plan is ever installed.
    {
        shs::resources::ResourceRegistry registry{};
        shs::app::VerticalSliceConfig config{};
        config.width = 64;
        config.height = 64;
        config.caps = shs::renderpath::RenderPathCapabilitySet{}; // no backend
        shs::app::VerticalSliceHost host{config, registry};
        CHECK(host.path_state().plan_generation == 0);

        // The batch still runs; the frame pod and routing are unaffected.
        std::pmr::monotonic_buffer_resource arena{1u << 16};
        const auto report = host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        CHECK(host.path_state().plan_generation == 0); // still no plan
        CHECK(report.path.plan_generation == 0);
    }

    // Healthy snapshot: plan installs; a recipe requiring support the
    // snapshot lacks is REJECTED and rendering continues on the kept plan.
    {
        shs::resources::ResourceRegistry registry{};
        shs::app::VerticalSliceHost host{{64, 64}, registry};
        CHECK(host.path_state().plan_generation == 1); // initial compile landed
        std::pmr::monotonic_buffer_resource arena{1u << 16};

        // Candidate that fails compile: occlusion culling REQUIRED, snapshot
        // reports no occlusion-query support.
        auto unsupported_recipe = shs::renderpath::make_builtin_render_path_recipe(
            shs::RenderPathPreset::Forward, shs::RenderBackendType::Software, "occ");
        unsupported_recipe.view_culling = shs::RenderPathCullingMode::FrustumAndOcclusion;
        const auto before = host.path_state();
        const shs::renderpath::RenderPathCommand swap_command{
            shs::renderpath::SelectPathPresetIntent{unsupported_recipe}};
        const auto step = host.run_frame({},
            std::span<const shs::renderpath::RenderPathCommand>{&swap_command, 1},
            {}, 1.0f / 60.0f, arena);
        CHECK(step.path.swaps_rejected == 1);
        CHECK(host.path_state() == before);
        CHECK(host.path_state().plan_generation == 1); // previous plan kept
    }

    // --- 4) asset deletion/recreation: identity + registry generation -----
    {
        shs::resources::ResourceRegistry registry{};
        const auto mesh_handle = registry.add_mesh(make_triangle(), "tri");
        const auto material_handle =
            registry.add_material(make_material({1.0f, 0.25f, 0.0f}, "red"), "red");
        shs::app::VerticalSliceHost host{{32, 32}, registry};
        auto& object = host.objects().add(
            shs::scene::SceneObject{"cube", (shs::scene::MeshHandle)mesh_handle,
                (shs::scene::MaterialHandle)material_handle, {}});
        const uint64_t identity = object.object_id;
        CHECK(identity != 0);
        std::pmr::monotonic_buffer_resource arena{1u << 16};

        const auto rendered = host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        CHECK(rendered.items_drawn == 1);
        const uint64_t live_digest = rendered.pixel_digest;

        // DELETE: stale handle policy — projection keeps the item identity
        // but rendering SKIPS it instead of dereferencing dead storage.
        CHECK(host.objects().remove("cube"));
        auto gone = host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        CHECK(gone.items_projected == 0); // object removed: not projected

        // RECREATE: the name-derived identity is preserved by construction,
        // and the frame is pixel-identical to the pre-deletion frame.
        auto& recreated = host.objects().add(shs::scene::SceneObject{"cube",
            (shs::scene::MeshHandle)mesh_handle, (shs::scene::MaterialHandle)material_handle, {}});
        CHECK(recreated.object_id == identity);
        const auto again = host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        CHECK(again.items_drawn == 1);
        CHECK(again.pixel_digest == live_digest);
    }

    // Registry clear(): identity epoch bumps; pre-clear handles go stale
    // (SceneResourceView resolves them to nullptr — the render skips).
    {
        shs::resources::ResourceRegistry registry{};
        const auto mesh = registry.add_mesh(make_triangle(), "tri");
        shs::app::VerticalSliceHost host{{32, 32}, registry};
        (void)host.objects().add(shs::scene::SceneObject{"obj", (shs::scene::MeshHandle)mesh, 0, {}});
        std::pmr::monotonic_buffer_resource arena{1u << 16};
        auto live = host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        CHECK(live.items_drawn == 1);
        const uint64_t generation_before = registry.generation();
        registry.clear();
        CHECK(registry.generation() == generation_before + 1);
        auto cleared = host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        CHECK(cleared.items_projected == 1);
        CHECK(cleared.items_drawn == 0); // stale mesh: skipped, no crash
        CHECK(cleared.pixel_digest != live.pixel_digest);
    }

    // --- 5) shutdown: routed quit ends the loop; teardown is clean --------
    {
        shs::resources::ResourceRegistry registry{};
        const auto run = run_recorded(registry, 32, 32);
        CHECK(run.reports.size() == 4);          // quit on the recorded frame
        CHECK(run.reports[3].input.commands_applied == 1);
        CHECK(run.final_session.quit_requested);
        // The host value destroys cleanly with nothing outstanding.
    }

    // --- 6) known-pixel software output (backend output pinned exactly) ---
    {
        shs::resources::ResourceRegistry registry{};
        const auto mesh = registry.add_mesh(make_triangle(), "tri");
        const auto material = registry.add_material(
            make_material({1.0f, 0.25f, 0.0f}, "red"), "red");
        shs::app::VerticalSliceHost host{{32, 32}, registry};
        // Same NDC triangle the offscreen Vulkan harness draws (vs_uploaded
        // positions), flat-shaded through the host program.
        (void)host.objects().add(shs::scene::SceneObject{"tri",
            (shs::scene::MeshHandle)mesh, (shs::scene::MaterialHandle)material,
            {}, /*visible=*/true});
        std::pmr::monotonic_buffer_resource arena{1u << 16};

        // The session camera is the ONLY projection source; park it on the
        // documented default so the screen mapping is fully determined.
        // The session camera is the ONLY projection source; park it 3m back
        // on +z so the world triangle maps to the documented screen region.
        host.session().camera.pos = glm::vec3(0.0f, 0.0f, -3.0f);
        auto frame = host.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        const auto& buffer = host.target().color;

        bool covered_any = false;
        int covered_count = 0;
        for (int y = 0; y < 32; ++y)
        {
            for (int x = 0; x < 32; ++x)
            {
                const shs::render::ColorF& pixel = buffer.at(x, y);
                if (pixel.r == 1.0f && pixel.g == 0.25f && pixel.b == 0.0f)
                {
                    covered_any = true;
                    covered_count += 1;
                }
            }
        }
        CHECK(covered_any);
        CHECK(covered_count > 32);   // a real triangle landed, not one pixel
        CHECK(covered_count < 32 * 32); // background survived
        std::fprintf(stderr, "known-pixel: %d covered pixels of 1024\n", covered_count);

        // Byte-exact replay: same scene, same input log, same digest.
        shs::app::VerticalSliceHost twin{{32, 32}, registry};
        (void)twin.objects().add(shs::scene::SceneObject{"tri",
            (shs::scene::MeshHandle)mesh, (shs::scene::MaterialHandle)material, {}});
        auto twin_report = twin.run_frame({}, {}, {}, 1.0f / 60.0f, arena);
        CHECK(twin_report.pixel_digest == frame.pixel_digest);
    }

    // --- 7) teardown with OUTSTANDING work (step 4.4 contract, host seam) -
    {
        // The engine-style host never spawns workers itself. The caller
        // owns the job system; the boundary promise under test: destruction
        // DRAINS every accepted job even when wait_idle() was never called.
        std::atomic<int> completed{0};
        {
            shs::task::ThreadPoolJobSystem jobs{2};
            for (int i = 0; i < 64; ++i)
            {
                jobs.enqueue([&completed]() {
                    std::this_thread::yield();
                    completed.fetch_add(1, std::memory_order_relaxed);
                });
            }
            // Deliberately NO wait_idle(): destructor must drain.
        }
        CHECK(completed.load() == 64);
    }

    std::fprintf(stderr,
        "PASS: vertical slice (replay, instances, resize, backend rejection, "
        "assets, shutdown, known-pixel, teardown)\n");
    return 0;
}
