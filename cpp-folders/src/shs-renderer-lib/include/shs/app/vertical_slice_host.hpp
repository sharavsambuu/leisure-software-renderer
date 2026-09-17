#pragma once

/*
    SHS RENDERER SAN

    FILE: vertical_slice_host.hpp
    MODULE: app
    PURPOSE: Step 6 (engine_domain_separation_migration.md): an ENGINE-STYLE
             VERTICAL-SLICE HOST built exclusively from public shs headers.
             Per frame it chains the full integration path:

               recorded input batch
                 -> action routing (shs::app::session_orchestrate, step 4.1)
                 -> camera/scene update (session_settings_sync funnels,
                    step 4.2)
                 -> render projection (SceneObjectSet::to_render_items +
                    SceneResourceView per-call asset resolution, step 4.3)
                 -> plan (shs::renderpath::renderpath_gateway pod)
                 -> backend output (shs::rasterize_mesh into RT_ColorHDR;
                    a stable FNV-1a digest over the color buffer is the
                    replay-verifiable backend output)

             GAME RULES STAY OUTSIDE the rendering library: the host carries
             no gameplay logic. Every per-frame decision is either a recorded
             command the caller supplied or a pure projection of session/
             scene/path state. No SDL, no Vulkan, no Context, no global event
             bus: the host is a value aggregate the caller instantiates any
             number of times (independent hosts are pinned by tests).

    COMPLETION / CANCELLATION BOUNDARIES (step 6):
      - run_frame is SYNCHRONOUS per host instance: when it returns, this
        frame's routing, plan swap, frame transition and raster are complete
        and the report/digest are valid.
      - Asset storage is an EXTERNAL ResourceRegistry the caller owns and
        mutates between frames. Unresolvable handles (deleted/cleared
        assets) render as SKIPPED items by policy — never a crash and never
        a stale-pointer read. Identity across deletion/recreation is the
        stable_object_id + registry generation() epoch contract
        (docs/spec/external_engine_seams.md).
      - Optional parallelism (rasterizer IJobSystem, asset loaders) is
        CALLER-OWNED: completion is guaranteed only by wait_idle(), teardown
        drains accepted work (shs::task contract, step 4.4). Tests pin
        teardown with outstanding work before any concurrency is promised.

    Pinned by vertical_slice_tests.
*/

#include <cstdint>
#include <memory_resource>
#include <memory>
#include <span>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shs/app/session_orchestrator.hpp"
#include "shs/app/session_settings_sync.hpp"
#include "shs/render/frame/frame.gateway.hpp"
#include "shs/render/software/rasterizer.hpp"
#include "shs/renderpath/planning/render_path_compiler.hpp"
#include "shs/renderpath/planning/render_path_presets.hpp"
#include "shs/renderpath/renderpath.gateway.hpp"
#include "shs/scene/scene_objects.hpp"
#include "shs/scene/scene_resource_view.hpp"
#include "shs/resources/storage/resource_registry.hpp"

namespace shs::app
{
    // Host configuration: value context fixed at construction (size may be
    // changed later through resize()).
    struct VerticalSliceConfig
    {
        int width = 160;
        int height = 120;

        // Initial render path recipe + the capability snapshot the plan is
        // compiled against. Backend availability is DATA here: a recipe the
        // snapshot cannot support is rejected by the compiler
        // (BackendUnavailable) and the previous plan survives — pinned.
        RenderPathRecipe recipe = shs::make_builtin_render_path_recipe(
            RenderPathPreset::Forward, RenderBackendType::Software, "slice");
        RenderPathCapabilitySet caps =
            make_render_path_capability_set(RenderBackendType::Software, BackendCapabilities{});
    };

    // Per-frame outcome summary (value type; comparable for replay pins).
    struct VerticalSliceFrameReport
    {
        shs::input::InputStep input{};          // action routing tally
        shs::renderpath::RenderPathStep path{}; // plan swap tally
        shs::frame::FrameStep frame{};          // frame pod transition
        uint32_t items_projected = 0;           // scene -> RenderItem count
        uint32_t items_drawn = 0;               // items that resolved + rasterized
        uint64_t pixel_digest = 0;              // FNV-1a over the HDR color buffer
        bool quit_requested = false;            // routed QuitIntent observed

        bool operator==(const VerticalSliceFrameReport&) const = default;
    };

    // Stable FNV-1a 64-bit digest over the raw color-buffer bytes. Same
    // scene + same recorded input log => same digest (replay goldens).
    inline uint64_t digest_render_target(const RT_ColorHDR& target)
    {
        uint64_t h = 1469598103934665603ull;
        const auto* bytes = reinterpret_cast<const unsigned char*>(target.color.data.data());
        const size_t count = target.color.data.size() * sizeof(ColorF);
        for (size_t i = 0; i < count; ++i)
        {
            h ^= (uint64_t)bytes[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    // Engine-style vertical-slice host (step 6). One instance == one
    // independent session; instances share no state. The caller owns the
    // asset registry; the host resolves handles through it per frame only.
    class VerticalSliceHost
    {
    public:
        VerticalSliceHost(VerticalSliceConfig config, ResourceRegistry& registry)
            : config_(config), registry_(registry)
        {
            rebuild_targets();
            // Initial plan through the canonical pod gateway: generation 0 =
            // no plan installed until the first batch compiles one.
            (void)swap_plan(config_.recipe);
        }

        // Run ONE frame of the vertical slice. `input_batch` is the recorded
        // input for this frame (may be empty); `path_commands` are recorded
        // renderpath intents (may be empty); events land on the caller's
        // frame arena.
        VerticalSliceFrameReport run_frame(
            std::span<const RuntimeCommand> input_batch,
            std::span<const renderpath::RenderPathCommand> path_commands,
            const shs::input::InputContext& input_context,
            float dt,
            std::pmr::memory_resource& arena)
        {
            VerticalSliceFrameReport report{};

            // 1) Action routing: recorded input -> session-owned state.
            std::pmr::vector<shs::input::InputEvent> input_events{&arena};
            report.input = session_orchestrate(
                session_, input_batch, input_context, input_events);

            // 2) Camera/scene update: the two canonical funnels (step 4.2).
            const float aspect =
                config_.height > 0 ? (float)config_.width / (float)config_.height : 1.0f;
            sync_session_to_scene(session_, scene_, aspect);

            frame_.w = config_.width;
            frame_.h = config_.height;
            frame_.dt = dt;
            frame_.time += dt;
            apply_session_render_settings(session_, frame_);

            // 3) Frame pod transition (the per-frame vehicle, C1.4).
            const shs::frame::FrameContext frame_context{};
            std::pmr::vector<shs::frame::FrameEvent> frame_events{&arena};
            report.frame = shs::frame::frame_gateway(
                frame_, std::span<const shs::frame::FrameCommand>{},
                frame_context, frame_events);

            // 4) Plan: recorded renderpath intents against the current pod.
            std::pmr::vector<renderpath::RenderPathEvent> path_events{&arena};
            report.path = renderpath::renderpath_gateway(
                path_, path_commands, compiler_, config_.caps, path_events);

            // 5) Render projection: scene objects -> render items (copies;
            //    vector-reference invalidation is contained by value semantics).
            scene_.items = objects_.to_render_items();
            report.items_projected = (uint32_t)scene_.items.size();

            // 6) Backend output: software raster of the projection through
            //    the EXTERNAL registry (per-call handle resolution).
            scene_.resources = &registry_;
            const SceneResourceView view{&registry_};
            const bool quit = session_.quit_requested;

            target_->clear(ColorF{0.0f, 0.0f, 0.0f, 0.0f});
            depth_->clear_all();

            ShaderUniforms uniforms{};
            uniforms.viewproj = scene_.cam.viewproj;
            uniforms.model = glm::mat4(1.0f);
            uniforms.light_dir_ws = glm::normalize(-scene_.sun.dir_ws);
            uniforms.light_color = scene_.sun.color;
            uniforms.light_intensity = scene_.sun.intensity;
            uniforms.camera_pos = scene_.cam.pos;
            uniforms.enable_motion_vectors = false;

            RasterizerConfig raster_config{};
            raster_config.cull_mode = RasterizerCullMode::None;
            raster_config.front_face_ccw = true;

            for (const RenderItem& item : scene_.items)
            {
                if (!item.visible) continue;
                const MeshData* mesh = view.mesh(item);
                if (!mesh || mesh->positions.empty()) continue; // stale/absent: skip
                const MaterialData* material = view.material(item);
                uniforms.base_color = material ? material->base_color : glm::vec3(1.0f);
                uniforms.model = item_transform(item.tr);
                rasterize_mesh(*mesh, host_program(), uniforms,
                    RasterizerTarget{target_.get(), depth_.get()}, raster_config);
                report.items_drawn += 1;
            }

            report.pixel_digest = digest_render_target(*target_);
            report.quit_requested = quit;
            return report;
        }

        // Resize: reallocates the backend output targets; the next run_frame
        // re-derives the camera aspect from the new size. Deterministic.
        void resize(int width, int height)
        {
            config_.width = width;
            config_.height = height;
            rebuild_targets();
        }

        int width() const { return config_.width; }
        int height() const { return config_.height; }

        // --- state accessors (the projections live HERE; game rules mutate
        // the session only through recorded commands) ------------------------
        SessionState& session() { return session_; }
        const SessionState& session() const { return session_; }
        Scene& scene() { return scene_; }
        const Scene& scene() const { return scene_; }
        SceneObjectSet& objects() { return objects_; }
        const SceneObjectSet& objects() const { return objects_; }
        renderpath::RenderPathPodState& path_state() { return path_; }
        const renderpath::RenderPathPodState& path_state() const { return path_; }
        const FrameParams& frame_params() const { return frame_; }
        const RT_ColorHDR& target() const { return *target_; }
        bool quit_requested() const { return session_.quit_requested; }

        // Explicit plan swap (used by the backend-unavailable rejection pin;
        // returns false when the compiler rejected the candidate).
        bool swap_plan(const RenderPathRecipe& candidate)
        {
            std::pmr::monotonic_buffer_resource arena{4096};
            std::pmr::vector<renderpath::RenderPathEvent> events{&arena};
            const renderpath::RenderPathCommand command =
                renderpath::SelectPathPresetIntent{candidate};
            const renderpath::RenderPathStep step = renderpath::renderpath_gateway(
                path_, std::span<const renderpath::RenderPathCommand>{&command, 1},
                compiler_, config_.caps, events);
            return step.swaps_rejected == 0;
        }

    private:
        // The host's single deterministic program: view-proj * model vertex
        // transform, flat base-color fragment. No time-varying math and no
        // environment sampling — every pixel is replay-reproducible.
        static ShaderProgram make_host_program()
        {
            ShaderProgram program{};
            program.vs = [](const ShaderVertex& v, const ShaderUniforms& u) -> VertexOut {
                VertexOut out{};
                const glm::vec4 world = u.model * glm::vec4(v.position, 1.0f);
                out.clip = u.viewproj * world;
                out.world_pos = glm::vec3(world);
                out.normal_ws = v.normal;
                out.uv = v.uv;
                return out;
            };
            program.fs = [](const FragmentIn&, const ShaderUniforms& u) -> FragmentOut {
                FragmentOut out{};
                out.color = ColorF{u.base_color.r, u.base_color.g, u.base_color.b, 1.0f};
                return out;
            };
            return program;
        }

        static const ShaderProgram& host_program()
        {
            static const ShaderProgram program = make_host_program();
            return program;
        }

        static glm::mat4 item_transform(const Transform& tr)
        {
            glm::mat4 model{1.0f};
            model = glm::translate(model, tr.pos);
            model = glm::rotate(model, tr.rot_euler.x, glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::rotate(model, tr.rot_euler.y, glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::rotate(model, tr.rot_euler.z, glm::vec3(0.0f, 0.0f, 1.0f));
            model = glm::scale(model, tr.scl);
            return model;
        }

        void rebuild_targets()
        {
            const ColorF clear{0.0f, 0.0f, 0.0f, 0.0f};
            target_ = std::make_unique<RT_ColorHDR>(config_.width, config_.height, clear);
            depth_ = std::make_unique<RT_ColorDepthMotion>(
                config_.width, config_.height, 0.1f, 1000.0f);
            frame_.w = config_.width;
            frame_.h = config_.height;
        }

        VerticalSliceConfig config_{};
        ResourceRegistry& registry_;
        RenderPathCompiler compiler_{};

        SessionState session_{};
        Scene scene_{};
        SceneObjectSet objects_{};
        renderpath::RenderPathPodState path_{};
        FrameParams frame_{};

        std::unique_ptr<RT_ColorHDR> target_{};
        std::unique_ptr<RT_ColorDepthMotion> depth_{};
    };
} // namespace shs::app

