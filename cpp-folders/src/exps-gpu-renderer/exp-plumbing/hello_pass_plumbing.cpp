#define SDL_MAIN_HANDLED

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <glm/glm.hpp>

#include <shs/app/camera_sync.hpp>
#include <shs/app/runtime_state.hpp>
#include <shs/camera/camera_rig.hpp>
#include <shs/core/context.hpp>
#include <shs/frame/frame_params.hpp>
#include <shs/gfx/rt_registry.hpp>
#include <shs/gfx/rt_shadow.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/input/value_actions.hpp>
#include <shs/job/thread_pool_job_system.hpp>
#include <shs/pipeline/pass_adapters.hpp>
#include <shs/pipeline/pluggable_pipeline.hpp>
#include <shs/platform/sdl/sdl_runtime.hpp>
#include <shs/resources/loaders/primitive_import.hpp>
#include <shs/resources/resource_registry.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/scene/scene_objects.hpp>
#include <shs/shader/types.hpp>

namespace
{
    constexpr int WINDOW_W = 1024;
    constexpr int WINDOW_H = 640;
    constexpr int CANVAS_W = 960;
    constexpr int CANVAS_H = 540;

    struct CaptureConfig
    {
        bool enabled = false;
        std::string path{};
        int after_frames = 8;
    };

    struct CameraPreset
    {
        glm::vec3 pos{0.0f};
        float yaw = 0.0f;
        float pitch = 0.0f;
    };

    constexpr CameraPreset kCameraPresets[3] = {
        {glm::vec3(0.0f, 4.8f, -9.8f), 1.40f, -0.24f},
        {glm::vec3(8.8f, 4.0f, -1.2f), 2.88f, -0.15f},
        {glm::vec3(-7.5f, 5.5f, 5.8f), -0.57f, -0.30f}
    };
    constexpr float LOOK_SENSITIVITY = 0.0025f;
    constexpr float MOVE_SPEED = 6.0f;
    constexpr float MOVE_SPEED_BOOST = 12.0f;
    constexpr float MOUSE_SPIKE_THRESHOLD = 240.0f;
    constexpr float MOUSE_DELTA_CLAMP = 90.0f;

    float probe_shader_varyings()
    {
        shs::VertexOut vs_out{};
        shs::set_varying(vs_out, shs::VaryingSemantic::WorldPos, glm::vec4(1.0f, 2.0f, 3.0f, 1.0f));
        shs::set_varying(vs_out, shs::VaryingSemantic::UV0, glm::vec4(0.2f, 0.8f, 0.0f, 0.0f));

        shs::FragmentIn fs_in{};
        fs_in.varyings = vs_out.varyings;
        fs_in.varying_mask = vs_out.varying_mask;

        const glm::vec4 wp = shs::get_varying(fs_in, shs::VaryingSemantic::WorldPos);
        const glm::vec4 uv = shs::get_varying(fs_in, shs::VaryingSemantic::UV0);
        return wp.x + wp.y + wp.z + uv.x + uv.y;
    }

    void upload_ldr_to_rgba8(std::vector<uint8_t>& rgba, const shs::RT_ColorLDR& ldr)
    {
        rgba.resize((size_t)ldr.w * (size_t)ldr.h * 4);
        for (int y_screen = 0; y_screen < ldr.h; ++y_screen)
        {
            const int y_canvas = ldr.h - 1 - y_screen;
            uint8_t* row = rgba.data() + (size_t)y_screen * (size_t)ldr.w * 4;
            for (int x = 0; x < ldr.w; ++x)
            {
                const shs::Color c = ldr.color.at(x, y_canvas);
                const int i = x * 4;
                row[i + 0] = c.r;
                row[i + 1] = c.g;
                row[i + 2] = c.b;
                row[i + 3] = 255;
            }
        }
    }

    bool write_ldr_to_ppm(const std::string& path, const shs::RT_ColorLDR& ldr)
    {
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        out << "P6\n" << ldr.w << " " << ldr.h << "\n255\n";
        for (int y_screen = 0; y_screen < ldr.h; ++y_screen)
        {
            const int y_canvas = ldr.h - 1 - y_screen;
            for (int x = 0; x < ldr.w; ++x)
            {
                const shs::Color c = ldr.color.at(x, y_canvas);
                const char rgb[3] = {
                    (char)c.r,
                    (char)c.g,
                    (char)c.b
                };
                out.write(rgb, 3);
            }
        }
        return out.good();
    }

    int clamp_preset_index(int idx)
    {
        if (idx < 0) return 0;
        if (idx > 2) return 2;
        return idx;
    }
}

int main(int argc, char* argv[])
{
    CaptureConfig capture{};
    int preset_index = 0;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "--capture" && i + 1 < argc)
        {
            capture.path = argv[++i];
            capture.enabled = !capture.path.empty();
        }
        else if (arg == "--capture-after" && i + 1 < argc)
        {
            capture.after_frames = std::max(1, std::atoi(argv[++i]));
        }
        else if (arg == "--preset" && i + 1 < argc)
        {
            preset_index = clamp_preset_index(std::atoi(argv[++i]));
        }
    }
    const CameraPreset preset = kCameraPresets[preset_index];

    shs::SdlRuntime runtime{
        shs::WindowDesc{"HelloPassPlumbing", WINDOW_W, WINDOW_H},
        shs::SurfaceDesc{CANVAS_W, CANVAS_H}
    };
    if (!runtime.valid()) return 1;

    shs::Context ctx{};
    const char* backend_env = std::getenv("SHS_RENDER_BACKEND");
    auto backend_result = shs::create_render_backend(backend_env ? backend_env : "software");
    std::vector<std::unique_ptr<shs::IRenderBackend>> backend_keepalive{};
    backend_keepalive.reserve(1 + backend_result.auxiliary_backends.size());
    backend_keepalive.push_back(std::move(backend_result.backend));
    for (auto& b : backend_result.auxiliary_backends)
    {
        backend_keepalive.push_back(std::move(b));
    }
    if (backend_keepalive.empty() || !backend_keepalive[0]) return 1;

    ctx.set_primary_backend(backend_keepalive[0].get());
    for (size_t i = 1; i < backend_keepalive.size(); ++i)
    {
        if (backend_keepalive[i]) ctx.register_backend(backend_keepalive[i].get());
    }
    if (!backend_result.note.empty())
    {
        std::fprintf(stderr, "[shs] %s\n", backend_result.note.c_str());
    }

    shs::ThreadPoolJobSystem jobs{std::max(1u, std::thread::hardware_concurrency())};
    ctx.job_system = &jobs;

    shs::ResourceRegistry resources{};
    shs::Scene scene{};
    scene.resources = &resources;
    scene.sun.dir_ws = glm::normalize(glm::vec3(-0.35f, -1.0f, -0.25f));
    scene.sun.color = glm::vec3(1.0f, 0.97f, 0.92f);
    scene.sun.intensity = 2.2f;

    const shs::MeshAssetHandle floor_mesh = shs::import_plane_primitive(
        resources, shs::PlaneDesc{28.0f, 28.0f, 24, 24}, "floor_mesh");
    const shs::MeshAssetHandle orb_mesh = shs::import_sphere_primitive(
        resources, shs::SphereDesc{1.0f, 40, 24}, "orb_mesh");

    const shs::MaterialAssetHandle floor_mat = resources.add_material(
        shs::MaterialData{"mat_floor", glm::vec3(0.42f, 0.45f, 0.50f), 0.0f, 0.95f, 1.0f}, "floor_mat");
    const shs::MaterialAssetHandle orb_mat = resources.add_material(
        shs::MaterialData{"mat_orb", glm::vec3(0.95f, 0.74f, 0.22f), 0.90f, 0.18f, 1.0f}, "orb_mat");

    shs::SceneObjectSet objects{};
    objects.add(shs::SceneObject{
        "floor",
        (shs::MeshHandle)floor_mesh,
        (shs::MaterialHandle)floor_mat,
        shs::Transform{glm::vec3(0.0f, -1.1f, 0.0f), glm::vec3(0.0f), glm::vec3(1.0f)},
        true,
        true
    });
    objects.add(shs::SceneObject{
        "orb",
        (shs::MeshHandle)orb_mesh,
        (shs::MaterialHandle)orb_mat,
        shs::Transform{glm::vec3(0.0f, 1.2f, 0.0f), glm::vec3(0.0f), glm::vec3(1.0f)},
        true,
        true
    });

    shs::RTRegistry rtr{};
    shs::RT_ShadowDepth shadow_rt{1024, 1024};
    shs::RT_ColorHDR hdr_rt{CANVAS_W, CANVAS_H};
    shs::RT_ColorDepthVelocity motion_rt{CANVAS_W, CANVAS_H, 0.1f, 500.0f};
    shs::RT_ColorLDR ldr_rt{CANVAS_W, CANVAS_H};
    shs::RT_ColorLDR shafts_tmp_rt{CANVAS_W, CANVAS_H};

    const shs::RT_Shadow rt_shadow_h = rtr.reg<shs::RT_Shadow>(&shadow_rt);
    const shs::RTHandle rt_hdr_h = rtr.reg<shs::RTHandle>(&hdr_rt);
    const shs::RT_Motion rt_motion_h = rtr.reg<shs::RT_Motion>(&motion_rt);
    const shs::RTHandle rt_ldr_h = rtr.reg<shs::RTHandle>(&ldr_rt);
    const shs::RTHandle rt_shafts_tmp_h = rtr.reg<shs::RTHandle>(&shafts_tmp_rt);

    const shs::PassFactoryRegistry pass_registry = shs::make_standard_pass_factory_registry(
        rt_shadow_h,
        rt_hdr_h,
        rt_motion_h,
        rt_ldr_h,
        rt_shafts_tmp_h,
        shs::RTHandle{}
    );
    shs::PluggablePipeline pipeline{};
    (void)pipeline.add_pass_from_registry(pass_registry, shs::PassId::ShadowMap);
    (void)pipeline.add_pass_from_registry(pass_registry, shs::PassId::PBRForward);
    (void)pipeline.add_pass_from_registry(pass_registry, shs::PassId::Tonemap);
    (void)pipeline.add_pass_from_registry(pass_registry, "light_shafts");
    pipeline.set_strict_graph_validation(true);

    shs::FrameParams fp{};
    fp.w = CANVAS_W;
    fp.h = CANVAS_H;
    fp.debug_view = shs::DebugViewMode::Final;
    fp.cull_mode = shs::CullMode::Back;
    fp.shading_model = shs::ShadingModel::PBRMetalRough;
    fp.pass.tonemap.exposure = 1.0f;
    fp.pass.tonemap.gamma = 2.2f;
    fp.pass.shadow.enable = true;
    fp.pass.shadow.pcf_radius = 2;
    fp.pass.shadow.pcf_step = 1.0f;
    fp.pass.shadow.strength = 0.82f;
    fp.pass.light_shafts.enable = true;
    fp.pass.light_shafts.steps = 20;
    fp.pass.light_shafts.density = 0.85f;
    fp.pass.light_shafts.weight = 0.25f;
    fp.pass.light_shafts.decay = 0.95f;
    fp.pass.motion_vectors.enable = true;
    fp.technique.mode = shs::TechniqueMode::Forward;
    fp.technique.active_modes_mask = shs::technique_mode_mask_all();
    fp.technique.depth_prepass = false;
    fp.technique.light_culling = false;

    shs::RuntimeState runtime_state{};
    runtime_state.camera.pos = preset.pos;
    runtime_state.camera.yaw = preset.yaw;
    runtime_state.camera.pitch = preset.pitch;
    runtime_state.enable_light_shafts = fp.pass.light_shafts.enable;
    runtime_state.bot_enabled = !capture.enabled;

    const float varying_probe_checksum = probe_shader_varyings();
    std::vector<shs::RuntimeAction> runtime_actions{};
    bool mouse_look_active = false;

    bool running = true;
    float time_s = 0.0f;
    float orbit = 0.0f;
    auto prev = std::chrono::steady_clock::now();
    int frame_count = 0;
    int frames = 0;
    float fps_accum = 0.0f;
    std::vector<uint8_t> rgba_staging{};

    while (running)
    {
        const auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - prev).count();
        prev = now;
        dt = std::clamp(dt, 0.0f, 0.1f);
        time_s += dt;
        fp.dt = dt;
        fp.time = time_s;

        shs::PlatformInputState pin{};
        if (!runtime.pump_input(pin)) break;

        const bool look_drag = pin.right_mouse_down || pin.left_mouse_down;
        if (look_drag != mouse_look_active)
        {
            mouse_look_active = look_drag;
            runtime.set_relative_mouse_mode(mouse_look_active);
            pin.mouse_dx = 0.0f;
            pin.mouse_dy = 0.0f;
        }

        runtime_actions.clear();
        shs::InputState input_actions{};
        input_actions.forward = pin.forward;
        input_actions.backward = pin.backward;
        input_actions.left = pin.left;
        input_actions.right = pin.right;
        input_actions.ascend = pin.ascend;
        input_actions.descend = pin.descend;
        input_actions.boost = pin.boost;
        input_actions.look_active = mouse_look_active;
        float mdx = pin.mouse_dx;
        float mdy = pin.mouse_dy;
        if (std::abs(mdx) > MOUSE_SPIKE_THRESHOLD || std::abs(mdy) > MOUSE_SPIKE_THRESHOLD)
        {
            mdx = 0.0f;
            mdy = 0.0f;
        }
        input_actions.look_dx = -std::clamp(mdx, -MOUSE_DELTA_CLAMP, MOUSE_DELTA_CLAMP);
        input_actions.look_dy = std::clamp(mdy, -MOUSE_DELTA_CLAMP, MOUSE_DELTA_CLAMP);
        input_actions.toggle_light_shafts = pin.toggle_light_shafts;
        input_actions.toggle_bot = pin.toggle_bot;
        input_actions.quit = pin.quit;
        shs::emit_human_actions(
            input_actions,
            runtime_actions,
            MOVE_SPEED,
            MOVE_SPEED_BOOST / MOVE_SPEED,
            LOOK_SENSITIVITY);
        runtime_state = shs::reduce_runtime_state(runtime_state, runtime_actions, dt);
        if (runtime_state.quit_requested) running = false;

        if (pin.cycle_debug_view)
        {
            const int next = (((int)fp.debug_view) + 1) % 4;
            fp.debug_view = (shs::DebugViewMode)next;
        }
        if (pin.cycle_cull_mode)
        {
            const int next = (((int)fp.cull_mode) + 1) % 3;
            fp.cull_mode = (shs::CullMode)next;
        }
        if (pin.toggle_front_face)
        {
            fp.front_face_ccw = !fp.front_face_ccw;
        }
        if (pin.toggle_shading_model)
        {
            fp.shading_model = (fp.shading_model == shs::ShadingModel::PBRMetalRough)
                ? shs::ShadingModel::BlinnPhong
                : shs::ShadingModel::PBRMetalRough;
        }
        fp.pass.light_shafts.enable = runtime_state.enable_light_shafts;

        if (auto* orb = objects.find("orb"))
        {
            orb->tr.pos.y = 1.25f + std::sin(time_s * 1.8f) * 0.30f;
            orb->tr.rot_euler.y += dt * 0.8f;
            orb->tr.rot_euler.x = 0.12f + std::sin(time_s * 0.9f) * 0.10f;
        }
        scene.sun.dir_ws = glm::normalize(glm::vec3(
            -0.30f + std::cos(time_s * 0.20f) * 0.10f,
            -1.0f,
            -0.24f + std::sin(time_s * 0.20f) * 0.10f
        ));

        if (capture.enabled)
        {
            runtime_state.camera.pos = preset.pos;
            runtime_state.camera.yaw = preset.yaw;
            runtime_state.camera.pitch = preset.pitch;
        }
        else if (runtime_state.bot_enabled)
        {
            orbit += dt * 0.22f;
            const float cam_radius = 9.8f;
            const glm::vec3 focus{0.0f, 1.0f, 0.0f};
            const glm::vec3 cam_pos{
                std::cos(orbit) * cam_radius,
                4.4f,
                std::sin(orbit) * cam_radius
            };
            const glm::vec3 to_focus = glm::normalize(focus - cam_pos);
            runtime_state.camera.pos = cam_pos;
            runtime_state.camera.yaw = std::atan2(to_focus.z, to_focus.x);
            runtime_state.camera.pitch = std::asin(glm::clamp(to_focus.y, -1.0f, 1.0f));
        }

        scene.items = objects.to_render_items();
        shs::sync_camera_to_scene(runtime_state.camera, scene, (float)CANVAS_W / (float)CANVAS_H);

        pipeline.execute(ctx, scene, fp, rtr);

        upload_ldr_to_rgba8(rgba_staging, ldr_rt);
        runtime.upload_rgba8(rgba_staging.data(), ldr_rt.w, ldr_rt.h, ldr_rt.w * 4);
        runtime.present();

        frame_count++;
        if (capture.enabled && frame_count >= capture.after_frames)
        {
            if (!write_ldr_to_ppm(capture.path, ldr_rt))
            {
                std::fprintf(stderr, "[shs] failed to write capture: %s\n", capture.path.c_str());
                return 2;
            }
            running = false;
        }

        frames++;
        fps_accum += dt;
        if (fps_accum >= 0.25f)
        {
            const float fps = (fps_accum > 1e-6f) ? ((float)frames / fps_accum) : 0.0f;
            const std::string title =
                std::string("HelloPassPlumbing | ")
                + "fps=" + std::to_string((int)std::lround(fps))
                + " | backend=" + ctx.active_backend_name()
                + " | shafts=" + (fp.pass.light_shafts.enable ? "on" : "off")
                + " | bot=" + (runtime_state.bot_enabled ? "on" : "off")
                + " | vary=" + std::to_string((int)std::lround(varying_probe_checksum));
            runtime.set_title(title);
            frames = 0;
            fps_accum = 0.0f;
        }
    }

    runtime.set_relative_mouse_mode(false);
    return 0;
}
