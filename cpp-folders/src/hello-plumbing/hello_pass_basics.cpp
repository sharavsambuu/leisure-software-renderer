#define SDL_MAIN_HANDLED

#include <chrono>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/app/camera_sync.hpp>
#include <shs/camera/follow_camera.hpp>
#include <shs/core/context.hpp>
#include <shs/frame/frame_params.hpp>
#include <shs/gfx/rt_registry.hpp>
#include <shs/gfx/rt_shadow.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/job/thread_pool_job_system.hpp>
#include <shs/pipeline/pass_adapters.hpp>
#include <shs/pipeline/pluggable_pipeline.hpp>
#include <shs/platform/sdl/sdl_runtime.hpp>
#include <shs/resources/loaders/primitive_import.hpp>
#include <shs/resources/loaders/resource_import.hpp>
#include <shs/resources/resource_registry.hpp>
#include <shs/scene/scene_bindings.hpp>
#include <shs/scene/scene_objects.hpp>
#include <shs/scene/system_processors.hpp>
#include <shs/scene/scene_types.hpp>
#include <shs/sky/cubemap_sky.hpp>
#include <shs/sky/loaders/cubemap_loader_sdl.hpp>
#include <shs/sky/procedural_sky.hpp>

/*
    HelloPassBasics demo
    - Pass pipeline: shadow -> PBR/Blinn forward -> tonemap
    - Scene: floor + subaru + monkey
    - Runtime toggle: debug/shading/sky/follow camera
*/

namespace
{
    constexpr int WINDOW_W = 800;
    constexpr int WINDOW_H = 620;
    constexpr int CANVAS_W = 640;
    constexpr int CANVAS_H = 360;
    constexpr float MOUSE_LOOK_SENS = 0.0025f;
    constexpr float FREE_CAM_BASE_SPEED = 8.0f;
    constexpr float CHASE_ORBIT_SENS = 0.0025f;

    // LDR render target-ийг SDL texture-д upload хийх RGBA8 буфер рүү хөрвүүлнэ.
    // Canvas координатын Y дээшээ тул дэлгэц рүү гаргахдаа босоогоор нь эргүүлж авна.
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

    float lerp_angle_rad(float a, float b, float t)
    {
        float d = b - a;
        while (d > 3.14159265f) d -= 6.2831853f;
        while (d < -3.14159265f) d += 6.2831853f;
        return a + d * t;
    }

    // Машиныг plane дээр санамсаргүй цэгүүд рүү зөөлөн эргэж зорчуулах логик систем.
    class SubaruCruiseSystem final : public shs::ILogicSystem
    {
    public:
        SubaruCruiseSystem(
            std::string object_name,
            float area_half_extent,
            float y_level,
            float cruise_speed = 6.5f,
            float max_turn_rate_rad = 1.9f,
            float visual_yaw_offset_rad = 3.14159265f
        )
            : object_name_(std::move(object_name))
            , area_half_extent_(area_half_extent)
            , y_level_(y_level)
            , cruise_speed_(cruise_speed)
            , max_turn_rate_rad_(max_turn_rate_rad)
            , visual_yaw_offset_rad_(visual_yaw_offset_rad)
            , rng_(0xC0FFEEu)
            , dist_(-area_half_extent_ * 0.92f, area_half_extent_ * 0.92f)
            , speed_jitter_(0.78f, 1.22f)
        {}

        void tick(shs::LogicSystemContext& ctx) override
        {
            if (!ctx.objects) return;
            auto* obj = ctx.objects->find(object_name_);
            if (!obj) return;
            const float dt = std::max(0.0f, ctx.dt);
            if (dt <= 1e-6f) return;

            if (!initialized_)
            {
                // Эхний tick дээр эхлэх төлөвөө бэлдэнэ.
                obj->tr.pos.y = y_level_;
                current_yaw_ = obj->tr.rot_euler.y;
                pick_new_target(obj->tr.pos);
                current_speed_ = cruise_speed_ * speed_jitter_(rng_);
                initialized_ = true;
            }

            obj->tr.pos.y = y_level_;
            if (glm::length(glm::vec2(obj->tr.pos.x - target_.x, obj->tr.pos.z - target_.z)) < 2.8f)
            {
                // Ойртсон үед дараагийн зорилтот цэг сонгоно.
                pick_new_target(obj->tr.pos);
                current_speed_ = cruise_speed_ * speed_jitter_(rng_);
            }

            const glm::vec3 to_target_ws = target_ - obj->tr.pos;
            const glm::vec2 to_target_xz{to_target_ws.x, to_target_ws.z};
            const float d = glm::length(to_target_xz);
            if (d > 1e-5f)
            {
                // Одоогийн yaw-г зорилтот чиглэл рүү max_turn_rate-аар хязгаарлан эргүүлнэ.
                const glm::vec2 dir = to_target_xz / d;
                const float target_yaw = std::atan2(dir.y, dir.x);
                float dy = target_yaw - current_yaw_;
                while (dy > 3.14159265f) dy -= 6.2831853f;
                while (dy < -3.14159265f) dy += 6.2831853f;
                const float max_step = max_turn_rate_rad_ * dt;
                dy = std::clamp(dy, -max_step, max_step);
                current_yaw_ += dy;
            }

            const glm::vec3 fwd{std::cos(current_yaw_), 0.0f, std::sin(current_yaw_)};
            float edge = std::max(std::abs(obj->tr.pos.x), std::abs(obj->tr.pos.z));
            // Ирмэг рүү дөхөх тусам хурдыг бага зэрэг бууруулж хөдөлгөөнийг тогтвортой болгоно.
            const float edge_ratio = std::clamp((edge - area_half_extent_ * 0.70f) / (area_half_extent_ * 0.30f), 0.0f, 1.0f);
            const float speed_scale = 1.0f - edge_ratio * 0.35f;
            obj->tr.pos += fwd * (current_speed_ * speed_scale * dt);
            obj->tr.pos.x = std::clamp(obj->tr.pos.x, -area_half_extent_, area_half_extent_);
            obj->tr.pos.z = std::clamp(obj->tr.pos.z, -area_half_extent_, area_half_extent_);
            obj->tr.pos.y = y_level_;
            obj->tr.rot_euler.y = current_yaw_ + visual_yaw_offset_rad_;
        }

    private:
        void pick_new_target(const glm::vec3& current_pos)
        {
            for (int i = 0; i < 32; ++i)
            {
                glm::vec3 c{dist_(rng_), y_level_, dist_(rng_)};
                if (glm::length(glm::vec2(c.x - current_pos.x, c.z - current_pos.z)) > area_half_extent_ * 0.35f)
                {
                    target_ = c;
                    return;
                }
            }
            target_ = glm::vec3(dist_(rng_), y_level_, dist_(rng_));
        }

        std::string object_name_{};
        float area_half_extent_ = 16.0f;
        float y_level_ = 0.0f;
        float cruise_speed_ = 6.5f;
        float max_turn_rate_rad_ = 1.9f;
        float visual_yaw_offset_rad_ = 3.14159265f;
        float current_speed_ = 6.5f;
        float current_yaw_ = 0.0f;
        bool initialized_ = false;
        glm::vec3 target_{0.0f};
        std::mt19937 rng_{};
        std::uniform_real_distribution<float> dist_{-12.0f, 12.0f};
        std::uniform_real_distribution<float> speed_jitter_{0.78f, 1.22f};
    };

    // Follow mode асаалттай үед камерыг машины араас зөөлөн дагуулах логик систем.
    class FollowCameraSystem final : public shs::ILogicSystem
    {
    public:
        FollowCameraSystem(
            shs::CameraRig* rig,
            bool* enabled,
            std::string target_name,
            float follow_distance,
            float follow_height,
            float look_ahead,
            float smoothing
        )
            : rig_(rig)
            , enabled_(enabled)
            , target_name_(std::move(target_name))
            , follow_distance_(follow_distance)
            , follow_height_(follow_height)
            , look_ahead_(look_ahead)
            , smoothing_(smoothing)
        {}

        void tick(shs::LogicSystemContext& ctx) override
        {
            if (!rig_ || !enabled_ || !(*enabled_) || !ctx.objects) return;
            const auto* target = ctx.objects->find(target_name_);
            if (!target) return;
            const glm::vec3 fwd{std::cos(target->tr.rot_euler.y), 0.0f, std::sin(target->tr.rot_euler.y)};
            // Камерын хүссэн байрлалыг объектын ар ба дээд талд тооцоолоод smooth байдлаар дөхүүлнэ.
            const glm::vec3 desired_cam = target->tr.pos - fwd * follow_distance_ + glm::vec3(0.0f, follow_height_, 0.0f);
            follow_target(*rig_, desired_cam, glm::vec3(0.0f), smoothing_, ctx.dt);

            const glm::vec3 look_point = target->tr.pos + fwd * look_ahead_ + glm::vec3(0.0f, 0.8f, 0.0f);
            const glm::vec3 v = look_point - rig_->pos;
            const float len = glm::length(v);
            if (len > 1e-6f)
            {
                const glm::vec3 d = v / len;
                const float target_yaw = std::atan2(d.z, d.x);
                const float target_pitch = std::asin(glm::clamp(d.y, -1.0f, 1.0f));
                const float t = std::clamp(smoothing_ * ctx.dt * 8.0f, 0.0f, 1.0f);

                float dy = target_yaw - rig_->yaw;
                while (dy > 3.14159265f) dy -= 6.2831853f;
                while (dy < -3.14159265f) dy += 6.2831853f;
                rig_->yaw += dy * t;
                rig_->pitch = glm::mix(rig_->pitch, target_pitch, t);
            }
        }

    private:
        shs::CameraRig* rig_ = nullptr;
        bool* enabled_ = nullptr;
        std::string target_name_{};
        float follow_distance_ = 8.0f;
        float follow_height_ = 3.0f;
        float look_ahead_ = 3.0f;
        float smoothing_ = 0.18f;
    };

    // Monkey объектод эргэлт + босоо чиглэлийн жижиг савлалт өгнө.
    class MonkeyWiggleSystem final : public shs::ILogicSystem
    {
    public:
        MonkeyWiggleSystem(std::string object_name, float spin_rps, float bob_amp, float bob_hz)
            : object_name_(std::move(object_name))
            , spin_rps_(spin_rps)
            , bob_amp_(bob_amp)
            , bob_hz_(bob_hz)
        {}

        void tick(shs::LogicSystemContext& ctx) override
        {
            if (!ctx.objects) return;
            auto* obj = ctx.objects->find(object_name_);
            if (!obj) return;
            if (!base_captured_)
            {
                base_pos_ = obj->tr.pos;
                base_captured_ = true;
            }

            time_ += std::max(0.0f, ctx.dt);
            obj->tr.rot_euler.y += (2.0f * 3.14159265f) * spin_rps_ * std::max(0.0f, ctx.dt);
            obj->tr.pos = base_pos_;
            obj->tr.pos.y += std::sin(time_ * (2.0f * 3.14159265f) * bob_hz_) * bob_amp_;
        }

    private:
        std::string object_name_{};
        float spin_rps_ = 0.25f;
        float bob_amp_ = 0.2f;
        float bob_hz_ = 1.7f;
        bool base_captured_ = false;
        float time_ = 0.0f;
        glm::vec3 base_pos_{0.0f};
    };
}

int main()
{
    // SDL runtime: window + software canvas.
    shs::SdlRuntime runtime{
        shs::WindowDesc{"HelloPassBasics", WINDOW_W, WINDOW_H},
        shs::SurfaceDesc{CANVAS_W, CANVAS_H}
    };
    if (!runtime.valid()) return 1;

    shs::Context ctx{};
    // Рендерийн parallel хэсгүүдэд ашиглагдах thread pool.
    shs::ThreadPoolJobSystem jobs{std::max(1u, std::thread::hardware_concurrency())};
    ctx.job_system = &jobs;

    shs::ResourceRegistry resources{};
    shs::RTRegistry rtr{};
    shs::PluggablePipeline pipeline{};
    shs::LogicSystemProcessor logic_systems{};
    shs::RenderSystemProcessor render_systems{};

    shs::RT_ShadowDepth shadow_rt{768, 768};
    shs::RT_ColorHDR hdr_rt{CANVAS_W, CANVAS_H};
    shs::RT_ColorDepthMotion motion_rt{CANVAS_W, CANVAS_H, 0.1f, 1000.0f};
    shs::RT_ColorLDR ldr_rt{CANVAS_W, CANVAS_H};
    shs::RT_ColorLDR shafts_tmp_rt{CANVAS_W, CANVAS_H};

    const shs::RT_Shadow rt_shadow_h = rtr.reg<shs::RT_Shadow>(&shadow_rt);
    const shs::RTHandle rt_hdr_h = rtr.reg<shs::RTHandle>(&hdr_rt);
    const shs::RT_Motion rt_motion_h = rtr.reg<shs::RT_Motion>(&motion_rt);
    const shs::RTHandle rt_ldr_h = rtr.reg<shs::RTHandle>(&ldr_rt);
    const shs::RTHandle rt_shafts_tmp_h = rtr.reg<shs::RTHandle>(&shafts_tmp_rt);

    // Render pass дараалал: shadow map -> forward shading -> tonemap.
    pipeline.add_pass<shs::PassShadowMapAdapter>(rt_shadow_h);
    pipeline.add_pass<shs::PassPBRForwardAdapter>(rt_hdr_h, rt_motion_h, rt_shadow_h);
    pipeline.add_pass<shs::PassTonemapAdapter>(rt_hdr_h, rt_ldr_h);
    pipeline.add_pass<shs::PassLightShaftsAdapter>(rt_ldr_h, rt_motion_h, rt_shafts_tmp_h);
    render_systems.add_system<shs::PipelineRenderSystem>(&pipeline);

    shs::Scene scene{};
    scene.resources = &resources;
    scene.sun.dir_ws = glm::normalize(glm::vec3(-0.35f, -1.0f, -0.25f));
    scene.sun.color = glm::vec3(1.0f, 0.97f, 0.92f);
    scene.sun.intensity = 2.2f;
    // Cubemap default; хэрэв cubemap уншигдахгүй бол procedural sky fallback.
    shs::ProceduralSky procedural_sky{scene.sun.dir_ws};
    const shs::CubemapData sky_cm = shs::load_cubemap_sdl_folder("./assets/images/skybox/water_scene", true);
    shs::CubemapSky cubemap_sky{sky_cm, 1.0f};
    bool use_cubemap_sky = sky_cm.valid();
    scene.sky = use_cubemap_sky ? static_cast<const shs::ISkyModel*>(&cubemap_sky)
                                : static_cast<const shs::ISkyModel*>(&procedural_sky);

    const float plane_extent = 64.0f;
    const shs::MeshAssetHandle plane_h = shs::import_plane_primitive(resources, shs::PlaneDesc{plane_extent, plane_extent, 64, 64}, "plane");
    shs::MeshAssetHandle subaru_h = shs::import_mesh_assimp(resources, "./assets/obj/subaru/SUBARU_1.rawobj", "subaru_mesh");
    const bool subaru_loaded = (subaru_h != 0);
    if (!subaru_loaded) subaru_h = shs::import_box_primitive(resources, shs::BoxDesc{glm::vec3(2.4f, 1.1f, 4.8f), 2, 1, 2}, "subaru_fallback");
    const shs::TextureAssetHandle subaru_albedo_h = shs::import_texture_sdl(resources, "./assets/obj/subaru/SUBARU1_M.bmp", "subaru_albedo", true);
    shs::MeshAssetHandle monkey_h = shs::import_mesh_assimp(resources, "./assets/obj/monkey/monkey.rawobj", "monkey_mesh");
    if (monkey_h == 0) monkey_h = shs::import_sphere_primitive(resources, shs::SphereDesc{1.0f, 28, 18}, "monkey_fallback");
    const glm::vec3 car_scale = subaru_loaded ? glm::vec3(0.020f) : glm::vec3(1.0f);

    // Scene материалууд: plastic floor, textured subaru, gold monkey.
    const shs::MaterialAssetHandle floor_mat_h = resources.add_material(
        shs::MaterialData{"mat_floor_plastic", glm::vec3(0.42f, 0.44f, 0.48f), 0.0f, 0.96f, 1.0f},
        "mat_floor"
    );
    const shs::MaterialAssetHandle subaru_mat_h = resources.add_material(
        shs::MaterialData{"mat_subaru", glm::vec3(1.0f), 0.28f, 0.44f, 1.0f, glm::vec3(0.0f), 0.0f, subaru_albedo_h, 0, 0, 0},
        "mat_subaru"
    );
    const shs::MaterialAssetHandle monkey_mat_h = resources.add_material(
        shs::MaterialData{"mat_monkey_gold", glm::vec3(240.0f / 255.0f, 195.0f / 255.0f, 75.0f / 255.0f), 0.95f, 0.20f, 1.0f},
        "mat_monkey_gold"
    );

    shs::SceneObjectSet objects{};
    objects.add(shs::SceneObject{
        "floor",
        (shs::MeshHandle)plane_h,
        (shs::MaterialHandle)floor_mat_h,
        shs::Transform{glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f), glm::vec3(1.0f)},
        true,
        false
    });
    objects.add(shs::SceneObject{
        "subaru",
        (shs::MeshHandle)subaru_h,
        (shs::MaterialHandle)subaru_mat_h,
        shs::Transform{glm::vec3(0.0f, -0.95f, 0.0f), glm::vec3(0.0f), car_scale},
        true,
        true
    });
    objects.add(shs::SceneObject{
        "monkey",
        (shs::MeshHandle)monkey_h,
        (shs::MaterialHandle)monkey_mat_h,
        shs::Transform{glm::vec3(0.0f, 1.45f, 0.0f), glm::vec3(0.0f), glm::vec3(1.05f)},
        true,
        true
    });
    objects.sync_to_scene(scene);

    // Frame-level render тохиргоонууд.
    shs::FrameParams fp{};
    fp.w = CANVAS_W;
    fp.h = CANVAS_H;
    fp.exposure = 1.0f;
    fp.gamma = 2.2f;
    fp.enable_light_shafts = true;
    fp.debug_view = shs::DebugViewMode::Final;
    fp.cull_mode = shs::CullMode::None;
    fp.shading_model = shs::ShadingModel::PBRMetalRough;
    fp.enable_shadows = true;
    fp.shadow_pcf_radius = 1;
    fp.shadow_pcf_step = 1.0f;
    fp.shadow_strength = 0.80f;
    fp.shafts_steps = 28;
    fp.shafts_density = 0.85f;
    fp.shafts_weight = 0.30f;
    fp.shafts_decay = 0.95f;

    shs::CameraRig cam{};
    cam.pos = glm::vec3(0.0f, 6.0f, -16.0f);
    cam.yaw = glm::radians(90.0f);
    cam.pitch = glm::radians(-16.0f);
    // Follow mode default асаалттай.
    bool follow_camera = true;
    // Free болон chase камерыг тусад нь хадгалж, эцсийн камераа blend хийж гаргана.
    shs::CameraRig free_cam = cam;
    shs::CameraRig chase_cam = cam;
    float follow_blend = follow_camera ? 1.0f : 0.0f;
    bool drag_look = false;
    bool left_mouse_held = false;
    bool right_mouse_held = false;
    const float chase_dist = 9.5f;
    const float chase_height = 1.0f;
    const float chase_look_ahead = 3.5f;
    const float chase_smoothing = 0.16f;
    const float mode_blend_speed = 6.0f;
    float chase_orbit_yaw = 0.0f;
    float chase_orbit_pitch = glm::radians(20.0f);
    glm::vec3 chase_forward{1.0f, 0.0f, 0.0f};
    glm::vec3 prev_subaru_pos{0.0f};
    bool has_prev_subaru_pos = false;
    logic_systems.add_system<SubaruCruiseSystem>("subaru", plane_extent * 0.48f, -0.95f, 6.8f, 1.9f, 3.14159265f);
    logic_systems.add_system<MonkeyWiggleSystem>("monkey", 0.32f, 0.22f, 1.9f);

    if (const auto* subaru_init = objects.find("subaru"))
    {
        prev_subaru_pos = subaru_init->tr.pos;
        has_prev_subaru_pos = true;
        const float logical_yaw = subaru_init->tr.rot_euler.y - 3.14159265f;
        chase_forward = glm::normalize(glm::vec3(std::cos(logical_yaw), 0.0f, std::sin(logical_yaw)));
    }

    bool running = true;
    auto prev = std::chrono::steady_clock::now();
    float time_s = 0.0f;
    int frames = 0;
    float fps_accum = 0.0f;
    float logic_ms_accum = 0.0f;
    float render_ms_accum = 0.0f;
    std::vector<uint8_t> rgba_staging{};

    // Main loop: input -> logic -> scene/camera sync -> render -> present.
    while (running)
    {
        const auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - prev).count();
        prev = now;
        if (dt > 0.1f) dt = 0.1f;
        time_s += dt;
        fp.dt = dt;
        fp.time = time_s;

        shs::PlatformInputState pin{};
        if (!runtime.pump_input(pin)) break;
        if (pin.quit) running = false;
        // F1: debug view цикл.
        if (pin.cycle_debug_view)
        {
            const int next = (((int)fp.debug_view) + 1) % 4;
            fp.debug_view = (shs::DebugViewMode)next;
        }
        // F4: PBR <-> BlinnPhong солих.
        if (pin.toggle_shading_model)
        {
            fp.shading_model = (fp.shading_model == shs::ShadingModel::PBRMetalRough)
                ? shs::ShadingModel::BlinnPhong
                : shs::ShadingModel::PBRMetalRough;
        }
        // F5: cubemap/procedural sky солих.
        if (pin.toggle_sky_mode)
        {
            if (sky_cm.valid()) use_cubemap_sky = !use_cubemap_sky;
        }
        // F6: camera follow mode toggle.
        if (pin.toggle_follow_camera)
        {
            const bool prev = follow_camera;
            follow_camera = !follow_camera;
            if (prev && !follow_camera)
            {
                // Chase -> Free: одоогийн харагдаж буй камераас free горим эхэлнэ.
                free_cam = cam;
            }
            else if (!prev && follow_camera)
            {
                // Free -> Chase: blend-г таслахгүй байлгахын тулд chase camera-г одоогийн байрлалаас эхлүүлнэ.
                chase_cam = cam;
            }
        }

        // Товчлууруудын hold төлөв.
        if (pin.left_mouse_down) left_mouse_held = true;
        if (pin.left_mouse_up) left_mouse_held = false;
        if (pin.right_mouse_down)
        {
            right_mouse_held = true;
            runtime.set_relative_mouse_mode(true);
        }
        if (pin.right_mouse_up)
        {
            right_mouse_held = false;
            runtime.set_relative_mouse_mode(false);
        }
        drag_look = left_mouse_held || right_mouse_held;

        // Left/Right drag хийхэд 2 горимд хоёуланд нь камер эргэлдэнэ.
        if (drag_look)
        {
            if (follow_camera)
            {
                chase_orbit_yaw -= pin.mouse_dx * CHASE_ORBIT_SENS;
                chase_orbit_pitch = std::clamp(
                    chase_orbit_pitch + pin.mouse_dy * CHASE_ORBIT_SENS,
                    glm::radians(5.0f),
                    glm::radians(70.0f)
                );
            }
            else
            {
                free_cam.yaw += pin.mouse_dx * MOUSE_LOOK_SENS;
                free_cam.pitch = std::clamp(
                    free_cam.pitch - pin.mouse_dy * MOUSE_LOOK_SENS,
                    glm::radians(-85.0f),
                    glm::radians(85.0f)
                );
            }
        }

        // Free camera хөдөлгөөн (WASD + QE).
        if (!follow_camera)
        {
            const float move_speed = FREE_CAM_BASE_SPEED * (pin.boost ? 2.5f : 1.0f) * dt;
            glm::vec3 fwd = free_cam.forward();
            fwd.y = 0.0f;
            const float fwd_len = glm::length(fwd);
            if (fwd_len > 1e-6f) fwd /= fwd_len;
            const glm::vec3 right = free_cam.right();
            if (pin.forward) free_cam.pos += fwd * move_speed;
            if (pin.backward) free_cam.pos -= fwd * move_speed;
            if (pin.right) free_cam.pos += right * move_speed;
            if (pin.left) free_cam.pos -= right * move_speed;
            if (pin.ascend) free_cam.pos.y += move_speed;
            if (pin.descend) free_cam.pos.y -= move_speed;
        }

        // Logic systems ажиллуулна (subaru cruise, follow camera, monkey wiggle).
        const auto t_logic0 = std::chrono::steady_clock::now();
        shs::LogicSystemContext logic_ctx{};
        logic_ctx.dt = dt;
        logic_ctx.time = time_s;
        logic_ctx.objects = &objects;
        logic_ctx.scene = &scene;
        logic_ctx.frame = &fp;
        logic_systems.tick(logic_ctx);
        const auto t_logic1 = std::chrono::steady_clock::now();
        logic_ms_accum += std::chrono::duration<float, std::milli>(t_logic1 - t_logic0).count();

        // Subaru-ийн transform-аас chase camera зорилтот байрлал/чиглэлийг frame бүр шинэчилнэ.
        if (const auto* subaru = objects.find("subaru"))
        {
            // Chase чиглэлийг model yaw бус, бодит хөдөлгөөний вектороос тооцно.
            glm::vec3 move = subaru->tr.pos - prev_subaru_pos;
            move.y = 0.0f;
            const float move_len = glm::length(move);
            if (has_prev_subaru_pos && move_len > 1e-4f)
            {
                const glm::vec3 move_dir = move / move_len;
                const float t_dir = 1.0f - std::exp(-std::max(0.0f, dt) * 10.0f);
                chase_forward = glm::normalize(glm::mix(chase_forward, move_dir, t_dir));
            }
            else
            {
                // Машин бараг зогссон үед visual yaw offset (pi)-ийг залруулж fallback чиглэл авна.
                const float logical_yaw = subaru->tr.rot_euler.y - 3.14159265f;
                const glm::vec3 fallback_fwd{std::cos(logical_yaw), 0.0f, std::sin(logical_yaw)};
                chase_forward = glm::normalize(glm::mix(chase_forward, fallback_fwd, 0.08f));
            }
            prev_subaru_pos = subaru->tr.pos;
            has_prev_subaru_pos = true;

            const float car_yaw = std::atan2(chase_forward.z, chase_forward.x);
            const float orbit_yaw = car_yaw + 3.14159265f + chase_orbit_yaw;
            const float orbit_pitch = std::clamp(chase_orbit_pitch, glm::radians(5.0f), glm::radians(70.0f));
            const float cp = std::cos(orbit_pitch);
            const glm::vec3 orbit_dir{
                cp * std::cos(orbit_yaw),
                std::sin(orbit_pitch),
                cp * std::sin(orbit_yaw)
            };
            const glm::vec3 focus = subaru->tr.pos + glm::vec3(0.0f, chase_height, 0.0f);
            const glm::vec3 desired_cam = focus + orbit_dir * chase_dist;
            follow_target(chase_cam, desired_cam, glm::vec3(0.0f), chase_smoothing, dt);

            const glm::vec3 look_point = subaru->tr.pos + chase_forward * chase_look_ahead + glm::vec3(0.0f, 0.8f, 0.0f);
            const glm::vec3 v = look_point - chase_cam.pos;
            const float len = glm::length(v);
            if (len > 1e-6f)
            {
                const glm::vec3 d = v / len;
                const float target_yaw = std::atan2(d.z, d.x);
                const float target_pitch = std::asin(glm::clamp(d.y, -1.0f, 1.0f));
                const float rot_t = std::clamp(chase_smoothing * dt * 8.0f, 0.0f, 1.0f);
                chase_cam.yaw = lerp_angle_rad(chase_cam.yaw, target_yaw, rot_t);
                chase_cam.pitch = glm::mix(chase_cam.pitch, target_pitch, rot_t);
            }
        }

        // Камерын mode шилжилтийг тасралтгүй, зөөлөн blend-ээр шийднэ.
        const float target_blend = follow_camera ? 1.0f : 0.0f;
        const float blend_t = 1.0f - std::exp(-mode_blend_speed * std::max(0.0f, dt));
        follow_blend = glm::mix(follow_blend, target_blend, blend_t);
        cam.pos = glm::mix(free_cam.pos, chase_cam.pos, follow_blend);
        cam.yaw = lerp_angle_rad(free_cam.yaw, chase_cam.yaw, follow_blend);
        cam.pitch = glm::mix(free_cam.pitch, chase_cam.pitch, follow_blend);

        // Logic-оор шинэчлэгдсэн object/camera төлөвийг render scene рүү sync хийнэ.
        objects.sync_to_scene(scene);
        shs::sync_camera_to_scene(cam, scene, (float)CANVAS_W / (float)CANVAS_H);
        procedural_sky.set_sun_direction(scene.sun.dir_ws);
        scene.sky = use_cubemap_sky ? static_cast<const shs::ISkyModel*>(&cubemap_sky)
                                    : static_cast<const shs::ISkyModel*>(&procedural_sky);

        // Render systems ажиллуулж LDR target гаргана.
        const auto t_render0 = std::chrono::steady_clock::now();
        shs::RenderSystemContext render_ctx{};
        render_ctx.ctx = &ctx;
        render_ctx.scene = &scene;
        render_ctx.frame = &fp;
        render_ctx.rtr = &rtr;
        render_systems.render(render_ctx);
        const auto t_render1 = std::chrono::steady_clock::now();
        render_ms_accum += std::chrono::duration<float, std::milli>(t_render1 - t_render0).count();

        upload_ldr_to_rgba8(rgba_staging, ldr_rt);
        runtime.upload_rgba8(rgba_staging.data(), ldr_rt.w, ldr_rt.h, ldr_rt.w * 4);
        runtime.present();

        // Богино хугацааны FPS/telemetry-ийг title дээр шинэчилнэ.
        frames++;
        fps_accum += dt;
        if (fps_accum >= 0.25f)
        {
            const int fps = (int)std::lround((float)frames / fps_accum);
            std::string title = "HelloPassBasics | FPS: " + std::to_string(fps)
                + " | dbg[F1]: " + std::to_string((int)fp.debug_view)
                + " | shade[F4]: " + (fp.shading_model == shs::ShadingModel::PBRMetalRough ? "PBR" : "Blinn")
                + " | sky[F5]: " + (use_cubemap_sky ? "cubemap" : "procedural")
                + " | follow[F6]: " + (follow_camera ? "on" : "off")
                + " | logic: " + std::to_string((int)std::lround(logic_ms_accum / std::max(1, frames))) + "ms"
                + " | render: " + std::to_string((int)std::lround(render_ms_accum / std::max(1, frames))) + "ms"
                + " | tri(in/clip/rast): "
                + std::to_string((unsigned long long)ctx.debug.tri_input) + "/"
                + std::to_string((unsigned long long)ctx.debug.tri_after_clip) + "/"
                + std::to_string((unsigned long long)ctx.debug.tri_raster);
            runtime.set_title(title);
            frames = 0;
            fps_accum = 0.0f;
            logic_ms_accum = 0.0f;
            render_ms_accum = 0.0f;
        }
    }

    return 0;
}
