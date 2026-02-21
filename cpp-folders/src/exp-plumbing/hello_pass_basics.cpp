#define SDL_MAIN_HANDLED

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/app/camera_sync.hpp>
#include <shs/camera/follow_camera.hpp>
#include <shs/core/context.hpp>
#include <shs/frame/technique_mode.hpp>
#include <shs/frame/frame_params.hpp>
#include <shs/gfx/rt_registry.hpp>
#include <shs/gfx/rt_shadow.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/job/thread_pool_job_system.hpp>
#include <shs/logic/fsm.hpp>
#include <shs/pipeline/pass_adapters.hpp>
#include <shs/pipeline/pluggable_pipeline.hpp>
#include <shs/pipeline/render_composition_presets.hpp>
#include <shs/pipeline/render_path_executor.hpp>
#include <shs/pipeline/render_technique_presets.hpp>
#include <shs/platform/sdl/sdl_runtime.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
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
    constexpr float PI = 3.14159265f;
    constexpr float TWO_PI = 6.2831853f;
    constexpr float MOUSE_LOOK_SENS = 0.0025f;
    constexpr float FREE_CAM_BASE_SPEED = 8.0f;
    constexpr float CHASE_ORBIT_SENS = 0.0025f;

    enum class ModelForwardAxis : uint8_t
    {
        PosX = 0,
        NegX = 1,
        PosZ = 2,
        NegZ = 3,
    };

    constexpr ModelForwardAxis SUBARU_VISUAL_FORWARD_AXIS = ModelForwardAxis::PosZ;

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
        while (d > PI) d -= TWO_PI;
        while (d < -PI) d += TWO_PI;
        return a + d * t;
    }

    float visual_yaw_from_world_forward(const glm::vec3& fwd_ws, ModelForwardAxis axis)
    {
        glm::vec2 d{fwd_ws.x, fwd_ws.z};
        const float len = glm::length(d);
        if (len <= 1e-6f) return 0.0f;
        d /= len;
        switch (axis)
        {
            case ModelForwardAxis::PosX: return std::atan2(d.y, d.x);
            case ModelForwardAxis::NegX: return std::atan2(-d.y, -d.x);
            case ModelForwardAxis::PosZ: return std::atan2(d.x, d.y);
            case ModelForwardAxis::NegZ: return std::atan2(-d.x, -d.y);
        }
        return 0.0f;
    }

    glm::vec3 world_forward_from_visual_yaw(float visual_yaw, ModelForwardAxis axis)
    {
        switch (axis)
        {
            case ModelForwardAxis::PosX: return glm::vec3(std::cos(visual_yaw), 0.0f, std::sin(visual_yaw));
            case ModelForwardAxis::NegX: return glm::vec3(-std::cos(visual_yaw), 0.0f, -std::sin(visual_yaw));
            case ModelForwardAxis::PosZ: return glm::vec3(std::sin(visual_yaw), 0.0f, std::cos(visual_yaw));
            case ModelForwardAxis::NegZ: return glm::vec3(-std::sin(visual_yaw), 0.0f, -std::cos(visual_yaw));
        }
        return glm::vec3(1.0f, 0.0f, 0.0f);
    }

    // Subaru машинд deterministic төлөвт автомат жолоодлого (Cruise/Turn/Recover/Idle) хэрэгжүүлнэ.
    class SubaruCruiseSystem final : public shs::ILogicSystem
    {
    public:
        enum class DriveState : uint8_t
        {
            Cruise = 0,
            Turn = 1,
            Recover = 2,
            Idle = 3,
        };

        struct FsmContext
        {
            SubaruCruiseSystem& self;
            shs::SceneObject& obj;
        };

        SubaruCruiseSystem(
            std::string object_name,
            float area_half_extent,
            float y_level,
            float cruise_speed = 6.5f,
            float max_turn_rate_rad = 1.9f,
            ModelForwardAxis visual_forward_axis = ModelForwardAxis::PosX,
            float visual_yaw_offset_rad = 0.0f,
            uint32_t seed = 0xC0FFEEu
        )
            : object_name_(std::move(object_name))
            , area_half_extent_(area_half_extent)
            , y_level_(y_level)
            , cruise_speed_(cruise_speed)
            , max_turn_rate_rad_(max_turn_rate_rad)
            , visual_forward_axis_(visual_forward_axis)
            , visual_yaw_offset_rad_(visual_yaw_offset_rad)
            , rng_(seed)
            , area_dist_(-area_half_extent_ * 0.90f, area_half_extent_ * 0.90f)
            , unit_dist_(0.0f, 1.0f)
            , turn_rate_dist_(0.95f, 1.80f)
            , cruise_yaw_bias_dist_(-0.46f, 0.46f)
            , speed_jitter_(0.82f, 1.18f)
        {
            configure_fsm();
        }

        const char* state_name() const
        {
            switch (current_state())
            {
                case DriveState::Cruise: return "Cruise";
                case DriveState::Turn: return "Turn";
                case DriveState::Recover: return "Recover";
                case DriveState::Idle: return "Idle";
            }
            return "Unknown";
        }

        float state_progress() const
        {
            if (!fsm_.started()) return 0.0f;
            if (state_duration_ <= 1e-6f) return 1.0f;
            return std::clamp(fsm_.state_time() / state_duration_, 0.0f, 1.0f);
        }

        glm::vec3 heading_ws() const
        {
            if (!initialized_) return glm::vec3(1.0f, 0.0f, 0.0f);
            return glm::normalize(glm::vec3(std::cos(current_yaw_), 0.0f, std::sin(current_yaw_)));
        }

        void tick(shs::LogicSystemContext& ctx) override
        {
            if (!ctx.objects) return;
            auto* obj = ctx.objects->find(object_name_);
            if (!obj) return;
            const float dt = std::max(0.0f, ctx.dt);
            if (dt <= 1e-6f) return;

            if (!initialized_)
            {
                // Эхний чиглэлийг model-ийн yaw-аас coordinate convention дагуу сэргээнэ.
                obj->tr.pos.y = y_level_;
                const glm::vec3 seed_fwd = world_forward_from_visual_yaw(
                    obj->tr.rot_euler.y - visual_yaw_offset_rad_,
                    visual_forward_axis_
                );
                current_yaw_ = std::atan2(seed_fwd.z, seed_fwd.x);
                current_speed_ = cruise_speed_;

                FsmContext fsm_ctx{*this, *obj};
                (void)fsm_.start(DriveState::Cruise, fsm_ctx);
                initialized_ = true;
            }

            obj->tr.pos.y = y_level_;

            desired_yaw_ = current_yaw_;
            desired_speed_ = cruise_speed_;
            FsmContext fsm_ctx{*this, *obj};
            fsm_.tick(fsm_ctx, dt);

            const float edge_ratio = boundary_ratio(obj->tr.pos);
            apply_boundary_steer(obj->tr.pos, desired_yaw_, desired_speed_);

            float dy = desired_yaw_ - current_yaw_;
            while (dy > PI) dy -= TWO_PI;
            while (dy < -PI) dy += TWO_PI;
            const float max_step = max_turn_rate_rad_ * dt;
            dy = std::clamp(dy, -max_step, max_step);
            current_yaw_ += dy;

            const float speed_lerp_t = 1.0f - std::exp(-dt * 6.0f);
            current_speed_ = glm::mix(current_speed_, desired_speed_, speed_lerp_t);

            const glm::vec3 fwd{std::cos(current_yaw_), 0.0f, std::sin(current_yaw_)};
            const float speed_scale = 1.0f - edge_ratio * 0.35f;
            obj->tr.pos += fwd * (current_speed_ * speed_scale * dt);
            obj->tr.pos.x = std::clamp(obj->tr.pos.x, -area_half_extent_, area_half_extent_);
            obj->tr.pos.z = std::clamp(obj->tr.pos.z, -area_half_extent_, area_half_extent_);
            obj->tr.pos.y = y_level_;
            obj->tr.rot_euler.y = visual_yaw_from_world_forward(fwd, visual_forward_axis_) + visual_yaw_offset_rad_;
        }

    private:
        DriveState current_state() const
        {
            const auto s = fsm_.current_state();
            return s.has_value() ? *s : DriveState::Cruise;
        }

        void configure_fsm()
        {
            using StateCallbacks = typename shs::StateMachine<DriveState, FsmContext>::StateCallbacks;

            fsm_.add_state(DriveState::Cruise, StateCallbacks{
                [this](FsmContext& fctx) { on_enter_state(DriveState::Cruise, fctx.obj.tr.pos); },
                [this](FsmContext&, float dt, float) { update_cruise(dt); },
                {}
            });
            fsm_.add_state(DriveState::Turn, StateCallbacks{
                [this](FsmContext& fctx) { on_enter_state(DriveState::Turn, fctx.obj.tr.pos); },
                [this](FsmContext&, float dt, float) { update_turn(dt); },
                {}
            });
            fsm_.add_state(DriveState::Recover, StateCallbacks{
                [this](FsmContext& fctx) { on_enter_state(DriveState::Recover, fctx.obj.tr.pos); },
                [this](FsmContext& fctx, float, float) { update_recover(fctx.obj); },
                {}
            });
            fsm_.add_state(DriveState::Idle, StateCallbacks{
                [this](FsmContext& fctx) { on_enter_state(DriveState::Idle, fctx.obj.tr.pos); },
                [this](FsmContext&, float, float) { update_idle(); },
                {}
            });

            // Нэг төлөвийн хугацаа дуусахад тухайн төлөв дээр урьдчилан тооцсон дараагийн төлөв рүү шилжинэ.
            fsm_.add_transition(DriveState::Cruise, DriveState::Idle, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Idle;
            });
            fsm_.add_transition(DriveState::Cruise, DriveState::Turn, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Turn;
            });
            fsm_.add_transition(DriveState::Turn, DriveState::Recover, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Recover;
            });
            fsm_.add_transition(DriveState::Recover, DriveState::Idle, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Idle;
            });
            fsm_.add_transition(DriveState::Recover, DriveState::Cruise, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Cruise;
            });
            fsm_.add_transition(DriveState::Idle, DriveState::Cruise, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Cruise;
            });
        }

        float rand01()
        {
            return unit_dist_(rng_);
        }

        float rand_range(float lo, float hi)
        {
            return lo + (hi - lo) * rand01();
        }

        float boundary_ratio(const glm::vec3& p) const
        {
            const float edge = std::max(std::abs(p.x), std::abs(p.z));
            return std::clamp((edge - area_half_extent_ * 0.66f) / (area_half_extent_ * 0.34f), 0.0f, 1.0f);
        }

        void apply_boundary_steer(const glm::vec3& p, float& desired_yaw, float& desired_speed)
        {
            const float edge_ratio = boundary_ratio(p);
            if (edge_ratio <= 0.0f) return;

            glm::vec2 to_center{-p.x, -p.z};
            const float len = glm::length(to_center);
            if (len > 1e-6f)
            {
                to_center /= len;
                const float center_yaw = std::atan2(to_center.y, to_center.x);
                const float steer_w = std::clamp(edge_ratio * (current_state() == DriveState::Recover ? 1.0f : 0.74f), 0.0f, 1.0f);
                desired_yaw = lerp_angle_rad(desired_yaw, center_yaw, steer_w);
            }
            desired_speed *= (1.0f - edge_ratio * 0.28f);

            // Ирмэгт хэт ойртох үед Recover рүү шууд request өгч буцааж төв рүү эргүүлнэ.
            if (edge_ratio > 0.92f && current_state() != DriveState::Recover)
            {
                fsm_.request_transition(DriveState::Recover);
            }
        }

        void pick_recover_target(const glm::vec3& current_pos)
        {
            for (int i = 0; i < 24; ++i)
            {
                const glm::vec3 c{area_dist_(rng_), y_level_, area_dist_(rng_)};
                if (glm::length(glm::vec2(c.x - current_pos.x, c.z - current_pos.z)) > area_half_extent_ * 0.24f)
                {
                    recover_target_ = c;
                    return;
                }
            }
            recover_target_ = glm::vec3(area_dist_(rng_), y_level_, area_dist_(rng_));
        }

        float duration_for_state(DriveState s)
        {
            switch (s)
            {
                case DriveState::Cruise: return rand_range(2.6f, 5.6f);
                case DriveState::Turn: return rand_range(0.55f, 1.65f);
                case DriveState::Recover: return rand_range(1.0f, 2.2f);
                case DriveState::Idle: return rand_range(0.25f, 0.95f);
            }
            return 1.0f;
        }

        DriveState timeout_next_for_state(DriveState s)
        {
            switch (s)
            {
                case DriveState::Cruise: return (rand01() < 0.16f) ? DriveState::Idle : DriveState::Turn;
                case DriveState::Turn: return DriveState::Recover;
                case DriveState::Recover: return (rand01() < 0.20f) ? DriveState::Idle : DriveState::Cruise;
                case DriveState::Idle: return DriveState::Cruise;
            }
            return DriveState::Cruise;
        }

        void on_enter_state(DriveState s, const glm::vec3& pos)
        {
            state_duration_ = duration_for_state(s);
            timeout_next_state_ = timeout_next_for_state(s);
            switch (s)
            {
                case DriveState::Cruise:
                    cruise_turn_rate_ = cruise_yaw_bias_dist_(rng_);
                    cruise_target_speed_ = cruise_speed_ * speed_jitter_(rng_);
                    break;
                case DriveState::Turn:
                {
                    const float sign = (rand01() < 0.5f) ? -1.0f : 1.0f;
                    turn_rate_ = turn_rate_dist_(rng_) * sign;
                    break;
                }
                case DriveState::Recover:
                    pick_recover_target(pos);
                    break;
                case DriveState::Idle:
                    break;
            }
        }

        void update_cruise(float dt)
        {
            desired_yaw_ = current_yaw_ + cruise_turn_rate_ * dt;
            desired_speed_ = cruise_target_speed_;
        }

        void update_turn(float dt)
        {
            desired_yaw_ = current_yaw_ + turn_rate_ * dt;
            desired_speed_ = cruise_speed_ * 0.76f;
        }

        void update_recover(const shs::SceneObject& obj)
        {
            const glm::vec3 to_goal = recover_target_ - obj.tr.pos;
            const glm::vec2 to_goal_xz{to_goal.x, to_goal.z};
            const float len = glm::length(to_goal_xz);
            if (len > 1e-5f)
            {
                const glm::vec2 d = to_goal_xz / len;
                desired_yaw_ = std::atan2(d.y, d.x);
            }
            desired_speed_ = cruise_speed_ * 0.92f;
            if (len < area_half_extent_ * 0.10f)
            {
                fsm_.request_transition(timeout_next_state_);
            }
        }

        void update_idle()
        {
            desired_yaw_ = current_yaw_;
            desired_speed_ = 0.0f;
        }

        std::string object_name_{};
        float area_half_extent_ = 16.0f;
        float y_level_ = 0.0f;
        float cruise_speed_ = 6.5f;
        float max_turn_rate_rad_ = 1.9f;
        ModelForwardAxis visual_forward_axis_ = ModelForwardAxis::PosX;
        float visual_yaw_offset_rad_ = 0.0f;
        float current_speed_ = 0.0f;
        float current_yaw_ = 0.0f;
        bool initialized_ = false;

        float state_duration_ = 1.0f;
        DriveState timeout_next_state_ = DriveState::Cruise;
        float desired_yaw_ = 0.0f;
        float desired_speed_ = 0.0f;

        float cruise_turn_rate_ = 0.0f;
        float cruise_target_speed_ = 6.5f;
        float turn_rate_ = 0.0f;
        glm::vec3 recover_target_{0.0f};

        shs::StateMachine<DriveState, FsmContext> fsm_{};
        std::mt19937 rng_{};
        std::uniform_real_distribution<float> area_dist_{-12.0f, 12.0f};
        std::uniform_real_distribution<float> unit_dist_{0.0f, 1.0f};
        std::uniform_real_distribution<float> turn_rate_dist_{0.95f, 1.80f};
        std::uniform_real_distribution<float> cruise_yaw_bias_dist_{-0.46f, 0.46f};
        std::uniform_real_distribution<float> speed_jitter_{0.82f, 1.18f};
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
    // Рендерийн parallel хэсгүүдэд ашиглагдах thread pool.
    shs::ThreadPoolJobSystem jobs{std::max(1u, std::thread::hardware_concurrency())};
    ctx.job_system = &jobs;

    shs::ResourceRegistry resources{};
    shs::RTRegistry rtr{};
    shs::PluggablePipeline pipeline{};
    shs::LogicSystemProcessor logic_systems{};
    shs::RenderSystemProcessor render_systems{};

    shs::RT_ShadowDepth shadow_rt{512, 512};
    shs::RT_ColorHDR hdr_rt{CANVAS_W, CANVAS_H};
    shs::RT_ColorDepthMotion motion_rt{CANVAS_W, CANVAS_H, 0.1f, 1000.0f};
    shs::RT_ColorLDR ldr_rt{CANVAS_W, CANVAS_H};
    shs::RT_ColorLDR shafts_tmp_rt{CANVAS_W, CANVAS_H};
    shs::RT_ColorLDR motion_blur_tmp_rt{CANVAS_W, CANVAS_H};

    const shs::RT_Shadow rt_shadow_h = rtr.reg<shs::RT_Shadow>(&shadow_rt);
    const shs::RTHandle rt_hdr_h = rtr.reg<shs::RTHandle>(&hdr_rt);
    const shs::RT_Motion rt_motion_h = rtr.reg<shs::RT_Motion>(&motion_rt);
    const shs::RTHandle rt_ldr_h = rtr.reg<shs::RTHandle>(&ldr_rt);
    const shs::RTHandle rt_shafts_tmp_h = rtr.reg<shs::RTHandle>(&shafts_tmp_rt);
    const shs::RTHandle rt_motion_blur_tmp_h = rtr.reg<shs::RTHandle>(&motion_blur_tmp_rt);

    // Pass registry-г shared pass adapter factory-аар байгуулна.
    const shs::PassFactoryRegistry pass_registry = shs::make_standard_pass_factory_registry(
        rt_shadow_h,
        rt_hdr_h,
        rt_motion_h,
        rt_ldr_h,
        rt_shafts_tmp_h,
        rt_motion_blur_tmp_h
    );
    shs::RenderPathExecutor render_path_executor{};
    std::vector<std::string> render_path_missing_passes{};
    std::string render_path_name = "fallback_forward";
    bool render_path_plan_valid = false;
    bool render_path_configured = false;

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
    const shs::MeshAssetHandle plane_h = shs::import_plane_primitive(resources, shs::PlaneDesc{plane_extent, plane_extent, 32, 32}, "plane");
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
    fp.debug_view = shs::DebugViewMode::Final;
    fp.cull_mode = shs::CullMode::None;
    fp.shading_model = shs::ShadingModel::PBRMetalRough;
    fp.pass.tonemap.exposure = 1.0f;
    fp.pass.tonemap.gamma = 2.2f;
    fp.pass.shadow.enable = true;
    fp.pass.shadow.pcf_radius = 1;
    fp.pass.shadow.pcf_step = 1.0f;
    fp.pass.shadow.strength = 0.80f;
    fp.pass.light_shafts.enable = true;
    fp.pass.light_shafts.steps = 14;
    fp.pass.light_shafts.density = 0.85f;
    fp.pass.light_shafts.weight = 0.26f;
    fp.pass.light_shafts.decay = 0.95f;
    fp.pass.motion_vectors.enable = true;
    fp.pass.motion_blur.enable = true;
    fp.pass.motion_blur.samples = 12;
    fp.pass.motion_blur.strength = 0.95f;
    fp.pass.motion_blur.max_velocity_px = 20.0f;
    fp.pass.motion_blur.min_velocity_px = 0.30f;
    fp.pass.motion_blur.depth_reject = 0.10f;
    fp.technique.mode = shs::TechniqueMode::Forward;
    fp.technique.active_modes_mask = shs::technique_mode_mask_all();
    fp.technique.depth_prepass = false;
    fp.technique.light_culling = false;
    shs::RenderTechniquePreset render_technique_preset =
        shs::render_technique_preset_from_shading_model(fp.shading_model);
    shs::RenderTechniqueRecipe render_technique_recipe =
        shs::make_builtin_render_technique_recipe(render_technique_preset, "composition_sw");
    shs::RenderCompositionRecipe active_composition_recipe =
        shs::make_builtin_render_composition_recipe(
            shs::render_path_preset_for_mode(fp.technique.mode),
            render_technique_preset,
            "composition_sw");
    std::vector<shs::RenderCompositionRecipe> composition_cycle_order{};
    size_t active_composition_index = 0u;

    const auto apply_render_technique_preset = [&](shs::RenderTechniquePreset preset)
    {
        render_technique_preset = preset;
        render_technique_recipe =
            shs::make_builtin_render_technique_recipe(render_technique_preset, "composition_sw");
        fp.shading_model = render_technique_recipe.shading_model;
        fp.pass.tonemap.exposure = render_technique_recipe.tonemap_exposure;
        fp.pass.tonemap.gamma = render_technique_recipe.tonemap_gamma;
    };
    apply_render_technique_preset(render_technique_preset);

    const auto technique_uses_light_culling = [](shs::TechniqueMode mode) -> bool
    {
        return mode == shs::TechniqueMode::ForwardPlus ||
               mode == shs::TechniqueMode::TiledDeferred ||
               mode == shs::TechniqueMode::ClusteredForward;
    };

    const auto plan_has_pass = [](const shs::RenderPathExecutionPlan& plan, const char* pass_id) -> bool
    {
        if (!pass_id || pass_id[0] == '\0') return false;
        for (const auto& e : plan.pass_chain)
        {
            if (e.id == pass_id) return true;
        }
        return false;
    };

    auto apply_fallback_technique_pipeline = [&](const char* fallback_tag) -> bool
    {
        render_path_missing_passes.clear();
        render_path_configured =
            pipeline.configure_for_technique(pass_registry, fp.technique.mode, &render_path_missing_passes);
        pipeline.set_strict_graph_validation(true);
        render_path_plan_valid = false;
        render_path_name = std::string(fallback_tag ? fallback_tag : "fallback")
            + "_"
            + shs::technique_mode_name(fp.technique.mode);
        fp.technique.depth_prepass = (fp.technique.mode != shs::TechniqueMode::Forward);
        fp.technique.light_culling = technique_uses_light_culling(fp.technique.mode);
        fp.pass.light_shafts.enable = false;
        return render_path_configured;
    };

    auto apply_render_path_index = [&](size_t index) -> bool
    {
        if (!render_path_executor.has_recipes()) return false;

        render_path_plan_valid = render_path_executor.apply_index(index, ctx, &pass_registry);
        const shs::RenderPathRecipe& recipe = render_path_executor.active_recipe();
        const shs::RenderPathExecutionPlan& plan = render_path_executor.active_plan();
        render_path_name = recipe.name.empty() ? std::string("unnamed_path") : recipe.name;

        if (plan.pass_chain.empty())
        {
            return false;
        }

        fp.technique.mode = plan.technique_mode;
        fp.technique.active_modes_mask = shs::technique_mode_mask_all();
        fp.technique.tile_size = std::max(1u, recipe.light_tile_size);
        fp.technique.depth_prepass = plan_has_pass(plan, "depth_prepass");
        fp.technique.light_culling = technique_uses_light_culling(plan.technique_mode);
        fp.pass.shadow.enable = recipe.wants_shadows && recipe.runtime_defaults.enable_shadows;
        fp.pass.light_shafts.enable = false;
        if (!plan_has_pass(plan, "motion_blur"))
        {
            fp.pass.motion_blur.enable = false;
        }

        render_path_missing_passes.clear();
        render_path_configured =
            pipeline.configure_from_render_path_plan(pass_registry, plan, &render_path_missing_passes);
        pipeline.set_strict_graph_validation(recipe.strict_validation);
        return render_path_plan_valid && render_path_configured;
    };

    auto refresh_active_composition_recipe = [&]()
    {
        const shs::RenderPathPreset active_path_preset =
            shs::render_path_preset_for_mode(fp.technique.mode);
        active_composition_recipe = shs::make_builtin_render_composition_recipe(
            active_path_preset,
            render_technique_preset,
            "composition_sw");
        for (size_t i = 0; i < composition_cycle_order.size(); ++i)
        {
            const auto& c = composition_cycle_order[i];
            if (c.path_preset == active_path_preset && c.technique_preset == render_technique_preset)
            {
                active_composition_index = i;
                active_composition_recipe = c;
                break;
            }
        }
    };

    auto apply_render_composition_by_index = [&](size_t index) -> bool
    {
        if (composition_cycle_order.empty() || !render_path_executor.has_recipes()) return false;
        const size_t resolved = index % composition_cycle_order.size();
        const shs::RenderCompositionRecipe& composition = composition_cycle_order[resolved];
        const size_t path_index = render_path_executor.find_recipe_index_by_mode(
            shs::render_path_preset_mode(composition.path_preset));
        apply_render_technique_preset(composition.technique_preset);
        if (!apply_render_path_index(path_index))
        {
            refresh_active_composition_recipe();
            return false;
        }

        active_composition_index = resolved;
        active_composition_recipe = composition;
        return true;
    };

    const bool have_builtin_paths =
        render_path_executor.register_builtin_presets(shs::RenderBackendType::Software, "sw_path");
    composition_cycle_order = shs::make_default_render_composition_recipes("composition_sw");
    if (!have_builtin_paths || !render_path_executor.has_recipes())
    {
        std::fprintf(stderr, "[shs] Render-path presets unavailable for software backend. Falling back.\n");
        (void)apply_fallback_technique_pipeline("fallback");
    }
    else
    {
        const size_t preferred_index =
            render_path_executor.find_recipe_index_by_mode(fp.technique.mode);
        if (!apply_render_path_index(preferred_index))
        {
            std::fprintf(stderr,
                "[shs] Render-path compile/config failed for '%s'. Falling back to profile.\n",
                render_path_name.c_str());
            (void)apply_fallback_technique_pipeline("fallback");
        }
    }
    refresh_active_composition_recipe();
    render_systems.add_system<shs::PipelineRenderSystem>(&pipeline);

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
    auto& subaru_ai = logic_systems.add_system<SubaruCruiseSystem>(
        "subaru",
        plane_extent * 0.48f,
        -0.95f,
        6.8f,
        1.9f,
        SUBARU_VISUAL_FORWARD_AXIS,
        0.0f
    );
    logic_systems.add_system<MonkeyWiggleSystem>("monkey", 0.32f, 0.22f, 1.9f);

    if (const auto* subaru_init = objects.find("subaru"))
    {
        prev_subaru_pos = subaru_init->tr.pos;
        has_prev_subaru_pos = true;
        chase_forward = world_forward_from_visual_yaw(subaru_init->tr.rot_euler.y, SUBARU_VISUAL_FORWARD_AXIS);
    }

    std::printf(
        "Controls: LMB/RMB drag look, WASD+QE move, Shift boost | "
        "F1 debug view, F2 cycle render path, F3 cycle composition, F4 cycle shading, "
        "M motion blur, F5 sky, F6 follow camera\n");

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
        // F2: render-path preset cycle.
        if (pin.cycle_cull_mode)
        {
            if (render_path_executor.has_recipes())
            {
                if (!apply_render_path_index(render_path_executor.active_index() + 1u))
                {
                    std::fprintf(stderr,
                        "[shs] Render-path cycle failed for '%s'. Falling back.\n",
                        render_path_name.c_str());
                    (void)apply_fallback_technique_pipeline("fallback");
                }
            }
            else
            {
                (void)apply_fallback_technique_pipeline("fallback");
            }
            refresh_active_composition_recipe();
        }
        // F3: explicit composition cycle (path + technique).
        if (pin.toggle_front_face)
        {
            if (composition_cycle_order.empty() || !render_path_executor.has_recipes())
            {
                std::fprintf(stderr, "[shs] Render composition cycle unavailable.\n");
            }
            else if (!apply_render_composition_by_index(active_composition_index + 1u))
            {
                std::fprintf(stderr, "[shs] Render composition cycle failed. Falling back.\n");
                (void)apply_fallback_technique_pipeline("fallback");
                refresh_active_composition_recipe();
            }
        }
        // F4: PBR <-> BlinnPhong солих.
        if (pin.toggle_shading_model)
        {
            apply_render_technique_preset(
                shs::next_render_technique_preset(render_technique_preset));
            refresh_active_composition_recipe();
        }
        // M: motion blur on/off.
        if (pin.toggle_motion_blur) fp.pass.motion_blur.enable = !fp.pass.motion_blur.enable;
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

        // Mouse hold төлөвийг SDL-ээс шууд уншиж drag-look/relative mode-ыг тогтвортой болгоно.
        const uint32_t mouse_state = SDL_GetMouseState(nullptr, nullptr);
        left_mouse_held = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
        const bool right_now = (mouse_state & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
        if (right_now != right_mouse_held)
        {
            right_mouse_held = right_now;
            runtime.set_relative_mouse_mode(right_mouse_held);
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
                free_cam.yaw -= pin.mouse_dx * MOUSE_LOOK_SENS;
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
            if (pin.right) free_cam.pos -= right * move_speed;
            if (pin.left) free_cam.pos += right * move_speed;
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
                // Машин бараг зогссон үед AI-ийн одоогийн heading-ийг fallback чиглэл болгон авна.
                const glm::vec3 fallback_fwd = subaru_ai.heading_ws();
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
            const auto& graph_rep = pipeline.execution_report();
            const size_t comp_total = composition_cycle_order.size();
            const size_t comp_slot = comp_total > 0u
                ? ((active_composition_index % comp_total) + 1u)
                : 0u;
            std::string title = "HelloPassBasics | FPS: " + std::to_string(fps)
                + " | backend: " + std::string(ctx.active_backend_name())
                + " | path[F2]: " + render_path_name
                + " | mode: " + shs::technique_mode_name(fp.technique.mode)
                + " | comp[F3]: " + active_composition_recipe.name
                + "(" + std::to_string((unsigned long long)comp_slot)
                + "/" + std::to_string((unsigned long long)comp_total) + ")"
                + " | path_state: " + ((render_path_plan_valid && render_path_configured) ? "ok" : "fallback")
                + " | missing: " + std::to_string((unsigned long long)render_path_missing_passes.size())
                + " | dbg[F1]: " + std::to_string((int)fp.debug_view)
                + " | shade[F4]: " + (fp.shading_model == shs::ShadingModel::PBRMetalRough ? "PBR" : "Blinn")
                + " | sky[F5]: " + (use_cubemap_sky ? "cubemap" : "procedural")
                + " | follow[F6]: " + (follow_camera ? "on" : "off")
                + " | ai: " + std::string(subaru_ai.state_name())
                + "(" + std::to_string((int)std::lround(subaru_ai.state_progress() * 100.0f)) + "%)"
                + " | mblur[M]: " + (fp.pass.motion_blur.enable ? "on" : "off")
                + " | graph: " + (graph_rep.valid ? "ok" : "err")
                + "(w" + std::to_string((unsigned long long)graph_rep.warnings.size())
                + "/e" + std::to_string((unsigned long long)graph_rep.errors.size()) + ")"
                + " | logic: " + std::to_string((int)std::lround(logic_ms_accum / std::max(1, frames))) + "ms"
                + " | render: " + std::to_string((int)std::lround(render_ms_accum / std::max(1, frames))) + "ms"
                + " | pass(s/p/t/m): "
                + std::to_string((int)std::lround(ctx.debug.ms_shadow)) + "/"
                + std::to_string((int)std::lround(ctx.debug.ms_pbr)) + "/"
                + std::to_string((int)std::lround(ctx.debug.ms_tonemap)) + "/"
                + std::to_string((int)std::lround(ctx.debug.ms_motion_blur)) + "ms"
                + " | vk-like(sub/task/stall): "
                + std::to_string((unsigned long long)ctx.debug.vk_like_submissions) + "/"
                + std::to_string((unsigned long long)ctx.debug.vk_like_tasks) + "/"
                + std::to_string((unsigned long long)ctx.debug.vk_like_stalls)
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
