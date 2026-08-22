#pragma once

// ============================================================================
// fps/domains/matrix/fps.contract.hpp — CORE TYPES (fps::matrix)
// Command variant, DOD SoA tables, snapshots, and events for the world
// state machine. Pure data + tiny inline helpers; no SDL, no audio, no I/O.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <memory_resource>
#include <variant>
#include <vector>

#include <glm/glm.hpp>

namespace fps::matrix {

    // ------------------------------------------------------------------
    // Commands (input edge -> matrix action)
    // ------------------------------------------------------------------
    struct MoveIntent  { glm::vec3 direction_xz{ 0.0f }; }; // local axes: x=strafe, z=forward
    struct LookIntent  { float delta_yaw = 0.0f; float delta_pitch = 0.0f; };
    struct JumpIntent  {};
    struct FireIntent  {};
    struct ResetIntent {};

    using UserCommand = std::variant<MoveIntent, LookIntent, JumpIntent, FireIntent, ResetIntent>;

    struct PlayerCommandFrame {
        glm::vec3 move_dir{ 0.0f };      // normalized xz intent (local space)
        float     delta_yaw     = 0.0f;
        float     delta_pitch   = 0.0f;
        bool      jump_pressed  = false;
        bool      fire_pressed  = false;
        bool      reset_pressed = false;
    };

    // ------------------------------------------------------------------
    // Events (matrix -> progression / audio / ui edges)
    // ------------------------------------------------------------------
    enum class EventType : uint8_t {
        PLAYER_FIRED,
        BOT_FIRED,       // emitted by the reducer (replaces the old in-reducer audio call)
        BOT_HIT,
        BOT_KILLED,
        PLAYER_DAMAGED,
        PLAYER_JUMPED
    };

    struct CombatEvent {
        EventType type;
        glm::vec3 position{ 0.0f };
        int       target_id = -1;
    };

    // ------------------------------------------------------------------
    // DOD tables (SoA)
    // ------------------------------------------------------------------
    enum class BotState : uint8_t {
        PATROL = 0,
        CHASE  = 1,
        ATTACK = 2,
        DEAD   = 3
    };

    struct BotTableSoA {
        std::pmr::vector<glm::vec3> position;
        std::pmr::vector<glm::vec3> target_waypoint;
        std::pmr::vector<float>     yaw;
        std::pmr::vector<int16_t>   hp;
        std::pmr::vector<BotState>  state;
        std::pmr::vector<float>     hit_flash_time;
        std::pmr::vector<float>     respawn_time;
        std::pmr::vector<float>     attack_cooldown;
        std::pmr::vector<float>     bob_phase;
        std::pmr::vector<float>     strafe_dir;

        explicit BotTableSoA(std::pmr::memory_resource* mr = std::pmr::get_default_resource())
            : position(mr), target_waypoint(mr), yaw(mr), hp(mr),
              state(mr), hit_flash_time(mr), respawn_time(mr),
              attack_cooldown(mr), bob_phase(mr), strafe_dir(mr) {
        }

        BotTableSoA(const BotTableSoA&)                = default;
        BotTableSoA& operator=(const BotTableSoA&)     = default;
        BotTableSoA(BotTableSoA&&) noexcept            = default;
        BotTableSoA& operator=(BotTableSoA&&) noexcept = default;

        size_t size() const { return position.size(); }

        void add_bot(glm::vec3 pos, glm::vec3 wp) {
            position.push_back(pos);
            target_waypoint.push_back(wp);
            yaw.push_back(0.0f);
            hp.push_back(100);
            state.push_back(BotState::PATROL);
            hit_flash_time.push_back(0.0f);
            respawn_time.push_back(0.0f);
            attack_cooldown.push_back(0.0f);
            bob_phase.push_back(0.0f);
            strafe_dir.push_back(1.0f);
        }
    };

    struct ProjectileTableSoA {
        std::pmr::vector<glm::vec3> position;
        std::pmr::vector<glm::vec3> velocity;
        std::pmr::vector<float>     life;

        explicit ProjectileTableSoA(std::pmr::memory_resource* mr = std::pmr::get_default_resource())
            : position(mr), velocity(mr), life(mr) {
        }

        ProjectileTableSoA(const ProjectileTableSoA&)                = default;
        ProjectileTableSoA& operator=(const ProjectileTableSoA&)     = default;
        ProjectileTableSoA(ProjectileTableSoA&&) noexcept            = default;
        ProjectileTableSoA& operator=(ProjectileTableSoA&&) noexcept = default;

        size_t size() const { return position.size(); }

        void add(glm::vec3 pos, glm::vec3 vel, float dur = 3.0f) {
            position.push_back(pos);
            velocity.push_back(vel);
            life.push_back(dur);
        }

        void remove_at(size_t i) {
            if (i < position.size()) {
                position.erase(position.begin() + static_cast<std::ptrdiff_t>(i));
                velocity.erase(velocity.begin() + static_cast<std::ptrdiff_t>(i));
                life.erase(life.begin() + static_cast<std::ptrdiff_t>(i));
            }
        }
    };

    struct BulletTracer {
        glm::vec3 start;
        glm::vec3 end;
        float     life = 0.08f;
    };

    // ------------------------------------------------------------------
    // Snapshots
    // ------------------------------------------------------------------
    struct PlayerSnapshot {
        glm::vec3 position{ 0.0f, 1.70f, -8.0f };
        float     velocity_y    = 0.0f;
        float     yaw           = 0.0f;
        float     pitch         = 0.0f;
        int16_t   hp            = 100;
        float     damage_flash  = 0.0f;
        float     fire_cooldown = 0.0f;
        float     recoil_offset = 0.0f;
        float     muzzle_flash  = 0.0f;
        bool      is_grounded   = true;

        glm::vec3 get_forward() const {
            return glm::normalize(glm::vec3(
                std::sin(yaw) * std::cos(pitch),
                std::sin(pitch),
                std::cos(yaw) * std::cos(pitch)
            ));
        }

        glm::vec3 get_right() const {
            return glm::normalize(glm::vec3(std::cos(yaw), 0.0f, -std::sin(yaw)));
        }
    };

    struct WorldSnapshot {
        PlayerSnapshot                 player;
        BotTableSoA                    bots;
        ProjectileTableSoA             projectiles;
        std::pmr::vector<BulletTracer> tracers;
        uint32_t                       rng_state = 0x853c49e6u; // deterministic bot-AI LCG

        explicit WorldSnapshot(std::pmr::memory_resource* mr = std::pmr::get_default_resource())
            : bots(mr), projectiles(mr), tracers(mr) {
        }

        WorldSnapshot(const WorldSnapshot&)                = default;
        WorldSnapshot& operator=(const WorldSnapshot&)     = default;
        WorldSnapshot(WorldSnapshot&&) noexcept            = default;
        WorldSnapshot& operator=(WorldSnapshot&&) noexcept = default;
    };

    struct WorldStepResult {
        WorldSnapshot                 next_world;
        std::pmr::vector<CombatEvent> events;

        WorldStepResult(std::pmr::memory_resource* persistent_mr, std::pmr::memory_resource* frame_mr)
            : next_world(persistent_mr), events(frame_mr) {
        }
    };

} // namespace fps::matrix