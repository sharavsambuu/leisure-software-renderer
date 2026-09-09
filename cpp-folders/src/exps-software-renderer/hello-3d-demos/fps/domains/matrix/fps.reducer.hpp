#pragma once

// ============================================================================
// fps/domains/matrix/fps.reducer.hpp — reduce_world (pure)
// (prev WorldSnapshot, commands, difficulty, level, dt) -> WorldStepResult.
//
// Purity contract: NO SDL, NO audio, NO globals. All randomness flows through
// the snapshot's rng_state (deterministic LCG). Audio-relevant happenings are
// emitted as CombatEvents and mapped to sounds by the main edge.
//
// Internally decomposed into private step functions, executed in order:
//   step_player -> step_hitscan -> step_bot_ai -> step_projectiles -> step_tracers
// ============================================================================

#include <algorithm>
#include <cmath>
#include <span>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "fps.contract.hpp"
#include "fps.action.hpp"

#include <config/difficulty.hpp>
#include <config/levels/fps_level_01.hpp>

namespace fps::matrix {

    namespace detail {

        // Deterministic LCG in [0,1) — replaces the old non-deterministic rand().
        inline float next_rand01(uint32_t& state) {
            state = state * 1664525u + 1013904223u;
            return static_cast<float>(state >> 8) / 16777216.0f;
        }

        inline bool ray_sphere_intersect(const glm::vec3& orig, const glm::vec3& dir,
                                         const glm::vec3& center, float rad, float& out_t) {
            const glm::vec3 oc   = orig - center;
            const float     b    = glm::dot(oc, dir);
            const float     c    = glm::dot(oc, oc) - rad * rad;
            const float     disc = b * b - c;
            if (disc < 0.0f) return false;

            const float sqrt_disc = std::sqrt(disc);
            const float t0        = -b - sqrt_disc;
            const float t1        = -b + sqrt_disc;

            if (t0 > 0.001f) { out_t = t0; return true; }
            if (t1 > 0.001f) { out_t = t1; return true; }
            return false;
        }

        // --- Step 1: player look / move / jump / timers -----------------------
        inline void step_player(WorldSnapshot& w, const PlayerCommandFrame& input,
                                const config::Difficulty& d, const config::FpsLevel01& lvl,
                                float dt, std::pmr::vector<CombatEvent>& events) {
            PlayerSnapshot& p = w.player;

            if (input.reset_pressed) {
                p.position   = lvl.player_spawn_position;
                p.yaw        = lvl.player_spawn_yaw;
                p.pitch      = lvl.player_spawn_pitch;
                p.hp         = 100;
                p.velocity_y = 0.0f;
            }

            p.yaw   += input.delta_yaw;
            p.pitch -= input.delta_pitch;
            p.pitch  = glm::clamp(p.pitch, -d.max_pitch, d.max_pitch);

            const glm::vec3 fwd_xz   = glm::normalize(glm::vec3(std::sin(p.yaw), 0.0f, std::cos(p.yaw)));
            const glm::vec3 right_xz = glm::normalize(glm::vec3(std::cos(p.yaw), 0.0f, -std::sin(p.yaw)));

            glm::vec3 move_dir = input.move_dir.z * fwd_xz + input.move_dir.x * right_xz;
            if (glm::length(move_dir) > 0.01f) {
                move_dir    = glm::normalize(move_dir);
                p.position += move_dir * d.player_speed * dt;
            }
            const float bound = lvl.player_clamp_bound();
            p.position.x = glm::clamp(p.position.x, -bound, bound);
            p.position.z = glm::clamp(p.position.z, -bound, bound);

            if (input.jump_pressed && p.is_grounded) {
                p.velocity_y  = d.jump_velocity;
                p.is_grounded = false;
                events.push_back({ EventType::PLAYER_JUMPED, p.position });
            }
            p.velocity_y -= d.gravity * dt;
            p.position.y += p.velocity_y * dt;

            if (p.position.y <= lvl.player_eye_height) {
                p.position.y  = lvl.player_eye_height;
                p.velocity_y  = 0.0f;
                p.is_grounded = true;
            }

            if (p.recoil_offset > 0.0f) p.recoil_offset  = std::max(0.0f, p.recoil_offset - dt * 2.5f);
            if (p.muzzle_flash  > 0.0f) p.muzzle_flash  -= dt;
            if (p.fire_cooldown > 0.0f) p.fire_cooldown -= dt;
            if (p.damage_flash  > 0.0f) p.damage_flash  -= dt;
        }

        // --- Step 2: hitscan fire --------------------------------------------
        inline void step_hitscan(WorldSnapshot& w, const PlayerCommandFrame& input,
                                 const config::Difficulty& d, float dt,
                                 std::pmr::vector<CombatEvent>& events) {
            (void)dt; // hitscan is instantaneous; dt kept for signature uniformity
            PlayerSnapshot& p = w.player;
            BotTableSoA& bots = w.bots;

            if (!(input.fire_pressed && p.fire_cooldown <= 0.0f)) return;

            p.fire_cooldown = d.fire_cooldown;
            p.recoil_offset = d.recoil_offset;
            p.muzzle_flash  = d.muzzle_flash_time;

            const glm::vec3 eye = p.position;
            const glm::vec3 dir = p.get_forward();

            events.push_back({ EventType::PLAYER_FIRED, eye });

            int   hit_idx  = -1;
            float closest_t = 1e6f;

            for (size_t i = 0; i < bots.size(); ++i) {
                if (bots.state[i] == BotState::DEAD) continue;

                const glm::vec3 chest = bots.position[i] + glm::vec3(0, 0.95f, 0);
                const glm::vec3 head  = bots.position[i] + glm::vec3(0, 1.55f, 0);

                float t_c = 1e6f, t_h = 1e6f;
                const bool hit_c = detail::ray_sphere_intersect(eye, dir, chest, 0.55f, t_c);
                const bool hit_h = detail::ray_sphere_intersect(eye, dir, head, 0.30f, t_h);

                float t_best = 1e6f;
                if (hit_c) t_best = std::min(t_best, t_c);
                if (hit_h) t_best = std::min(t_best, t_h);

                if (t_best < closest_t) {
                    closest_t = t_best;
                    hit_idx   = static_cast<int>(i);
                }
            }

            const glm::vec3 hit_pos      = (hit_idx >= 0) ? (eye + dir * closest_t) : (eye + dir * 60.0f);
            const glm::vec3 muzzle_world = eye + dir * 0.4f + p.get_right() * 0.18f - glm::vec3(0, 0.12f, 0);
            w.tracers.push_back({ muzzle_world, hit_pos, 0.06f });

            if (hit_idx >= 0) {
                const size_t idx = static_cast<size_t>(hit_idx);
                bots.hp[idx]            -= static_cast<int16_t>(d.fire_damage);
                bots.hit_flash_time[idx] = 0.12f;
                events.push_back({ EventType::BOT_HIT, hit_pos, hit_idx });

                if (bots.hp[idx] <= 0 && bots.state[idx] != BotState::DEAD) {
                    bots.state[idx]        = BotState::DEAD;
                    bots.respawn_time[idx] = d.respawn_delay;
                    events.push_back({ EventType::BOT_KILLED, bots.position[idx], hit_idx });
                }
            }
        }

        // --- Step 3: bot AI FSM (PATROL / CHASE / ATTACK / DEAD) --------------
        inline void step_bot_ai(WorldSnapshot& w, const config::Difficulty& d,
                                const config::FpsLevel01& lvl, float dt,
                                std::pmr::vector<CombatEvent>& events) {
            const PlayerSnapshot& p = w.player;
            BotTableSoA& bots       = w.bots;
            ProjectileTableSoA& projs = w.projectiles;

            for (size_t i = 0; i < bots.size(); ++i) {
                bots.bob_phase[i] += dt * 3.0f;

                if (bots.state[i] == BotState::DEAD) {
                    bots.respawn_time[i] -= dt;
                    if (bots.respawn_time[i] <= 0.0f) {
                        bots.state[i]          = BotState::PATROL;
                        bots.hp[i]             = d.bot_max_hp;
                        bots.hit_flash_time[i] = 0.0f;
                    }
                    continue;
                }

                if (bots.hit_flash_time[i] > 0.0f)  bots.hit_flash_time[i]  -= dt;
                if (bots.attack_cooldown[i] > 0.0f) bots.attack_cooldown[i] -= dt;

                const glm::vec3 to_player      = p.position - bots.position[i];
                const float     dist_to_player = glm::length(glm::vec2(to_player.x, to_player.z));
                bots.yaw[i]                    = std::atan2(to_player.x, to_player.z);

                if (dist_to_player > d.chase_range)      bots.state[i] = BotState::PATROL;
                else if (dist_to_player > d.attack_range) bots.state[i] = BotState::CHASE;
                else                                      bots.state[i] = BotState::ATTACK;

                const glm::vec3 fwd(std::sin(bots.yaw[i]), 0.0f, std::cos(bots.yaw[i]));
                const glm::vec3 right(std::cos(bots.yaw[i]), 0.0f, -std::sin(bots.yaw[i]));

                switch (bots.state[i]) {
                case BotState::PATROL: {
                    glm::vec3 to_wp = bots.target_waypoint[i] - bots.position[i];
                    if (glm::length(to_wp) < 1.0f) {
                        const float r1 = detail::next_rand01(w.rng_state);
                        const float r2 = detail::next_rand01(w.rng_state);
                        bots.target_waypoint[i] = glm::vec3(r1 * 24.0f - 12.0f, 0.0f, r2 * 24.0f - 12.0f);
                        to_wp = bots.target_waypoint[i] - bots.position[i];
                    }
                    bots.position[i] += glm::normalize(to_wp) * (d.bot_speed * 0.5f) * dt;
                    break;
                }
                case BotState::CHASE: {
                    bots.position[i] += fwd * d.bot_speed * dt;
                    break;
                }
                case BotState::ATTACK: {
                    if (detail::next_rand01(w.rng_state) < 0.02f) {
                        bots.strafe_dir[i] = -bots.strafe_dir[i];
                    }
                    bots.position[i] += right * (bots.strafe_dir[i] * d.bot_speed * 0.8f) * dt;

                    if (bots.attack_cooldown[i] <= 0.0f) {
                        bots.attack_cooldown[i] = d.attack_cooldown_base
                                                + detail::next_rand01(w.rng_state) * d.attack_cooldown_jitter;
                        const glm::vec3 muzzle  = bots.position[i] + glm::vec3(0, 0.95f, 0) + fwd * 0.5f;
                        const glm::vec3 aim_dir = glm::normalize((p.position + glm::vec3(0, 0.2f, 0)) - muzzle);
                        projs.add(muzzle, aim_dir * d.projectile_speed, d.projectile_lifetime);
                        events.push_back({ EventType::BOT_FIRED, muzzle, static_cast<int>(i) });
                    }
                    break;
                }
                case BotState::DEAD:
                    break;
                }

                const float bound = lvl.bot_clamp_bound();
                bots.position[i].x = glm::clamp(bots.position[i].x, -bound, bound);
                bots.position[i].z = glm::clamp(bots.position[i].z, -bound, bound);
            }
        }

        // --- Step 4: projectile integration + player collision -----------------
        inline void step_projectiles(WorldSnapshot& w, const config::Difficulty& d,
                                     const config::FpsLevel01& lvl, float dt,
                                     std::pmr::vector<CombatEvent>& events) {
            PlayerSnapshot& p     = w.player;
            ProjectileTableSoA& projs = w.projectiles;
            const float arena_bound = lvl.arena_half_size;

            for (size_t i = 0; i < projs.size();) {
                projs.position[i] += projs.velocity[i] * dt;
                projs.life[i]     -= dt;

                const float dist = glm::length(projs.position[i] - (p.position - glm::vec3(0, 0.6f, 0)));
                if (dist < 0.85f) {
                    p.hp -= static_cast<int16_t>(d.projectile_damage);
                    p.damage_flash = 0.22f;
                    events.push_back({ EventType::PLAYER_DAMAGED, p.position });
                    if (p.hp <= 0) {
                        p.hp         = 100;
                        p.position   = lvl.player_spawn_position;
                        p.velocity_y = 0.0f;
                    }
                    projs.remove_at(i);
                    continue;
                }

                if (projs.life[i] <= 0.0f
                    || std::abs(projs.position[i].x) > arena_bound
                    || std::abs(projs.position[i].z) > arena_bound) {
                    projs.remove_at(i);
                } else {
                    ++i;
                }
            }
        }

        // --- Step 5: tracer decay ------------------------------------------------
        inline void step_tracers(const WorldSnapshot& prev, WorldSnapshot& next, float dt) {
            for (const auto& tr : prev.tracers) {
                if (tr.life - dt > 0.0f) {
                    next.tracers.push_back({ tr.start, tr.end, tr.life - dt });
                }
            }
        }

    } // namespace detail

    inline WorldStepResult reduce_world(
        const WorldSnapshot&         prev,
        std::span<const UserCommand> commands,
        const config::Difficulty&    diff,
        const config::FpsLevel01&    level,
        float                        dt,
        std::pmr::memory_resource*   frame_arena
    ) {
        WorldStepResult result(std::pmr::get_default_resource(), frame_arena);
        result.next_world.player       = prev.player;
        result.next_world.bots         = prev.bots;
        result.next_world.projectiles  = prev.projectiles;
        result.next_world.rng_state    = prev.rng_state;

        const PlayerCommandFrame input = reduce_user_commands(commands);

        detail::step_player(result.next_world, input, diff, level, dt, result.events);
        detail::step_hitscan(result.next_world, input, diff, dt, result.events);
        detail::step_bot_ai(result.next_world, diff, level, dt, result.events);
        detail::step_projectiles(result.next_world, diff, level, dt, result.events);
        detail::step_tracers(prev, result.next_world, dt);

        return result;
    }

} // namespace fps::matrix