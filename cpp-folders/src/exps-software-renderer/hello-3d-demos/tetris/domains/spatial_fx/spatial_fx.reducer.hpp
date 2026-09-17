#pragma once
// tetris/domains/spatial_fx/spatial_fx.reducer.hpp — FX STEP (tetris::spatial_fx)
// Event-fed particle physics + camera spring/pulse + shockwave rings.
// Deterministic xorshift replaces rand() (headless reproducibility).
// Consumes BOTH matrix raw facts and progression derived events — listeners
// only; never touches the grid or score state (Rule 8.1).
#include <algorithm>
#include <cstdint>
#include <span>

#include <domains/matrix/matrix.contract.hpp>
#include <domains/matrix/matrix.event.hpp>
#include <domains/progression/progression.event.hpp>
#include <domains/spatial_fx/spatial_fx.contract.hpp>

namespace tetris::spatial_fx {
using tetris::matrix::MatrixEvent;
using tetris::matrix::MatrixEventType;
using tetris::matrix::GRID_W;
using tetris::progression::ProgressionEvent;
using tetris::progression::ProgressionEventType;

    static inline uint32_t fx_rand(uint32_t& s) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }

    // Amber spark trail along the hard-drop column (speed feel).
    static inline void spark_trail(FxState& fx, const MatrixEvent& ev) {
        const float x = ev.world_position.x;
        const int cells = std::min(static_cast<int>(ev.cells), 18);
        for (int i = 0; i < cells; ++i) {
            for (int k = 0; k < 2; ++k) {
                glm::vec3 p(x + ((float)(fx_rand(fx.rng_state) % 100) / 100.0f - 0.5f),
                            ev.world_position.y + (float)i + 0.5f, 0.0f);
                glm::vec3 vel(((float)(fx_rand(fx.rng_state) % 100) / 50.0f - 1.0f) * 2.0f,
                              1.0f + (float)(fx_rand(fx.rng_state) % 100) / 60.0f,
                              -0.5f);
                fx.particles.add(p, vel, shs::render::Color{ 255, 190, 80, 255 }, 0.35f);
            }
        }
    }

    // --- L3 dig-feel recipes (Garbage Canyon) --------------------------------

    // Canyon rubble palette: dust browns + grays.
    static const shs::render::Color RUBBLE_PAL[4] = {
        shs::render::Color{ 138, 106,  74, 255 },   // mud brown
        shs::render::Color{ 170, 140, 100, 255 },   // dust tan
        shs::render::Color{ 128, 118, 108, 255 },   // gray rock
        shs::render::Color{ 196, 164, 120, 255 }    // pale sand
    };

    // Dust bursts + rubble debris scaled to the cleared garbage mass; screen
    // rumble grows with the mass too (multi-row collapses hit harder).
    static inline void rubble_burst(FxState& fx, const MatrixEvent& ev) {
        const int mass = std::min(static_cast<int>(ev.garbage_cells), 40);
        if (mass <= 0) return;
        for (int i = 0; i < mass * 2; ++i) {
            const int   row_i = static_cast<int>(fx_rand(fx.rng_state)) % std::max(1, (int)ev.lines_cleared_count);
            const float row_y = (float)ev.cleared_rows[row_i];
            glm::vec3 p((float)(fx_rand(fx.rng_state) % GRID_W) - 4.5f, row_y + 0.4f,
                        (float)(fx_rand(fx.rng_state) % 100) / 60.0f - 0.8f);
            glm::vec3 vel(((float)(fx_rand(fx.rng_state) % 100) / 25.0f - 2.0f),
                          1.5f + (float)(fx_rand(fx.rng_state) % 100) / 45.0f,
                          -1.0f - (float)(fx_rand(fx.rng_state) % 100) / 50.0f);
            fx.particles.add(p, vel, RUBBLE_PAL[fx_rand(fx.rng_state) % 4], 1.0f);
        }
        // Rumble scaled to excavated mass (heavier collapses shake harder).
        const float rumble = 0.30f + 0.05f * (float)mass;
        fx.camera_shake = std::max(fx.camera_shake, std::min(rumble, 0.9f));
    }

    // Horizontal dust wave sweeping outward on 3+ row collapses.
    static inline void dust_wave(FxState& fx, const MatrixEvent& ev) {
        const float row_y = (float)ev.cleared_rows[0] + 0.5f;
        for (int col = 0; col < GRID_W; ++col) {
            const float side = (col < GRID_W / 2) ? -1.0f : 1.0f;
            glm::vec3 p((float)col - 4.5f, row_y, 0.2f);
            glm::vec3 vel(side * (6.0f + (float)(fx_rand(fx.rng_state) % 100) / 25.0f),
                          0.8f + (float)(fx_rand(fx.rng_state) % 100) / 60.0f,
                          -0.4f);
            fx.particles.add(p, vel, RUBBLE_PAL[3], 0.9f);
        }
    }

    // Pebble trickle falling from disturbed rows just above the clear.
    static inline void pebble_trickle(FxState& fx, const MatrixEvent& ev) {
        uint8_t lowest = ev.cleared_rows[0];
        for (int i = 1; i < ev.lines_cleared_count && i < 4; ++i) {
            lowest = std::max(lowest, ev.cleared_rows[i]);
        }
        for (int k = 0; k < 6; ++k) {
            const float py = (float)lowest + 1.0f + (float)(k % 3);
            glm::vec3 p((float)(fx_rand(fx.rng_state) % GRID_W) - 4.5f, py, 0.1f);
            glm::vec3 vel((float)(fx_rand(fx.rng_state) % 100) / 80.0f - 0.6f,
                          -1.0f - (float)(fx_rand(fx.rng_state) % 100) / 90.0f,
                          0.0f);
            fx.particles.add(p, vel, RUBBLE_PAL[2], 0.8f);
        }
    }

    // Small dust puff where a piece locks/slams against canyon rubble.
    static inline void impact_dust_puff(FxState& fx, const MatrixEvent& ev) {
        for (int i = 0; i < 6; ++i) {
            glm::vec3 p(ev.world_position.x + ((float)(fx_rand(fx.rng_state) % 100) / 40.0f - 1.2f),
                        ev.world_position.y - 0.3f,
                        0.1f);
            glm::vec3 vel(((float)(fx_rand(fx.rng_state) % 100) / 35.0f - 1.4f),
                          0.8f + (float)(fx_rand(fx.rng_state) % 100) / 70.0f,
                          -0.3f);
            fx.particles.add(p, vel, RUBBLE_PAL[1], 0.55f);
        }
    }

    // --- L4 cyber-storm recipes (special-piece detonations) ------------------

    // Bomb: radial fireball — orange core, yellow mid, white-hot flecks, plus
    // an expanding shockwave ring and a heavy shake.
    static inline void bomb_blast(FxState& fx, const MatrixEvent& ev) {
        for (int i = 0; i < 90; ++i) {
            const float ang = (float)(fx_rand(fx.rng_state) % 628) / 100.0f;
            const float spd = 2.5f + (float)(fx_rand(fx.rng_state) % 100) / 22.0f;
            glm::vec3 vel(std::cos(ang) * spd, std::sin(ang) * spd * 0.8f,
                          -1.0f - (float)(fx_rand(fx.rng_state) % 100) / 60.0f);
            shs::render::Color c = (i % 7 == 0) ? shs::render::Color{ 255, 245, 200, 255 }
                         : (i % 3 == 0) ? shs::render::Color{ 255, 210, 70, 255 }
                                        : shs::render::Color{ 255, 120, 30, 255 };
            fx.particles.add(ev.world_position, vel, c, 0.9f);
        }
        fx.rings.add(ev.world_position, 0.4f, 14.0f,
                     shs::render::Color{ 255, 160, 60, 255 }, 0.55f);
        fx.camera_shake = std::max(fx.camera_shake, 0.75f);
        fx.screen_flash = std::max(fx.screen_flash, 0.55f);
    }

    // Laser: horizontal beam sweep along the locked row — bright streaks
    // racing outward from the anchor with magenta/white cores.
    static inline void laser_sweep(FxState& fx, const MatrixEvent& ev) {
        const float row_y = (float)ev.lock_y + 0.5f;
        for (int col = 0; col < GRID_W; ++col) {
            const int   reps = 2 + (int)(fx_rand(fx.rng_state) % 2);
            for (int k = 0; k < reps; ++k) {
                glm::vec3 p((float)col - 4.5f, row_y + ((float)(fx_rand(fx.rng_state) % 40) - 20.0f) / 40.0f, 0.15f);
                const float dir = (col >= ev.lock_x) ? 1.0f : -1.0f;
                glm::vec3 vel(dir * (9.0f + (float)(fx_rand(fx.rng_state) % 100) / 18.0f),
                              (float)(fx_rand(fx.rng_state) % 30) / 30.0f - 0.05f, 0.0f);
                shs::render::Color c = (k == 0) ? shs::render::Color{ 255, 240, 250, 255 }
                                        : shs::render::Color{ 255,  60, 200, 255 };
                fx.particles.add(p, vel, c, 0.45f);
            }
        }
        fx.camera_shake = std::max(fx.camera_shake, 0.35f);
        fx.screen_flash = std::max(fx.screen_flash, 0.35f);
    }

    // Freeze: time-stop frost — slow ice crystals drifting down over the well
    // plus a cold expanding ring (no shake; the world goes quiet).
    static inline void frost_bloom(FxState& fx, const MatrixEvent& ev) {
        for (int i = 0; i < 70; ++i) {
            glm::vec3 p((float)(fx_rand(fx.rng_state) % 110) / 10.0f - 5.5f,
                        (float)(fx_rand(fx.rng_state) % 190) / 10.0f + 0.5f,
                        (float)(fx_rand(fx.rng_state) % 100) / 50.0f - 1.0f);
            glm::vec3 vel((float)(fx_rand(fx.rng_state) % 20) / 20.0f - 0.5f,
                          -0.8f - (float)(fx_rand(fx.rng_state) % 40) / 40.0f, 0.0f);
            shs::render::Color c = (i % 4 == 0) ? shs::render::Color{ 230, 250, 255, 255 }
                                        : shs::render::Color{ 140, 230, 255, 255 };
            fx.particles.add(p, vel, c, 1.8f);
        }
        fx.rings.add(glm::vec3(0.0f, 9.5f, 0.4f), 1.0f, 10.0f,
                     shs::render::Color{ 140, 230, 255, 255 }, 0.9f);
    }

    // Golden confetti/firework burst on the victory crescendo ("photo finish").
    static inline void victory_fireworks(FxState& fx) {
        static const shs::render::Color GOLD[] = {
            shs::render::Color{ 255, 215,  80, 255 },
            shs::render::Color{ 255, 170,  40, 255 },
            shs::render::Color{ 255, 240, 180, 255 },
            shs::render::Color{ 255, 225,  45, 255 }
        };
        for (int i = 0; i < 140; ++i) {
            glm::vec3 p((float)(fx_rand(fx.rng_state) % 200) / 10.0f - 10.0f,
                        (float)(fx_rand(fx.rng_state) % 160) / 10.0f + 2.0f,
                        (float)(fx_rand(fx.rng_state) % 100) / 40.0f - 1.0f);
            glm::vec3 vel((float)(fx_rand(fx.rng_state) % 100) / 20.0f - 2.5f,
                          4.0f + (float)(fx_rand(fx.rng_state) % 100) / 25.0f,
                          (float)(fx_rand(fx.rng_state) % 100) / 50.0f - 1.0f);
            fx.particles.add(p, vel, GOLD[fx_rand(fx.rng_state) % 4], 1.6f);
        }
    }

    // Steps FxState in place (long-lived PMR-backed particles/rings).
    static inline void step_fx(FxState& fx,
                               std::span<const MatrixEvent> matrix_events,
                               std::span<const ProgressionEvent> progression_events,
                               float dt) {
        fx.time += dt;
        if (fx.camera_shake > 0.0f) {
            fx.camera_shake = std::max(0.0f, fx.camera_shake - dt * 4.0f);
        }
        if (fx.camera_pulse > 0.0f) {
            fx.camera_pulse = std::max(0.0f, fx.camera_pulse - dt * 2.2f);
        }
        if (fx.screen_flash > 0.0f) {
            fx.screen_flash = std::max(0.0f, fx.screen_flash - dt * 2.8f);
        }
        // L5 victory crescendo: slow celebratory orbit (angle accumulates
        // only while the orbit is live — deterministic from event timing).
        if (fx.victory_orbit > 0.0f) {
            fx.victory_orbit = std::max(0.0f, fx.victory_orbit - dt);
            fx.orbit_elapsed += dt;
        }

        // --- Matrix raw facts --------------------------------------------------
        for (const auto& ev : matrix_events) {
            switch (ev.type) {
            case MatrixEventType::HARD_DROP_SLAM:
                fx.camera_shake = 0.35f;
                spark_trail(fx, ev);
                if (ev.garbage_cells > 0) impact_dust_puff(fx, ev);   // canyon dig feel
                break;
            case MatrixEventType::PIECE_LOCK_IMPACT:
                if (ev.garbage_cells > 0) impact_dust_puff(fx, ev);
                break;
            case MatrixEventType::SPECIAL_LOCKED:
                switch (ev.special_type) {
                case 9:  bomb_blast(fx, ev); break;    // PieceType::Bomb
                case 10: laser_sweep(fx, ev); break;   // PieceType::Laser
                case 11: frost_bloom(fx, ev); break;   // PieceType::Freeze
                default: break;
                }
                break;
            case MatrixEventType::GARBAGE_RAINED: {
                // L5 rain impact: tremor scaled to the volley + dust plumes
                // rising off the new bottom rows.
                fx.camera_shake = std::max(fx.camera_shake,
                                           0.45f + 0.15f * ev.rain_rows);
                const int rows = std::min<int>(ev.rain_rows, 4);
                for (int r = 0; r < rows; ++r) {
                    for (int col = 0; col < GRID_W; ++col) {
                        if ((col + r) % 3 == 0) continue;   // sparse plumes
                        glm::vec3 p((float)col - 4.5f, (float)r + 0.3f, 0.2f);
                        glm::vec3 vel(
                            ((float)(fx_rand(fx.rng_state) % 100) / 50.0f - 1.0f),
                            2.0f + ((float)(fx_rand(fx.rng_state) % 100) / 60.0f),
                            -1.0f - ((float)(fx_rand(fx.rng_state) % 100) / 80.0f));
                        fx.particles.add(p, vel, shs::render::Color{ 120, 100, 80, 255 }, 0.9f);
                    }
                }
                break;
            }
            case MatrixEventType::LINES_CLEARED: {
                const bool tetris = (ev.lines_cleared_count >= 4);
                fx.camera_shake = tetris ? 0.65f : 0.25f;
                if (tetris) fx.camera_pulse = 1.0f;   // zoom punch on 4-line clears
                for (int i = 0; i < ev.lines_cleared_count; ++i) {
                    float row_y = (float)ev.cleared_rows[i];
                    for (int col = 0; col < GRID_W; ++col) {
                        glm::vec3 p((float)col - 4.5f, row_y, 0.0f);
                        const float energy = tetris ? 1.6f : 1.0f;   // oversized burst
                        glm::vec3 vel(
                            ((col - 4.5f) * 1.2f) + ((float)(fx_rand(fx.rng_state) % 100) / 50.0f - 1.0f),
                            (3.0f + ((float)(fx_rand(fx.rng_state) % 100) / 30.0f)) * energy,
                            -2.5f - ((float)(fx_rand(fx.rng_state) % 100) / 40.0f)
                        );
                        // Gold flecks mixed into tetris bursts
                        shs::render::Color pc = (tetris && (col & 1) == 0)
                            ? shs::render::Color{ 255, 210, 70, 255 }
                            : shs::render::Color{ 40, 220, 240, 255 };
                        fx.particles.add(p, vel, pc, 1.2f);
                    }
                }
                // L3 dig feel: rubble scaled to cleared garbage mass, pebbles
                // trickling from disturbed rows above, and a horizontal dust
                // wave when a big collapse (3+ rows) tears through the canyon.
                if (ev.garbage_cells > 0) {
                    rubble_burst(fx, ev);
                    pebble_trickle(fx, ev);
                    if (ev.lines_cleared_count >= 3) dust_wave(fx, ev);
                }
                break;
            }
            default:
                break;
            }
        }

        // --- Progression derived events -----------------------------------------
        for (const auto& pev : progression_events) {
            switch (pev.type) {
            case ProgressionEventType::CLOCK_TICK:
                // Threshold shockwave ring from the board every 30-second tick.
                fx.rings.add(glm::vec3(0.0f, 9.5f, 0.4f), 1.0f, 16.0f,
                             shs::render::Color{ 120, 220, 255, 255 }, 0.8f);
                break;
            case ProgressionEventType::OBJECTIVE_COMPLETED:
                victory_fireworks(fx);
                fx.camera_pulse = 1.0f;
                if (fx.env_finale > 0.5f) fx.victory_orbit = 7.0f;   // L5 slow orbit
                break;
            default:
                break;
            }
        }

        // Integrate + compact particles (erase pattern preserved from original demo)
        auto& P = fx.particles;
        for (size_t i = 0; i < P.position.size();) {
            P.position[i] += P.velocity[i] * dt;
            P.velocity[i].y -= 18.0f * dt; // Gravity
            P.life[i] -= dt;
            if (P.life[i] <= 0.0f) {
                P.position.erase(P.position.begin() + i);
                P.velocity.erase(P.velocity.begin() + i);
                P.color.erase(P.color.begin() + i);
                P.life.erase(P.life.begin() + i);
            } else {
                ++i;
            }
        }

        // Integrate + compact rings
        auto& R = fx.rings;
        for (size_t i = 0; i < R.center.size();) {
            R.radius[i] += R.speed[i] * dt;
            R.life[i] -= dt;
            if (R.life[i] <= 0.0f) {
                R.center.erase(R.center.begin() + i);
                R.radius.erase(R.radius.begin() + i);
                R.speed.erase(R.speed.begin() + i);
                R.life.erase(R.life.begin() + i);
                R.max_life.erase(R.max_life.begin() + i);
                R.color.erase(R.color.begin() + i);
            } else {
                ++i;
            }
        }
    }

} // namespace tetris::spatial_fx