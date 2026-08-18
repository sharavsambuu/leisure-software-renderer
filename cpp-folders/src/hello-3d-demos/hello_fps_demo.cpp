#define SDL_MAIN_HANDLED

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <atomic>
#include <memory>
#include <thread>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include "shs_renderer.hpp"

// ============================================================================
// CONFIGURATION & CONSTANTS
// ============================================================================
static const int WINDOW_WIDTH          = 1280;
static const int WINDOW_HEIGHT         = 720;
static const int CANVAS_WIDTH          = 1280;
static const int CANVAS_HEIGHT         = 720;

// Reserve 2 CPU cores for OS & real-time audio thread to prevent underruns
static const int THREAD_COUNT = std::max(2u, std::thread::hardware_concurrency() > 2 ? std::thread::hardware_concurrency() - 2 : 2u);

static const int TILE_SIZE_X           = 80;
static const int TILE_SIZE_Y           = 80;

static const float Z_NEAR              = 0.15f;
static const float Z_FAR               = 200.0f;

static const float PLAYER_EYE_HEIGHT   = 1.70f; // 1.7m eye level
static const float PLAYER_SPEED        = 7.0f;  // m/s
static const float MOUSE_SENSITIVITY   = 0.0035f;

static const glm::vec3 SUN_DIR_WORLD   = glm::normalize(glm::vec3(0.45f, -0.85f, 0.35f));

// ============================================================================
// LOCK-FREE PROCEDURAL AUDIO SYNTHESIZER (CONSTITUTION II COMPLIANT)
// ============================================================================
enum SoundType : uint8_t {
    SND_NONE = 0,
    SND_PLAYER_SHOOT,
    SND_HITMARKER,
    SND_ENEMY_SHOOT,
    SND_ENEMY_EXPLODE,
    SND_PLAYER_HURT,
    SND_PLAYER_JUMP
};

// Lock-Free SPSC Ring Queue for Audio Triggers
struct AudioEventRing {
    static const uint32_t CAP = 64;
    SoundType buffer[CAP]{};
    alignas(64) std::atomic<uint32_t> write_idx{0};
    alignas(64) uint32_t read_idx{0};

    inline void push(SoundType type) {
        uint32_t wi = write_idx.load(std::memory_order_relaxed);
        buffer[wi % CAP] = type;
        write_idx.store(wi + 1, std::memory_order_release);
    }

    inline bool pop(SoundType& out) {
        uint32_t wi = write_idx.load(std::memory_order_acquire);
        if (read_idx == wi) return false;
        out = buffer[read_idx % CAP];
        read_idx++;
        return true;
    }
};

struct SoundVoice {
    SoundType type     = SND_NONE;
    float     time     = 0.0f;
    float     duration = 0.1f;
    float     phase    = 0.0f;
    bool      active   = false;
};

struct FPSSoundSynth {
    static const int MAX_VOICES = 16;
    SoundVoice voices[MAX_VOICES];
    AudioEventRing event_queue;

    uint32_t rng_state = 0x853c49e6u;
    inline float noise() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        return (float)(rng_state & 0xFFFF) / 32768.0f - 1.0f;
    }

    // Lock-free trigger from main thread
    inline void play(SoundType type) {
        event_queue.push(type);
    }

    // Mixing executed exclusively on the SDL audio thread
    void mix(float* stream, int frames, int channels, int sample_rate) {
        // 1. Process new sound triggers with smart voice stealing
        SoundType new_type;
        while (event_queue.pop(new_type)) {
            if (new_type == SND_NONE) continue;

            int slot = -1;
            float oldest_time = -1.0f;
            int oldest_slot = 0;

            for (int i = 0; i < MAX_VOICES; ++i) {
                if (!voices[i].active) {
                    slot = i;
                    break;
                }
                if (voices[i].type == new_type && voices[i].time > oldest_time) {
                    oldest_time = voices[i].time;
                    oldest_slot = i;
                }
            }

            if (slot < 0) slot = oldest_slot;

            SoundVoice& v = voices[slot];
            v.type   = new_type;
            v.time   = 0.0f;
            v.phase  = 0.0f;
            v.active = true;

            switch (new_type) {
                case SND_PLAYER_SHOOT:  v.duration = 0.10f; break;
                case SND_HITMARKER:     v.duration = 0.07f; break;
                case SND_ENEMY_SHOOT:   v.duration = 0.14f; break;
                case SND_ENEMY_EXPLODE: v.duration = 0.35f; break;
                case SND_PLAYER_HURT:   v.duration = 0.18f; break;
                case SND_PLAYER_JUMP:   v.duration = 0.11f; break;
                default: break;
            }
        }

        // 2. Synthesize audio frames
        float dt = 1.0f / (float)sample_rate;

        for (int f = 0; f < frames; ++f) {
            float mono_sample = 0.0f;
            int active_count = 0;

            for (int v = 0; v < MAX_VOICES; ++v) {
                if (!voices[v].active) continue;

                SoundVoice& vox = voices[v];
                vox.time += dt;
                float progress = vox.time / vox.duration;

                if (progress >= 1.0f) {
                    vox.active = false;
                    continue;
                }

                active_count++;

                // 2ms anti-click attack and cubic release
                float attack  = std::min(1.0f, vox.time / 0.002f);
                float release = (1.0f - progress);

                switch (vox.type) {
                    case SND_PLAYER_SHOOT: {
                        float freq = 120.0f + 850.0f * std::exp(-35.0f * vox.time);
                        vox.phase += freq * dt;
                        float env = attack * release * release;
                        float s = std::sin(vox.phase * glm::two_pi<float>());
                        if (vox.time < 0.010f) s += noise() * 0.25f;
                        mono_sample += s * env * 0.18f;
                        break;
                    }
                    case SND_HITMARKER: {
                        vox.phase += 2500.0f * dt;
                        float env = attack * std::exp(-55.0f * vox.time);
                        float s = std::sin(vox.phase * glm::two_pi<float>()) * 0.7f
                                + std::sin(vox.phase * 1.5f * glm::two_pi<float>()) * 0.3f;
                        mono_sample += s * env * 0.16f;
                        break;
                    }
                    case SND_ENEMY_SHOOT: {
                        float freq = 80.0f + 320.0f * std::exp(-18.0f * vox.time);
                        vox.phase += freq * dt;
                        float env = attack * release;
                        float s = std::sin(vox.phase * glm::two_pi<float>());
                        mono_sample += s * env * 0.14f;
                        break;
                    }
                    case SND_ENEMY_EXPLODE: {
                        float freq = 30.0f + 110.0f * std::exp(-9.0f * vox.time);
                        vox.phase += freq * dt;
                        float env = attack * std::exp(-8.0f * vox.time);
                        float s = std::sin(vox.phase * glm::two_pi<float>()) * 0.6f + noise() * 0.4f;
                        mono_sample += s * env * 0.25f;
                        break;
                    }
                    case SND_PLAYER_HURT: {
                        vox.phase += 75.0f * dt;
                        float env = attack * std::exp(-18.0f * vox.time);
                        mono_sample += (std::sin(vox.phase * glm::two_pi<float>()) + noise() * 0.25f) * env * 0.22f;
                        break;
                    }
                    case SND_PLAYER_JUMP: {
                        float freq = 150.0f + 280.0f * progress;
                        vox.phase += freq * dt;
                        float env = attack * release;
                        mono_sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.12f;
                        break;
                    }
                    default: break;
                }
            }

            // Headroom compression (prevents volume stacking distortion)
            if (active_count > 1) {
                mono_sample /= std::sqrt((float)active_count);
            }

            // Soft-clipping compression
            mono_sample = mono_sample / (1.0f + 0.8f * std::abs(mono_sample));

            for (int c = 0; c < channels; ++c) {
                stream[f * channels + c] = mono_sample;
            }
        }
    }
};

static FPSSoundSynth g_audio;

static void fps_audio_callback(void* userdata, Uint8* stream, int len) {
    FPSSoundSynth* synth = reinterpret_cast<FPSSoundSynth*>(userdata);
    float* out = reinterpret_cast<float*>(stream);
    const int channels = 2;
    int frames = len / (int)(sizeof(float) * channels);
    synth->mix(out, frames, channels, 44100);
}

// ============================================================================
// LOW-POLY GEOMETRY DATA TYPES & BUILDERS
// ============================================================================
struct LowPolyTriangle {
    glm::vec3  p0;
    glm::vec3  p1;
    glm::vec3  p2;
    shs::Color color;
    float      depth_bias;

    LowPolyTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, shs::Color col, float bias = 0.0f)
        : p0(a), p1(b), p2(c), color(col), depth_bias(bias) {}
};

namespace MeshBuilder {
    static inline void add_quad(std::vector<LowPolyTriangle>& tris,
                                glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
                                shs::Color c, float bias = 0.0f) {
        tris.emplace_back(v0, v1, v2, c, bias);
        tris.emplace_back(v0, v2, v3, c, bias);
    }

    static inline void add_box(std::vector<LowPolyTriangle>& tris,
                               glm::vec3 center, glm::vec3 size,
                               shs::Color c_top, shs::Color c_side, shs::Color c_bot,
                               float bias = 0.0f) {
        glm::vec3 h = size * 0.5f;
        glm::vec3 p000 = center + glm::vec3(-h.x, -h.y, -h.z);
        glm::vec3 p100 = center + glm::vec3( h.x, -h.y, -h.z);
        glm::vec3 p110 = center + glm::vec3( h.x,  h.y, -h.z);
        glm::vec3 p010 = center + glm::vec3(-h.x,  h.y, -h.z);
        glm::vec3 p001 = center + glm::vec3(-h.x, -h.y,  h.z);
        glm::vec3 p101 = center + glm::vec3( h.x, -h.y,  h.z);
        glm::vec3 p111 = center + glm::vec3( h.x,  h.y,  h.z);
        glm::vec3 p011 = center + glm::vec3(-h.x,  h.y,  h.z);

        add_quad(tris, p001, p101, p111, p011, c_side, bias); // Front (+Z)
        add_quad(tris, p100, p000, p010, p110, c_side, bias); // Back (-Z)
        add_quad(tris, p010, p011, p111, p110, c_top,  bias); // Top (+Y)
        add_quad(tris, p000, p100, p101, p001, c_bot,  bias); // Bottom (-Y)
        add_quad(tris, p100, p110, p111, p101, c_side, bias); // Right (+X)
        add_quad(tris, p000, p001, p011, p010, c_side, bias); // Left (-X)
    }

    static inline void add_cylinder(std::vector<LowPolyTriangle>& tris,
                                    glm::vec3 base_center, float radius, float height, int segments,
                                    shs::Color color) {
        glm::vec3 top_center = base_center + glm::vec3(0, height, 0);
        float step = glm::two_pi<float>() / (float)segments;

        for (int i = 0; i < segments; ++i) {
            float a0 = (float)i * step;
            float a1 = (float)(i + 1) * step;

            glm::vec3 b0 = base_center + glm::vec3(std::cos(a0) * radius, 0.0f, std::sin(a0) * radius);
            glm::vec3 b1 = base_center + glm::vec3(std::cos(a1) * radius, 0.0f, std::sin(a1) * radius);
            glm::vec3 t0 = b0 + glm::vec3(0, height, 0);
            glm::vec3 t1 = b1 + glm::vec3(0, height, 0);

            // Exterior side walls
            tris.emplace_back(b0, t0, t1, color, 0.0f);
            tris.emplace_back(b0, t1, b1, color, 0.0f);

            // Caps
            tris.emplace_back(top_center, t1, t0, color, 0.0f);
            tris.emplace_back(base_center, b0, b1, color, 0.0f);
        }
    }
}

// ============================================================================
// ARENA ENVIRONMENT GENERATOR
// ============================================================================
class ArenaWorld {
public:
    static std::vector<LowPolyTriangle> build_mesh() {
        std::vector<LowPolyTriangle> tris;

        const float ARENA_HALF_SIZE = 16.0f; // 32m x 32m
        const float WALL_HEIGHT     = 4.5f;

        // 1. Checkerboard Tiled Floor
        const int TILES = 16;
        const float TILE_SIZE = (ARENA_HALF_SIZE * 2.0f) / (float)TILES;

        shs::Color floor_dark  = shs::Color{45, 52, 60, 255};
        shs::Color floor_light = shs::Color{65, 75, 88, 255};

        for (int iz = 0; iz < TILES; ++iz) {
            float z0 = -ARENA_HALF_SIZE + (float)iz * TILE_SIZE;
            float z1 = z0 + TILE_SIZE;
            for (int ix = 0; ix < TILES; ++ix) {
                float x0 = -ARENA_HALF_SIZE + (float)ix * TILE_SIZE;
                float x1 = x0 + TILE_SIZE;

                shs::Color c = ((ix + iz) % 2 == 0) ? floor_dark : floor_light;

                glm::vec3 p00(x0, 0.0f, z0);
                glm::vec3 p10(x1, 0.0f, z0);
                glm::vec3 p11(x1, 0.0f, z1);
                glm::vec3 p01(x0, 0.0f, z1);

                MeshBuilder::add_quad(tris, p00, p01, p11, p10, c);
            }
        }

        // 2. Perimeter Inward-Facing Walls
        shs::Color wall_base = shs::Color{95, 105, 118, 255};
        shs::Color wall_trim = shs::Color{130, 140, 155, 255};
        float S = ARENA_HALF_SIZE;
        float H = WALL_HEIGHT;

        MeshBuilder::add_quad(tris, glm::vec3(-S, 0, -S), glm::vec3(-S, H, -S), glm::vec3( S, H, -S), glm::vec3( S, 0, -S), wall_base); // North
        MeshBuilder::add_quad(tris, glm::vec3( S, 0,  S), glm::vec3( S, H,  S), glm::vec3(-S, H,  S), glm::vec3(-S, 0,  S), wall_base); // South
        MeshBuilder::add_quad(tris, glm::vec3( S, 0, -S), glm::vec3( S, 0,  S), glm::vec3( S, H,  S), glm::vec3( S, H, -S), wall_base); // East
        MeshBuilder::add_quad(tris, glm::vec3(-S, 0,  S), glm::vec3(-S, 0, -S), glm::vec3(-S, H, -S), glm::vec3(-S, H,  S), wall_base); // West

        MeshBuilder::add_box(tris, glm::vec3(0, H + 0.15f, -S), glm::vec3(S * 2.0f, 0.3f, 0.6f), wall_trim, wall_trim, wall_trim);
        MeshBuilder::add_box(tris, glm::vec3(0, H + 0.15f,  S), glm::vec3(S * 2.0f, 0.3f, 0.6f), wall_trim, wall_trim, wall_trim);
        MeshBuilder::add_box(tris, glm::vec3( S, H + 0.15f, 0), glm::vec3(0.6f, 0.3f, S * 2.0f), wall_trim, wall_trim, wall_trim);
        MeshBuilder::add_box(tris, glm::vec3(-S, H + 0.15f, 0), glm::vec3(0.6f, 0.3f, S * 2.0f), wall_trim, wall_trim, wall_trim);

        // 3. Central Platform
        shs::Color plat_top  = shs::Color{180, 140, 80, 255};
        shs::Color plat_side = shs::Color{120, 95, 60, 255};
        MeshBuilder::add_box(tris, glm::vec3(0, 0.25f, 0), glm::vec3(7.0f, 0.5f, 7.0f), plat_top, plat_side, plat_side);

        // 4. Pillars
        shs::Color pillar_c = shs::Color{140, 145, 155, 255};
        float P_OFF = 8.5f;
        MeshBuilder::add_cylinder(tris, glm::vec3(-P_OFF, 0, -P_OFF), 1.1f, WALL_HEIGHT, 8, pillar_c);
        MeshBuilder::add_cylinder(tris, glm::vec3( P_OFF, 0, -P_OFF), 1.1f, WALL_HEIGHT, 8, pillar_c);
        MeshBuilder::add_cylinder(tris, glm::vec3(-P_OFF, 0,  P_OFF), 1.1f, WALL_HEIGHT, 8, pillar_c);
        MeshBuilder::add_cylinder(tris, glm::vec3( P_OFF, 0,  P_OFF), 1.1f, WALL_HEIGHT, 8, pillar_c);

        // 5. Tactical Crates
        shs::Color crate_wood = shs::Color{165, 110, 60, 255};
        shs::Color crate_dark = shs::Color{120, 75, 40, 255};
        MeshBuilder::add_box(tris, glm::vec3(-4.5f, 0.6f,  3.0f), glm::vec3(1.2f, 1.2f, 1.2f), crate_wood, crate_dark, crate_dark);
        MeshBuilder::add_box(tris, glm::vec3( 4.5f, 0.6f, -3.5f), glm::vec3(1.2f, 1.2f, 1.2f), crate_wood, crate_dark, crate_dark);
        MeshBuilder::add_box(tris, glm::vec3( 5.5f, 0.5f,  6.0f), glm::vec3(1.0f, 1.0f, 1.0f), crate_wood, crate_dark, crate_dark);

        return tris;
    }
};

// ============================================================================
// LOW-POLY ACTOR MESHES
// ============================================================================
class SyntheticActorMeshes {
public:
    static std::vector<LowPolyTriangle> build_bot_mesh(bool hit_flash) {
        std::vector<LowPolyTriangle> tris;

        shs::Color armor   = hit_flash ? shs::Color{255, 255, 255, 255} : shs::Color{60, 120, 190, 255};
        shs::Color joints  = hit_flash ? shs::Color{255, 200, 200, 255} : shs::Color{40, 45, 50, 255};
        shs::Color visor   = hit_flash ? shs::Color{255, 255, 255, 255} : shs::Color{240, 60, 50, 255};
        shs::Color metal   = hit_flash ? shs::Color{255, 255, 255, 255} : shs::Color{160, 170, 180, 255};

        MeshBuilder::add_box(tris, glm::vec3(0, 0.95f, 0), glm::vec3(0.65f, 0.70f, 0.38f), armor, armor, joints);
        MeshBuilder::add_box(tris, glm::vec3(0, 1.55f, 0), glm::vec3(0.42f, 0.42f, 0.42f), armor, metal, joints);
        MeshBuilder::add_box(tris, glm::vec3(0, 1.58f, 0.22f), glm::vec3(0.32f, 0.14f, 0.06f), visor, visor, visor, -0.001f);
        MeshBuilder::add_box(tris, glm::vec3(0, 1.05f, -0.24f), glm::vec3(0.35f, 0.45f, 0.15f), visor, metal, joints);

        MeshBuilder::add_box(tris, glm::vec3(-0.48f, 0.95f, 0.0f), glm::vec3(0.20f, 0.65f, 0.20f), metal, joints, metal);
        MeshBuilder::add_box(tris, glm::vec3( 0.48f, 0.95f, 0.0f), glm::vec3(0.20f, 0.65f, 0.20f), metal, joints, metal);

        MeshBuilder::add_box(tris, glm::vec3(-0.20f, 0.32f, 0.0f), glm::vec3(0.22f, 0.65f, 0.22f), metal, joints, joints);
        MeshBuilder::add_box(tris, glm::vec3( 0.20f, 0.32f, 0.0f), glm::vec3(0.22f, 0.65f, 0.22f), metal, joints, joints);

        return tris;
    }

    static std::vector<LowPolyTriangle> build_gun_mesh() {
        std::vector<LowPolyTriangle> tris;

        shs::Color metal_dark = shs::Color{45, 48, 55, 255};
        shs::Color metal_body = shs::Color{80, 85, 95, 255};
        shs::Color grip_wood  = shs::Color{130, 75, 45, 255};
        shs::Color glow_cyan  = shs::Color{40, 220, 240, 255};

        MeshBuilder::add_box(tris, glm::vec3(0, -0.15f, -0.05f), glm::vec3(0.08f, 0.25f, 0.12f), grip_wood, grip_wood, grip_wood);
        MeshBuilder::add_box(tris, glm::vec3(0, 0.02f, 0.08f), glm::vec3(0.10f, 0.12f, 0.35f), metal_body, metal_dark, metal_dark);
        MeshBuilder::add_box(tris, glm::vec3(0, 0.04f, 0.32f), glm::vec3(0.06f, 0.06f, 0.25f), metal_dark, metal_dark, metal_dark);
        MeshBuilder::add_box(tris, glm::vec3(0, 0.10f, 0.05f), glm::vec3(0.04f, 0.04f, 0.22f), glow_cyan, metal_dark, metal_dark);

        return tris;
    }

    static std::vector<LowPolyTriangle> build_muzzle_flash() {
        std::vector<LowPolyTriangle> tris;
        shs::Color c_bright = shs::Color{255, 240, 150, 255};
        shs::Color c_orange = shs::Color{255, 120, 30, 255};

        auto add_spike = [&](glm::vec3 dir, float len, float w) {
            glm::vec3 side = glm::normalize(glm::cross(dir, glm::vec3(0, 1, 0))) * w;
            glm::vec3 tip  = dir * len;
            tris.emplace_back(-side, side, tip, c_bright, -0.002f);
            tris.emplace_back(side, -side, tip, c_orange, -0.002f);
        };

        add_spike(glm::vec3(0, 0, 1.0f), 0.35f, 0.08f);
        add_spike(glm::vec3( 0.7f,  0.3f, 0.6f), 0.25f, 0.06f);
        add_spike(glm::vec3(-0.7f,  0.3f, 0.6f), 0.25f, 0.06f);
        add_spike(glm::vec3( 0.0f,  0.8f, 0.5f), 0.22f, 0.06f);
        add_spike(glm::vec3( 0.0f, -0.6f, 0.5f), 0.22f, 0.06f);

        return tris;
    }

    static std::vector<LowPolyTriangle> build_projectile_mesh() {
        std::vector<LowPolyTriangle> tris;
        shs::Color plasma_core   = shs::Color{255, 60, 40, 255};
        shs::Color plasma_orange = shs::Color{255, 180, 50, 255};
        MeshBuilder::add_box(tris, glm::vec3(0), glm::vec3(0.20f, 0.20f, 0.35f), plasma_orange, plasma_core, plasma_core);
        return tris;
    }
};

// ============================================================================
// ENEMY AI STATE MACHINE & COMBAT TYPES
// ============================================================================
enum class BotState { PATROL, CHASE, ATTACK, DEAD };

struct BotProjectile {
    glm::vec3 pos;
    glm::vec3 vel;
    float     life = 2.5f;
};

struct TargetBot {
    glm::vec3 position;
    glm::vec3 target_waypoint;
    float     yaw             = 0.0f;
    int       hp              = 100;
    BotState  state           = BotState::PATROL;
    float     hit_flash_time  = 0.0f;
    float     respawn_time    = 0.0f;
    float     attack_cooldown = 0.0f;
    float     bob_phase       = 0.0f;
    float     strafe_dir      = 1.0f;

    glm::vec3 get_chest_center() const { return position + glm::vec3(0, 0.95f, 0); }
    glm::vec3 get_head_center()  const { return position + glm::vec3(0, 1.55f, 0); }

    void update(float dt, const glm::vec3& player_pos, std::vector<BotProjectile>& projectiles) {
        bob_phase += dt * 3.0f;

        if (state == BotState::DEAD) {
            respawn_time -= dt;
            if (respawn_time <= 0.0f) {
                state = BotState::PATROL;
                hp    = 100;
                hit_flash_time = 0.0f;
            }
            return;
        }

        if (hit_flash_time > 0.0f)  hit_flash_time -= dt;
        if (attack_cooldown > 0.0f) attack_cooldown -= dt;

        glm::vec3 to_player = player_pos - position;
        float dist_to_player = glm::length(glm::vec2(to_player.x, to_player.z));
        yaw = std::atan2(to_player.x, to_player.z);

        const float CHASE_RANGE  = 18.0f;
        const float ATTACK_RANGE = 9.0f;
        const float BOT_SPEED    = 3.8f;

        if (dist_to_player > CHASE_RANGE) {
            state = BotState::PATROL;
        } else if (dist_to_player > ATTACK_RANGE) {
            state = BotState::CHASE;
        } else {
            state = BotState::ATTACK;
        }

        glm::vec3 fwd(std::sin(yaw), 0.0f, std::cos(yaw));
        glm::vec3 right(std::cos(yaw), 0.0f, -std::sin(yaw));

        switch (state) {
            case BotState::PATROL: {
                glm::vec3 to_wp = target_waypoint - position;
                if (glm::length(to_wp) < 1.0f) {
                    target_waypoint = glm::vec3((rand() % 24) - 12.0f, 0.0f, (rand() % 24) - 12.0f);
                }
                position += glm::normalize(to_wp) * (BOT_SPEED * 0.5f) * dt;
                break;
            }
            case BotState::CHASE: {
                position += fwd * BOT_SPEED * dt;
                break;
            }
            case BotState::ATTACK: {
                if ((rand() % 100) < 2) strafe_dir = -strafe_dir;
                position += right * (strafe_dir * BOT_SPEED * 0.8f) * dt;

                if (attack_cooldown <= 0.0f) {
                    attack_cooldown = 1.25f + ((rand() % 40) / 100.0f);
                    glm::vec3 muzzle = get_chest_center() + fwd * 0.5f;
                    glm::vec3 aim_dir = glm::normalize((player_pos + glm::vec3(0, 0.2f, 0)) - muzzle);
                    projectiles.push_back({ muzzle, aim_dir * 18.0f, 3.0f });
                    g_audio.play(SND_ENEMY_SHOOT);
                }
                break;
            }
            default: break;
        }

        position.x = glm::clamp(position.x, -14.0f, 14.0f);
        position.z = glm::clamp(position.z, -14.0f, 14.0f);
    }
};

struct BulletTracer {
    glm::vec3 start;
    glm::vec3 end;
    float     life = 0.08f;
};

struct HitscanResult {
    bool      hit = false;
    int       bot_index = -1;
    glm::vec3 hit_pos;
    float     distance = 1e6f;
};

static inline bool ray_sphere_intersect(const glm::vec3& ray_orig, const glm::vec3& ray_dir,
                                        const glm::vec3& sphere_center, float sphere_rad,
                                        float& out_t) {
    glm::vec3 oc = ray_orig - sphere_center;
    float b = glm::dot(oc, ray_dir);
    float c = glm::dot(oc, oc) - sphere_rad * sphere_rad;
    float disc = b * b - c;
    if (disc < 0.0f) return false;

    float sqrt_disc = std::sqrt(disc);
    float t0 = -b - sqrt_disc;
    float t1 = -b + sqrt_disc;

    if (t0 > 0.001f) { out_t = t0; return true; }
    if (t1 > 0.001f) { out_t = t1; return true; }
    return false;
}

static HitscanResult perform_hitscan(const glm::vec3& eye, const glm::vec3& forward,
                                     const std::vector<TargetBot>& bots) {
    HitscanResult res;

    for (size_t i = 0; i < bots.size(); ++i) {
        if (bots[i].state == BotState::DEAD) continue;

        float t_chest = 1e6f;
        float t_head  = 1e6f;
        bool hit_chest = ray_sphere_intersect(eye, forward, bots[i].get_chest_center(), 0.55f, t_chest);
        bool hit_head  = ray_sphere_intersect(eye, forward, bots[i].get_head_center(),  0.30f, t_head);

        float closest_t = 1e6f;
        if (hit_chest) closest_t = std::min(closest_t, t_chest);
        if (hit_head)  closest_t = std::min(closest_t, t_head);

        if (closest_t < res.distance) {
            res.hit       = true;
            res.bot_index = (int)i;
            res.distance  = closest_t;
            res.hit_pos   = eye + forward * closest_t;
        }
    }

    if (!res.hit) {
        res.hit_pos = eye + forward * 60.0f;
    }

    return res;
}

// ============================================================================
// PLAYER STATE WITH JUMPING & DAMAGE
// ============================================================================
struct FPSPlayer {
    glm::vec3 position           = glm::vec3(0.0f, PLAYER_EYE_HEIGHT, -8.0f);
    float     velocity_y         = 0.0f;
    bool      is_grounded        = true;

    float     yaw                = 0.0f;
    float     pitch              = 0.0f;

    int       hp                 = 100;
    int       max_hp             = 100;
    float     damage_flash_timer = 0.0f;
    float     fire_cooldown      = 0.0f;

    float     recoil_offset      = 0.0f;
    float     muzzle_flash_timer = 0.0f;
    int       score              = 0;

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

    void take_damage(int amount) {
        hp -= amount;
        damage_flash_timer = 0.22f;
        if (hp <= 0) {
            hp = 100;
            position = glm::vec3(0.0f, PLAYER_EYE_HEIGHT, -8.0f);
            velocity_y = 0.0f;
        }
    }

    void update(float dt) {
        const float GRAVITY = 24.0f;
        velocity_y -= GRAVITY * dt;
        position.y += velocity_y * dt;

        if (position.y <= PLAYER_EYE_HEIGHT) {
            position.y   = PLAYER_EYE_HEIGHT;
            velocity_y   = 0.0f;
            is_grounded  = true;
        }

        if (recoil_offset > 0.0f)       recoil_offset = std::max(0.0f, recoil_offset - dt * 2.5f);
        if (muzzle_flash_timer > 0.0f)  muzzle_flash_timer -= dt;
        if (fire_cooldown > 0.0f)       fire_cooldown -= dt;
        if (damage_flash_timer > 0.0f)  damage_flash_timer -= dt;
    }
};

// ============================================================================
// HIGH PRECISION PERSPECTIVE RASTERIZER
// ============================================================================
static inline glm::vec4 clip_to_screen_vec4(const glm::vec4& clip, int W, int H) {
    float inv_w = 1.0f / clip.w;
    glm::vec3 ndc = glm::vec3(clip) * inv_w;
    glm::vec4 s;
    s.x = (ndc.x + 1.0f) * 0.5f * (float)(W - 1);
    s.y = (1.0f - ndc.y) * 0.5f * (float)(H - 1);
    s.z = ndc.z;
    s.w = inv_w;
    return s;
}

static void rasterize_perspective_triangle_tile(
    shs::Canvas& canvas, shs::ZBuffer& z_buffer,
    const glm::vec4& sc0, const glm::vec4& sc1, const glm::vec4& sc2,
    shs::Color lit_color, float depth_bias,
    glm::ivec2 tile_min, glm::ivec2 tile_max)
{
    glm::vec2 v0(sc0.x, sc0.y);
    glm::vec2 v1(sc1.x, sc1.y);
    glm::vec2 v2(sc2.x, sc2.y);

    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (!shs::Raster::is_front_facing_screen(area, shs::Raster::FrontFace::CounterClockwise)) return;

    glm::vec2 bboxmin = glm::max(glm::vec2(tile_min), glm::min(v0, glm::min(v1, v2)));
    glm::vec2 bboxmax = glm::min(glm::vec2(tile_max), glm::max(v0, glm::max(v1, v2)));
    if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

    std::vector<glm::vec2> v2d = { v0, v1, v2 };
    int min_x = (int)bboxmin.x; int max_x = (int)bboxmax.x;
    int min_y = (int)bboxmin.y; int max_y = (int)bboxmax.y;

    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2((float)px + 0.5f, (float)py + 0.5f), v2d);
            if (bc.x < 0.0f || bc.y < 0.0f || bc.z < 0.0f) continue;

            float interp_z = shs::Raster::interpolate_ndc_depth(bc, sc0.z, sc1.z, sc2.z);
            float final_z = interp_z + depth_bias;
            if (final_z < -1.0f || final_z > 1.0f) continue;

            if (z_buffer.test_and_set_depth_screen_space(px, py, final_z)) {
                canvas.draw_pixel_screen_space(px, py, lit_color);
            }
        }
    }
}

// ============================================================================
// UI, 7-SEGMENT VECTOR NUMBERS & SCORE HUD
// ============================================================================
static void draw_rect_fill_screen(shs::Canvas& canvas, int x, int y, int w, int h, shs::Color col) {
    int x0 = std::max(0, x);
    int y0 = std::max(0, y);
    int x1 = std::min(canvas.get_width() - 1, x + w);
    int y1 = std::min(canvas.get_height() - 1, y + h);
    for (int py = y0; py <= y1; ++py) {
        for (int px = x0; px <= x1; ++px) {
            canvas.draw_pixel_screen_space(px, py, col);
        }
    }
}

static void draw_rect_border_screen(shs::Canvas& canvas, int x, int y, int w, int h, shs::Color col) {
    int x1 = std::min(canvas.get_width() - 1, x + w);
    int y1 = std::min(canvas.get_height() - 1, y + h);
    for (int px = std::max(0, x); px <= x1; ++px) {
        canvas.draw_pixel_screen_space(px, y, col);
        canvas.draw_pixel_screen_space(px, y1, col);
    }
    for (int py = std::max(0, y); py <= y1; ++py) {
        canvas.draw_pixel_screen_space(x, py, col);
        canvas.draw_pixel_screen_space(x1, py, col);
    }
}

static void draw_digit_screen(shs::Canvas& canvas, int x, int y, int d, int w, int h, shs::Color col) {
    static const uint8_t segs[10] = {
        0b00111111, // 0
        0b00000110, // 1
        0b01011011, // 2
        0b01001111, // 3
        0b01100110, // 4
        0b01101101, // 5
        0b01111101, // 6
        0b00000111, // 7
        0b01111111, // 8
        0b01101111  // 9
    };
    if (d < 0 || d > 9) return;
    uint8_t mask = segs[d];
    int mid_y = y + h / 2;

    auto line = [&](int x0, int y0, int x1, int y1) {
        shs::Canvas::draw_line(canvas, x0, y0, x1, y1, col);
    };

    if (mask & (1 << 0)) line(x, y, x + w, y);                 // a (top)
    if (mask & (1 << 1)) line(x + w, y, x + w, mid_y);         // b (top-right)
    if (mask & (1 << 2)) line(x + w, mid_y, x + w, y + h);     // c (bot-right)
    if (mask & (1 << 3)) line(x, y + h, x + w, y + h);         // d (bottom)
    if (mask & (1 << 4)) line(x, mid_y, x, y + h);             // e (bot-left)
    if (mask & (1 << 5)) line(x, y, x, mid_y);                 // f (top-left)
    if (mask & (1 << 6)) line(x, mid_y, x + w, mid_y);         // g (middle)
}

static void draw_number_screen(shs::Canvas& canvas, int x, int y, int val, int digits, shs::Color col) {
    int w = 10, h = 18, gap = 5;
    for (int i = digits - 1; i >= 0; --i) {
        int d = val % 10;
        val /= 10;
        draw_digit_screen(canvas, x + i * (w + gap), y, d, w, h, col);
    }
}

static void draw_enemy_health_bars(shs::Canvas& canvas, const glm::mat4& vp, const std::vector<TargetBot>& bots) {
    int W = canvas.get_width();
    int H = canvas.get_height();

    for (const auto& bot : bots) {
        if (bot.state == BotState::DEAD) continue;

        glm::vec3 head_top = bot.position + glm::vec3(0, 2.05f, 0);
        glm::vec4 clip = vp * glm::vec4(head_top, 1.0f);

        if (clip.w <= 0.15f) continue;

        glm::vec3 sc = shs::Canvas::clip_to_screen(clip, W, H);
        if (sc.x < -60 || sc.x > W + 60 || sc.y < -60 || sc.y > H + 60) continue;

        int bar_w = (int)glm::clamp(54.0f / (clip.w * 0.08f + 0.5f), 22.0f, 50.0f);
        int bar_h = 4;
        int bx = (int)sc.x - bar_w / 2;
        int by = (int)sc.y - 10;

        draw_rect_fill_screen(canvas, bx - 1, by - 1, bar_w + 2, bar_h + 2, shs::Color{10, 12, 16, 230});
        draw_rect_fill_screen(canvas, bx, by, bar_w, bar_h, shs::Color{80, 20, 20, 255});

        float hp_pct = glm::clamp((float)bot.hp / 100.0f, 0.0f, 1.0f);
        int fill_w   = (int)(hp_pct * (float)bar_w);

        shs::Color hp_col;
        if (bot.hit_flash_time > 0.0f) {
            hp_col = shs::Color{255, 255, 255, 255};
        } else if (hp_pct > 0.5f) {
            hp_col = shs::Color{240, 70, 50, 255};
        } else {
            hp_col = shs::Color{255, 25, 25, 255};
        }

        draw_rect_fill_screen(canvas, bx, by, fill_w, bar_h, hp_col);
    }
}

static void draw_fps_hud(shs::Canvas& canvas, const FPSPlayer& player, int kills, float hitmarker_timer) {
    int W  = canvas.get_width();
    int H  = canvas.get_height();
    int cx = W / 2;
    int cy = H / 2;

    // 1. Damage Screen Vignette
    if (player.damage_flash_timer > 0.0f) {
        shs::Color red_border{255, 30, 30, 200};
        for (int i = 0; i < 8; ++i) {
            draw_rect_border_screen(canvas, i, i, W - 1 - i * 2, H - 1 - i * 2, red_border);
        }
    }

    // 2. Score & Kill Counter (Top-Right HUD Card)
    int sc_x = W - 180, sc_y = 25;
    draw_rect_fill_screen(canvas, sc_x - 10, sc_y - 8, 165, 40, shs::Color{15, 18, 25, 230});
    draw_rect_border_screen(canvas, sc_x - 10, sc_y - 8, 165, 40, shs::Color{90, 100, 120, 255});
    draw_number_screen(canvas, sc_x, sc_y + 2, player.score, 6, shs::Color{255, 215, 60, 255});

    // 3. Player Health Gauge (Bottom-Left)
    int hp_x = 35;
    int hp_y = H - 55;
    int hp_w = 220;
    int hp_h = 18;

    draw_rect_fill_screen(canvas, hp_x - 4, hp_y - 4, hp_w + 8, hp_h + 8, shs::Color{15, 18, 25, 230});
    draw_rect_border_screen(canvas, hp_x - 4, hp_y - 4, hp_w + 8, hp_h + 8, shs::Color{90, 100, 120, 255});
    draw_rect_fill_screen(canvas, hp_x, hp_y, hp_w, hp_h, shs::Color{45, 20, 20, 255});

    float hp_ratio = glm::clamp((float)player.hp / (float)player.max_hp, 0.0f, 1.0f);
    int fill_w     = (int)(hp_ratio * (float)hp_w);
    shs::Color hp_fill = (player.hp > 35) ? shs::Color{45, 220, 95, 255} : shs::Color{240, 40, 40, 255};
    draw_rect_fill_screen(canvas, hp_x, hp_y, fill_w, hp_h, hp_fill);
    draw_number_screen(canvas, hp_x + hp_w + 14, hp_y, player.hp, 3, hp_fill);

    // 4. Crosshair & Hitmarker
    shs::Color ch_color = (hitmarker_timer > 0.0f) ? shs::Color{255, 50, 50, 255} : shs::Color{255, 255, 255, 220};
    int ch_size = 7, ch_gap = 3;
    shs::Canvas::draw_line(canvas, cx - ch_size - ch_gap, cy, cx - ch_gap, cy, ch_color);
    shs::Canvas::draw_line(canvas, cx + ch_gap, cy, cx + ch_size + ch_gap, cy, ch_color);
    shs::Canvas::draw_line(canvas, cx, cy - ch_size - ch_gap, cx, cy - ch_gap, ch_color);
    shs::Canvas::draw_line(canvas, cx, cy + ch_gap, cx, cy + ch_size + ch_gap, ch_color);

    if (hitmarker_timer > 0.0f) {
        shs::Color hm_col{255, 60, 60, 255};
        int s = 5;
        shs::Canvas::draw_line(canvas, cx - s, cy - s, cx - 2, cy - 2, hm_col);
        shs::Canvas::draw_line(canvas, cx + 2, cy + 2, cx + s, cy + s, hm_col);
        shs::Canvas::draw_line(canvas, cx + 2, cy - 2, cx + s, cy - s, hm_col);
        shs::Canvas::draw_line(canvas, cx - s, cy + s, cx - 2, cy + 2, hm_col);
    }
}

// ============================================================================
// MAIN APPLICATION
// ============================================================================
int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    // WSL2 Audio & Mouse Stability Hints [1.1.1, 1.1.8, 1.1.9, 1.2.4]
    SDL_setenv("PULSE_LATENCY_MSEC", "60", 1);
    SDL_SetHintWithPriority(SDL_HINT_MOUSE_RELATIVE_MODE_WARP, "1", SDL_HINT_OVERRIDE);
    SDL_SetHint(SDL_HINT_GRAB_KEYBOARD, "1");

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "SHS Renderer - Low-Poly FPS Combat Arena",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN
    );

    SDL_Renderer* sdl_renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* screen_texture = SDL_CreateTexture(
        sdl_renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        CANVAS_WIDTH, CANVAS_HEIGHT
    );
    SDL_Surface* screen_surface = SDL_CreateRGBSurfaceWithFormat(0, CANVAS_WIDTH, CANVAS_HEIGHT, 32, SDL_PIXELFORMAT_RGBA32);

    // Audio Setup (WSLg Native 44.1kHz Stereo with 4096-sample buffer) [1.1.1, 1.2.4]
    SDL_AudioSpec want{}, have{};
    want.freq     = 44100;
    want.format   = AUDIO_F32SYS;
    want.channels = 2;
    want.samples  = 4096;
    want.callback = fps_audio_callback;
    want.userdata = &g_audio;

    SDL_AudioDeviceID audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
    if (audio_dev) SDL_PauseAudioDevice(audio_dev, 0);

    SDL_SetRelativeMouseMode(SDL_TRUE);

    shs::Canvas canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20, 25, 35, 255});
    shs::ZBuffer z_buffer(CANVAS_WIDTH, CANVAS_HEIGHT, -1.0f, 1.0f);

    shs::Job::ThreadedPriorityJobSystem job_system(THREAD_COUNT);
    shs::Job::WaitGroup wg_render;

    std::cout << "Building Low-Poly FPS Combat Arena..." << std::endl;
    std::vector<LowPolyTriangle> arena_tris = ArenaWorld::build_mesh();
    std::vector<LowPolyTriangle> gun_tris   = SyntheticActorMeshes::build_gun_mesh();
    std::vector<LowPolyTriangle> flash_tris = SyntheticActorMeshes::build_muzzle_flash();
    std::vector<LowPolyTriangle> bolt_tris  = SyntheticActorMeshes::build_projectile_mesh();

    std::vector<TargetBot> bots = {
        { glm::vec3(-7.0f, 0.0f,  6.0f), glm::vec3(-7.0f, 0.0f,  6.0f) },
        { glm::vec3( 7.0f, 0.0f,  6.0f), glm::vec3( 7.0f, 0.0f,  6.0f) },
        { glm::vec3(-6.0f, 0.0f, -4.0f), glm::vec3(-6.0f, 0.0f, -4.0f) },
        { glm::vec3( 6.0f, 0.0f, -4.0f), glm::vec3( 6.0f, 0.0f, -4.0f) }
    };

    std::vector<BotProjectile> bot_projectiles;
    FPSPlayer player;
    std::vector<BulletTracer> tracers;
    float hitmarker_timer = 0.0f;
    int total_kills = 0;

    auto fire_weapon = [&]() {
        if (player.fire_cooldown > 0.0f) return;
        player.fire_cooldown      = 0.18f;
        player.recoil_offset      = 0.08f;
        player.muzzle_flash_timer = 0.05f;

        g_audio.play(SND_PLAYER_SHOOT);

        glm::vec3 eye = player.position;
        glm::vec3 dir = player.get_forward();

        HitscanResult hr = perform_hitscan(eye, dir, bots);
        if (hr.hit && hr.bot_index >= 0) {
            TargetBot& target = bots[hr.bot_index];
            target.hp -= 35;
            target.hit_flash_time = 0.12f;
            hitmarker_timer = 0.15f;
            g_audio.play(SND_HITMARKER);

            if (target.hp <= 0 && target.state != BotState::DEAD) {
                target.state = BotState::DEAD;
                target.respawn_time = 4.0f;
                player.score += 100;
                total_kills++;
                g_audio.play(SND_ENEMY_EXPLODE);
            }
        }

        glm::vec3 muzzle_world = eye + dir * 0.4f + player.get_right() * 0.18f - glm::vec3(0, 0.12f, 0);
        tracers.push_back({ muzzle_world, hr.hit_pos, 0.06f });
    };

    bool quit = false;
    SDL_Event e;
    Uint32 last_tick = SDL_GetTicks();
    int frame_count = 0;
    float fps_timer = 0.0f;

    while (!quit) {
        Uint32 current_tick = SDL_GetTicks();
        float dt = (current_tick - last_tick) / 1000.0f;
        last_tick = current_tick;
        if (dt > 0.1f) dt = 0.1f;

        // --------------------------------------------------------------------
        // INPUT PROCESSING
        // --------------------------------------------------------------------
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;

            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) quit = true;
                if (e.key.keysym.sym == SDLK_r) {
                    player.position = glm::vec3(0.0f, PLAYER_EYE_HEIGHT, -8.0f);
                    player.yaw = 0.0f; player.pitch = 0.0f; player.hp = 100;
                }
                if (e.key.keysym.sym == SDLK_f || e.key.keysym.sym == SDLK_LCTRL || 
                    e.key.keysym.sym == SDLK_RCTRL || e.key.keysym.sym == SDLK_RETURN) {
                    fire_weapon();
                }
                if (e.key.keysym.sym == SDLK_SPACE && player.is_grounded) {
                    player.velocity_y = 8.5f;
                    player.is_grounded = false;
                    g_audio.play(SND_PLAYER_JUMP);
                }
            }

            if (e.type == SDL_MOUSEBUTTONDOWN) {
                SDL_SetRelativeMouseMode(SDL_TRUE);
                if (e.button.button == SDL_BUTTON_LEFT) fire_weapon();
            }
        }

        // Relative Mouse Look
        int mouse_dx = 0, mouse_dy = 0;
        SDL_GetRelativeMouseState(&mouse_dx, &mouse_dy);
        if (mouse_dx != 0 || mouse_dy != 0) {
            player.yaw   += (float)mouse_dx * MOUSE_SENSITIVITY;
            player.pitch -= (float)mouse_dy * MOUSE_SENSITIVITY;
        }

        // Keyboard Look (Arrow Keys)
        const Uint8* keys = SDL_GetKeyboardState(NULL);
        if (keys[SDL_SCANCODE_LEFT])  player.yaw   -= 2.4f * dt;
        if (keys[SDL_SCANCODE_RIGHT]) player.yaw   += 2.4f * dt;
        if (keys[SDL_SCANCODE_UP])    player.pitch += 2.0f * dt;
        if (keys[SDL_SCANCODE_DOWN])  player.pitch -= 2.0f * dt;
        player.pitch = glm::clamp(player.pitch, -glm::radians(85.0f), glm::radians(85.0f));

        // Movement (WASD)
        glm::vec3 move_dir(0.0f);
        glm::vec3 fwd_xz = glm::normalize(glm::vec3(std::sin(player.yaw), 0.0f, std::cos(player.yaw)));
        glm::vec3 right_xz = glm::normalize(glm::vec3(std::cos(player.yaw), 0.0f, -std::sin(player.yaw)));

        if (keys[SDL_SCANCODE_W]) move_dir += fwd_xz;
        if (keys[SDL_SCANCODE_S]) move_dir -= fwd_xz;
        if (keys[SDL_SCANCODE_D]) move_dir += right_xz;
        if (keys[SDL_SCANCODE_A]) move_dir -= right_xz;

        if (glm::length(move_dir) > 0.01f) {
            move_dir = glm::normalize(move_dir);
            player.position += move_dir * PLAYER_SPEED * dt;
        }

        player.position.x = glm::clamp(player.position.x, -14.5f, 14.5f);
        player.position.z = glm::clamp(player.position.z, -14.5f, 14.5f);

        // Simulation Update
        player.update(dt);
        for (auto& bot : bots) bot.update(dt, player.position, bot_projectiles);
        if (hitmarker_timer > 0.0f) hitmarker_timer -= dt;

        // Projectile Update & Hit Detection
        for (size_t i = 0; i < bot_projectiles.size();) {
            bot_projectiles[i].pos += bot_projectiles[i].vel * dt;
            bot_projectiles[i].life -= dt;

            float dist_to_player = glm::length(bot_projectiles[i].pos - (player.position - glm::vec3(0, 0.6f, 0)));
            if (dist_to_player < 0.85f) {
                player.take_damage(15);
                g_audio.play(SND_PLAYER_HURT);
                bot_projectiles.erase(bot_projectiles.begin() + i);
                continue;
            }

            if (bot_projectiles[i].life <= 0.0f || std::abs(bot_projectiles[i].pos.x) > 16.0f || std::abs(bot_projectiles[i].pos.z) > 16.0f) {
                bot_projectiles.erase(bot_projectiles.begin() + i);
            } else {
                ++i;
            }
        }

        for (size_t i = 0; i < tracers.size();) {
            tracers[i].life -= dt;
            if (tracers[i].life <= 0.0f) tracers.erase(tracers.begin() + i);
            else ++i;
        }

        // --------------------------------------------------------------------
        // SCENE MATRICES & SHADING PASS
        // --------------------------------------------------------------------
        canvas.buffer().clear(shs::Color{22, 28, 38, 255});
        z_buffer.clear();

        glm::vec3 eye = player.position;
        glm::vec3 fwd = player.get_forward();
        glm::mat4 view = glm::lookAtLH(eye, eye + fwd, glm::vec3(0, 1, 0));
        glm::mat4 proj = glm::perspectiveLH_NO(glm::radians(75.0f), (float)CANVAS_WIDTH / (float)CANVAS_HEIGHT, Z_NEAR, Z_FAR);
        glm::mat4 vp   = proj * view;

        struct ProcessedTriangle {
            glm::vec4 c0, c1, c2;
            shs::Color lit_color;
            float depth_bias;
        };

        std::vector<ProcessedTriangle> active_tris;
        active_tris.reserve(arena_tris.size() + bots.size() * 120 + bot_projectiles.size() * 12 + 200);

        auto process_batch = [&](const std::vector<LowPolyTriangle>& batch_tris, const glm::mat4& model) {
            glm::mat4 mvp = vp * model;
            glm::vec3 L = -SUN_DIR_WORLD;

            for (const auto& tri : batch_tris) {
                glm::vec3 w0 = glm::vec3(model * glm::vec4(tri.p0, 1.0f));
                glm::vec3 w1 = glm::vec3(model * glm::vec4(tri.p1, 1.0f));
                glm::vec3 w2 = glm::vec3(model * glm::vec4(tri.p2, 1.0f));

                glm::vec3 N = glm::cross(w1 - w0, w2 - w0);
                float len = glm::length(N);
                if (len < 1e-6f) continue;
                N /= len;

                float NdotL   = std::max(0.0f, glm::dot(N, L));
                float diffuse = NdotL * 0.75f + 0.25f;
                float ambient = std::max(0.0f, N.y) * 0.20f + 0.15f;

                glm::vec3 base_col = glm::vec3(tri.color.r, tri.color.g, tri.color.b) / 255.0f;
                glm::vec3 lit_rgb  = base_col * (diffuse * glm::vec3(1.0f, 0.98f, 0.92f) + ambient * glm::vec3(0.50f, 0.70f, 1.0f));

                shs::Color final_c = shs::rgb01_to_color(lit_rgb);

                glm::vec4 c0 = mvp * glm::vec4(tri.p0, 1.0f);
                glm::vec4 c1 = mvp * glm::vec4(tri.p1, 1.0f);
                glm::vec4 c2 = mvp * glm::vec4(tri.p2, 1.0f);

                active_tris.push_back({ c0, c1, c2, final_c, tri.depth_bias });
            }
        };

        // 1. Arena
        process_batch(arena_tris, glm::mat4(1.0f));

        // 2. Target Bots
        for (const auto& bot : bots) {
            if (bot.state == BotState::DEAD) {
                glm::mat4 model = glm::translate(glm::mat4(1.0f), bot.position + glm::vec3(0, 0.2f, 0))
                                * glm::rotate(glm::mat4(1.0f), bot.yaw, glm::vec3(0, 1, 0))
                                * glm::rotate(glm::mat4(1.0f), glm::radians(-80.0f), glm::vec3(1, 0, 0));
                std::vector<LowPolyTriangle> dead_mesh = SyntheticActorMeshes::build_bot_mesh(false);
                process_batch(dead_mesh, model);
            } else {
                float hover_y = std::sin(bot.bob_phase) * 0.05f;
                glm::mat4 model = glm::translate(glm::mat4(1.0f), bot.position + glm::vec3(0, hover_y, 0))
                                * glm::rotate(glm::mat4(1.0f), bot.yaw, glm::vec3(0, 1, 0));
                std::vector<LowPolyTriangle> bot_mesh = SyntheticActorMeshes::build_bot_mesh(bot.hit_flash_time > 0.0f);
                process_batch(bot_mesh, model);
            }
        }

        // 3. Bot Projectiles
        for (const auto& bolt : bot_projectiles) {
            glm::mat4 bolt_model = glm::translate(glm::mat4(1.0f), bolt.pos)
                                 * glm::scale(glm::mat4(1.0f), glm::vec3(1.2f));
            process_batch(bolt_tris, bolt_model);
        }

        // 4. Viewmodel Gun & Muzzle Flash
        glm::vec3 gun_offset(0.22f, -0.22f, 0.45f - player.recoil_offset);
        glm::mat4 gun_rot = glm::rotate(glm::mat4(1.0f), player.yaw, glm::vec3(0, 1, 0))
                          * glm::rotate(glm::mat4(1.0f), player.pitch, glm::vec3(-1, 0, 0));
        glm::vec3 gun_world_pos = eye + glm::vec3(gun_rot * glm::vec4(gun_offset, 1.0f));
        glm::mat4 gun_model = glm::translate(glm::mat4(1.0f), gun_world_pos)
                            * gun_rot
                            * glm::scale(glm::mat4(1.0f), glm::vec3(0.9f));
        process_batch(gun_tris, gun_model);

        if (player.muzzle_flash_timer > 0.0f) {
            glm::mat4 flash_model = gun_model * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.04f, 0.46f));
            process_batch(flash_tris, flash_model);
        }

        // --------------------------------------------------------------------
        // MULTI-THREADED TILED RASTERIZATION
        // --------------------------------------------------------------------
        int W = canvas.get_width();
        int H = canvas.get_height();
        int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wg_render.reset();

        for (int ty = 0; ty < rows; ++ty) {
            for (int tx = 0; tx < cols; ++tx) {
                wg_render.add(1);
                job_system.submit({[&, tx, ty, W, H]() {
                    glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                    glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, W) - 1,
                                     std::min((ty + 1) * TILE_SIZE_Y, H) - 1);

                    for (const auto& tri : active_tris) {
                        const shs::Raster::FrustumClipPolygon poly =
                            shs::Raster::clip_triangle_to_frustum(tri.c0, tri.c1, tri.c2);
                        if (poly.count < 3) continue;

                        glm::vec4 s0 = clip_to_screen_vec4(poly.vertices[0], W, H);
                        for (int i = 1; i + 1 < poly.count; ++i) {
                            glm::vec4 s1 = clip_to_screen_vec4(poly.vertices[i], W, H);
                            glm::vec4 s2 = clip_to_screen_vec4(poly.vertices[i + 1], W, H);
                            rasterize_perspective_triangle_tile(canvas, z_buffer, s0, s1, s2, tri.lit_color, tri.depth_bias, t_min, t_max);
                        }
                    }

                    wg_render.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }

        wg_render.wait();

        // 5. Render Bullet Tracers
        for (const auto& tr : tracers) {
            glm::vec4 c_start = vp * glm::vec4(tr.start, 1.0f);
            glm::vec4 c_end   = vp * glm::vec4(tr.end,   1.0f);
            if (c_start.w > 0.1f && c_end.w > 0.1f) {
                glm::vec3 s0 = shs::Canvas::clip_to_screen(c_start, W, H);
                glm::vec3 s1 = shs::Canvas::clip_to_screen(c_end,   W, H);
                shs::Canvas::draw_line(canvas, (int)s0.x, (int)s0.y, (int)s1.x, (int)s1.y, shs::Color{255, 230, 100, 255});
            }
        }

        // 6. Draw 3D Floating Enemy Health Bars
        draw_enemy_health_bars(canvas, vp, bots);

        // 7. Draw HUD, Score, Crosshair & Player HP
        draw_fps_hud(canvas, player, total_kills, hitmarker_timer);

        // --------------------------------------------------------------------
        // SWAPCHAIN PRESENTATION
        // --------------------------------------------------------------------
        shs::Canvas::copy_to_SDLSurface(screen_surface, &canvas);
        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderClear(sdl_renderer);
        SDL_RenderCopy(sdl_renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(sdl_renderer);

        frame_count++;
        fps_timer += dt;
        if (fps_timer >= 0.5f) {
            int alive_count = 0;
            for (const auto& b : bots) if (b.state != BotState::DEAD) alive_count++;

            std::ostringstream ss;
            ss << "Low-Poly FPS Arena | FPS: " << (int)((float)frame_count / fps_timer)
               << " | HP: " << player.hp
               << " | Score: " << player.score
               << " | Kills: " << total_kills
               << " | Targets: " << alive_count << "/4"
               << " | [WASD: Move, Space: Jump, F/LMB: Fire, Arrows/Mouse: Aim]";
            SDL_SetWindowTitle(window, ss.str().c_str());
            frame_count = 0;
            fps_timer = 0.0f;
        }
    }

    if (audio_dev) SDL_CloseAudioDevice(audio_dev);
    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(screen_surface);
    SDL_DestroyRenderer(sdl_renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}