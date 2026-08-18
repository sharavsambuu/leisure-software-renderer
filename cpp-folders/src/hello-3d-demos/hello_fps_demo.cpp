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
#include <variant>
#include <span>
#include <memory_resource>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include "shs_renderer.hpp"

// ============================================================================
// CONFIGURATION & CONSTANTS (Constitution I)
// ============================================================================
static const int WINDOW_WIDTH          = 1280;
static const int WINDOW_HEIGHT         = 720;
static const int CANVAS_WIDTH          = 1280;
static const int CANVAS_HEIGHT         = 720;

// Reserve 2 CPU cores for OS scheduler and real-time audio threads
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
// PHASE 1: PMR FRAME MEMORY ARENA (Constitution II, Rule 6 & 8)
// ============================================================================
namespace vop {
    // Cross-platform, MSVC-safe linear bump allocator for per-frame transients
    class FrameMemoryResource : public std::pmr::memory_resource {
    public:
        static constexpr size_t CAPACITY = 8 * 1024 * 1024; // 8 MB

        FrameMemoryResource()
            : buffer_(std::make_unique<uint8_t[]>(CAPACITY)), offset_(0) {
        }

        inline void reset() noexcept {
            offset_ = 0; // True O(1) instant reset
        }

        inline std::pmr::memory_resource* get() noexcept { return this; }

    protected:
        void* do_allocate(size_t bytes, size_t alignment) override {
            size_t current = offset_;
            size_t aligned = (current + (alignment - 1)) & ~(alignment - 1);
            if (aligned + bytes > CAPACITY) {
                // Safe fallback to default heap if frame exceeds 8MB
                return std::pmr::get_default_resource()->allocate(bytes, alignment);
            }
            offset_ = aligned + bytes;
            return buffer_.get() + aligned;
        }

        void do_deallocate(void* p, size_t bytes, size_t alignment) noexcept override {
            // Fallback allocations are freed; arena allocations are no-op
            if (p < buffer_.get() || p >= buffer_.get() + CAPACITY) {
                std::pmr::get_default_resource()->deallocate(p, bytes, alignment);
            }
        }

        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return this == &other;
        }

    private:
        std::unique_ptr<uint8_t[]> buffer_;
        size_t offset_ = 0;
    };
}

// ============================================================================
// PHASE 2: LOCK-FREE AUDIO SYNTHESIZER (Wait-Free Audio Edge)
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

    inline void play(SoundType type) {
        event_queue.push(type);
    }

    void mix(float* stream, int frames, int channels, int sample_rate) {
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

            if (active_count > 1) {
                mono_sample /= std::sqrt((float)active_count);
            }
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
// PHASE 3: GEOMETRY & VALUE GENERATORS
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

        add_quad(tris, p001, p101, p111, p011, c_side, bias);
        add_quad(tris, p100, p000, p010, p110, c_side, bias);
        add_quad(tris, p010, p011, p111, p110, c_top,  bias);
        add_quad(tris, p000, p100, p101, p001, c_bot,  bias);
        add_quad(tris, p100, p110, p111, p101, c_side, bias);
        add_quad(tris, p000, p001, p011, p010, c_side, bias);
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

            tris.emplace_back(b0, t0, t1, color, 0.0f);
            tris.emplace_back(b0, t1, b1, color, 0.0f);

            tris.emplace_back(top_center, t1, t0, color, 0.0f);
            tris.emplace_back(base_center, b0, b1, color, 0.0f);
        }
    }
}

class ArenaWorld {
public:
    static std::vector<LowPolyTriangle> build_mesh() {
        std::vector<LowPolyTriangle> tris;

        const float ARENA_HALF_SIZE = 16.0f;
        const float WALL_HEIGHT     = 4.5f;

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

        shs::Color wall_base = shs::Color{95, 105, 118, 255};
        shs::Color wall_trim = shs::Color{130, 140, 155, 255};
        float S = ARENA_HALF_SIZE;
        float H = WALL_HEIGHT;

        MeshBuilder::add_quad(tris, glm::vec3(-S, 0, -S), glm::vec3(-S, H, -S), glm::vec3( S, H, -S), glm::vec3( S, 0, -S), wall_base);
        MeshBuilder::add_quad(tris, glm::vec3( S, 0,  S), glm::vec3( S, H,  S), glm::vec3(-S, H,  S), glm::vec3(-S, 0,  S), wall_base);
        MeshBuilder::add_quad(tris, glm::vec3( S, 0, -S), glm::vec3( S, 0,  S), glm::vec3( S, H,  S), glm::vec3( S, H, -S), wall_base);
        MeshBuilder::add_quad(tris, glm::vec3(-S, 0,  S), glm::vec3(-S, 0, -S), glm::vec3(-S, H, -S), glm::vec3(-S, H,  S), wall_base);

        MeshBuilder::add_box(tris, glm::vec3(0, H + 0.15f, -S), glm::vec3(S * 2.0f, 0.3f, 0.6f), wall_trim, wall_trim, wall_trim);
        MeshBuilder::add_box(tris, glm::vec3(0, H + 0.15f,  S), glm::vec3(S * 2.0f, 0.3f, 0.6f), wall_trim, wall_trim, wall_trim);
        MeshBuilder::add_box(tris, glm::vec3( S, H + 0.15f, 0), glm::vec3(0.6f, 0.3f, S * 2.0f), wall_trim, wall_trim, wall_trim);
        MeshBuilder::add_box(tris, glm::vec3(-S, H + 0.15f, 0), glm::vec3(0.6f, 0.3f, S * 2.0f), wall_trim, wall_trim, wall_trim);

        shs::Color plat_top  = shs::Color{180, 140, 80, 255};
        shs::Color plat_side = shs::Color{120, 95, 60, 255};
        MeshBuilder::add_box(tris, glm::vec3(0, 0.25f, 0), glm::vec3(7.0f, 0.5f, 7.0f), plat_top, plat_side, plat_side);

        shs::Color pillar_c = shs::Color{140, 145, 155, 255};
        float P_OFF = 8.5f;
        MeshBuilder::add_cylinder(tris, glm::vec3(-P_OFF, 0, -P_OFF), 1.1f, WALL_HEIGHT, 8, pillar_c);
        MeshBuilder::add_cylinder(tris, glm::vec3( P_OFF, 0, -P_OFF), 1.1f, WALL_HEIGHT, 8, pillar_c);
        MeshBuilder::add_cylinder(tris, glm::vec3(-P_OFF, 0,  P_OFF), 1.1f, WALL_HEIGHT, 8, pillar_c);
        MeshBuilder::add_cylinder(tris, glm::vec3( P_OFF, 0,  P_OFF), 1.1f, WALL_HEIGHT, 8, pillar_c);

        shs::Color crate_wood = shs::Color{165, 110, 60, 255};
        shs::Color crate_dark = shs::Color{120, 75, 40, 255};
        MeshBuilder::add_box(tris, glm::vec3(-4.5f, 0.6f,  3.0f), glm::vec3(1.2f, 1.2f, 1.2f), crate_wood, crate_dark, crate_dark);
        MeshBuilder::add_box(tris, glm::vec3( 4.5f, 0.6f, -3.5f), glm::vec3(1.2f, 1.2f, 1.2f), crate_wood, crate_dark, crate_dark);
        MeshBuilder::add_box(tris, glm::vec3( 5.5f, 0.5f,  6.0f), glm::vec3(1.0f, 1.0f, 1.0f), crate_wood, crate_dark, crate_dark);

        return tris;
    }
};

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
// PHASE 4: COMMAND PATTERN & ACTION SYSTEM
// ============================================================================
namespace vop {
    struct MoveIntent  { glm::vec3 direction_xz; };
    struct LookIntent  { float delta_yaw; float delta_pitch; };
    struct JumpIntent  {};
    struct FireIntent  {};
    struct ResetIntent {};

    using UserCommand = std::variant<MoveIntent, LookIntent, JumpIntent, FireIntent, ResetIntent>;

    struct PlayerCommandFrame {
        glm::vec3 move_dir{0.0f};
        float delta_yaw    = 0.0f;
        float delta_pitch  = 0.0f;
        bool  jump_pressed = false;
        bool  fire_pressed = false;
        bool  reset_pressed= false;
    };

    static inline PlayerCommandFrame reduce_user_commands(std::span<const UserCommand> commands) {
        PlayerCommandFrame out{};
        for (const auto& cmd : commands) {
            std::visit([&out](auto&& c) {
                using T = std::decay_t<decltype(c)>;
                if constexpr (std::is_same_v<T, MoveIntent>) {
                    out.move_dir += c.direction_xz;
                } else if constexpr (std::is_same_v<T, LookIntent>) {
                    out.delta_yaw   += c.delta_yaw;
                    out.delta_pitch += c.delta_pitch;
                } else if constexpr (std::is_same_v<T, JumpIntent>) {
                    out.jump_pressed = true;
                } else if constexpr (std::is_same_v<T, FireIntent>) {
                    out.fire_pressed = true;
                } else if constexpr (std::is_same_v<T, ResetIntent>) {
                    out.reset_pressed = true;
                }
            }, cmd);
        }
        if (glm::length(out.move_dir) > 0.01f) out.move_dir = glm::normalize(out.move_dir);
        return out;
    }
}

// ============================================================================
// PHASE 5: DOD SIMULATION & DATA MODELS (SoA)
// ============================================================================
namespace vop {
    enum class EventType : uint8_t {
        PLAYER_FIRED,
        BOT_HIT,
        BOT_KILLED,
        PLAYER_DAMAGED,
        PLAYER_JUMPED
    };

    struct CombatEvent {
        EventType type;
        glm::vec3 position{0.0f};
        int       target_id = -1;
    };

    struct BulletTracer {
        glm::vec3 start;
        glm::vec3 end;
        float     life = 0.08f;
    };

    // Data-Oriented Structure of Arrays for Bots
    struct BotTableSoA {
        std::pmr::vector<glm::vec3> position;
        std::pmr::vector<glm::vec3> target_waypoint;
        std::pmr::vector<float>     yaw;
        std::pmr::vector<int16_t>   hp;
        std::pmr::vector<uint8_t>   state; // 0: Patrol, 1: Chase, 2: Attack, 3: Dead
        std::pmr::vector<float>     hit_flash_time;
        std::pmr::vector<float>     respawn_time;
        std::pmr::vector<float>     attack_cooldown;
        std::pmr::vector<float>     bob_phase;
        std::pmr::vector<float>     strafe_dir;

        explicit BotTableSoA(std::pmr::memory_resource* mr = std::pmr::get_default_resource())
            : position(mr), target_waypoint(mr), yaw(mr), hp(mr),
              state(mr), hit_flash_time(mr), respawn_time(mr),
              attack_cooldown(mr), bob_phase(mr), strafe_dir(mr) {}

        inline size_t size() const { return position.size(); }

        inline void add_bot(glm::vec3 pos, glm::vec3 wp) {
            position.push_back(pos);
            target_waypoint.push_back(wp);
            yaw.push_back(0.0f);
            hp.push_back(100);
            state.push_back(0); // Patrol
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
            : position(mr), velocity(mr), life(mr) {}

        inline size_t size() const { return position.size(); }

        inline void add(glm::vec3 pos, glm::vec3 vel, float dur = 3.0f) {
            position.push_back(pos);
            velocity.push_back(vel);
            life.push_back(dur);
        }

        inline void remove_at(size_t i) {
            if (i < position.size()) {
                position.erase(position.begin() + i);
                velocity.erase(velocity.begin() + i);
                life.erase(life.begin() + i);
            }
        }
    };

    struct PlayerSnapshot {
        glm::vec3 position{0.0f, PLAYER_EYE_HEIGHT, -8.0f};
        float     velocity_y    = 0.0f;
        float     yaw           = 0.0f;
        float     pitch         = 0.0f;
        int16_t   hp            = 100;
        float     damage_flash  = 0.0f;
        float     fire_cooldown = 0.0f;
        float     recoil_offset = 0.0f;
        float     muzzle_flash  = 0.0f;
        bool      is_grounded   = true;
        int       score         = 0;
        int       kills         = 0;

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

    // Persistent World State Snapshot
    struct WorldSnapshot {
        PlayerSnapshot                 player;
        BotTableSoA                    bots;
        ProjectileTableSoA             projectiles;
        std::pmr::vector<BulletTracer> tracers;

        explicit WorldSnapshot(std::pmr::memory_resource* mr = std::pmr::get_default_resource())
            : bots(mr), projectiles(mr), tracers(mr) {}
    };

    struct WorldStepResult {
        WorldSnapshot                 next_world;
        std::pmr::vector<CombatEvent> events;
        float                         hitmarker_timer = 0.0f;

        WorldStepResult(std::pmr::memory_resource* persistent_mr, std::pmr::memory_resource* frame_mr)
            : next_world(persistent_mr), events(frame_mr) {}
    };

    static inline bool ray_sphere_intersect(const glm::vec3& orig, const glm::vec3& dir,
                                            const glm::vec3& center, float rad, float& out_t) {
        glm::vec3 oc = orig - center;
        float b = glm::dot(oc, dir);
        float c = glm::dot(oc, oc) - rad * rad;
        float disc = b * b - c;
        if (disc < 0.0f) return false;

        float sqrt_disc = std::sqrt(disc);
        float t0 = -b - sqrt_disc;
        float t1 = -b + sqrt_disc;

        if (t0 > 0.001f) { out_t = t0; return true; }
        if (t1 > 0.001f) { out_t = t1; return true; }
        return false;
    }

    // Pure World Reducer: (State, Commands, dt) -> (NextState, Events)
    static WorldStepResult reduce_world(
        const WorldSnapshot& prev,
        std::span<const UserCommand>       commands,
        float                              dt,
        std::pmr::memory_resource* frame_arena
    ) {
        WorldStepResult result(std::pmr::get_default_resource(), frame_arena);
        result.next_world.player = prev.player;
        PlayerSnapshot& p = result.next_world.player;

        PlayerCommandFrame input = reduce_user_commands(commands);

        // 1. Player Reset
        if (input.reset_pressed) {
            p.position = glm::vec3(0.0f, PLAYER_EYE_HEIGHT, -8.0f);
            p.yaw = 0.0f; p.pitch = 0.0f; p.hp = 100; p.velocity_y = 0.0f;
        }

        // 2. Player Look & Movement
        p.yaw   += input.delta_yaw;
        p.pitch -= input.delta_pitch;
        p.pitch  = glm::clamp(p.pitch, -glm::radians(85.0f), glm::radians(85.0f));

        glm::vec3 fwd_xz   = glm::normalize(glm::vec3(std::sin(p.yaw), 0.0f, std::cos(p.yaw)));
        glm::vec3 right_xz = glm::normalize(glm::vec3(std::cos(p.yaw), 0.0f, -std::sin(p.yaw)));

        glm::vec3 move_dir = input.move_dir.z * fwd_xz + input.move_dir.x * right_xz;
        if (glm::length(move_dir) > 0.01f) {
            move_dir = glm::normalize(move_dir);
            p.position += move_dir * PLAYER_SPEED * dt;
        }
        p.position.x = glm::clamp(p.position.x, -14.5f, 14.5f);
        p.position.z = glm::clamp(p.position.z, -14.5f, 14.5f);

        // 3. Jump & Vertical Gravity Physics
        const float GRAVITY = 24.0f;
        if (input.jump_pressed && p.is_grounded) {
            p.velocity_y  = 8.5f;
            p.is_grounded = false;
            result.events.push_back({ EventType::PLAYER_JUMPED, p.position });
        }
        p.velocity_y -= GRAVITY * dt;
        p.position.y += p.velocity_y * dt;

        if (p.position.y <= PLAYER_EYE_HEIGHT) {
            p.position.y   = PLAYER_EYE_HEIGHT;
            p.velocity_y   = 0.0f;
            p.is_grounded  = true;
        }

        // 4. Timers decay
        if (p.recoil_offset > 0.0f)       p.recoil_offset = std::max(0.0f, p.recoil_offset - dt * 2.5f);
        if (p.muzzle_flash > 0.0f)        p.muzzle_flash -= dt;
        if (p.fire_cooldown > 0.0f)       p.fire_cooldown -= dt;
        if (p.damage_flash > 0.0f)        p.damage_flash -= dt;

        // 5. Clone and Step Bots (SoA)
        result.next_world.bots = prev.bots;
        BotTableSoA& bots = result.next_world.bots;

        // 6. Player Weapon Firing & Hitscan
        if (input.fire_pressed && p.fire_cooldown <= 0.0f) {
            p.fire_cooldown = 0.18f;
            p.recoil_offset = 0.08f;
            p.muzzle_flash  = 0.05f;

            glm::vec3 eye = p.position;
            glm::vec3 dir = p.get_forward();

            result.events.push_back({ EventType::PLAYER_FIRED, eye });

            int hit_idx = -1;
            float closest_t = 1e6f;

            for (size_t i = 0; i < bots.size(); ++i) {
                if (bots.state[i] == 3) continue; // Dead

                glm::vec3 chest = bots.position[i] + glm::vec3(0, 0.95f, 0);
                glm::vec3 head  = bots.position[i] + glm::vec3(0, 1.55f, 0);

                float t_c = 1e6f, t_h = 1e6f;
                bool hit_c = ray_sphere_intersect(eye, dir, chest, 0.55f, t_c);
                bool hit_h = ray_sphere_intersect(eye, dir, head,  0.30f, t_h);

                float t_best = 1e6f;
                if (hit_c) t_best = std::min(t_best, t_c);
                if (hit_h) t_best = std::min(t_best, t_h);

                if (t_best < closest_t) {
                    closest_t = t_best;
                    hit_idx = (int)i;
                }
            }

            glm::vec3 hit_pos = (hit_idx >= 0) ? (eye + dir * closest_t) : (eye + dir * 60.0f);

            glm::vec3 muzzle_world = eye + dir * 0.4f + p.get_right() * 0.18f - glm::vec3(0, 0.12f, 0);
            result.next_world.tracers.push_back({ muzzle_world, hit_pos, 0.06f });

            if (hit_idx >= 0) {
                bots.hp[hit_idx] -= 35;
                bots.hit_flash_time[hit_idx] = 0.12f;
                result.hitmarker_timer = 0.15f;
                result.events.push_back({ EventType::BOT_HIT, hit_pos, hit_idx });

                if (bots.hp[hit_idx] <= 0 && bots.state[hit_idx] != 3) {
                    bots.state[hit_idx] = 3; // Dead
                    bots.respawn_time[hit_idx] = 4.0f;
                    p.score += 100;
                    p.kills += 1;
                    result.events.push_back({ EventType::BOT_KILLED, bots.position[hit_idx], hit_idx });
                }
            }
        }

        // 7. Step Bot AI & Projectile Spawns
        result.next_world.projectiles = prev.projectiles;
        ProjectileTableSoA& projs = result.next_world.projectiles;

        for (size_t i = 0; i < bots.size(); ++i) {
            bots.bob_phase[i] += dt * 3.0f;

            if (bots.state[i] == 3) { // Dead
                bots.respawn_time[i] -= dt;
                if (bots.respawn_time[i] <= 0.0f) {
                    bots.state[i] = 0; // Patrol
                    bots.hp[i] = 100;
                    bots.hit_flash_time[i] = 0.0f;
                }
                continue;
            }

            if (bots.hit_flash_time[i] > 0.0f)  bots.hit_flash_time[i] -= dt;
            if (bots.attack_cooldown[i] > 0.0f) bots.attack_cooldown[i] -= dt;

            glm::vec3 to_player = p.position - bots.position[i];
            float dist_to_player = glm::length(glm::vec2(to_player.x, to_player.z));
            bots.yaw[i] = std::atan2(to_player.x, to_player.z);

            const float CHASE_RANGE  = 18.0f;
            const float ATTACK_RANGE = 9.0f;
            const float BOT_SPEED    = 3.8f;

            if (dist_to_player > CHASE_RANGE)       bots.state[i] = 0; // Patrol
            else if (dist_to_player > ATTACK_RANGE) bots.state[i] = 1; // Chase
            else                                    bots.state[i] = 2; // Attack

            glm::vec3 fwd(std::sin(bots.yaw[i]), 0.0f, std::cos(bots.yaw[i]));
            glm::vec3 right(std::cos(bots.yaw[i]), 0.0f, -std::sin(bots.yaw[i]));

            switch (bots.state[i]) {
                case 0: { // Patrol
                    glm::vec3 to_wp = bots.target_waypoint[i] - bots.position[i];
                    if (glm::length(to_wp) < 1.0f) {
                        bots.target_waypoint[i] = glm::vec3((rand() % 24) - 12.0f, 0.0f, (rand() % 24) - 12.0f);
                    }
                    bots.position[i] += glm::normalize(to_wp) * (BOT_SPEED * 0.5f) * dt;
                    break;
                }
                case 1: { // Chase
                    bots.position[i] += fwd * BOT_SPEED * dt;
                    break;
                }
                case 2: { // Attack
                    if ((rand() % 100) < 2) bots.strafe_dir[i] = -bots.strafe_dir[i];
                    bots.position[i] += right * (bots.strafe_dir[i] * BOT_SPEED * 0.8f) * dt;

                    if (bots.attack_cooldown[i] <= 0.0f) {
                        bots.attack_cooldown[i] = 1.25f + ((rand() % 40) / 100.0f);
                        glm::vec3 muzzle = bots.position[i] + glm::vec3(0, 0.95f, 0) + fwd * 0.5f;
                        glm::vec3 aim_dir = glm::normalize((p.position + glm::vec3(0, 0.2f, 0)) - muzzle);
                        projs.add(muzzle, aim_dir * 18.0f, 3.0f);
                        g_audio.play(SND_ENEMY_SHOOT);
                    }
                    break;
                }
            }

            bots.position[i].x = glm::clamp(bots.position[i].x, -14.0f, 14.0f);
            bots.position[i].z = glm::clamp(bots.position[i].z, -14.0f, 14.0f);
        }

        // 8. Step Projectiles & Collisions with Player
        for (size_t i = 0; i < projs.size();) {
            projs.position[i] += projs.velocity[i] * dt;
            projs.life[i] -= dt;

            float dist = glm::length(projs.position[i] - (p.position - glm::vec3(0, 0.6f, 0)));
            if (dist < 0.85f) {
                p.hp -= 15;
                p.damage_flash = 0.22f;
                result.events.push_back({ EventType::PLAYER_DAMAGED, p.position });
                if (p.hp <= 0) {
                    p.hp = 100;
                    p.position = glm::vec3(0.0f, PLAYER_EYE_HEIGHT, -8.0f);
                    p.velocity_y = 0.0f;
                }
                projs.remove_at(i);
                continue;
            }

            if (projs.life[i] <= 0.0f || std::abs(projs.position[i].x) > 16.0f || std::abs(projs.position[i].z) > 16.0f) {
                projs.remove_at(i);
            } else {
                ++i;
            }
        }

        // 9. Step Tracers
        for (const auto& tr : prev.tracers) {
            if (tr.life - dt > 0.0f) {
                result.next_world.tracers.push_back({ tr.start, tr.end, tr.life - dt });
            }
        }

        return result;
    }

    static inline void is_grounded(PlayerSnapshot& p) {
        p.position.y   = PLAYER_EYE_HEIGHT;
        p.velocity_y   = 0.0f;
        p.is_grounded  = true;
    }
}

// ============================================================================
// PHASE 6: PURE SCENE-TO-BATCH PLANNER (`to_render_items`)
// ============================================================================
namespace vop {
    struct ProcessedTriangle {
        glm::vec4  c0, c1, c2; // Clip coordinates in [-w, +w]
        shs::Color lit_color;
        float      depth_bias;
    };

    struct PipelineExecutionPlan {
        std::pmr::vector<ProcessedTriangle> triangles;
        glm::mat4                          view_matrix;
        glm::mat4                          proj_matrix;
        glm::mat4                          vp_matrix;

        explicit PipelineExecutionPlan(std::pmr::memory_resource* mr)
            : triangles(mr) {}
    };

    static PipelineExecutionPlan build_render_plan(
        const WorldSnapshot&                world,
        const std::vector<LowPolyTriangle>& arena_geometry,
        const std::vector<LowPolyTriangle>& gun_geometry,
        const std::vector<LowPolyTriangle>& flash_geometry,
        const std::vector<LowPolyTriangle>& bolt_geometry,
        int                                 canvas_w,
        int                                 canvas_h,
        std::pmr::memory_resource*          arena
    ) {
        PipelineExecutionPlan plan(arena);
        plan.triangles.reserve(arena_geometry.size() + world.bots.size() * 120 + world.projectiles.size() * 12 + 200);

        glm::vec3 eye = world.player.position;
        glm::vec3 fwd = world.player.get_forward();

        plan.view_matrix = glm::lookAtLH(eye, eye + fwd, glm::vec3(0, 1, 0));
        plan.proj_matrix = glm::perspectiveLH_NO(glm::radians(75.0f), (float)canvas_w / (float)canvas_h, Z_NEAR, Z_FAR);
        plan.vp_matrix   = plan.proj_matrix * plan.view_matrix;

        auto process_batch = [&](const std::vector<LowPolyTriangle>& batch, const glm::mat4& model) {
            glm::mat4 mvp = plan.vp_matrix * model;
            glm::vec3 L = -SUN_DIR_WORLD;

            for (const auto& tri : batch) {
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

                plan.triangles.push_back({
                    mvp * glm::vec4(tri.p0, 1.0f),
                    mvp * glm::vec4(tri.p1, 1.0f),
                    mvp * glm::vec4(tri.p2, 1.0f),
                    shs::rgb01_to_color(lit_rgb),
                    tri.depth_bias
                });
            }
        };

        // 1. Arena
        process_batch(arena_geometry, glm::mat4(1.0f));

        // 2. Bots (SoA)
        for (size_t i = 0; i < world.bots.size(); ++i) {
            if (world.bots.state[i] == 3) { // Dead
                glm::mat4 m = glm::translate(glm::mat4(1.0f), world.bots.position[i] + glm::vec3(0, 0.2f, 0))
                            * glm::rotate(glm::mat4(1.0f), world.bots.yaw[i], glm::vec3(0, 1, 0))
                            * glm::rotate(glm::mat4(1.0f), glm::radians(-80.0f), glm::vec3(1, 0, 0));
                std::vector<LowPolyTriangle> mesh = SyntheticActorMeshes::build_bot_mesh(false);
                process_batch(mesh, m);
            } else {
                float hover_y = std::sin(world.bots.bob_phase[i]) * 0.05f;
                glm::mat4 m = glm::translate(glm::mat4(1.0f), world.bots.position[i] + glm::vec3(0, hover_y, 0))
                            * glm::rotate(glm::mat4(1.0f), world.bots.yaw[i], glm::vec3(0, 1, 0));
                std::vector<LowPolyTriangle> mesh = SyntheticActorMeshes::build_bot_mesh(world.bots.hit_flash_time[i] > 0.0f);
                process_batch(mesh, m);
            }
        }

        // 3. Projectiles
        for (size_t i = 0; i < world.projectiles.size(); ++i) {
            glm::mat4 m = glm::translate(glm::mat4(1.0f), world.projectiles.position[i])
                        * glm::scale(glm::mat4(1.0f), glm::vec3(1.2f));
            process_batch(bolt_geometry, m);
        }

        // 4. Viewmodel Gun & Muzzle Flash
        glm::vec3 gun_offset(0.22f, -0.22f, 0.45f - world.player.recoil_offset);
        glm::mat4 gun_rot = glm::rotate(glm::mat4(1.0f), world.player.yaw, glm::vec3(0, 1, 0))
                          * glm::rotate(glm::mat4(1.0f), world.player.pitch, glm::vec3(-1, 0, 0));
        glm::vec3 gun_world_pos = eye + glm::vec3(gun_rot * glm::vec4(gun_offset, 1.0f));
        glm::mat4 gun_model = glm::translate(glm::mat4(1.0f), gun_world_pos)
                            * gun_rot
                            * glm::scale(glm::mat4(1.0f), glm::vec3(0.9f));
        process_batch(gun_geometry, gun_model);

        if (world.player.muzzle_flash > 0.0f) {
            glm::mat4 flash_model = gun_model * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.04f, 0.46f));
            process_batch(flash_geometry, flash_model);
        }

        return plan;
    }
}

// ============================================================================
// PHASE 7: WAIT-FREE TILED RASTERIZER (Constitution II, Rule 9)
// ============================================================================
namespace vop {
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

    struct TileRasterContract {
        std::span<const ProcessedTriangle> active_triangles;
        glm::ivec2                         tile_min;
        glm::ivec2                         tile_max;
        int                                canvas_w;
        int                                canvas_h;
    };

    static void execute_tile_raster_job(
        shs::Canvas&             canvas,
        shs::ZBuffer&            z_buffer,
        const TileRasterContract contract
    ) {
        for (const auto& tri : contract.active_triangles) {
            const shs::Raster::FrustumClipPolygon poly =
                shs::Raster::clip_triangle_to_frustum(tri.c0, tri.c1, tri.c2);
            if (poly.count < 3) continue;

            glm::vec4 s0 = clip_to_screen_vec4(poly.vertices[0], contract.canvas_w, contract.canvas_h);
            for (int i = 1; i + 1 < poly.count; ++i) {
                glm::vec4 s1 = clip_to_screen_vec4(poly.vertices[i], contract.canvas_w, contract.canvas_h);
                glm::vec4 s2 = clip_to_screen_vec4(poly.vertices[i + 1], contract.canvas_w, contract.canvas_h);
                rasterize_perspective_triangle_tile(canvas, z_buffer, s0, s1, s2, tri.lit_color, tri.depth_bias, contract.tile_min, contract.tile_max);
            }
        }
    }
}

// ============================================================================
// PHASE 8: UI & VECTOR HUD (Screen-Space Top-Left Origin)
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
        0b00111111, 0b00000110, 0b01011011, 0b01001111, 0b01100110,
        0b01101101, 0b01111101, 0b00000111, 0b01111111, 0b01101111
    };
    if (d < 0 || d > 9) return;
    uint8_t mask = segs[d];
    int mid_y = y + h / 2;

    auto line = [&](int x0, int y0, int x1, int y1) {
        shs::Canvas::draw_line(canvas, x0, y0, x1, y1, col);
    };

    if (mask & (1 << 0)) line(x, y, x + w, y);
    if (mask & (1 << 1)) line(x + w, y, x + w, mid_y);
    if (mask & (1 << 2)) line(x + w, mid_y, x + w, y + h);
    if (mask & (1 << 3)) line(x, y + h, x + w, y + h);
    if (mask & (1 << 4)) line(x, mid_y, x, y + h);
    if (mask & (1 << 5)) line(x, y, x, mid_y);
    if (mask & (1 << 6)) line(x, mid_y, x + w, mid_y);
}

static void draw_number_screen(shs::Canvas& canvas, int x, int y, int val, int digits, shs::Color col) {
    int w = 10, h = 18, gap = 5;
    for (int i = digits - 1; i >= 0; --i) {
        int d = val % 10;
        val /= 10;
        draw_digit_screen(canvas, x + i * (w + gap), y, d, w, h, col);
    }
}

static void draw_enemy_health_bars(shs::Canvas& canvas, const glm::mat4& vp, const vop::BotTableSoA& bots) {
    int W = canvas.get_width();
    int H = canvas.get_height();

    for (size_t i = 0; i < bots.size(); ++i) {
        if (bots.state[i] == 3) continue; // Dead

        glm::vec3 head_top = bots.position[i] + glm::vec3(0, 2.05f, 0);
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

        float hp_pct = glm::clamp((float)bots.hp[i] / 100.0f, 0.0f, 1.0f);
        int fill_w   = (int)(hp_pct * (float)bar_w);

        shs::Color hp_col = (bots.hit_flash_time[i] > 0.0f) ? shs::Color{255, 255, 255, 255}
                          : (hp_pct > 0.5f) ? shs::Color{240, 70, 50, 255} : shs::Color{255, 255, 255, 255};

        draw_rect_fill_screen(canvas, bx, by, fill_w, bar_h, hp_col);
    }
}

static void draw_fps_hud(shs::Canvas& canvas, const vop::PlayerSnapshot& player, float hitmarker_timer) {
    int W  = canvas.get_width();
    int H  = canvas.get_height();
    int cx = W / 2;
    int cy = H / 2;

    // 1. Damage Screen Vignette
    if (player.damage_flash > 0.0f) {
        shs::Color red_border{255, 30, 30, 200};
        for (int i = 0; i < 8; ++i) {
            draw_rect_border_screen(canvas, i, i, W - 1 - i * 2, H - 1 - i * 2, red_border);
        }
    }

    // 2. Score & Kill Counter (Top-Right HUD Card)
    int sc_x = W - 180, sc_y = 25;
    draw_rect_fill_screen(canvas, sc_x - 10, sc_y - 8, 165, 40, shs::Color{15, 18, 25, 230});
    draw_rect_border_screen(canvas, sc_x - 10, sc_y - 8, 165, 40, shs::Color{90, 100, 120, 255});
    draw_number_screen(canvas, sc_x, sc_y + 3, player.score, 6, shs::Color{255, 215, 60, 255});

    // 3. Player Health Gauge (Bottom-Left)
    int hp_x = 35;
    int hp_y = H - 55;
    int hp_w = 220;
    int hp_h = 18;

    draw_rect_fill_screen(canvas, hp_x - 4, hp_y - 4, hp_w + 8, hp_h + 8, shs::Color{15, 18, 25, 230});
    draw_rect_border_screen(canvas, hp_x - 4, hp_y - 4, hp_w + 8, hp_h + 8, shs::Color{90, 100, 120, 255});
    draw_rect_fill_screen(canvas, hp_x, hp_y, hp_w, hp_h, shs::Color{45, 20, 20, 255});

    float hp_ratio = glm::clamp((float)player.hp / 100.0f, 0.0f, 1.0f);
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
// PHASE 9: APPLICATION & EXECUTION EDGES
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
        "SHS Renderer - VOP Low-Poly FPS Combat Arena",
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

    std::cout << "Compiling Pure Geometry Batches..." << std::endl;
    std::vector<LowPolyTriangle> arena_mesh = ArenaWorld::build_mesh();
    std::vector<LowPolyTriangle> gun_mesh   = SyntheticActorMeshes::build_gun_mesh();
    std::vector<LowPolyTriangle> flash_mesh = SyntheticActorMeshes::build_muzzle_flash();
    std::vector<LowPolyTriangle> bolt_mesh  = SyntheticActorMeshes::build_projectile_mesh();

    // Heap-Allocated Frame Memory Arena (8 MB)
    vop::FrameMemoryResource frame_memory;

    // Build initial WorldSnapshot (Persistent heap memory)
    vop::WorldSnapshot current_world(std::pmr::get_default_resource());
    current_world.bots.add_bot(glm::vec3(-7.0f, 0.0f,  6.0f), glm::vec3(-7.0f, 0.0f,  6.0f));
    current_world.bots.add_bot(glm::vec3( 7.0f, 0.0f,  6.0f), glm::vec3( 7.0f, 0.0f,  6.0f));
    current_world.bots.add_bot(glm::vec3(-6.0f, 0.0f, -4.0f), glm::vec3(-6.0f, 0.0f, -4.0f));
    current_world.bots.add_bot(glm::vec3( 6.0f, 0.0f, -4.0f), glm::vec3( 6.0f, 0.0f, -4.0f));

    float hitmarker_timer = 0.0f;
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
        // 1. FRAME ARENA RESET (O(1) Memory Cycle)
        // --------------------------------------------------------------------
        frame_memory.reset();
        auto* arena = frame_memory.get();

        // --------------------------------------------------------------------
        // 2. INPUT COLLECTION EDGE -> PURE COMMAND STREAM
        // --------------------------------------------------------------------
        std::pmr::vector<vop::UserCommand> commands(arena);

        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;

            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) quit = true;
                if (e.key.keysym.sym == SDLK_r)      commands.push_back(vop::ResetIntent{});
                if (e.key.keysym.sym == SDLK_SPACE)  commands.push_back(vop::JumpIntent{});
                if (e.key.keysym.sym == SDLK_f || e.key.keysym.sym == SDLK_LCTRL || 
                    e.key.keysym.sym == SDLK_RCTRL || e.key.keysym.sym == SDLK_RETURN) {
                    commands.push_back(vop::FireIntent{});
                }
            }

            if (e.type == SDL_MOUSEBUTTONDOWN) {
                SDL_SetRelativeMouseMode(SDL_TRUE);
                if (e.button.button == SDL_BUTTON_LEFT) commands.push_back(vop::FireIntent{});
            }
        }

        // Mouse Delta Look
        int mouse_dx = 0, mouse_dy = 0;
        SDL_GetRelativeMouseState(&mouse_dx, &mouse_dy);
        if (mouse_dx != 0 || mouse_dy != 0) {
            commands.push_back(vop::LookIntent{ (float)mouse_dx * MOUSE_SENSITIVITY, (float)mouse_dy * MOUSE_SENSITIVITY });
        }

        // Keyboard Look (Arrow Keys)
        const Uint8* keys = SDL_GetKeyboardState(NULL);
        float k_yaw = 0.0f, k_pitch = 0.0f;
        if (keys[SDL_SCANCODE_LEFT])  k_yaw   -= 2.4f * dt;
        if (keys[SDL_SCANCODE_RIGHT]) k_yaw   += 2.4f * dt;
        if (keys[SDL_SCANCODE_UP])    k_pitch += 2.0f * dt;
        if (keys[SDL_SCANCODE_DOWN])  k_pitch -= 2.0f * dt;
        if (k_yaw != 0.0f || k_pitch != 0.0f) {
            commands.push_back(vop::LookIntent{ k_yaw, -k_pitch });
        }

        // Keyboard Movement (WASD)
        glm::vec3 move_intent{0.0f};
        if (keys[SDL_SCANCODE_W]) move_intent.z += 1.0f;
        if (keys[SDL_SCANCODE_S]) move_intent.z -= 1.0f;
        if (keys[SDL_SCANCODE_D]) move_intent.x += 1.0f;
        if (keys[SDL_SCANCODE_A]) move_intent.x -= 1.0f;
        if (glm::length(move_intent) > 0.01f) {
            commands.push_back(vop::MoveIntent{ move_intent });
        }

        // --------------------------------------------------------------------
        // 3. PURE SIMULATION REDUCER CENTER
        // --------------------------------------------------------------------
        vop::WorldStepResult step = vop::reduce_world(current_world, commands, dt, arena);
        current_world = std::move(step.next_world);

        if (step.hitmarker_timer > 0.0f) hitmarker_timer = step.hitmarker_timer;
        else if (hitmarker_timer > 0.0f) hitmarker_timer -= dt;

        // --------------------------------------------------------------------
        // 4. EVENT-SOURCED AUDIO DISPATCHER EDGE
        // --------------------------------------------------------------------
        for (const auto& ev : step.events) {
            switch (ev.type) {
                case vop::EventType::PLAYER_FIRED:   g_audio.play(SND_PLAYER_SHOOT);  break;
                case vop::EventType::BOT_HIT:        g_audio.play(SND_HITMARKER);     break;
                case vop::EventType::BOT_KILLED:     g_audio.play(SND_ENEMY_EXPLODE); break;
                case vop::EventType::PLAYER_DAMAGED: g_audio.play(SND_PLAYER_HURT);   break;
                case vop::EventType::PLAYER_JUMPED:  g_audio.play(SND_PLAYER_JUMP);   break;
            }
        }

        // --------------------------------------------------------------------
        // 5. PURE SCENE & PIPELINE EXECUTION PLANNER
        // --------------------------------------------------------------------
        vop::PipelineExecutionPlan plan = vop::build_render_plan(
            current_world, arena_mesh, gun_mesh, flash_mesh, bolt_mesh,
            CANVAS_WIDTH, CANVAS_HEIGHT, arena
        );

        // --------------------------------------------------------------------
        // 6. WAIT-FREE TILED RASTERIZATION EDGE (Span Contract)
        // --------------------------------------------------------------------
        canvas.buffer().clear(shs::Color{22, 28, 38, 255});
        z_buffer.clear();

        int W = canvas.get_width();
        int H = canvas.get_height();
        int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wg_render.reset();

        for (int ty = 0; ty < rows; ++ty) {
            for (int tx = 0; tx < cols; ++tx) {
                wg_render.add(1);
                job_system.submit({[&, tx, ty, W, H]() {
                    vop::TileRasterContract contract{
                        .active_triangles = plan.triangles,
                        .tile_min = glm::ivec2(tx * TILE_SIZE_X, ty * TILE_SIZE_Y),
                        .tile_max = glm::ivec2(std::min((tx + 1) * TILE_SIZE_X, W) - 1,
                                               std::min((ty + 1) * TILE_SIZE_Y, H) - 1),
                        .canvas_w = W,
                        .canvas_h = H
                    };
                    vop::execute_tile_raster_job(canvas, z_buffer, contract);
                    wg_render.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }

        wg_render.wait();

        // --------------------------------------------------------------------
        // 7. TRACERS & UI DRAWING (Pure Canvas Overlay)
        // --------------------------------------------------------------------
        for (const auto& tr : current_world.tracers) {
            glm::vec4 c_start = plan.vp_matrix * glm::vec4(tr.start, 1.0f);
            glm::vec4 c_end   = plan.vp_matrix * glm::vec4(tr.end,   1.0f);
            if (c_start.w > 0.1f && c_end.w > 0.1f) {
                glm::vec3 s0 = shs::Canvas::clip_to_screen(c_start, W, H);
                glm::vec3 s1 = shs::Canvas::clip_to_screen(c_end,   W, H);
                shs::Canvas::draw_line(canvas, (int)s0.x, (int)s0.y, (int)s1.x, (int)s1.y, shs::Color{255, 230, 100, 255});
            }
        }

        draw_enemy_health_bars(canvas, plan.vp_matrix, current_world.bots);
        draw_fps_hud(canvas, current_world.player, hitmarker_timer);

        // --------------------------------------------------------------------
        // 8. SWAPCHAIN PRESENTATION EDGE
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
            for (size_t i = 0; i < current_world.bots.size(); ++i) {
                if (current_world.bots.state[i] != 3) alive_count++;
            }

            std::ostringstream ss;
            ss << "VOP FPS Arena | FPS: " << (int)((float)frame_count / fps_timer)
               << " | HP: " << current_world.player.hp
               << " | Score: " << current_world.player.score
               << " | Kills: " << current_world.player.kills
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