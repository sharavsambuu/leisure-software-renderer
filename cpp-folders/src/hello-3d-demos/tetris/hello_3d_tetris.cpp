#define SDL_MAIN_HANDLED

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <atomic>
#include <thread>
#include <span>
#include <memory_resource>

#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shs_renderer.hpp"
#include "tetris.contract.hpp"
#include "tetris.action.hpp"
#include "tetris.event.hpp"
#include "tetris.reducer.hpp"
#include "tetris.plan.hpp"

// Window & Rasterizer Config
static const int CANVAS_WIDTH  = 1280;
static const int CANVAS_HEIGHT = 720;
static const int TILE_SIZE_X   = 80;
static const int TILE_SIZE_Y   = 80;
static const int THREAD_COUNT  = std::max(2u, std::thread::hardware_concurrency() > 2 ? std::thread::hardware_concurrency() - 2 : 2u);

// ============================================================================
// PMR FRAME ARENA (Constitution II)
// ============================================================================
namespace vop {
    class FrameMemoryResource : public std::pmr::memory_resource {
    public:
        static constexpr size_t CAPACITY = 8 * 1024 * 1024; // 8 MB
        FrameMemoryResource() : buffer_(std::make_unique<uint8_t[]>(CAPACITY)), offset_(0) {}
        inline void reset() noexcept { offset_ = 0; }
        inline std::pmr::memory_resource* get() noexcept { return this; }
    protected:
        void* do_allocate(size_t bytes, size_t alignment) override {
            uintptr_t base = reinterpret_cast<uintptr_t>(buffer_.get());
            uintptr_t current_addr = base + offset_;
            uintptr_t aligned_addr = (current_addr + (alignment - 1)) & ~(alignment - 1);
            size_t new_offset = (aligned_addr - base) + bytes;
            if (new_offset > CAPACITY) return std::pmr::get_default_resource()->allocate(bytes, alignment);
            offset_ = new_offset;
            return reinterpret_cast<void*>(aligned_addr);
        }
        void do_deallocate(void* p, size_t bytes, size_t alignment) noexcept override {
            uintptr_t base = reinterpret_cast<uintptr_t>(buffer_.get());
            uintptr_t ptr  = reinterpret_cast<uintptr_t>(p);
            if (ptr < base || ptr >= base + CAPACITY) std::pmr::get_default_resource()->deallocate(p, bytes, alignment);
        }
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
    private:
        std::unique_ptr<uint8_t[]> buffer_;
        size_t offset_ = 0;
    };
}

// ============================================================================
// LOCK-FREE PROCEDURAL AUDIO ENGINE
// ============================================================================
enum SoundType : uint8_t {
    SND_NONE        = 0,
    SND_MOVE        = 1,
    SND_ROTATE      = 2,
    SND_DROP_SLAM   = 3,
    SND_LINE_CLEAR  = 4,
    SND_TETRIS_FOUR = 5,
    SND_HOLD        = 6,
    SND_GAME_OVER   = 7
};

struct AudioEventRing {
    static const uint32_t CAP = 64;
    SoundType buffer[CAP]{};
    alignas(64) std::atomic<uint32_t> write_idx{ 0 };
    alignas(64) uint32_t              read_idx { 0 };

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

struct TetrisAudioSynth {
    static const int MAX_VOICES = 12;
    SoundVoice       voices[MAX_VOICES];
    AudioEventRing   event_queue;

    inline void play(SoundType type) { event_queue.push(type); }

    void mix(float* stream, int frames, int channels, int sample_rate) {
        SoundType new_type;
        while (event_queue.pop(new_type)) {
            if (new_type == SND_NONE) continue;
            for (int i = 0; i < MAX_VOICES; ++i) {
                if (!voices[i].active) {
                    voices[i].type     = new_type;
                    voices[i].time     = 0.0f;
                    voices[i].phase    = 0.0f;
                    voices[i].active   = true;
                    voices[i].duration = (new_type == SND_TETRIS_FOUR) ? 0.45f : 0.12f;
                    break;
                }
            }
        }

        float dt = 1.0f / (float)sample_rate;
        for (int f = 0; f < frames; ++f) {
            float sample = 0.0f;
            for (int v = 0; v < MAX_VOICES; ++v) {
                if (!voices[v].active) continue;
                SoundVoice& vox = voices[v];
                vox.time += dt;
                float p = vox.time / vox.duration;
                if (p >= 1.0f) { vox.active = false; continue; }

                float env = (1.0f - p);
                switch (vox.type) {
                    case SND_MOVE:
                        vox.phase += 400.0f * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.12f;
                        break;
                    case SND_ROTATE:
                        vox.phase += (600.0f + p * 200.0f) * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.14f;
                        break;
                    case SND_DROP_SLAM:
                        vox.phase += (140.0f - p * 80.0f) * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.28f;
                        break;
                    case SND_LINE_CLEAR:
                        vox.phase += (523.25f + p * 400.0f) * dt; // C5 to G5
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.22f;
                        break;
                    case SND_TETRIS_FOUR:
                        vox.phase += (659.25f + std::sin(p * 20.0f) * 100.0f) * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.30f;
                        break;
                    case SND_HOLD:
                        vox.phase += (320.0f + p * 150.0f) * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.15f;
                        break;
                    case SND_GAME_OVER:
                        vox.phase += (220.0f - p * 140.0f) * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.25f;
                        break;
                    default: break;
                }
            }
            sample = std::tanh(sample);
            for (int c = 0; c < channels; ++c) stream[f * channels + c] = sample;
        }
    }
};

static TetrisAudioSynth g_audio;
static void audio_callback(void* userdata, Uint8* stream, int len) {
    TetrisAudioSynth* synth = reinterpret_cast<TetrisAudioSynth*>(userdata);
    float* out = reinterpret_cast<float*>(stream);
    synth->mix(out, len / (int)(sizeof(float) * 2), 2, 44100);
}

// ============================================================================
// MULTITHREADED TILED RASTERIZER (Constitution III)
// ============================================================================
namespace vop {
    static inline glm::vec4 clip_to_screen_vec4(const glm::vec4& clip, int W, int H) {
        float inv_w = 1.0f / clip.w;
        glm::vec3 ndc = glm::vec3(clip) * inv_w;
        return glm::vec4(
            (ndc.x + 1.0f) * 0.5f * (float)(W - 1),
            (1.0f - ndc.y) * 0.5f * (float)(H - 1),
            ndc.z,
            inv_w
        );
    }

    static void rasterize_triangle_tile(
        shs::Canvas& canvas, shs::ZBuffer& z_buffer,
        const glm::vec4& sc0, const glm::vec4& sc1, const glm::vec4& sc2,
        shs::Color lit_color, float depth_bias,
        glm::ivec2 tile_min, glm::ivec2 tile_max
    ) {
        glm::vec2 v0(sc0.x, sc0.y), v1(sc1.x, sc1.y), v2(sc2.x, sc2.y);
        float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
        if (!shs::Raster::is_front_facing_screen(area, shs::Raster::FrontFace::CounterClockwise)) return;

        glm::vec2 bboxmin = glm::max(glm::vec2(tile_min), glm::min(v0, glm::min(v1, v2)));
        glm::vec2 bboxmax = glm::min(glm::vec2(tile_max), glm::max(v0, glm::max(v1, v2)));
        if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

        std::vector<glm::vec2> v2d = { v0, v1, v2 };
        int min_x = (int)bboxmin.x, max_x = (int)bboxmax.x;
        int min_y = (int)bboxmin.y, max_y = (int)bboxmax.y;

        for (int py = min_y; py <= max_y; ++py) {
            for (int px = min_x; px <= max_x; ++px) {
                glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2((float)px + 0.5f, (float)py + 0.5f), v2d);
                if (bc.x < 0.0f || bc.y < 0.0f || bc.z < 0.0f) continue;

                float interp_z = shs::Raster::interpolate_ndc_depth(bc, sc0.z, sc1.z, sc2.z);
                float final_z  = interp_z + depth_bias;
                if (final_z < -1.0f || final_z > 1.0f) continue;

                if (z_buffer.test_and_set_depth_screen_space(px, py, final_z)) {
                    canvas.draw_pixel_screen_space(px, py, lit_color);
                }
            }
        }
    }
}

// ============================================================================
// 2D RETRO HUD & 2X COMPACT FONT
// ============================================================================
static void draw_line_screen(shs::Canvas& c, int x0, int y0, int x1, int y1, shs::Color col) {
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    for (;;) {
        c.draw_pixel_screen_space(x0, y0, col);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

static void draw_rect_fill(shs::Canvas& c, int x, int y, int w, int h, shs::Color col) {
    int x0 = std::max(0, x), y0 = std::max(0, y);
    int x1 = std::min(c.get_width() - 1, x + w), y1 = std::min(c.get_height() - 1, y + h);
    for (int py = y0; py <= y1; ++py) {
        for (int px = x0; px <= x1; ++px) c.draw_pixel_screen_space(px, py, col);
    }
}

static void draw_rect_border(shs::Canvas& c, int x, int y, int w, int h, shs::Color col) {
    int x1 = std::min(c.get_width() - 1, x + w), y1 = std::min(c.get_height() - 1, y + h);
    for (int px = std::max(0, x); px <= x1; ++px) {
        c.draw_pixel_screen_space(px, y, col);
        c.draw_pixel_screen_space(px, y1, col);
    }
    for (int py = std::max(0, y); py <= y1; ++py) {
        c.draw_pixel_screen_space(x, py, col);
        c.draw_pixel_screen_space(x1, py, col);
    }
}

// Compact 5x7 ASCII font glyphs
static const uint8_t FONT_5X7[][5] = {
    {0x00,0x00,0x00,0x00,0x00}, // Space
    {0x00,0x00,0x5F,0x00,0x00}, // !
    {0x00,0x07,0x00,0x07,0x00}, // "
    {0x14,0x7F,0x14,0x7F,0x14}, // #
    {0x24,0x2A,0x7F,0x2A,0x12}, // $
    {0x23,0x13,0x08,0x64,0x62}, // %
    {0x36,0x49,0x55,0x22,0x50}, // &
    {0x00,0x05,0x03,0x00,0x00}, // '
    {0x00,0x1C,0x22,0x41,0x00}, // (
    {0x00,0x41,0x22,0x1C,0x00}, // )
    {0x08,0x2A,0x1C,0x2A,0x08}, // *
    {0x08,0x08,0x3E,0x08,0x08}, // +
    {0x00,0x50,0x30,0x00,0x00}, // ,
    {0x08,0x08,0x08,0x08,0x08}, // -
    {0x00,0x60,0x60,0x00,0x00}, // .
    {0x20,0x10,0x08,0x04,0x02}, // /
    {0x3E,0x51,0x49,0x45,0x3E}, // 0
    {0x00,0x42,0x7F,0x40,0x00}, // 1
    {0x42,0x61,0x51,0x49,0x46}, // 2
    {0x21,0x41,0x45,0x4B,0x31}, // 3
    {0x18,0x14,0x12,0x7F,0x10}, // 4
    {0x27,0x45,0x45,0x45,0x39}, // 5
    {0x3C,0x4A,0x49,0x49,0x30}, // 6
    {0x01,0x71,0x09,0x05,0x03}, // 7
    {0x36,0x49,0x49,0x49,0x36}, // 8
    {0x06,0x49,0x49,0x29,0x1E}, // 9
    {0x00,0x36,0x36,0x00,0x00}, // :
    {0x00,0x56,0x36,0x00,0x00}, // ;
    {0x08,0x14,0x22,0x41,0x00}, // <
    {0x14,0x14,0x14,0x14,0x14}, // =
    {0x00,0x41,0x22,0x14,0x08}, // >
    {0x02,0x01,0x51,0x09,0x06}, // ?
    {0x32,0x49,0x79,0x41,0x3E}, // @
    {0x7E,0x11,0x11,0x11,0x7E}, // A
    {0x7F,0x49,0x49,0x49,0x36}, // B
    {0x3E,0x41,0x41,0x41,0x22}, // C
    {0x7F,0x41,0x41,0x22,0x1C}, // D
    {0x7F,0x49,0x49,0x49,0x41}, // E
    {0x7F,0x09,0x09,0x09,0x01}, // F
    {0x3E,0x41,0x49,0x49,0x7A}, // G
    {0x7F,0x08,0x08,0x08,0x7F}, // H
    {0x00,0x41,0x7F,0x41,0x00}, // I
    {0x20,0x40,0x41,0x3F,0x01}, // J
    {0x7F,0x08,0x14,0x22,0x41}, // K
    {0x7F,0x40,0x40,0x40,0x40}, // L
    {0x7F,0x02,0x0C,0x02,0x7F}, // M
    {0x7F,0x04,0x08,0x10,0x7F}, // N
    {0x3E,0x41,0x41,0x41,0x3E}, // O
    {0x7F,0x09,0x09,0x09,0x06}, // P
    {0x3E,0x41,0x51,0x21,0x5E}, // Q
    {0x7F,0x09,0x19,0x29,0x46}, // R
    {0x46,0x49,0x49,0x49,0x31}, // S
    {0x01,0x01,0x7F,0x01,0x01}, // T
    {0x3F,0x40,0x40,0x40,0x3F}, // U
    {0x1F,0x20,0x40,0x20,0x1F}, // V
    {0x7F,0x20,0x18,0x20,0x7F}, // W
    {0x63,0x14,0x08,0x14,0x63}, // X
    {0x07,0x08,0x70,0x08,0x07}, // Y
    {0x61,0x51,0x49,0x45,0x43}  // Z
};

static void draw_text(shs::Canvas& c, int x, int y, const std::string& str, shs::Color col, int scale = 2) {
    int cur_x = x;
    for (char ch : str) {
        char upper = (ch >= 'a' && ch <= 'z') ? (ch - 'a' + 'A') : ch;
        if (upper >= ' ' && upper <= 'Z') {
            int idx = upper - ' ';
            for (int col_i = 0; col_i < 5; ++col_i) {
                uint8_t bits = FONT_5X7[idx][col_i];
                for (int row_i = 0; row_i < 7; ++row_i) {
                    if (bits & (1 << row_i)) {
                        draw_rect_fill(c, cur_x + col_i * scale, y + row_i * scale, scale, scale, col);
                    }
                }
            }
        }
        cur_x += (5 + 1) * scale;
    }
}

// Bold 7-segment digit drawing with 2px thick strokes
static void draw_digit_bold(shs::Canvas& c, int x, int y, int d, int w, int h, shs::Color col) {
    static const uint8_t segs[10] = {
        0b00111111, 0b00000110, 0b01011011, 0b01001111, 0b01100110,
        0b01101101, 0b01111101, 0b00000111, 0b01111111, 0b01101111
    };
    if (d < 0 || d > 9) return;
    uint8_t mask = segs[d];
    int my = y + h / 2;

    auto h_seg = [&](int sx, int sy) { draw_rect_fill(c, sx, sy, w, 2, col); };
    auto v_seg = [&](int sx, int sy, int len) { draw_rect_fill(c, sx, sy, 2, len, col); };

    if (mask & (1 << 0)) h_seg(x, y);                      // top
    if (mask & (1 << 1)) v_seg(x + w - 2, y, my - y);      // top-right
    if (mask & (1 << 2)) v_seg(x + w - 2, my, y + h - my); // bot-right
    if (mask & (1 << 3)) h_seg(x, y + h - 2);              // bottom
    if (mask & (1 << 4)) v_seg(x, my, y + h - my);         // bot-left
    if (mask & (1 << 5)) v_seg(x, y, my - y);              // top-left
    if (mask & (1 << 6)) h_seg(x, my - 1);                 // middle
}

static void draw_number_bold(shs::Canvas& c, int x, int y, int val, int digits, shs::Color col) {
    int w = 12, h = 20, gap = 5;
    for (int i = digits - 1; i >= 0; --i) {
        int d = val % 10;
        val /= 10;
        draw_digit_bold(c, x + i * (w + gap), y, d, w, h, col);
    }
}

// Polished HUD with 2x scale and proper alignment
static void draw_hud(shs::Canvas& canvas, const tetris::TetrisSnapshot& state) {
    int W = canvas.get_width();
    int H = canvas.get_height();

    // ------------------------------------------------------------------------
    // 1. TOP RIGHT: SCORE CARD
    // ------------------------------------------------------------------------
    int sx = W - 265, sy = 18, sw = 245, sh = 88;
    draw_rect_fill(canvas, sx, sy, sw, sh, shs::Color{ 15, 18, 26, 230 });
    draw_rect_border(canvas, sx, sy, sw, sh, shs::Color{ 60, 140, 220, 255 });

    draw_text(canvas, sx + 14, sy + 14, "SCORE", shs::Color{ 255, 225, 45, 255 }, 2);
    draw_number_bold(canvas, sx + 125, sy + 12, state.score, 6, shs::Color{ 255, 225, 45, 255 });

    draw_text(canvas, sx + 14, sy + 48, "BEST", shs::Color{ 140, 155, 175, 255 }, 2);
    draw_number_bold(canvas, sx + 125, sy + 46, state.high_score, 6, shs::Color{ 140, 155, 175, 255 });

    // ------------------------------------------------------------------------
    // 2. TOP LEFT: GOAL & STATS CARD
    // ------------------------------------------------------------------------
    int ox = 20, oy = 18, ow = 280, oh = 88;
    draw_rect_fill(canvas, ox, oy, ow, oh, shs::Color{ 15, 18, 26, 230 });
    draw_rect_border(canvas, ox, oy, ow, oh, shs::Color{ 60, 140, 220, 255 });

    // Target Progress Bar
    draw_text(canvas, ox + 14, oy + 12, "GOAL", shs::Color{ 45, 220, 120, 255 }, 2);
    int bar_x = ox + 75, bar_y = oy + 12, bar_w = ow - 90, bar_h = 14;
    float progress = glm::clamp((float)state.score / (float)state.target_score, 0.0f, 1.0f);
    draw_rect_fill(canvas, bar_x, bar_y, bar_w, bar_h, shs::Color{ 35, 40, 52, 255 });
    draw_rect_fill(canvas, bar_x, bar_y, (int)(progress * (float)bar_w), bar_h, shs::Color{ 45, 220, 120, 255 });
    draw_rect_border(canvas, bar_x, bar_y, bar_w, bar_h, shs::Color{ 80, 95, 115, 255 });

    // Lines & Level
    draw_text(canvas, ox + 14, oy + 48, "LINES", shs::Color{ 40, 220, 240, 255 }, 2);
    draw_number_bold(canvas, ox + 85, oy + 46, state.lines_cleared, 3, shs::Color{ 40, 220, 240, 255 });

    draw_text(canvas, ox + 165, oy + 48, "LVL", shs::Color{ 255, 140, 35, 255 }, 2);
    draw_number_bold(canvas, ox + 225, oy + 46, state.level, 2, shs::Color{ 255, 140, 35, 255 });

    // ------------------------------------------------------------------------
    // 3. 3D PLATFORM LABELS (Centered above shelves)
    // ------------------------------------------------------------------------
    draw_text(canvas, 105, 120, "HOLD [C]", shs::Color{ 80, 200, 255, 240 }, 2);
    draw_text(canvas, W - 200, 120, "NEXT", shs::Color{ 80, 200, 255, 240 }, 2);

    // ------------------------------------------------------------------------
    // 4. NON-INTRUSIVE BOTTOM CONTROLS FOOTER
    // ------------------------------------------------------------------------
    draw_text(canvas, (W - 860) / 2, H - 24, "A/D: MOVE | W: ROTATE | S: DROP | SPACE: HARD DROP | C: HOLD | R: RETRY", shs::Color{ 140, 155, 175, 220 }, 2);

    // ------------------------------------------------------------------------
    // 5. GAME OVER / VICTORY MODAL OVERLAY
    // ------------------------------------------------------------------------
    if (state.game_over || state.victory) {
        int mw = 460, mh = 200;
        int mx = (W - mw) / 2, my = (H - mh) / 2;

        draw_rect_fill(canvas, mx, my, mw, mh, shs::Color{ 10, 12, 18, 245 });
        shs::Color bc = state.victory ? shs::Color{ 45, 240, 110, 255 } : shs::Color{ 245, 55, 55, 255 };
        draw_rect_border(canvas, mx, my, mw, mh, bc);
        draw_rect_border(canvas, mx + 2, my + 2, mw - 4, mh - 4, bc);

        if (state.victory) {
            draw_text(canvas, mx + 80, my + 25, "OBJECTIVE COMPLETE!", shs::Color{ 45, 240, 110, 255 }, 2);
        }
        else {
            draw_text(canvas, mx + 155, my + 25, "GAME OVER", shs::Color{ 245, 55, 55, 255 }, 2);
        }

        draw_text(canvas, mx + 80, my + 80, "FINAL SCORE:", shs::Color{ 220, 220, 220, 255 }, 2);
        draw_number_bold(canvas, mx + 240, my + 76, state.score, 6, shs::Color{ 255, 230, 80, 255 });

        draw_text(canvas, mx + 105, my + 140, "PRESS [R] TO RETRY", shs::Color{ 140, 160, 190, 255 }, 2);
    }
}

// ============================================================================
// MAIN ENTRY EDGE
// ============================================================================
int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "SHS Renderer - Semi-3D Low-Poly Cyber Tetris",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        CANVAS_WIDTH, CANVAS_HEIGHT, SDL_WINDOW_SHOWN
    );

    SDL_Renderer* sdl_renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* screen_texture = SDL_CreateTexture(
        sdl_renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING,
        CANVAS_WIDTH, CANVAS_HEIGHT
    );
    SDL_Surface* screen_surface = SDL_CreateRGBSurfaceWithFormat(0, CANVAS_WIDTH, CANVAS_HEIGHT, 32, SDL_PIXELFORMAT_RGBA32);

    // Audio Setup
    SDL_AudioSpec want{}, have{};
    want.freq     = 44100;
    want.format   = AUDIO_F32SYS;
    want.channels = 2;
    want.samples  = 2048;
    want.callback = audio_callback;
    want.userdata = &g_audio;

    SDL_AudioDeviceID audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
    if (audio_dev) SDL_PauseAudioDevice(audio_dev, 0);

    shs::Canvas   canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{ 14, 16, 22, 255 });
    shs::ZBuffer  z_buffer(CANVAS_WIDTH, CANVAS_HEIGHT, -1.0f, 1.0f);

    shs::Job::ThreadedPriorityJobSystem job_system(THREAD_COUNT);
    shs::Job::WaitGroup                 wg_render;

    vop::FrameMemoryResource frame_memory;

    // Persistent State
    tetris::TetrisSnapshot world;
    world.active.type = tetris::pull_next_piece(world.rng_state, world.next_queue);

    tetris::ShatterParticleSoA particles(std::pmr::get_default_resource());
    float camera_shake = 0.0f;

    bool quit = false;
    SDL_Event e;
    Uint32 last_tick = SDL_GetTicks();

    while (!quit) {
        Uint32 cur_tick = SDL_GetTicks();
        float dt = (cur_tick - last_tick) / 1000.0f;
        last_tick = cur_tick;
        if (dt > 0.05f) dt = 0.05f;

        frame_memory.reset();
        auto* arena = frame_memory.get();

        // 1. INPUT POLLING
        std::pmr::vector<tetris::TetrisCommand> commands(arena);
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) quit = true;
                if (e.key.keysym.sym == SDLK_r)     commands.push_back(tetris::RestartIntent{});
                if (e.key.keysym.sym == SDLK_LEFT  || e.key.keysym.sym == SDLK_a) commands.push_back(tetris::MoveLeftIntent{});
                if (e.key.keysym.sym == SDLK_RIGHT || e.key.keysym.sym == SDLK_d) commands.push_back(tetris::MoveRightIntent{});
                if (e.key.keysym.sym == SDLK_UP    || e.key.keysym.sym == SDLK_w) commands.push_back(tetris::RotateCWIntent{});
                if (e.key.keysym.sym == SDLK_z)     commands.push_back(tetris::RotateCCWIntent{});
                if (e.key.keysym.sym == SDLK_DOWN  || e.key.keysym.sym == SDLK_s) commands.push_back(tetris::SoftDropIntent{});
                if (e.key.keysym.sym == SDLK_SPACE) commands.push_back(tetris::HardDropIntent{});
                if (e.key.keysym.sym == SDLK_c     || e.key.keysym.sym == SDLK_LSHIFT) commands.push_back(tetris::HoldPieceIntent{});
            }
        }

        // 2. PURE REDUCTION (Calling reduce_tetris)
        tetris::TetrisStepResult step = tetris::reduce_tetris(world, commands, dt, arena);
        world = step.next_state;

        // 3. DISCRETE EVENT CONSUMPTION
        if (camera_shake > 0.0f) camera_shake = std::max(0.0f, camera_shake - dt * 4.0f);

        for (const auto& ev : step.events) {
            switch (ev.type) {
                case tetris::TetrisEventType::PIECE_MOVED:       g_audio.play(SND_MOVE); break;
                case tetris::TetrisEventType::PIECE_ROTATED:     g_audio.play(SND_ROTATE); break;
                case tetris::TetrisEventType::PIECE_LOCK_IMPACT: g_audio.play(SND_DROP_SLAM); break;
                case tetris::TetrisEventType::HOLD_SWAPPED:      g_audio.play(SND_HOLD); break;
                case tetris::TetrisEventType::GAME_OVER:         g_audio.play(SND_GAME_OVER); break;
                case tetris::TetrisEventType::HARD_DROP_SLAM:
                    g_audio.play(SND_DROP_SLAM);
                    camera_shake = 0.35f;
                    break;
                case tetris::TetrisEventType::LINES_CLEARED:
                    if (ev.lines_cleared_count >= 4) {
                        g_audio.play(SND_TETRIS_FOUR);
                        camera_shake = 0.65f;
                    } else {
                        g_audio.play(SND_LINE_CLEAR);
                        camera_shake = 0.25f;
                    }
                    // Spawn 3D Shatter Voxel Particles
                    for (int i = 0; i < ev.lines_cleared_count; ++i) {
                        float row_y = (float)ev.cleared_rows[i];
                        for (int col = 0; col < tetris::GRID_W; ++col) {
                            glm::vec3 p((float)col - 4.5f, row_y, 0.0f);
                            glm::vec3 vel(
                                ((col - 4.5f) * 1.2f) + ((rand() % 100) / 50.0f - 1.0f),
                                3.0f + ((rand() % 100) / 30.0f),
                                -2.5f - ((rand() % 100) / 40.0f)
                            );
                            particles.add(p, vel, shs::Color{ 40, 220, 240, 255 }, 1.2f);
                        }
                    }
                    break;
                default: break;
            }
        }

        // 4. UPDATE PARTICLES
        for (size_t i = 0; i < particles.position.size();) {
            particles.position[i] += particles.velocity[i] * dt;
            particles.velocity[i].y -= 18.0f * dt; // Gravity
            particles.life[i] -= dt;
            if (particles.life[i] <= 0.0f) {
                particles.position.erase(particles.position.begin() + i);
                particles.velocity.erase(particles.velocity.begin() + i);
                particles.color.erase(particles.color.begin() + i);
                particles.life.erase(particles.life.begin() + i);
            } else {
                ++i;
            }
        }

        // 5. PURE 3D BATCH PLANNER
        tetris::PipelineExecutionPlan plan = tetris::plan_tetris_scene(
            world, particles, CANVAS_WIDTH, CANVAS_HEIGHT, camera_shake, arena
        );

        // 6. TILED PARALLEL RASTERIZATION
        canvas.buffer().clear(shs::Color{ 14, 16, 22, 255 });
        z_buffer.clear();

        int W    = canvas.get_width();
        int H    = canvas.get_height();
        int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wg_render.reset();
        for (int ty = 0; ty < rows; ++ty) {
            for (int tx = 0; tx < cols; ++tx) {
                wg_render.add(1);
                job_system.submit({ [&, tx, ty, W, H]() {
                    glm::ivec2 tmin(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                    glm::ivec2 tmax(std::min((tx + 1) * TILE_SIZE_X, W) - 1, std::min((ty + 1) * TILE_SIZE_Y, H) - 1);

                    for (const auto& tri : plan.triangles) {
                        const shs::Raster::FrustumClipPolygon poly =
                            shs::Raster::clip_triangle_to_frustum(tri.c0, tri.c1, tri.c2);
                        if (poly.count < 3) continue;

                        glm::vec4 s0 = vop::clip_to_screen_vec4(poly.vertices[0], W, H);
                        for (int i = 1; i + 1 < poly.count; ++i) {
                            glm::vec4 s1 = vop::clip_to_screen_vec4(poly.vertices[i], W, H);
                            glm::vec4 s2 = vop::clip_to_screen_vec4(poly.vertices[i + 1], W, H);
                            vop::rasterize_triangle_tile(canvas, z_buffer, s0, s1, s2, tri.lit_color, tri.depth_bias, tmin, tmax);
                        }
                    }
                    wg_render.done();
                }, shs::Job::PRIORITY_HIGH });
            }
        }
        wg_render.wait();

        // 7. DRAW 2D HUD
        draw_hud(canvas, world);

        // 8. SWAPCHAIN PRESENTATION
        shs::Canvas::copy_to_SDLSurface(screen_surface, &canvas);
        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderClear(sdl_renderer);
        SDL_RenderCopy(sdl_renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(sdl_renderer);
    }

    if (audio_dev) SDL_CloseAudioDevice(audio_dev);
    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(screen_surface);
    SDL_DestroyRenderer(sdl_renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}