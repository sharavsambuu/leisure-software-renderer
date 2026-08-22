#pragma once

// ============================================================================
// fps/edges/ui/fps.hud.hpp — 2D HUD / OVERLAY EDGE (fps::ui)
// Screen-space (top-left origin) drawing: Bresenham lines, filled/bordered
// rects, 7-segment digits, enemy health bars, tracer overlay, and the main
// HUD (crosshair, hitmarker, score panel, HP bar, damage vignette).
// Runs AFTER rasterization; draws directly onto the Canvas.
// ============================================================================

#include <cstdint>

#include <glm/glm.hpp>

#include "shs_renderer.hpp"

#include <domains/matrix/fps.contract.hpp>
#include <domains/spatial_fx/spatial_fx.contract.hpp>

namespace fps::ui {

    // --- Primitives -----------------------------------------------------------
    inline void draw_line_screen_space(shs::Canvas& canvas, int x0, int y0, int x1, int y1, shs::Color color) {
        int dx  = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy  = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2;
        for (;;) {
            canvas.draw_pixel_screen_space(x0, y0, color);
            if (x0 == x1 && y0 == y1) break;
            e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }

    inline void draw_rect_fill_screen(shs::Canvas& canvas, int x, int y, int w, int h, shs::Color col) {
        const int x0 = std::max(0, x);
        const int y0 = std::max(0, y);
        const int x1 = std::min(canvas.get_width() - 1, x + w);
        const int y1 = std::min(canvas.get_height() - 1, y + h);
        for (int py = y0; py <= y1; ++py) {
            for (int px = x0; px <= x1; ++px) {
                canvas.draw_pixel_screen_space(px, py, col);
            }
        }
    }

    inline void draw_rect_border_screen(shs::Canvas& canvas, int x, int y, int w, int h, shs::Color col) {
        const int x1 = std::min(canvas.get_width() - 1, x + w);
        const int y1 = std::min(canvas.get_height() - 1, y + h);
        for (int px = std::max(0, x); px <= x1; ++px) {
            canvas.draw_pixel_screen_space(px, y, col);
            canvas.draw_pixel_screen_space(px, y1, col);
        }
        for (int py = std::max(0, y); py <= y1; ++py) {
            canvas.draw_pixel_screen_space(x, py, col);
            canvas.draw_pixel_screen_space(x1, py, col);
        }
    }

    inline void draw_digit_screen(shs::Canvas& canvas, int x, int y, int d, int w, int h, shs::Color col) {
        static constexpr uint8_t segs[10] = {
            0b00111111, 0b00000110, 0b01011011, 0b01001111, 0b01100110,
            0b01101101, 0b01111101, 0b00000111, 0b01111111, 0b01101111
        };
        if (d < 0 || d > 9) return;
        const uint8_t mask  = segs[d];
        const int     mid_y = y + h / 2;

        auto line = [&](int lx0, int ly0, int lx1, int ly1) {
            draw_line_screen_space(canvas, lx0, ly0, lx1, ly1, col);
        };

        if (mask & (1 << 0)) line(x    , y    , x + w, y    );
        if (mask & (1 << 1)) line(x + w, y    , x + w, mid_y);
        if (mask & (1 << 2)) line(x + w, mid_y, x + w, y + h);
        if (mask & (1 << 3)) line(x    , y + h, x + w, y + h);
        if (mask & (1 << 4)) line(x    , mid_y, x    , y + h);
        if (mask & (1 << 5)) line(x    , y    , x    , mid_y);
        if (mask & (1 << 6)) line(x    , mid_y, x + w, mid_y);
    }

    inline void draw_number_screen(shs::Canvas& canvas, int x, int y, int val, int digits, shs::Color col) {
        constexpr int w   = 10;
        constexpr int h   = 18;
        constexpr int gap = 5;
        for (int i = digits - 1; i >= 0; --i) {
            const int d = val % 10;
            val /= 10;
            draw_digit_screen(canvas, x + i * (w + gap), y, d, w, h, col);
        }
    }

    // --- World-anchored overlays -------------------------------------------------
    inline void draw_enemy_health_bars(shs::Canvas& canvas, const glm::mat4& vp,
                                       const matrix::BotTableSoA& bots) {
        const int W = canvas.get_width();
        const int H = canvas.get_height();

        for (size_t i = 0; i < bots.size(); ++i) {
            if (bots.state[i] == matrix::BotState::DEAD) continue;

            const glm::vec3 head_top = bots.position[i] + glm::vec3(0, 2.05f, 0);
            const glm::vec4 clip     = vp * glm::vec4(head_top, 1.0f);

            if (clip.w <= 0.15f) continue;

            const glm::vec3 sc = shs::Canvas::clip_to_screen(clip, W, H);
            if (sc.x < -60.0f || sc.x > static_cast<float>(W) + 60.0f
                || sc.y < -60.0f || sc.y > static_cast<float>(H) + 60.0f) continue;

            const int bar_w = static_cast<int>(glm::clamp(54.0f / (clip.w * 0.08f + 0.5f), 22.0f, 50.0f));
            constexpr int bar_h = 4;
            const int bx = static_cast<int>(sc.x) - bar_w / 2;
            const int by = static_cast<int>(sc.y) - 10;

            draw_rect_fill_screen(canvas, bx - 1, by - 1, bar_w + 2, bar_h + 2, shs::Color{ 10, 12, 16, 230 });
            draw_rect_fill_screen(canvas, bx, by, bar_w, bar_h, shs::Color{ 80, 20, 20, 255 });

            const float hp_pct = glm::clamp(static_cast<float>(bots.hp[i]) / 100.0f, 0.0f, 1.0f);
            const int   fill_w = static_cast<int>(hp_pct * static_cast<float>(bar_w));

            const shs::Color hp_col = (bots.hit_flash_time[i] > 0.0f) ? shs::Color{ 255, 255, 255, 255 }
                                    : (hp_pct > 0.5f)                 ? shs::Color{ 240, 70, 50, 255 }
                                                                      : shs::Color{ 255, 255, 255, 255 };

            draw_rect_fill_screen(canvas, bx, by, fill_w, bar_h, hp_col);
        }
    }

    inline void draw_tracers(shs::Canvas& canvas, const glm::mat4& vp,
                             std::span<const matrix::BulletTracer> tracers) {
        const int W = canvas.get_width();
        const int H = canvas.get_height();
        for (const auto& tr : tracers) {
            const glm::vec4 c_start = vp * glm::vec4(tr.start, 1.0f);
            const glm::vec4 c_end   = vp * glm::vec4(tr.end, 1.0f);
            if (c_start.w > 0.1f && c_end.w > 0.1f) {
                const glm::vec3 s0 = shs::Canvas::clip_to_screen(c_start, W, H);
                const glm::vec3 s1 = shs::Canvas::clip_to_screen(c_end, W, H);
                draw_line_screen_space(canvas, static_cast<int>(s0.x), static_cast<int>(s0.y),
                                       static_cast<int>(s1.x), static_cast<int>(s1.y),
                                       shs::Color{ 255, 230, 100, 255 });
            }
        }
    }

    // --- Main HUD -------------------------------------------------------------------
    inline void draw_fps_hud(shs::Canvas& canvas, const matrix::PlayerSnapshot& player,
                             float hitmarker_timer, int32_t score) {
        const int W  = canvas.get_width();
        const int H  = canvas.get_height();
        const int cx = W / 2;
        const int cy = H / 2;

        if (player.damage_flash > 0.0f) {
            const shs::Color red_border{ 255, 30, 30, 200 };
            for (int i = 0; i < 8; ++i) {
                draw_rect_border_screen(canvas, i, i, W - 1 - i * 2, H - 1 - i * 2, red_border);
            }
        }

        // Score panel (top-right)
        const int sc_x = W - 180;
        const int sc_y = 25;
        draw_rect_fill_screen(canvas, sc_x - 10, sc_y - 8, 165, 40, shs::Color{ 15, 18, 25, 230 });
        draw_rect_border_screen(canvas, sc_x - 10, sc_y - 8, 165, 40, shs::Color{ 90, 100, 120, 255 });
        draw_number_screen(canvas, sc_x, sc_y + 3, score, 6, shs::Color{ 255, 215, 60, 255 });

        // HP bar (bottom-left)
        const int hp_x = 35;
        const int hp_y = H - 55;
        constexpr int hp_w = 220;
        constexpr int hp_h = 18;

        draw_rect_fill_screen(canvas, hp_x - 4, hp_y - 4, hp_w + 8, hp_h + 8, shs::Color{ 15, 18, 25, 230 });
        draw_rect_border_screen(canvas, hp_x - 4, hp_y - 4, hp_w + 8, hp_h + 8, shs::Color{ 90, 100, 120, 255 });
        draw_rect_fill_screen(canvas, hp_x, hp_y, hp_w, hp_h, shs::Color{ 45, 20, 20, 255 });

        const float hp_ratio = glm::clamp(static_cast<float>(player.hp) / 100.0f, 0.0f, 1.0f);
        const int   fill_w   = static_cast<int>(hp_ratio * static_cast<float>(hp_w));
        const shs::Color hp_fill = (player.hp > 35) ? shs::Color{ 45, 220, 95, 255 } : shs::Color{ 240, 40, 40, 255 };
        draw_rect_fill_screen(canvas, hp_x, hp_y, fill_w, hp_h, hp_fill);
        draw_number_screen(canvas, hp_x + hp_w + 14, hp_y, player.hp, 3, hp_fill);

        // Crosshair + hitmarker
        const shs::Color ch_color = (hitmarker_timer > 0.0f) ? shs::Color{ 255, 50, 50, 255 } : shs::Color{ 255, 255, 255, 220 };
        constexpr int ch_size = 7;
        constexpr int ch_gap  = 3;
        draw_line_screen_space(canvas, cx - ch_size - ch_gap, cy, cx - ch_gap, cy, ch_color);
        draw_line_screen_space(canvas, cx + ch_gap, cy, cx + ch_size + ch_gap, cy, ch_color);
        draw_line_screen_space(canvas, cx, cy - ch_size - ch_gap, cx, cy - ch_gap, ch_color);
        draw_line_screen_space(canvas, cx, cy + ch_gap, cx, cy + ch_size + ch_gap, ch_color);

        if (hitmarker_timer > 0.0f) {
            const shs::Color hm_col{ 255, 60, 60, 255 };
            constexpr int s = 5;
            draw_line_screen_space(canvas, cx - s, cy - s, cx - 2, cy - 2, hm_col);
            draw_line_screen_space(canvas, cx + 2, cy + 2, cx + s, cy + s, hm_col);
            draw_line_screen_space(canvas, cx + 2, cy - 2, cx + s, cy - s, hm_col);
            draw_line_screen_space(canvas, cx - s, cy + s, cx - 2, cy + 2, hm_col);
        }
    }

} // namespace fps::ui