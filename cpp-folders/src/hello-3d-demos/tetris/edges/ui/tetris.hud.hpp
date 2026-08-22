#pragma once
// tetris/edges/ui/tetris.hud.hpp — 2D HUD + UTF-8 FONT ENGINE (tetris::ui)
// Pure projections of snapshots + edge-owned transient presentation state
// (HudState: banners/floaters — same ownership model as audio voices).
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <span>
#include <string>
#include "shs_renderer.hpp"
#include <domains/matrix/matrix.contract.hpp>
#include <domains/progression/progression.contract.hpp>
#include <domains/progression/progression.event.hpp>
#include <domains/session/session.contract.hpp>

namespace tetris::ui {

// 1. Immutable UTF-8 Mongolian String Constants (Immune to compiler codepage conversion)
// 
// These weird characters are raw hexadecimal UTF-8 bytes of Cyrillic Mongolian letter
// 
// ============================================================================
// MONGOLIAN CYRILLIC UTF - 8 LOOKUP TABLE(35 LETTERS)                      
// --------------------- + ----------------------------------------------------
// Letter(Upper / Lower) | UTF - 8 Hex Escape(Upper / Lower)                  
// --------------------- + ----------------------------------------------------
//  А / а                | \xD0\x90 / \xD0\xB0                                
//  Б / б                | \xD0\x91 / \xD0\xB1
//  В / в                | \xD0\x92 / \xD0\xB2
//  Г / г                | \xD0\x93 / \xD0\xB3
//  Д / д                | \xD0\x94 / \xD0\xB4
//  Е / е                | \xD0\x95 / \xD0\xB5
//  Ё / ё                | \xD0\x81 / \xD1\x91
//  Ж / ж                | \xD0\x96 / \xD0\xB6
//  З / з                | \xD0\x97 / \xD0\xB7
//  И / и                | \xD0\x98 / \xD0\xB8
//  Й / й                | \xD0\x99 / \xD0\xB9
//  К / к                | \xD0\x9A / \xD0\xBA
//  Л / л                | \xD0\x9B / \xD0\xBB
//  М / м                | \xD0\x9C / \xD0\xBC
//  Н / н                | \xD0\x9D / \xD0\xBD
//  О / о                | \xD0\x9E / \xD0\xBE
//  Ө / ө(Special)       | \xD3\xA8 / \xD3\xA9
//  П / п                | \xD0\x9F / \xD0\xBF
//  Р / р                | \xD0\xA0 / \xD1\x80
//  С / с                | \xD0\xA1 / \xD1\x81
//  Т / т                | \xD0\xA2 / \xD1\x82
//  У / у                | \xD0\xA3 / \xD1\x83
//  Ү / ү(Special)       | \xD2\xAE / \xD2\xAF
//  Ф / ф                | \xD0\xA4 / \xD1\x84
//  Х / х                | \xD0\xA5 / \xD1\x85
//  Ц / ц                | \xD0\xA6 / \xD1\x86
//  Ч / ч                | \xD0\xA7 / \xD1\x87
//  Ш / ш                | \xD0\xA8 / \xD1\x88
//  Щ / щ                | \xD0\xA9 / \xD1\x89
//  Ъ / ъ                | \xD0\xAA / \xD1\x8A
//  Ы / ы                | \xD0\xAB / \xD1\x8B
//  Ь / ь                | \xD0\xAC / \xD1\x8C
//  Э / э                | \xD0\xAD / \xD1\x8D
//  Ю / ю                | \xD0\xAE / \xD1\x8E
//  Я / я                | \xD0\xAF / \xD1\x8F
// --------------------- + ----------------------------------------------------
// 
// To generate characters on Linux shell:
//      $ printf "ЗОРИЛГО" | hexdump -ve '1/1 "\\\\x%02X"'
// To check what's on it
//      $ printf "\xD0\x97\xD0\x9E\xD0\xA0\xD0\x98\xD0\x9B\xD0\x93\xD0\x9E\n"
//
//

// ОНОО
static const char* TXT_SCORE       = "\xD0\x9E\xD0\x9D\xD0\x9E\xD0\x9E";             
// ДЭЭД
static const char* TXT_BEST        = "\xD0\x94\xD0\xAD\xD0\xAD\xD0\x94";             
// ЗОРИЛГО
static const char* TXT_GOAL        = "\xD0\x97\xD0\x9E\xD0\xA0\xD0\x98\xD0\x9B\xD0\x93\xD0\x9E"; 
// МӨР
static const char* TXT_LINES       = "\xD0\x9C\xD3\xA8\xD0\xA0";                     
// ҮЕ  (үе = level/stage; \xD2\xAE is Ү — \xD0\xAE would render Ю)
static const char* TXT_LEVEL       = "\xD2\xAE\xD0\x95";                             
// НӨӨЦ [C]
static const char* TXT_HOLD        = "\xD0\x9D\xD3\xA8\xD3\xA8\xD0\xA6 [C]";         
// ДАРААГИЙН
static const char* TXT_NEXT        = "\xD0\x94\xD0\x90\xD0\xA0\xD0\x90\xD0\x90\xD0\x93\xD0\x98\xD0\x99\xD0\x9D"; 
// ТОГЛООМ ДУУСЛАА
static const char* TXT_GAME_OVER   = "\xD0\xA2\xD0\x9E\xD0\x93\xD0\x9B\xD0\x9E\xD0\x9E\xD0\x9C \xD0\x94\xD0\xA3\xD0\xA3\xD0\xA1\xD0\x9B\xD0\x90\xD0\x90"; 
// ЗОРИЛГО БИЕЛЛЭЭ!
static const char* TXT_VICTORY     = "\xD0\x97\xD0\x9E\xD0\xA0\xD0\x98\xD0\x9B\xD0\x93\xD0\x9E \xD0\x91\xD0\x98\xD0\x95\xD0\x9B\xD0\x9B\xD0\xAD\xD0\xAD!"; 
// ЭЦСИЙН ОНОО:
static const char* TXT_FINAL_SCORE = "\xD0\xAD\xD0\xA6\xD0\xA1\xD0\x98\xD0\x99\xD0\x9D \xD0\x9E\xD0\x9D\xD0\x9E\xD0\x9E:"; 
// [R] - ДАХИН ТОГЛОХ
static const char* TXT_RETRY       = "[R] - \xD0\x94\xD0\x90\xD0\xA5\xD0\x98\xD0\x9D \xD0\xA2\xD0\x9E\xD0\x93\xD0\x9B\xD0\x9E\xD0\xA5"; 
static const char* TXT_FOOTER      = "A/D: \xD0\xA5\xD3\xA8\xD0\x94\xD0\x9B\xD3\xA8\xD0\xA5 | W: \xD0\xAD\xD0\xA0\xD0\x93\xD2\xAE\xD2\xAE\xD0\x9B\xD0\xAD\xD0\xA5 | S: \xD0\x91\xD0\xA3\xD0\xA3\xD0\x9B\xD0\x93\xD0\x90\xD0\xA5 | SPACE: \xD0\xA3\xD0\x9D\xD0\x90\xD0\x93\xD0\x90\xD0\x90\xD0\xA5 | C: \xD0\x9D\xD3\xA8\xD3\xA8\xD0\xA6 | R: \xD0\xAD\xD0\xA5\xD0\x9B\xD0\xAD\xD0\xA5";

// --- L1/L2 additions --------------------------------------------------------
// ЦАГ (time)
static const char* TXT_TIME        = "\xD0\xA6\xD0\x90\xD0\x93";
// ХУРДАЦГАА! (hurry up! — imperative; "хурдаа" is not a word)
static const char* TXT_HURRY       = "\xD0\xA5\xD0\xA3\xD0\xA0\xD0\x94\xD0\x90\xD0\xA6\xD0\x93\xD0\x90\xD0\x90!";
// КОМБО (combo)
static const char* TXT_COMBO       = "\xD0\x9A\xD0\x9E\xD0\x9C\xD0\x91\xD0\x9E";
// ДЭЭД КОМБО (best combo — matches ДЭЭД = best on the score card)
static const char* TXT_MAX_COMBO   = "\xD0\x94\xD0\xAD\xD0\xAD\xD0\x94 \xD0\x9A\xD0\x9E\xD0\x9C\xD0\x91\xD0\x9E";
// [R] - ДАРААГИЙН ҮЕ (next level — shown on the victory modal when a
// further campaign stage exists; R advances instead of restarting)
static const char* TXT_NEXT_STAGE  = "[R] - \xD0\x94\xD0\x90\xD0\xA0\xD0\x90\xD0\x90\xD0\x93\xD0\x98\xD0\x99\xD0\x9D \xD2\xAE\xD0\x95";
// НЭМЭЛТ ЦАГ (bonus time collected)
static const char* TXT_BONUS_TIME  = "\xD0\x9D\xD0\xAD\xD0\x9C\xD0\xAD\xD0\x9B\xD0\xA2 \xD0\xA6\xD0\x90\xD0\x93";
// МӨР ОНОО (line-clear points)
static const char* TXT_CLEAR_SCORE = "\xD0\x9C\xD3\xA8\xD0\xA0 \xD0\x9E\xD0\x9D\xD0\x9E\xD0\x9E";
// УНАлТ ОНОО (drop points)
static const char* TXT_DROP_SCORE  = "\xD0\xA3\xD0\x9D\xD0\x90\xD0\x9B\xD0\xA2 \xD0\x9E\xD0\x9D\xD0\x9E\xD0\x9E";
// ЦАГ ДУУСЛАА (time up)
static const char* TXT_TIME_UP     = "\xD0\xA6\xD0\x90\xD0\x93 \xD0\x94\xD0\xA3\xD0\xA3\xD0\xA1\xD0\x9B\xD0\x90\xD0\x90";

// --- Session / menu strings -------------------------------------------------
// ТЕТРИС (logo)
static const char* TXT_LOGO        = "\xD0\xA2\xD0\x95\xD0\xA2\xD0\xA0\xD0\x98\xD0\xA1";
// ЭХЛҮҮЛЭХ (start)
static const char* TXT_START       = "\xD0\xAD\xD0\xA5\xD0\x9B\xD2\xAE\xD2\xAE\xD0\x9B\xD0\xAD\xD0\xA5";
// ҮЕ СОНГОХ (select level)
static const char* TXT_SELECT_LVL  = "\xD2\xAE\xD0\x95 \xD0\xA1\xD0\x9E\xD0\x9D\xD0\x93\xD0\x9E\xD0\xA5";
// ДУУН (sound)
static const char* TXT_SOUND       = "\xD0\x94\xD0\xA3\xD0\xA3\xD0\x9D";
// ИДЭВХТЭЙ (on)
static const char* TXT_ON          = "\xD0\x98\xD0\x94\xD0\xAD\xD0\x92\xD0\xA5\xD0\xA2\xD0\xAD\xD0\x99";
// ИДЭВХГҮЙ (off)
static const char* TXT_OFF         = "\xD0\x98\xD0\x94\xD0\xAD\xD0\x92\xD0\xA5\xD0\x93\xD2\xAE\xD0\x99";
// ГАРАХ (exit)
static const char* TXT_EXIT        = "\xD0\x93\xD0\x90\xD0\xA0\xD0\x90\xD0\xA5";
// ҮРГЭЛЖЛҮҮЛЭХ (resume)
static const char* TXT_RESUME      = "\xD2\xAE\xD0\xA0\xD0\x93\xD0\xAD\xD0\x9B\xD0\x96\xD0\x9B\xD2\xAE\xD2\xAE\xD0\x9B\xD0\xAD\xD0\xA5";
// ДАХИН ЭХЛҮҮЛЭХ (restart stage)
static const char* TXT_RESTART     = "\xD0\x94\xD0\x90\xD0\xA5\xD0\x98\xD0\x9D \xD0\xAD\xD0\xA5\xD0\x9B\xD2\xAE\xD2\xAE\xD0\x9B\xD0\xAD\xD0\xA5";
// ТҮР ЗОГСООХ (paused)
static const char* TXT_PAUSED      = "\xD0\xA2\xD2\xAE\xD0\xA0 \xD0\x97\xD0\x9E\xD0\x93\xD0\xA1\xD0\x9E\xD0\x9E\xD0\xA5";
// ТҮГЖЭЭ (locked)
static const char* TXT_LOCKED      = "\xD0\xA2\xD2\xAE\xD0\x93\xD0\x96\xD0\xAD\xD0\xAD";
// БУЦАХ (back to title)
static const char* TXT_BACK        = "\xD0\x91\xD0\xA3\xD0\xA6\xD0\x90\xD0\xA5";
// ДАРААГИЙН ҮЕ (next level, plain — menu row form)
static const char* TXT_NEXT_PLAIN  = "\xD0\x94\xD0\x90\xD0\xA0\xD0\x90\xD0\x90\xD0\x93\xD0\x98\xD0\x99\xD0\x9D \xD2\xAE\xD0\x95";
// ЦЭВЭР C++ (pure-C++ tier tag)
static const char* TAG_PURE        = "\xD0\xA6\xD0\xAD\xD0\x92\xD0\xAD\xD0\xA0 C++";
// СКРИПТТЭЙ (Lua-scripted tier tag)
static const char* TAG_SCRIPTED    = "\xD0\xA1\xD0\x9A\xD0\xA0\xD0\x98\xD0\x9F\xD0\xA2\xD0\xA2\xD0\xAD\xD0\x99";
// Footer hints: "W/S - СОНГОХ", "ENTER - БАТАЛГАА", "ESC - БУЦАХ", "M - ДУУН"
static const char* HINT_NAV        = "W/S/A/D - \xD0\xA1\xD0\x9E\xD0\x9D\xD0\x93\xD0\x9E\xD0\xA5";
static const char* HINT_CONFIRM    = "ENTER - \xD0\x91\xD0\x90\xD0\xA2\xD0\x90\xD0\x9B\xD0\x93\xD0\x90\xD0\x90";
static const char* HINT_BACK       = "ESC/P - \xD0\x91\xD0\xA3\xD0\xA6\xD0\x90\xD0\xA5";
static const char* HINT_SOUND      = "M - \xD0\x94\xD0\xA3\xD0\xA3\xD0\x9D";

// Line drawing
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

// Filled rectangle
static void draw_rect_fill(shs::Canvas& c, int x, int y, int w, int h, shs::Color col) {
    int x0 = std::max(0, x), y0 = std::max(0, y);
    int x1 = std::min(c.get_width() - 1, x + w), y1 = std::min(c.get_height() - 1, y + h);
    for (int py = y0; py <= y1; ++py) {
        for (int px = x0; px <= x1; ++px) c.draw_pixel_screen_space(px, py, col);
    }
}

// Rectangle outline
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

// Dithered fill (checkerboard) — fake translucency on the opaque canvas.
static void draw_rect_fill_dithered(shs::Canvas& c, int x, int y, int w, int h, shs::Color col, int phase = 0) {
    int x0 = std::max(0, x), y0 = std::max(0, y);
    int x1 = std::min(c.get_width() - 1, x + w), y1 = std::min(c.get_height() - 1, y + h);
    for (int py = y0; py <= y1; ++py) {
        for (int px = x0; px <= x1; ++px) {
            if (((px + py + phase) & 1) == 0) c.draw_pixel_screen_space(px, py, col);
        }
    }
}

// UTF-8 codepoint count (for centering: Cyrillic is 2 bytes but 1 glyph)
static int utf8_char_count(const char* s) {
    int n = 0;
    for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
        if ((*p & 0xC0) != 0x80) ++n;
    }
    return n;
}

static int text_width_px(const char* s, int scale) {
    return utf8_char_count(s) * 6 * scale - scale;
}

// Defined below (after the font engine); forward-declared for the helpers.
static void draw_text(shs::Canvas& c, int x, int y, const char* str, shs::Color col, int scale);

static void draw_text_centered(shs::Canvas& c, int cx, int y, const char* s, shs::Color col, int scale = 2) {
    draw_text(c, cx - text_width_px(s, scale) / 2, y, s, col, scale);
}

// M:SS clock formatter ("2:00", floor semantics: 119.27 -> "1:59")
static void format_clock(char* buf, int cap, float seconds) {
    if (seconds < 0.0f) seconds = 0.0f;
    const int total = (int)seconds;
    std::snprintf(buf, (size_t)cap, "%d:%02d", total / 60, total % 60);
}

// Local color lerp (same math as spatial_fx vocabulary; kept edge-local)
static shs::Color hud_lerp_color(shs::Color a, shs::Color b, float t) {
    t = glm::clamp(t, 0.0f, 1.0f);
    return shs::Color{
        (uint8_t)(a.r + (b.r - a.r) * t + 0.5f),
        (uint8_t)(a.g + (b.g - a.g) * t + 0.5f),
        (uint8_t)(a.b + (b.b - a.b) * t + 0.5f),
        a.a
    };
}

// Standard ASCII Font Glyphs (ASCII 32 to 90)
static const uint8_t FONT_ASCII[][5] = {
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

// Complete Cyrillic Font Glyphs (35 Mongolian Letters)
static const uint8_t FONT_CYRILLIC[][5] = {
    {0x7E,0x11,0x11,0x11,0x7E}, // 0:  А
    {0x7F,0x49,0x49,0x49,0x31}, // 1:  Б
    {0x7F,0x49,0x49,0x49,0x36}, // 2:  В
    {0x7F,0x01,0x01,0x01,0x01}, // 3:  Г
    {0x60,0x3E,0x21,0x3E,0x60}, // 4:  Д
    {0x7F,0x49,0x49,0x49,0x41}, // 5:  Е
    {0x77,0x08,0x7F,0x08,0x77}, // 6:  Ж
    {0x21,0x41,0x45,0x4B,0x31}, // 7:  З
    {0x7F,0x20,0x10,0x08,0x7F}, // 8:  И
    {0x7D,0x21,0x12,0x09,0x7D}, // 9:  Й
    {0x7F,0x08,0x14,0x22,0x41}, // 10: К
    {0x70,0x0E,0x01,0x01,0x7F}, // 11: Л
    {0x7F,0x02,0x0C,0x02,0x7F}, // 12: М
    {0x7F,0x08,0x08,0x08,0x7F}, // 13: Н
    {0x3E,0x41,0x41,0x41,0x3E}, // 14: О
    {0x7F,0x01,0x01,0x01,0x7F}, // 15: П
    {0x7F,0x09,0x09,0x09,0x06}, // 16: Р
    {0x3E,0x41,0x41,0x41,0x22}, // 17: С
    {0x01,0x01,0x7F,0x01,0x01}, // 18: Т
    {0x07,0x08,0x70,0x08,0x07}, // 19: У
    {0x1C,0x22,0x7F,0x22,0x1C}, // 20: Ф
    {0x63,0x14,0x08,0x14,0x63}, // 21: Х
    {0x7F,0x40,0x40,0x7F,0x60}, // 22: Ц
    {0x07,0x04,0x04,0x04,0x7F}, // 23: Ч
    {0x7F,0x40,0x7F,0x40,0x7F}, // 24: Ш
    {0x7F,0x40,0x7F,0x7F,0x60}, // 25: Щ
    {0x01,0x7F,0x48,0x48,0x30}, // 26: Ъ
    {0x7F,0x48,0x30,0x00,0x7F}, // 27: Ы
    {0x7F,0x48,0x48,0x48,0x30}, // 28: Ь
    {0x22,0x41,0x49,0x49,0x3E}, // 29: Э
    {0x7F,0x08,0x3E,0x41,0x3E}, // 30: Ю
    {0x46,0x29,0x19,0x09,0x7F}, // 31: Я
    {0x7D,0x48,0x49,0x48,0x41}, // 32: Ё
    {0x3E,0x49,0x49,0x49,0x3E}, // 33: Ө
    {0x07,0x08,0x7F,0x08,0x07}  // 34: Ү
};

// Codepoint to Font Glyph Resolver
static const uint8_t* get_font_glyph(uint32_t cp) {
    if (cp >= ' ' && cp <= 'Z') return FONT_ASCII[cp - ' '];
    if (cp >= 'a' && cp <= 'z') return FONT_ASCII[cp - 'a' + ('A' - ' ')];
    if (cp == '[') return FONT_ASCII['(' - ' '];
    if (cp == ']') return FONT_ASCII[')' - ' '];
    if (cp == '|') return FONT_ASCII['/' - ' '];
    if (cp == '_') return FONT_ASCII['-' - ' '];

    // Cyrillic Lowercase to Uppercase Normalization
    if (cp >= 0x0430 && cp <= 0x044F) cp -= 0x20;
    if (cp == 0x0451) cp = 0x0401; // ё -> Ё
    if (cp == 0x04E9) cp = 0x04E8; // ө -> Ө
    if (cp == 0x04AF) cp = 0x04AE; // ү -> Ү

    // Cyrillic Unicode Mapping
    if (cp >= 0x0410 && cp <= 0x042F) return FONT_CYRILLIC[cp - 0x0410];
    if (cp == 0x0401) return FONT_CYRILLIC[32]; // Ё
    if (cp == 0x04E8) return FONT_CYRILLIC[33]; // Ө
    if (cp == 0x04AE) return FONT_CYRILLIC[34]; // Ү

    return FONT_ASCII['?' - ' '];
}

// UTF-8 Text Drawing Function
static void draw_text(shs::Canvas& c, int x, int y, const char* str, shs::Color col, int scale = 2) {
    if (!str) return;
    int cur_x = x;
    const unsigned char* p = (const unsigned char*)str;

    while (*p) {
        uint32_t cp = 0;
        unsigned char c0 = *p++;

        if (c0 < 0x80) {
            // 1-byte ASCII
            cp = c0;
        }
        else if ((c0 & 0xE0) == 0xC0 && *p && (*p & 0xC0) == 0x80) {
            // 2-byte UTF-8 Sequence
            unsigned char c1 = *p++;
            cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        }
        else if ((c0 & 0xF0) == 0xE0 && *p && (*(p + 1)) && (*p & 0xC0) == 0x80 && (*(p + 1) & 0xC0) == 0x80) {
            // 3-byte UTF-8 Sequence
            unsigned char c1 = *p++;
            unsigned char c2 = *p++;
            cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        }
        else {
            cp = c0;
        }

        const uint8_t* glyph = get_font_glyph(cp);
        if (glyph) {
            for (int col_i = 0; col_i < 5; ++col_i) {
                uint8_t bits = glyph[col_i];
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

// Bold 7-segment digit drawing (2px stroke thickness)
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

    if (mask & (1 << 0)) h_seg(x, y);
    if (mask & (1 << 1)) v_seg(x + w - 2, y, my - y);
    if (mask & (1 << 2)) v_seg(x + w - 2, my, y + h - my);
    if (mask & (1 << 3)) h_seg(x, y + h - 2);
    if (mask & (1 << 4)) v_seg(x, my, y + h - my);
    if (mask & (1 << 5)) v_seg(x, y, my - y);
    if (mask & (1 << 6)) h_seg(x, my - 1);
}

// Multi-digit integer drawer
static void draw_number_bold(shs::Canvas& c, int x, int y, int val, int digits, shs::Color col) {
    int w = 12, h = 20, gap = 5;
    for (int i = digits - 1; i >= 0; --i) {
        int d = val % 10;
        val /= 10;
        draw_digit_bold(c, x + i * (w + gap), y, d, w, h, col);
    }
}

// ============================================================================
// HUD TRANSIENT PRESENTATION STATE (edge-owned, like audio voices)
// Pure projection inputs: progression events + score snapshot. No gameplay
// state lives here — timers drive banners/floaters only.
// ============================================================================
struct Floater {
    float      life     = 0.0f;
    float      max_life = 1.3f;
    char       text[24] = {};
    shs::Color color{ 255, 255, 255, 255 };
};

struct HudState {
    float   time          = 0.0f;   // hud animation clock
    float   levelup_timer = 0.0f;   // >0 while the LEVEL-UP banner shows
    int     levelup_level = 0;
    Floater floaters[8];
    int     next_floater  = 0;

    void spawn_floater(const char* txt, shs::Color col, float life = 1.3f) {
        Floater& f = floaters[next_floater];
        next_floater = (next_floater + 1) % 8;
        f.life = life;
        f.max_life = life;
        f.color = col;
        std::snprintf(f.text, sizeof(f.text), "%s", txt);
    }
};

static void step_hud(HudState& hud,
                     std::span<const progression::ProgressionEvent> events,
                     float dt) {
    hud.time += dt;
    if (hud.levelup_timer > 0.0f) {
        hud.levelup_timer = std::max(0.0f, hud.levelup_timer - dt);
    }

    for (const auto& ev : events) {
        switch (ev.type) {
        case progression::ProgressionEventType::LEVEL_UP:
            hud.levelup_timer = 2.2f;
            hud.levelup_level = ev.new_level;
            break;
        case progression::ProgressionEventType::COMBO_STREAK: {
            char buf[24];
            std::snprintf(buf, sizeof(buf),
                          "\xD0\x9A\xD0\x9E\xD0\x9C\xD0\x91\xD0\x9E x%d", ev.combo);
            hud.spawn_floater(buf, shs::Color{ 40, 220, 240, 255 });
            break;
        }
        case progression::ProgressionEventType::TIME_BONUS: {
            char buf[24];
            std::snprintf(buf, sizeof(buf), "+%ds", (int)ev.seconds);
            hud.spawn_floater(buf, shs::Color{ 45, 240, 110, 255 });
            break;
        }
        case progression::ProgressionEventType::TIME_UP:
            hud.spawn_floater(TXT_TIME_UP, shs::Color{ 245, 55, 55, 255 }, 2.0f);
            break;
        default:
            break;
        }
    }

    for (auto& f : hud.floaters) {
        if (f.life > 0.0f) f.life = std::max(0.0f, f.life - dt);
    }
}

// ============================================================================
// MONGOLIAN CYRILLIC HUD (Layout & Presentation)
// ============================================================================
static void draw_hud(shs::Canvas& canvas, const matrix::MatrixSnapshot& m,
                     const progression::ScoreState& sc, HudState& hud,
                     bool campaign_has_next = false) {
    int W = canvas.get_width();
    int H = canvas.get_height();

    const bool blitz = (sc.mode_id == progression::MODE_BLITZ_120);
    const bool timed = blitz && !sc.time_up;   // countdown visible while clock runs

    // Level-up flash factor → brief gold palette shift on card accents.
    const float lvl_flash = (hud.levelup_timer > 0.0f) ? (hud.levelup_timer / 2.2f) : 0.0f;
    auto accent = [&](shs::Color base) {
        return hud_lerp_color(base, shs::Color{ 255, 200, 60, 255 }, lvl_flash * 0.8f);
    };

    // ------------------------------------------------------------------------
    // 1. TOP RIGHT: SCORE CARD (ОНОО / ДЭЭД)
    // ------------------------------------------------------------------------
    int sx = W - 265, sy = 18, sw = 245, sh = 88;
    draw_rect_fill(canvas, sx, sy, sw, sh, shs::Color{ 15, 18, 26, 230 });
    draw_rect_border(canvas, sx, sy, sw, sh, accent(shs::Color{ 60, 140, 220, 255 }));

    draw_text(canvas, sx + 14, sy + 14, TXT_SCORE, shs::Color{ 255, 225, 45, 255 }, 2);
    draw_number_bold(canvas, sx + 125, sy + 12, sc.score, 6, shs::Color{ 255, 225, 45, 255 });

    draw_text(canvas, sx + 14, sy + 48, TXT_BEST, shs::Color{ 140, 155, 175, 255 }, 2);
    draw_number_bold(canvas, sx + 125, sy + 46, sc.high_score, 6, shs::Color{ 140, 155, 175, 255 });

    // ------------------------------------------------------------------------
    // 2. TOP LEFT: GOAL & STATS CARD (ЗОРИЛГО / МӨР / ҮЕ)
    // ------------------------------------------------------------------------
    int ox = 20, oy = 18, ow = 280, oh = 88;
    draw_rect_fill(canvas, ox, oy, ow, oh, shs::Color{ 15, 18, 26, 230 });
    draw_rect_border(canvas, ox, oy, ow, oh, accent(shs::Color{ 60, 140, 220, 255 }));

    // Target Progress Bar
    draw_text(canvas, ox + 12, oy + 12, TXT_GOAL, shs::Color{ 45, 220, 120, 255 }, 2);
    int bar_x = ox + 105, bar_y = oy + 12, bar_w = ow - 120, bar_h = 14;
    float progress = glm::clamp((float)sc.score / (float)sc.target_score, 0.0f, 1.0f);
    draw_rect_fill(canvas, bar_x, bar_y, bar_w, bar_h, shs::Color{ 35, 40, 52, 255 });
    draw_rect_fill(canvas, bar_x, bar_y, (int)(progress * (float)bar_w), bar_h, shs::Color{ 45, 220, 120, 255 });
    draw_rect_border(canvas, bar_x, bar_y, bar_w, bar_h, shs::Color{ 80, 95, 115, 255 });

    // Lines & Level
    draw_text(canvas, ox + 14, oy + 48, TXT_LINES, shs::Color{ 40, 220, 240, 255 }, 2);
    draw_number_bold(canvas, ox + 65, oy + 46, sc.lines_cleared, 3, shs::Color{ 40, 220, 240, 255 });

    draw_text(canvas, ox + 155, oy + 48, TXT_LEVEL, shs::Color{ 255, 140, 35, 255 }, 2);
    draw_number_bold(canvas, ox + 205, oy + 46, sc.level, 2, shs::Color{ 255, 140, 35, 255 });

    // ------------------------------------------------------------------------
    // 3. BLITZ COUNTDOWN PANEL (top-center, large digits)
    //    amber → red < 30s → pulsing red < 10s
    // ------------------------------------------------------------------------
    if (timed) {
        char buf[16];
        format_clock(buf, sizeof(buf), sc.time_left);
        const float t = sc.time_left;
        const float pulse = 0.5f + 0.5f * std::sin(hud.time * 8.0f);
        shs::Color dig = (t <= 10.0f)
            ? shs::Color{ 245, (uint8_t)(70 + 90 * pulse), 55, 255 }
            : (t <= 30.0f) ? shs::Color{ 245, 60, 60, 255 }
                           : shs::Color{ 255, 180, 40, 255 };
        const int pw = 190, ph = 58;
        const int px = (W - pw) / 2, py = 14;
        draw_rect_fill(canvas, px, py, pw, ph, shs::Color{ 15, 18, 26, 230 });
        draw_rect_border(canvas, px, py, pw, ph,
                         (t <= 10.0f) ? dig : accent(shs::Color{ 60, 140, 220, 255 }));
        draw_text(canvas, px + 12, py + 8, TXT_TIME, shs::Color{ 140, 155, 175, 255 }, 2);

        int dx = px + 12, dy = py + 26;
        for (const char* q = buf; *q; ++q) {
            if (*q == ':') {
                draw_rect_fill(canvas, dx + 5, dy + 8, 4, 4, dig);
                draw_rect_fill(canvas, dx + 5, dy + 18, 4, 4, dig);
                dx += 15;
            } else if (*q >= '0' && *q <= '9') {
                draw_digit_bold(canvas, dx, dy, *q - '0', 16, 28, dig);
                dx += 23;
            }
        }
    }

    // ------------------------------------------------------------------------
    // 3b. COMBO METER (bottom-left, tier ticks at 2/4/6/8)
    // ------------------------------------------------------------------------
    if (blitz) {
        int cx = 20, cy = H - 64, cw = 280, ch = 36;
        draw_rect_fill(canvas, cx, cy, cw, ch, shs::Color{ 15, 18, 26, 230 });
        draw_rect_border(canvas, cx, cy, cw, ch, accent(shs::Color{ 60, 140, 220, 255 }));
        draw_text(canvas, cx + 12, cy + 11, TXT_COMBO, shs::Color{ 40, 220, 240, 255 }, 2);

        int bx = cx + 100, by = cy + 11, bw2 = cw - 118, bh2 = 14;
        draw_rect_fill(canvas, bx, by, bw2, bh2, shs::Color{ 35, 40, 52, 255 });
        const float combo_fill = glm::clamp((float)sc.combo_count / 8.0f, 0.0f, 1.0f);
        shs::Color combo_col = (sc.combo_count >= 4)
            ? shs::Color{ 255, 200, 60, 255 } : shs::Color{ 40, 220, 240, 255 };
        draw_rect_fill(canvas, bx, by, (int)(combo_fill * (float)bw2), bh2, combo_col);
        for (int tier = 1; tier <= 3; ++tier) {   // ticks at 2/4/6
            int tx = bx + (int)(bw2 * (tier * 2 / 8.0f));
            draw_rect_fill(canvas, tx, by, 2, bh2, shs::Color{ 80, 95, 115, 255 });
        }
        draw_rect_border(canvas, bx, by, bw2, bh2, shs::Color{ 80, 95, 115, 255 });
    }

    // ------------------------------------------------------------------------
    // 3c. LEVEL-UP BANNER (centered gold flash + palette shift driver)
    // ------------------------------------------------------------------------
    if (hud.levelup_timer > 0.0f) {
        char buf[24];
        std::snprintf(buf, sizeof(buf), "%s %d!", TXT_LEVEL, hud.levelup_level);
        const float pulse = 0.5f + 0.5f * std::sin(hud.time * 12.0f);
        shs::Color bc{ (uint8_t)(220 + 35 * pulse), (uint8_t)(170 + 55 * pulse), 50, 255 };
        const int bw3 = text_width_px(buf, 3) + 44;
        const int bx2 = (W - bw3) / 2, by2 = timed ? 84 : 24;
        draw_rect_fill(canvas, bx2, by2, bw3, 46, shs::Color{ 20, 16, 8, 235 });
        draw_rect_border(canvas, bx2, by2, bw3, 46, bc);
        draw_text_centered(canvas, W / 2, by2 + 12, buf, bc, 3);
    }

    // ------------------------------------------------------------------------
    // 3d. HURRY! BANNER + SCREEN-BORDER PULSE (final 10 seconds)
    // ------------------------------------------------------------------------
    if (sc.clock_hurry && !m.game_over && !sc.victory && !sc.time_up) {
        const float blink = std::sin(hud.time * 10.0f);
        if (blink > -0.2f) {
            draw_text_centered(canvas, W / 2, 160, TXT_HURRY, shs::Color{ 245, 55, 55, 255 }, 3);
        }
        const float bpulse = 0.5f + 0.5f * std::sin(hud.time * 6.0f);
        const uint8_t bb = (uint8_t)(120 + 120 * bpulse);
        const shs::Color pc{ bb, 25, 25, 255 };
        const int bw4 = 12;
        const int phase = (int)(hud.time * 30.0f);
        draw_rect_fill_dithered(canvas, 0, 0, W, bw4, pc, phase);
        draw_rect_fill_dithered(canvas, 0, H - bw4, W, bw4, pc, phase + 1);
        draw_rect_fill_dithered(canvas, 0, 0, bw4, H, pc, phase + 2);
        draw_rect_fill_dithered(canvas, W - bw4, 0, bw4, H, pc, phase + 3);
    }

    // ------------------------------------------------------------------------
    // 3e. DANGER VIGNETTE (stack height projection, breathing crimson bands)
    // ------------------------------------------------------------------------
    int stack = 0;
    for (int y = matrix::GRID_H - 1; y >= 0; --y) {
        bool any = false;
        for (int x = 0; x < matrix::GRID_W; ++x) {
            if (m.grid[y][x] != 0) { any = true; break; }
        }
        if (any) { stack = y + 1; break; }
    }
    if (stack >= 15 && !m.game_over && !sc.victory && !sc.time_up) {
        const float strength = glm::clamp((stack - 14) / 6.0f, 0.0f, 1.0f);
        const float breathe = 0.5f + 0.5f * std::sin(hud.time * 4.0f);
        const int band_w = (int)(14 + 22 * strength);
        for (int i = 0; i < 3; ++i) {
            const uint8_t vb = (uint8_t)((28 + i * 34) * strength * (0.6f + 0.4f * breathe));
            const shs::Color vc{ vb, 6, 14, 255 };
            const int off = i * (band_w / 3);
            draw_rect_fill_dithered(canvas, off, off, W - 2 * off, band_w - off, vc, i);                    // top
            draw_rect_fill_dithered(canvas, off, H - band_w + off, W - 2 * off, band_w - off, vc, i + 1);   // bottom
            draw_rect_fill_dithered(canvas, off, off, band_w - off, H - 2 * off, vc, i + 2);                // left
            draw_rect_fill_dithered(canvas, W - band_w + off, off, band_w - off, H - 2 * off, vc, i + 3);   // right
        }
    }

    // ------------------------------------------------------------------------
    // 3f. FLOATING POPUPS (COMBO ×N / +Ns / TIME UP — rising + fading)
    // ------------------------------------------------------------------------
    for (const auto& f : hud.floaters) {
        if (f.life <= 0.0f) continue;
        const float age01 = 1.0f - f.life / f.max_life;
        const int fy = (int)(H * 0.40f - age01 * 52.0f);
        const shs::Color fc = hud_lerp_color(f.color, shs::Color{ 14, 16, 22, 255 }, age01 * 0.85f);
        draw_text_centered(canvas, W / 2, fy, f.text, fc, 3);
    }

    // ------------------------------------------------------------------------
    // 4. 3D PLATFORM LABELS (НӨӨЦ / ДАРААГИЙН)
    // ------------------------------------------------------------------------
    draw_text(canvas, 95, 120, TXT_HOLD, shs::Color{ 80, 200, 255, 240 }, 2);
    draw_text(canvas, W - 230, 120, TXT_NEXT, shs::Color{ 80, 200, 255, 240 }, 2);

    // ------------------------------------------------------------------------
    // 5. BOTTOM CONTROLS FOOTER
    // ------------------------------------------------------------------------
    draw_text(canvas, (W - 980) / 2, H - 24, TXT_FOOTER, shs::Color{ 140, 155, 175, 220 }, 2);

    // ------------------------------------------------------------------------
    // 6. GAME OVER / VICTORY / TIME-UP MODAL OVERLAY (+ RESULTS breakdown)
    // ------------------------------------------------------------------------
    if (m.game_over || sc.victory || sc.time_up) {
        const int mh = blitz ? 310 : 200;
        int mw = 520, mx = (W - mw) / 2, my = (H - mh) / 2;

        draw_rect_fill(canvas, mx, my, mw, mh, shs::Color{ 10, 12, 18, 245 });
        shs::Color bc = sc.victory ? shs::Color{ 45, 240, 110, 255 }
                      : sc.time_up ? shs::Color{ 255, 180, 40, 255 }
                                   : shs::Color{ 245, 55, 55, 255 };
        draw_rect_border(canvas, mx, my, mw, mh, bc);
        draw_rect_border(canvas, mx + 2, my + 2, mw - 4, mh - 4, bc);

        if (sc.victory) {
            draw_text_centered(canvas, W / 2, my + 25, TXT_VICTORY, shs::Color{ 45, 240, 110, 255 }, 2);
        }
        else if (sc.time_up) {
            draw_text_centered(canvas, W / 2, my + 25, TXT_TIME_UP, shs::Color{ 255, 180, 40, 255 }, 2);
        }
        else {
            draw_text_centered(canvas, W / 2, my + 25, TXT_GAME_OVER, shs::Color{ 245, 55, 55, 255 }, 2);
        }

        draw_text(canvas, mx + 60, my + 75, TXT_FINAL_SCORE, shs::Color{ 220, 220, 220, 255 }, 2);
        draw_number_bold(canvas, mx + 300, my + 71, sc.score, 6, shs::Color{ 255, 230, 80, 255 });

        if (blitz) {
            // RESULTS time breakdown panel (clears vs drops vs time economy)
            draw_line_screen(canvas, mx + 40, my + 112, mx + mw - 40, my + 112, shs::Color{ 60, 70, 90, 255 });

            draw_text(canvas, mx + 45, my + 128, TXT_LINES, shs::Color{ 40, 220, 240, 255 }, 2);
            draw_number_bold(canvas, mx + 130, my + 124, sc.lines_cleared, 3, shs::Color{ 40, 220, 240, 255 });
            draw_text(canvas, mx + 270, my + 128, TXT_MAX_COMBO, shs::Color{ 185, 70, 240, 255 }, 2);
            draw_number_bold(canvas, mx + 420, my + 124, sc.max_combo, 2, shs::Color{ 185, 70, 240, 255 });

            char tbuf[16];
            format_clock(tbuf, sizeof(tbuf), m.game_time);
            draw_text(canvas, mx + 45, my + 158, TXT_TIME, shs::Color{ 255, 180, 40, 255 }, 2);
            draw_text(canvas, mx + 130, my + 158, tbuf, shs::Color{ 255, 180, 40, 255 }, 2);
            char bbuf[16];
            std::snprintf(bbuf, sizeof(bbuf), "+%ds", (int)sc.time_bonus_total);
            draw_text(canvas, mx + 270, my + 158, TXT_BONUS_TIME, shs::Color{ 45, 240, 110, 255 }, 2);
            draw_text(canvas, mx + 430, my + 158, bbuf, shs::Color{ 45, 240, 110, 255 }, 2);

            draw_text(canvas, mx + 45, my + 188, TXT_CLEAR_SCORE, shs::Color{ 220, 220, 220, 255 }, 2);
            draw_number_bold(canvas, mx + 210, my + 184, sc.score_clears, 6, shs::Color{ 220, 220, 220, 255 });
            draw_text(canvas, mx + 45, my + 218, TXT_DROP_SCORE, shs::Color{ 150, 160, 175, 255 }, 2);
            draw_number_bold(canvas, mx + 210, my + 214, sc.score_drops, 6, shs::Color{ 150, 160, 175, 255 });

            // Victory over a non-final stage: R rolls into the next campaign
            // stage (main consumes it); otherwise the usual retry hint.
            const char* hint = (sc.victory && campaign_has_next) ? TXT_NEXT_STAGE : TXT_RETRY;
            draw_text_centered(canvas, W / 2, my + 262, hint, shs::Color{ 140, 160, 190, 255 }, 2);
        } else {
            const char* hint = (sc.victory && campaign_has_next) ? TXT_NEXT_STAGE : TXT_RETRY;
            draw_text_centered(canvas, W / 2, my + 140, hint, shs::Color{ 140, 160, 190, 255 }, 2);
        }
    }
}

// ============================================================================
// SESSION MENU RENDERERS (pure projections of SessionSnapshot — M1)
// ============================================================================

struct MenuItem {
    const char* label;
    bool        enabled = true;
};

// Vertical menu list with a pulsing gold cursor chip. Rows are centered on cx.
static void draw_menu_list(shs::Canvas& canvas, int cx, int y0,
                           const MenuItem* items, int count, int cursor,
                           float anim_time) {
    const int row_h = 48;
    for (int i = 0; i < count; ++i) {
        const int  y   = y0 + i * row_h;
        const bool sel = (i == cursor);
        const shs::Color col = !items[i].enabled ? shs::Color{ 95, 105, 120, 255 }
                             : sel               ? shs::Color{ 255, 205, 70, 255 }
                                                 : shs::Color{ 185, 198, 215, 255 };
        if (sel) {
            const float pulse = 0.5f + 0.5f * std::sin(anim_time * 6.0f);
            const int   bw    = text_width_px(items[i].label, 2) + 64;
            draw_rect_fill(canvas, cx - bw / 2, y - 8, bw, 36, shs::Color{ 28, 23, 10, 235 });
            draw_rect_border(canvas, cx - bw / 2, y - 8, bw, 36,
                             shs::Color{ (uint8_t)(215 + 40 * pulse),
                                         (uint8_t)(165 + 60 * pulse), 50, 255 });
            if (std::sin(anim_time * 8.0f) > -0.3f) {
                draw_text(canvas, cx - bw / 2 + 14, y, ">", col, 2);
                draw_text(canvas, cx + bw / 2 - 26, y, "<", col, 2);
            }
        }
        draw_text_centered(canvas, cx, y, items[i].label, col, 2);
    }
}

// TITLE: drifting tetromino silhouettes + logo + [start / sound / exit].
static void draw_title_screen(shs::Canvas& canvas, const session::SessionSnapshot& s) {
    const int W = canvas.get_width(), H = canvas.get_height();

    // Deterministic attract background: four tetromino shapes drifting down.
    static const int SHAPE[4][4][2] = {
        { {0,0},{1,0},{2,0},{3,0} },   // I
        { {0,0},{1,0},{0,1},{1,1} },   // O
        { {0,0},{1,0},{2,0},{1,1} },   // T
        { {0,0},{1,0},{2,0},{2,1} },   // L
    };
    static const shs::Color DRIFT_COL[4] = {
        shs::Color{ 40, 220, 240, 60 }, shs::Color{ 255, 225, 45, 55 },
        shs::Color{ 185, 70, 240, 55 }, shs::Color{ 45, 110, 245, 55 },
    };
    for (int k = 0; k < 10; ++k) {
        const float spd  = 0.020f + 0.013f * (k % 4);
        const float ph   = 0.37f * k;
        float fx = 0.11f * (k % 7) + 0.03f * std::sin(s.anim_time * 0.35f + ph);
        float fy = 0.16f * k + s.anim_time * spd;
        fx = fx - std::floor(fx);
        fy = fy - std::floor(fy);
        const int cs = 16 + 6 * (k % 3);
        const int bx = (int)(fx * W), by = (int)(fy * H);
        const auto& cells = SHAPE[k % 4];
        for (int ci = 0; ci < 4; ++ci) {
            draw_rect_border(canvas, bx + cells[ci][0] * cs, by + cells[ci][1] * cs,
                             cs, cs, DRIFT_COL[k % 4]);
        }
    }

    // Logo + accent bar.
    draw_text_centered(canvas, W / 2, (int)(H * 0.14f), TXT_LOGO,
                       shs::Color{ 255, 210, 60, 255 }, 8);
    const int lw = text_width_px(TXT_LOGO, 8);
    draw_rect_fill(canvas, W / 2 - lw / 2, (int)(H * 0.14f) + 66, lw, 4,
                   shs::Color{ 60, 140, 220, 255 });

    char snd_buf[48];
    std::snprintf(snd_buf, sizeof(snd_buf), "%s: %s", TXT_SOUND,
                  s.sound_enabled ? TXT_ON : TXT_OFF);
    MenuItem items[session::TITLE_MENU_COUNT] = {
        { TXT_START },
        { snd_buf   },
        { TXT_EXIT  },
    };
    draw_menu_list(canvas, W / 2, (int)(H * 0.50f), items, session::TITLE_MENU_COUNT,
                   s.cursor, s.anim_time);

    char foot[128];
    std::snprintf(foot, sizeof(foot), "%s   |   %s   |   %s",
                  HINT_NAV, HINT_CONFIRM, HINT_SOUND);
    draw_text_centered(canvas, W / 2, H - 30, foot, shs::Color{ 130, 145, 165, 220 }, 2);
}

// LEVEL SELECT: carousel card over the manifest stages.
static void draw_level_select(shs::Canvas& canvas, const session::SessionSnapshot& s,
                              const char* const* names, const char* const* tiers,
                              int count) {
    const int W = canvas.get_width(), H = canvas.get_height();
    draw_rect_fill_dithered(canvas, 0, 0, W, H, shs::Color{ 8, 10, 15, 140 },
                            (int)(s.anim_time * 20.0f));

    draw_text_centered(canvas, W / 2, (int)(H * 0.16f), TXT_SELECT_LVL,
                       shs::Color{ 80, 200, 255, 255 }, 4);

    const int cw = 660, chh = 260;
    const int cx = (W - cw) / 2, cy = (int)(H * 0.32f);
    draw_rect_fill(canvas, cx, cy, cw, chh, shs::Color{ 13, 16, 24, 240 });
    draw_rect_border(canvas, cx, cy, cw, chh, shs::Color{ 60, 140, 220, 255 });
    draw_rect_border(canvas, cx + 3, cy + 3, cw - 6, chh - 6, shs::Color{ 35, 60, 90, 255 });

    const int idx = ((s.stage_cursor % count) + count) % count;
    char idx_buf[24];
    std::snprintf(idx_buf, sizeof(idx_buf), "%d / %d", idx + 1, count);

    draw_text_centered(canvas, W / 2, cy + 42, names[idx], shs::Color{ 255, 210, 60, 255 }, 4);
    draw_text_centered(canvas, W / 2, cy + 96, tiers[idx], shs::Color{ 140, 155, 175, 255 }, 2);

    // Progress dots (one per stage; filled up to the cursor).
    const int dots_w = count * 30;
    for (int i = 0; i < count; ++i) {
        const int dx = W / 2 - dots_w / 2 + i * 30 + 8;
        const shs::Color dc = (i == idx) ? shs::Color{ 255, 205, 70, 255 }
                                         : shs::Color{ 70, 85, 105, 255 };
        draw_rect_fill(canvas, dx, cy + 150, 14, 14, dc);
    }

    // Side arrows.
    draw_text_centered(canvas, cx - 44, cy + chh / 2 - 14, "<", shs::Color{ 80, 200, 255, 255 }, 3);
    draw_text_centered(canvas, cx + cw + 22, cy + chh / 2 - 14, ">", shs::Color{ 80, 200, 255, 255 }, 3);

    char foot[128];
    std::snprintf(foot, sizeof(foot), "%s   |   %s   |   %s",
                  HINT_NAV, HINT_CONFIRM, HINT_BACK);
    draw_text_centered(canvas, W / 2, H - 30, foot, shs::Color{ 130, 145, 165, 220 }, 2);
}

// PAUSED: dimmed frozen board + pause menu.
static void draw_pause_overlay(shs::Canvas& canvas, const session::SessionSnapshot& s) {
    const int W = canvas.get_width(), H = canvas.get_height();
    draw_rect_fill_dithered(canvas, 0, 0, W, H, shs::Color{ 6, 8, 12, 150 },
                            (int)(s.anim_time * 24.0f));

    const int pw = 520, ph = 380;
    const int px = (W - pw) / 2, py = (H - ph) / 2;
    draw_rect_fill(canvas, px, py, pw, ph, shs::Color{ 12, 14, 21, 245 });
    draw_rect_border(canvas, px, py, pw, ph, shs::Color{ 60, 140, 220, 255 });
    draw_text_centered(canvas, W / 2, py + 26, TXT_PAUSED, shs::Color{ 80, 200, 255, 255 }, 3);

    char snd_buf[48];
    std::snprintf(snd_buf, sizeof(snd_buf), "%s: %s", TXT_SOUND,
                  s.sound_enabled ? TXT_ON : TXT_OFF);
    MenuItem items[session::PAUSE_MENU_COUNT] = {
        { TXT_RESUME },
        { TXT_RESTART },
        { TXT_SELECT_LVL },
        { snd_buf },
        { TXT_EXIT },
    };
    draw_menu_list(canvas, W / 2, py + 84, items, session::PAUSE_MENU_COUNT,
                   s.cursor, s.anim_time);
}

// RESULTS: end-of-run breakdown + contextual next/retry/select/title menu.
static void draw_results_screen(shs::Canvas& canvas, const session::SessionSnapshot& s) {
    const int W = canvas.get_width(), H = canvas.get_height();
    draw_rect_fill_dithered(canvas, 0, 0, W, H, shs::Color{ 8, 10, 15, 150 },
                            (int)(s.anim_time * 20.0f));

    const int pw = 600, ph = 430;
    const int px = (W - pw) / 2, py = (H - ph) / 2;
    draw_rect_fill(canvas, px, py, pw, ph, shs::Color{ 10, 12, 18, 248 });
    const shs::Color bc = s.run_victory ? shs::Color{ 45, 240, 110, 255 }
                        : s.run_time_up ? shs::Color{ 255, 180, 40, 255 }
                                        : shs::Color{ 245, 55, 55, 255 };
    draw_rect_border(canvas, px, py, pw, ph, bc);
    draw_rect_border(canvas, px + 2, py + 2, pw - 4, ph - 4, bc);

    if      (s.run_victory) draw_text_centered(canvas, W / 2, py + 24, TXT_VICTORY, shs::Color{ 45, 240, 110, 255 }, 2);
    else if (s.run_time_up) draw_text_centered(canvas, W / 2, py + 24, TXT_TIME_UP,  shs::Color{ 255, 180, 40, 255 }, 2);
    else                    draw_text_centered(canvas, W / 2, py + 24, TXT_GAME_OVER, shs::Color{ 245, 55, 55, 255 }, 2);

    draw_text(canvas, px + 60, py + 78, TXT_FINAL_SCORE, shs::Color{ 220, 220, 220, 255 }, 2);
    draw_number_bold(canvas, px + 320, py + 74, s.final_score, 6, shs::Color{ 255, 230, 80, 255 });

    draw_text(canvas, px + 60, py + 116, TXT_LINES, shs::Color{ 40, 220, 240, 255 }, 2);
    draw_number_bold(canvas, px + 320, py + 112, s.final_lines, 3, shs::Color{ 40, 220, 240, 255 });

    draw_text(canvas, px + 60, py + 154, TXT_MAX_COMBO, shs::Color{ 185, 70, 240, 255 }, 2);
    draw_number_bold(canvas, px + 320, py + 150, s.final_max_combo, 2, shs::Color{ 185, 70, 240, 255 });

    char tbuf[16];
    format_clock(tbuf, sizeof(tbuf), s.final_seconds);
    draw_text(canvas, px + 60, py + 192, TXT_TIME, shs::Color{ 255, 180, 40, 255 }, 2);
    draw_text(canvas, px + 320, py + 192, tbuf, shs::Color{ 255, 180, 40, 255 }, 2);

    const bool has_next = s.run_victory && (s.current_stage + 1 < s.stage_count);
    MenuItem items[session::RESULTS_MENU_BASE + 1] = {
        { has_next ? TXT_NEXT_PLAIN : TXT_RESTART },
        { TXT_SELECT_LVL },
        { TXT_BACK },
    };
    draw_menu_list(canvas, W / 2, py + 250, items, session::RESULTS_MENU_BASE + 1,
                   s.cursor, s.anim_time);
}
} // namespace tetris::ui
