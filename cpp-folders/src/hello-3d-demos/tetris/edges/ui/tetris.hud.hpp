#pragma once
// tetris/edges/ui/tetris.hud.hpp — 2D HUD + UTF-8 FONT ENGINE (tetris::ui)
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include "shs_renderer.hpp"
#include <domains/matrix/matrix.contract.hpp>
#include <domains/progression/progression.contract.hpp>

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
// ҮЕ
static const char* TXT_LEVEL       = "\xD0\xAE\xD0\x95";                             
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
// MONGOLIAN CYRILLIC HUD (Layout & Presentation)
// ============================================================================
static void draw_hud(shs::Canvas& canvas, const matrix::MatrixSnapshot& m, const progression::ScoreState& sc) {
    int W = canvas.get_width();
    int H = canvas.get_height();

    // ------------------------------------------------------------------------
    // 1. TOP RIGHT: SCORE CARD (ОНОО / ДЭЭД)
    // ------------------------------------------------------------------------
    int sx = W - 265, sy = 18, sw = 245, sh = 88;
    draw_rect_fill(canvas, sx, sy, sw, sh, shs::Color{ 15, 18, 26, 230 });
    draw_rect_border(canvas, sx, sy, sw, sh, shs::Color{ 60, 140, 220, 255 });

    draw_text(canvas, sx + 14, sy + 14, TXT_SCORE, shs::Color{ 255, 225, 45, 255 }, 2);
    draw_number_bold(canvas, sx + 125, sy + 12, sc.score, 6, shs::Color{ 255, 225, 45, 255 });

    draw_text(canvas, sx + 14, sy + 48, TXT_BEST, shs::Color{ 140, 155, 175, 255 }, 2);
    draw_number_bold(canvas, sx + 125, sy + 46, sc.high_score, 6, shs::Color{ 140, 155, 175, 255 });

    // ------------------------------------------------------------------------
    // 2. TOP LEFT: GOAL & STATS CARD (ЗОРИЛГО / МӨР / ҮЕ)
    // ------------------------------------------------------------------------
    int ox = 20, oy = 18, ow = 280, oh = 88;
    draw_rect_fill(canvas, ox, oy, ow, oh, shs::Color{ 15, 18, 26, 230 });
    draw_rect_border(canvas, ox, oy, ow, oh, shs::Color{ 60, 140, 220, 255 });

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
    // 3. 3D PLATFORM LABELS (НӨӨЦ / ДАРААГИЙН)
    // ------------------------------------------------------------------------
    draw_text(canvas, 95, 120, TXT_HOLD, shs::Color{ 80, 200, 255, 240 }, 2);
    draw_text(canvas, W - 230, 120, TXT_NEXT, shs::Color{ 80, 200, 255, 240 }, 2);

    // ------------------------------------------------------------------------
    // 4. BOTTOM CONTROLS FOOTER
    // ------------------------------------------------------------------------
    draw_text(canvas, (W - 980) / 2, H - 24, TXT_FOOTER, shs::Color{ 140, 155, 175, 220 }, 2);

    // ------------------------------------------------------------------------
    // 5. GAME OVER / VICTORY MODAL OVERLAY
    // ------------------------------------------------------------------------
    if (m.game_over || sc.victory) {
        int mw = 480, mh = 200;
        int mx = (W - mw) / 2, my = (H - mh) / 2;

        draw_rect_fill(canvas, mx, my, mw, mh, shs::Color{ 10, 12, 18, 245 });
        shs::Color bc = sc.victory ? shs::Color{ 45, 240, 110, 255 } : shs::Color{ 245, 55, 55, 255 };
        draw_rect_border(canvas, mx, my, mw, mh, bc);
        draw_rect_border(canvas, mx + 2, my + 2, mw - 4, mh - 4, bc);

        if (sc.victory) {
            draw_text(canvas, mx + 90, my + 25, TXT_VICTORY, shs::Color{ 45, 240, 110, 255 }, 2);
        }
        else {
            draw_text(canvas, mx + 95, my + 25, TXT_GAME_OVER, shs::Color{ 245, 55, 55, 255 }, 2);
        }

        draw_text(canvas, mx + 80, my + 80, TXT_FINAL_SCORE, shs::Color{ 220, 220, 220, 255 }, 2);
        draw_number_bold(canvas, mx + 260, my + 76, sc.score, 6, shs::Color{ 255, 230, 80, 255 });

        draw_text(canvas, mx + 115, my + 140, TXT_RETRY, shs::Color{ 140, 160, 190, 255 }, 2);
    }
}
} // namespace tetris::ui
