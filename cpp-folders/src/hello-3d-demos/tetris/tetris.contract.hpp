#pragma once

#include <cstdint>
#include <array>
#include <span>
#include <memory_resource>
#include <glm/glm.hpp>
#include "shs_renderer.hpp"

namespace tetris {

    static constexpr int GRID_W      = 10;
    static constexpr int GRID_H      = 22; // 0..19 visible, 20..21 spawn buffer
    static constexpr int VISIBLE_H   = 20;

    static constexpr float CELL_SIZE = 1.0f;
    static constexpr float BLOCK_GAP = 0.06f;

    enum class PieceType : uint8_t {
        None    = 0,
        I       = 1, // Cyan
        O       = 2, // Yellow
        T       = 3, // Purple
        S       = 4, // Green
        Z       = 5, // Red
        J       = 6, // Blue
        L       = 7  // Orange
    };

    static inline shs::Color get_piece_color(PieceType type) {
        switch (type) {
            case PieceType::I: return shs::Color{  40, 220, 240, 255 }; // Cyan
            case PieceType::O: return shs::Color{ 255, 225,  45, 255 }; // Yellow
            case PieceType::T: return shs::Color{ 185,  70, 240, 255 }; // Purple
            case PieceType::S: return shs::Color{  60, 230,  95, 255 }; // Green
            case PieceType::Z: return shs::Color{ 245,  55,  55, 255 }; // Red
            case PieceType::J: return shs::Color{  45, 110, 245, 255 }; // Blue
            case PieceType::L: return shs::Color{ 255, 140,  35, 255 }; // Orange
            default:           return shs::Color{  80,  90, 105, 255 };
        }
    }

    struct ActivePiece {
        PieceType  type        = PieceType::None;
        glm::ivec2 pos         = { 4, 19 }; // Grid X, Y (Bottom-left origin)
        uint8_t    rotation    = 0;         // 0: 0 deg, 1: 90 deg, 2: 180 deg, 3: 270 deg
        float      lock_timer  = 0.0f;      // Max 0.5s before hard lock
        uint8_t    lock_resets = 0;         // Max 15 moves on floor
    };

    struct TetrisSnapshot {
        std::array<std::array<uint8_t, GRID_W>, GRID_H> grid{};
        ActivePiece                                     active;
        PieceType                                       hold_piece   = PieceType::None;
        bool                                            hold_locked  = false;
        std::array<PieceType, 5>                        next_queue   = { PieceType::I, PieceType::T, PieceType::L, PieceType::S, PieceType::O };
        uint32_t                                        rng_state    = 0x9e3779b9u;

        int   score             = 0;
        int   high_score        = 0;
        int   lines_cleared     = 0;
        int   level             = 1;
        int   combo_count       = 0;
        int   target_score      = 12000;

        float gravity_timer     = 0.0f;
        float drop_interval     = 0.80f;
        float game_time         = 0.0f;
        float danger_pulse      = 0.0f;

        bool  game_over         = false;
        bool  victory           = false;

        TetrisSnapshot() {
            for (auto& row : grid) row.fill(0);
        }
    };

} // namespace tetris