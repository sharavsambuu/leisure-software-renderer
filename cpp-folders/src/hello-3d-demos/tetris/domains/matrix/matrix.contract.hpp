#pragma once

#include <cstdint>
#include <array>
#include <span>
#include "shs_renderer.hpp"

namespace tetris {
    namespace matrix {

        // ============================================================================
        // POD 1: MATRIX — Core Grid Simulation (SRS, Gravity, Lock Delay)
        // ============================================================================

        static constexpr int GRID_W = 10;
        static constexpr int GRID_H = 22;   // Rows 0..19 visible, 20..21 spawn buffer

        enum class PieceType : uint8_t {
            None    = 0,
            I       = 1,
            O       = 2,
            T       = 3,
            S       = 4,
            Z       = 5,
            J       = 6,
            L       = 7
        };

        static inline shs::Color get_piece_color(PieceType type) {
            switch (type) {
                case PieceType::I: return shs::Color{ 40, 220, 240, 255 }; // Cyan
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
            PieceType  type   = PieceType::None;
            glm::ivec2 pos    = { 4, 19 }; // Grid X,Y (bottom-left origin)
            uint8_t    rot    = 0;         // 0: 0deg, 1:90deg, 2:180deg, 3:270deg
            float      lock_timer   = 0.0f;
            uint8_t    lock_resets  = 0;
        };

        struct MatrixSnapshot {
            std::array<std::array<uint8_t, GRID_W>, GRID_H> grid{};

            ActivePiece active;
            PieceType hold_piece = PieceType::None;
            bool hold_locked     = false;

            // 7-bag next piece queue (top of stack is next drop)
            std::array<PieceType, 5> next_queue = { PieceType::I, PieceType::T, PieceType::L, PieceType::S, PieceType::O };

            uint32_t rng_state = 0x9e3779b9u; // LCG seed

            MatrixSnapshot() {
                for (auto& row : grid) row.fill(0);
            }
        };

        struct MatrixStepResult {
            MatrixSnapshot next_snapshot;
            std::pmr::vector<matrix::MatrixEvent> events;
        };

    } // namespace matrix
} // namespace tetris
