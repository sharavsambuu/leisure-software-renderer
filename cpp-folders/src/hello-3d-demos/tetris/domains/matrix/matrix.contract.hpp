#pragma once
// tetris/domains/matrix/matrix.contract.hpp — GRID RULEBOOK STATE (tetris::matrix)
// Plain-data schema only: grid lifecycle, active piece, hold, 7-bag queue, RNG.
// NO scoring, NO render vocabulary (Constitution II purity).
#include <array>
#include <cstdint>
#include <glm/glm.hpp>

namespace tetris::matrix {

    static constexpr int GRID_W      = 10;
    static constexpr int GRID_H      = 22; // 0..19 visible, 20..21 spawn buffer
    static constexpr int VISIBLE_H   = 20;

    static constexpr float CELL_SIZE = 1.0f;
    static constexpr float BLOCK_GAP = 0.06f;

    enum class PieceType : uint8_t {
        None = 0, I = 1, O = 2, T = 3, S = 4, Z = 5, J = 6, L = 7,
        Garbage = 8,  // L3 canyon rubble: never spawns from the 7-bag; only
                      // arrives via the scripted initial-board stamp (raw fact)
        // L4 Cyber Storm specials: never spawn from the 7-bag either — they
        // arrive via QueueSpecialIntent (script-scheduled raw facts). On lock
        // they do NOT write the grid; they emit SPECIAL_LOCKED and the scripted
        // ruling decides the mutation (blast cells / vaporized row / freeze).
        Bomb   = 9,
        Laser  = 10,
        Freeze = 11
    };

    // L4 special-piece identity test (grid rulebook vocabulary).
    static inline bool is_special_piece(PieceType t) {
        return t == PieceType::Bomb || t == PieceType::Laser || t == PieceType::Freeze;
    }

    // Row-major cell block: [y][x], y 0 = grid bottom. Shared by the snapshot
    // and the initial-board stamp command payload.
    using CellGrid = std::array<std::array<uint8_t, GRID_W>, GRID_H>;

    struct ActivePiece {
        PieceType  type        = PieceType::None;
        glm::ivec2 pos         = { 4, 19 }; // Grid X, Y (Bottom-left origin)
        uint8_t    rotation    = 0;         // 0: 0 deg .. 3: 270 deg
        float      lock_timer  = 0.0f;
        uint8_t    lock_resets = 0;
    };

    struct MatrixSnapshot {
        CellGrid                   grid{};
        // L3 injection seam: pristine copy of a scripted initial board. The
        // stamp command fills BOTH grid and this backup; RestartIntent then
        // restores from it so R-restarts keep the pre-ruined layout.
        CellGrid                   initial_grid{};
        bool                       has_initial_board = false;

        ActivePiece                active;
        PieceType                  hold_piece    = PieceType::None;
        bool                       hold_locked   = false;
        std::array<PieceType, 5>   next_queue    = { PieceType::I, PieceType::T, PieceType::L, PieceType::S, PieceType::O };
        uint32_t                   rng_state     = 0x9e3779b9u;

        // Gravity cadence is OWNED BY MAIN WIRING (set from progression.level
        // via config::Rules::gravity_for_level each frame); reducer only reads.
        float drop_interval  = 0.80f;
        float gravity_timer  = 0.0f;
        float game_time      = 0.0f;

        // L4 powerup seam: pending_special overrides the NEXT pulled piece's
        // type (set by QueueSpecialIntent); gravity_freeze pauses the gravity
        // step while > 0 (set by FreezeGravityIntent). Both are plain fields —
        // no logic lives here.
        uint8_t pending_special = 0;
        float   gravity_freeze  = 0.0f;

        bool  game_over      = false;

        MatrixSnapshot() {
            for (auto& row : grid) row.fill(0);
        }
    };

} // namespace tetris::matrix
