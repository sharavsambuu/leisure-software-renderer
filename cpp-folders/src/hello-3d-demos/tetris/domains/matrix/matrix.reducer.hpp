#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <span>
#include "matrix.contract.hpp"
#include "matrix.event.hpp"

namespace tetris {
    namespace matrix {

        // ============================================================================
        // PURE REDUCER: Matrix Grid State Transition
        // (StateSnapshot + Commands) → (NextState, Discrete Events)
        // ============================================================================

        static inline std::array<std::array<uint8_t, GRID_W>, GRID_H> get_piece_blocks(
            PieceType type, uint8_t rot
        ) {
            rot = rot % 4;
            switch (type) {
                case PieceType::I:
                    if (rot == 0) return {{-1,0},{0,0},{1,0},{2,0}};
                    if (rot == 1) return {{1,1},{1,0},{1,-1},{1,-2}};
                    if (rot == 2) return {{-1,-1},{0,-1},{1,-1},{2,-1}};
                    return {{0,1},{0,0},{0,-1},{0,-2}};
                case PieceType::O: return {{0,0},{1,0},{0,1},{1,1}};
                case PieceType::T: {
                    if (rot == 0) return {{-1,0},{0,0},{1,0},{0,1}};
                    if (rot == 1) return {{0,1},{0,0},{0,-1},{1,0}};
                    if (rot == 2) return {{-1,0},{0,0},{1,0},{0,-1}};
                    return {{0,1},{0,0},{0,-1},{-1,0}};
                }
                case PieceType::S: {
                    if (rot == 0 || rot == 2) return {{-1,0},{0,0},{0,1},{1,1}};
                    return {{0,1},{0,0},{1,0},{1,-1}};
                }
                case PieceType::Z: {
                    if (rot == 0 || rot == 2) return {{-1,1},{0,1},{0,0},{1,0}};
                    return {{1,1},{1,0},{0,0},{0,-1}};
                }
                case PieceType::J: {
                    if (rot == 0) return {{-1,1},{-1,0},{0,0},{1,0}};
                    if (rot == 1) return {{1,1},{0,1},{0,0},{0,-1}};
                    if (rot == 2) return {{-1,0},{0,0},{1,0},{1,-1}};
                    return {{0,1},{0,0},{0,-1},{-1,-1}};
                }
                case PieceType::L: {
                    if (rot == 0) return {{1,1},{-1,0},{0,0},{1,0}};
                    if (rot == 1) return {{0,1},{0,0},{0,-1},{1,-1}};
                    if (rot == 2) return {{-1,0},{0,0},{1,0},{-1,-1}};
                    return {{-1,1},{0,1},{0,0},{0,-1}};
                }
            }
            return {};
        }

        static inline bool is_valid_position(
            const std::array<std::array<uint8_t, GRID_W>, GRID_H>& grid,
            PieceType type, glm::ivec2 pos, uint8_t rot
        ) {
            if (type == PieceType::None) return true;
            auto blocks = get_piece_blocks(type, rot);
            for (const auto& b : blocks) {
                int gx = pos.x + b.x;
                int gy = pos.y + b.y;
                if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H) return false;
                if (grid[gy][gx] != 0) return false;
            }
            return true;
        }

        static inline int get_ghost_y(
            const std::array<std::array<uint8_t, GRID_W>, GRID_H>& grid,
            const ActivePiece& piece
        ) {
            if (piece.type == PieceType::None) return piece.pos.y;
            int gy = piece.pos.y;
            while (gy > 0 && is_valid_position(grid, piece.type, {piece.pos.x, gy - 1}, piece.rot)) {
                gy--;
            }
            return gy;
        }

        static inline PieceType pull_next_piece(uint32_t& rng, std::array<PieceType, 5>& next_queue) {
            PieceType next = next_queue[0];
            for (size_t i = 0; i < 4; ++i) next_queue[i] = next_queue[i + 1];

            // LCG RNG — deterministic, no std::random needed in hot path
            rng = rng * 1664525u + 1013904223u;
            uint8_t roll = static_cast<uint8_t>((rng >> 24) % 7 + 1);
            next_queue[4] = static_cast<PieceType>(roll);
            return next;
        }

        static inline tetris::matrix::MatrixCommandFrame reduce_matrix_commands(std::span<const MatrixCommand> cmds) {
            MatrixCommandFrame frame{};
            for (const auto& c : cmds) {
                if (c.type == MoveLeftIntent{})       frame.move_left = true;
                else if (c.type == MoveRightIntent{}) frame.move_right = true;
                else if (c.type == RotateCWIntent{})  frame.rotate_cw = true;
                else if (c.type == RotateCCWIntent{}) frame.rotate_ccw = true;
                else if (c.type == SoftDropIntent{})  frame.soft_drop = true;
                else if (c.type == HardDropIntent{})  frame.hard_drop = true;
                else if (c.type == HoldPieceIntent{})frame.hold_pressed = true;
                else if (c.type == RestartIntent{})   frame.reset_pressed = true;
            }
            return frame;
        }

        static inline MatrixStepResult reduce_matrix(
            const MatrixSnapshot&  prev,
            std::span<const MatrixCommand> cmds,
            float dt,
            std::pmr::memory_resource* mr
        ) {
            MatrixStepResult result{mr};
            MatrixSnapshot& next = result.next_snapshot;

            // --- Restart command ---
            if (prev.active.type == PieceType::None && !prev.hold_locked) {
                next.active = prev.active;  // carry over hold piece if not locked
                next.active.type = pull_next_piece(next.rng_state, next.next_queue);
                next.active.pos = glm::ivec2{4, 19};
                result.events.push_back({MatrixEventType::PIECE_SPAWNED});
            }

            // --- Hold Piece (consumes event from progression or direct command) ---
            if (!prev.hold_locked && next.active.type != PieceType::None) {
                PieceType current = next.active.type;
                if (next.hold_piece == PieceType::None) {
                    next.hold_piece  = current;
                    next.active.type = pull_next_piece(next.rng_state, next.next_queue);
                } else {
                    next.active.type   = next.hold_piece;
                    next.hold_piece    = current;
                }
                next.active.pos        = glm::ivec2{4, 19};
                next.active.rotation   = 0;
                next.active.lock_timer = 0.0f;
                result.events.push_back({MatrixEventType::HOLD_SWAPPED});
            }

            // --- Horizontal Movement (with SRS wall-kicks) ---
            if (next.active.type != PieceType::None && !next.game_over) {
                int move_dir = 0;
                if (prev.frame.move_left)  move_dir = -1;
                if (prev.frame.move_right) move_dir = +1;

                if (move_dir != 0) {
                    glm::ivec2 target = next.active.pos + glm::ivec2(move_dir, 0);
                    if (is_valid_position(next.grid, next.active.type, target, next.active.rot)) {
                        next.active.pos = target;
                        result.events.push_back({MatrixEventType::PIECE_MOVED});
                        if (next.active.lock_resets < 15) {
                            next.active.lock_timer = 0.0f;
                            next.active.lock_resets++;
                        }
                    } else {
                        // Wall kick attempt on collision
                        glm::ivec2 kicks[] = {{-1,0},{+1,0},{0,-1},{0,+1}};
                        for (auto& k : kicks) {
                            if (is_valid_position(next.grid, next.active.type, next.active.pos + k, next.active.rot)) {
                                next.active.pos += k;
                                result.events.push_back({MatrixEventType::PIECE_MOVED});
                                break;
                            }
                        }
                    }
                }

                // --- Rotation (SRS 5-point wall kicks) ---
                if (next.active.type != PieceType::None && !next.game_over) {
                    int rot_dir = prev.frame.rotate_cw ? +1 : -1;
                    uint8_t target_rot = (next.active.rot + (rot_dir == +1)) % 4;

                    static const glm::ivec2 KICKS[5] = {{0,0},{-1,0},{+1,0},{0,-1},{0,+1}};
                    for (const auto& kick : KICKS) {
                        if (is_valid_position(next.grid, next.active.type, next.active.pos + kick, target_rot)) {
                            next.active.pos += kick;
                            next.active.rot = target_rot;
                            result.events.push_back({MatrixEventType::PIECE_ROTATED});
                            break;
                        }
                    }
                }

                // --- Hard Drop (instant lock) ---
                if (next.active.type != PieceType::None && !next.game_over && prev.frame.hard_drop) {
                    int ghost_y = get_ghost_y(next.grid, next.active);
                    int dropped_cells = next.active.pos.y - ghost_y;
                    next.active.pos.y = ghost_y;
                    next.score += dropped_cells * 2;

                    // Immediate lock
                    auto blocks = get_piece_blocks(next.active.type, next.active.rot);
                    for (const auto& b : blocks) {
                        int gx = next.active.pos.x + b.x;
                        int gy = next.active.pos.y + b.y;
                        if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H) {
                            next.grid[gy][gx] = static_cast<uint8_t>(next.active.type);
                        }
                    }
                    next.active.lock_timer += dt * 60.0f; // convert to frames
                    result.events.push_back({MatrixEventType::HARD_DROP_SLAM, dropped_cells});
                }

                // --- Gravity Step ---
                float interval = prev.drop_interval;
                if (prev.frame.soft_drop) interval *= 0.12f;
                next.gravity_timer += dt * 60.0f;

                if (next.gravity_timer >= interval) {
                    next.gravity_timer -= interval;
                    glm::ivec2 down_pos = {next.active.pos.x, next.active.pos.y - 1};

                    if (is_valid_position(next.grid, next.active.type, down_pos, next.active.rot)) {
                        next.active.pos = down_pos;
                        if (prev.frame.soft_drop) next.score += 1;
                        result.events.push_back({MatrixEventType::GRAVITY_STEP});
                    } else {
                        // Landed on ground or another piece
                        float remaining = interval - next.gravity_timer;
                        if (!prev.frame.hard_drop && remaining >= 0.5f) {
                            next.active.lock_timer += dt * 60.0f;
                        }
                    }
                }

                // --- Locking & Line Clear ---
                if (next.active.lock_timer >= 0.5f || prev.frame.hard_drop) {
                    auto blocks = get_piece_blocks(next.active.type, next.active.rot);
                    for (const auto& b : blocks) {
                        int gx = next.active.pos.x + b.x;
                        int gy = next.active.pos.y + b.y;
                        if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H) {
                            next.grid[gy][gx] = static_cast<uint8_t>(next.active.type);
                        }
                    }

                    result.events.push_back({MatrixEventType::PIECE_LOCK_IMPACT});

                    // Find cleared lines (top to bottom so shifts are correct)
                    int cleared_count = 0;
                    std::array<int, 4> cleared_indices{};

                    for (int y = GRID_H - 1; y >= 0; --y) {
                        bool full = true;
                        for (int x = 0; x < GRID_W; ++x) {
                            if (next.grid[y][x] == 0) { full = false; break; }
                        }
                        if (full) {
                            cleared_indices[cleared_count] = y;
                            cleared_count++;
                            // Shift down: copy row below into current row
                            for (int x = 0; x < GRID_W; ++x) next.grid[y][x] = next.grid[y + 1][x];
                        }
                    }

                    if (cleared_count > 0) {
                        // Spawn shatter particles — emitted to spatial_fx domain via event span
                        result.events.push_back({MatrixEventType::LINES_CLEARED, cleared_count, cleared_indices});
                    } else {
                        next.active.lock_timer = 0.0f;
                    }

                    // Spawn next piece after lock
                    if (next.active.type == PieceType::None) {
                        next.active.type       = pull_next_piece(next.rng_state, next.next_queue);
                        next.active.pos        = glm::ivec2{4, 19};
                        next.active.rotation   = 0;
                        result.events.push_back({MatrixEventType::PIECE_SPAWNED});
                    } else {
                        // Already spawned — reset lock timer for new piece
                        next.active.lock_timer = 0.0f;
                    }

                }
            }

            return result;
        }

    } // namespace matrix
} // namespace tetris
