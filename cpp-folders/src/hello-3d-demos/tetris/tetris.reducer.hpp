#pragma once

#include <algorithm>
#include <cmath>
#include "tetris.contract.hpp"
#include "tetris.action.hpp"
#include "tetris.event.hpp"

namespace tetris {

    // 4 standard block offsets per rotation state (0, 1, 2, 3)
    static inline std::array<glm::ivec2, 4> get_piece_blocks(PieceType type, uint8_t rot) {
        rot = rot % 4;
        switch (type) {
            case PieceType::I: {
                if (rot == 0) return { glm::ivec2{-1, 0}, {0, 0}, {1, 0}, {2, 0} };
                if (rot == 1) return { glm::ivec2{ 1, 1}, {1, 0}, {1,-1}, {1,-2} };
                if (rot == 2) return { glm::ivec2{-1,-1}, {0,-1}, {1,-1}, {2,-1} };
                return               { glm::ivec2{ 0, 1}, {0, 0}, {0,-1}, {0,-2} };
            }
            case PieceType::O:
                return { glm::ivec2{0, 0}, {1, 0}, {0, 1}, {1, 1} };
            case PieceType::T: {
                if (rot == 0) return { glm::ivec2{-1, 0}, {0, 0}, {1, 0}, {0, 1} };
                if (rot == 1) return { glm::ivec2{ 0, 1}, {0, 0}, {0,-1}, {1, 0} };
                if (rot == 2) return { glm::ivec2{-1, 0}, {0, 0}, {1, 0}, {0,-1} };
                return               { glm::ivec2{ 0, 1}, {0, 0}, {0,-1}, {-1,0} };
            }
            case PieceType::S: {
                if (rot == 0 || rot == 2) return { glm::ivec2{-1, 0}, {0, 0}, {0, 1}, {1, 1} };
                return                           { glm::ivec2{ 0, 1}, {0, 0}, {1, 0}, {1,-1} };
            }
            case PieceType::Z: {
                if (rot == 0 || rot == 2) return { glm::ivec2{-1, 1}, {0, 1}, {0, 0}, {1, 0} };
                return                           { glm::ivec2{ 1, 1}, {1, 0}, {0, 0}, {0,-1} };
            }
            case PieceType::J: {
                if (rot == 0) return { glm::ivec2{-1, 1}, {-1, 0}, {0, 0}, {1, 0} };
                if (rot == 1) return { glm::ivec2{ 1, 1}, { 0, 1}, {0, 0}, {0,-1} };
                if (rot == 2) return { glm::ivec2{-1, 0}, { 0, 0}, {1, 0}, {1,-1} };
                return               { glm::ivec2{ 0, 1}, { 0, 0}, {0,-1}, {-1,-1} };
            }
            case PieceType::L: {
                if (rot == 0) return { glm::ivec2{ 1, 1}, {-1, 0}, {0, 0}, {1, 0} };
                if (rot == 1) return { glm::ivec2{ 0, 1}, { 0, 0}, {0,-1}, {1,-1} };
                if (rot == 2) return { glm::ivec2{-1, 0}, { 0, 0}, {1, 0}, {-1,-1} };
                return               { glm::ivec2{-1, 1}, { 0, 1}, {0, 0}, {0,-1} };
            }
            default: return { glm::ivec2{0,0}, {0,0}, {0,0}, {0,0} };
        }
    }

    // Checks grid boundaries and block occupancy
    static inline bool is_valid_position(
        const std::array<std::array<uint8_t, GRID_W>, GRID_H>& grid,
        PieceType type,
        glm::ivec2 pos,
        uint8_t rot
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

    // Calculates real-time ghost piece landing Y
    static inline int get_ghost_y(
        const std::array<std::array<uint8_t, GRID_W>, GRID_H>& grid,
        const ActivePiece& piece
    ) {
        if (piece.type == PieceType::None) return piece.pos.y;
        int gy = piece.pos.y;
        while (gy > 0 && is_valid_position(grid, piece.type, { piece.pos.x, gy - 1 }, piece.rotation)) {
            gy--;
        }
        return gy;
    }

    // 7-Bag generator
    static inline PieceType pull_next_piece(uint32_t& rng, std::array<PieceType, 5>& next_queue) {
        PieceType next = next_queue[0];
        for (size_t i = 0; i < 4; ++i) next_queue[i] = next_queue[i + 1];

        // LCG RNG
        rng = rng * 1664525u + 1013904223u;
        uint8_t roll = static_cast<uint8_t>((rng >> 24) % 7 + 1);
        next_queue[4] = static_cast<PieceType>(roll);
        return next;
    }

    struct TetrisStepResult {
        TetrisSnapshot                next_state;
        std::pmr::vector<TetrisEvent> events;

        explicit TetrisStepResult(std::pmr::memory_resource* mr)
            : events(mr) {}
    };

    // PURE SIMULATION REDUCER
    static inline TetrisStepResult reduce_tetris(
        const TetrisSnapshot&          prev,
        std::span<const TetrisCommand> commands,
        float                          dt,
        std::pmr::memory_resource*     frame_mr
    ) {
        TetrisStepResult result(frame_mr);
        result.next_state = prev;
        TetrisSnapshot& s = result.next_state;

        TetrisCommandFrame input = reduce_tetris_commands(commands);

        // Restart
        if (input.reset_pressed) {
            int high = s.high_score;
            s = TetrisSnapshot();
            s.high_score = high;
            s.active.type = pull_next_piece(s.rng_state, s.next_queue);
            s.active.pos  = { 4, 19 };
            result.events.push_back({ TetrisEventType::PIECE_SPAWNED });
            return result;
        }

        if (s.game_over || s.victory) return result;

        s.game_time += dt;

        // Initialize first piece if empty
        if (s.active.type == PieceType::None) {
            s.active.type = pull_next_piece(s.rng_state, s.next_queue);
            s.active.pos  = { 4, 19 };
            s.active.rotation = 0;
            result.events.push_back({ TetrisEventType::PIECE_SPAWNED });
        }

        // 1. HOLD PIECE
        if (input.hold_pressed && !s.hold_locked) {
            PieceType current = s.active.type;
            if (s.hold_piece == PieceType::None) {
                s.hold_piece  = current;
                s.active.type = pull_next_piece(s.rng_state, s.next_queue);
            } else {
                s.active.type = s.hold_piece;
                s.hold_piece  = current;
            }
            s.active.pos        = { 4, 19 };
            s.active.rotation   = 0;
            s.active.lock_timer = 0.0f;
            s.hold_locked       = true;
            result.events.push_back({ TetrisEventType::HOLD_SWAPPED });
        }

        // 2. HORIZONTAL MOVEMENT
        if (input.move_x != 0) {
            glm::ivec2 target_pos = { s.active.pos.x + input.move_x, s.active.pos.y };
            if (is_valid_position(s.grid, s.active.type, target_pos, s.active.rotation)) {
                s.active.pos = target_pos;
                result.events.push_back({ TetrisEventType::PIECE_MOVED });
                if (s.active.lock_resets < 15) {
                    s.active.lock_timer = 0.0f;
                    s.active.lock_resets++;
                }
            }
        }

        // 3. ROTATION (SRS with 5-point wall-kicks)
        if (input.rotate_dir != 0) {
            uint8_t target_rot = (s.active.rotation + (input.rotate_dir > 0 ? 1 : 3)) % 4;
            static const glm::ivec2 KICKS[5] = { {0,0}, {-1,0}, {1,0}, {0,-1}, {0,1} };

            for (const auto& kick : KICKS) {
                glm::ivec2 kick_pos = s.active.pos + kick;
                if (is_valid_position(s.grid, s.active.type, kick_pos, target_rot)) {
                    s.active.pos      = kick_pos;
                    s.active.rotation = target_rot;
                    result.events.push_back({ TetrisEventType::PIECE_ROTATED });
                    if (s.active.lock_resets < 15) {
                        s.active.lock_timer = 0.0f;
                        s.active.lock_resets++;
                    }
                    break;
                }
            }
        }

        // 4. HARD DROP
        if (input.hard_drop) {
            int ghost_y = get_ghost_y(s.grid, s.active);
            int dropped_cells = s.active.pos.y - ghost_y;
            s.active.pos.y = ghost_y;
            s.score += dropped_cells * 2;

            // Immediate lock
            s.active.lock_timer = 1.0f;
            result.events.push_back({
                .type = TetrisEventType::HARD_DROP_SLAM,
                .world_position = glm::vec3((float)s.active.pos.x - 4.5f, (float)s.active.pos.y + 0.5f, 0.0f),
                .score_delta = dropped_cells * 2
            });
        }

        // 5. GRAVITY STEP
        float current_interval = input.soft_drop ? (s.drop_interval * 0.12f) : s.drop_interval;
        s.gravity_timer += dt;

        if (s.gravity_timer >= current_interval) {
            s.gravity_timer = 0.0f;
            glm::ivec2 down_pos = { s.active.pos.x, s.active.pos.y - 1 };

            if (is_valid_position(s.grid, s.active.type, down_pos, s.active.rotation)) {
                s.active.pos = down_pos;
                if (input.soft_drop) s.score += 1;
            } else {
                s.active.lock_timer += current_interval;
            }
        }

        // Check if resting on surface
        bool on_ground = !is_valid_position(s.grid, s.active.type, { s.active.pos.x, s.active.pos.y - 1 }, s.active.rotation);
        if (on_ground) {
            s.active.lock_timer += dt;
        }

        // 6. PIECE LOCKING & LINE CLEARING
        if (on_ground && s.active.lock_timer >= 0.5f) {
            auto blocks = get_piece_blocks(s.active.type, s.active.rotation);
            for (const auto& b : blocks) {
                int gx = s.active.pos.x + b.x;
                int gy = s.active.pos.y + b.y;
                if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H) {
                    s.grid[gy][gx] = static_cast<uint8_t>(s.active.type);
                }
            }

            result.events.push_back({
                .type = TetrisEventType::PIECE_LOCK_IMPACT,
                .world_position = glm::vec3((float)s.active.pos.x - 4.5f, (float)s.active.pos.y + 0.5f, 0.0f)
            });

            // Find cleared lines
            uint8_t cleared_count = 0;
            uint8_t cleared_indices[4]{ 0 };

            for (int y = 0; y < GRID_H; ++y) {
                bool full = true;
                for (int x = 0; x < GRID_W; ++x) {
                    if (s.grid[y][x] == 0) { full = false; break; }
                }
                if (full) {
                    if (cleared_count < 4) cleared_indices[cleared_count] = static_cast<uint8_t>(y);
                    cleared_count++;

                    // Shift down
                    for (int ny = y; ny < GRID_H - 1; ++ny) {
                        s.grid[ny] = s.grid[ny + 1];
                    }
                    s.grid[GRID_H - 1].fill(0);
                    y--; // Re-check shifted row
                }
            }

            if (cleared_count > 0) {
                s.combo_count++;
                s.lines_cleared += cleared_count;

                int base_scores[5] = { 0, 100, 300, 500, 800 };
                int added_score = base_scores[cleared_count] * s.level + (s.combo_count * 50 * s.level);
                s.score += added_score;

                // Level up every 10 lines
                int new_level = 1 + (s.lines_cleared / 10);
                if (new_level > s.level) {
                    s.level = new_level;
                    s.drop_interval = std::max(0.08f, 0.80f * std::pow(0.85f, (float)(s.level - 1)));
                    result.events.push_back({ TetrisEventType::LEVEL_UP });
                }

                result.events.push_back({
                    .type = TetrisEventType::LINES_CLEARED,
                    .lines_cleared_count = cleared_count,
                    .cleared_rows = { cleared_indices[0], cleared_indices[1], cleared_indices[2], cleared_indices[3] },
                    .world_position = glm::vec3(0.0f, (float)cleared_indices[0] + 0.5f, 0.0f),
                    .score_delta = added_score,
                    .combo = s.combo_count
                });
            } else {
                s.combo_count = 0;
            }

            // Spawn next piece
            s.active.type       = pull_next_piece(s.rng_state, s.next_queue);
            s.active.pos        = { 4, 19 };
            s.active.rotation   = 0;
            s.active.lock_timer = 0.0f;
            s.active.lock_resets= 0;
            s.hold_locked       = false;

            // Top-out Game Over check
            if (!is_valid_position(s.grid, s.active.type, s.active.pos, s.active.rotation)) {
                s.game_over = true;
                result.events.push_back({ TetrisEventType::GAME_OVER });
            }
        }

        // High score & Victory condition
        s.high_score = std::max(s.high_score, s.score);
        if (s.score >= s.target_score && !s.victory) {
            s.victory = true;
            result.events.push_back({ TetrisEventType::OBJECTIVE_COMPLETED });
        }

        return result;
    }

} // namespace tetris