#pragma once
// tetris/domains/matrix/matrix.reducer.hpp — PURE TRANSITION (tetris::matrix)
// State_{t+1}, Events = f(State_t, Commands, dt). No platform headers, zero scoring,
// zero allocation beyond the caller's PMR arena for the event log.
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <glm/glm.hpp>

#include <domains/matrix/matrix.contract.hpp>
#include <domains/matrix/matrix.action.hpp>
#include <domains/matrix/matrix.event.hpp>

namespace tetris::matrix {

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
            // L4 specials: fixed footprints (rotation-invariant; distinct
            // colors carry identity)
            case PieceType::Bomb:
            case PieceType::Freeze:
                return { glm::ivec2{0,0}, glm::ivec2{1,0}, glm::ivec2{0,1}, glm::ivec2{1,1} };
            case PieceType::Laser:
                return { glm::ivec2{0,0}, glm::ivec2{1,0}, glm::ivec2{2,0}, glm::ivec2{3,0} };
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

    // L3 raw fact helper: does the piece footprint (+1 cell halo) touch any
    // Garbage block? Pure read of the grid; feeds dig-feel FX/audio mapping.
    static inline uint8_t touches_garbage(
        const std::array<std::array<uint8_t, GRID_W>, GRID_H>& grid,
        const ActivePiece& piece
    ) {
        if (piece.type == PieceType::None) return 0;
        auto blocks = get_piece_blocks(piece.type, piece.rotation);
        for (const auto& b : blocks) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int gx = piece.pos.x + b.x + dx;
                    const int gy = piece.pos.y + b.y + dy;
                    if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H) continue;
                    if (grid[gy][gx] == static_cast<uint8_t>(PieceType::Garbage)) return 1;
                }
            }
        }
        return 0;
    }

    struct MatrixStepResult {
        MatrixSnapshot                next_state;
        std::pmr::vector<MatrixEvent> events;

        explicit MatrixStepResult(std::pmr::memory_resource* mr)
            : events(mr) {}
    };

    // PURE SIMULATION REDUCER
    static inline MatrixStepResult reduce_matrix(
        const MatrixSnapshot&          prev,
        std::span<const TetrisCommand> commands,
        float                          dt,
        std::pmr::memory_resource*     frame_mr
    ) {
        MatrixStepResult result(frame_mr);
        result.next_state = prev;
        MatrixSnapshot& s = result.next_state;

        // L3 injection seam: apply any initial-board stamp FIRST (raw facts —
        // fills the live grid AND the pristine restart backup).
        for (const auto& cmd : commands) {
            if (std::holds_alternative<StampInitialBoardIntent>(cmd)) {
                const auto& stamp = std::get<StampInitialBoardIntent>(cmd);
                s.grid          = stamp.cells;
                s.initial_grid  = stamp.cells;
                s.has_initial_board = true;
            }
        }

        // L4 powerup seams (raw facts only — the scripted ruling decided WHAT):
        // QueueSpecial arms the next pull; ClearCells zeroes cells then sweeps
        // any rows the removal completed; FreezeGravity pauses the gravity step.
        for (const auto& cmd : commands) {
            if (const auto* qs = std::get_if<QueueSpecialIntent>(&cmd)) {
                s.pending_special = qs->special_type;
            } else if (const auto* fg = std::get_if<FreezeGravityIntent>(&cmd)) {
                s.gravity_freeze = std::max(s.gravity_freeze, fg->seconds);
            } else if (const auto* cc = std::get_if<ClearCellsIntent>(&cmd)) {
                for (int i = 0; i < cc->count && i < ClearCellsIntent::MAX_CELLS; ++i) {
                    const int cx = cc->cells[i].x, cy = cc->cells[i].y;
                    if (cx >= 0 && cx < GRID_W && cy >= 0 && cy < GRID_H) s.grid[cy][cx] = 0;
                }
                // Rows completed by the removal collapse immediately (same
                // sweep semantics as the lock path).
                uint8_t cleared_count = 0;
                uint8_t cleared_indices[4]{ 0 };
                uint8_t cleared_garbage_mass = 0;
                for (int y = 0; y < GRID_H; ++y) {
                    bool full = true;
                    for (int x = 0; x < GRID_W; ++x) {
                        if (s.grid[y][x] == 0) { full = false; break; }
                    }
                    if (full) {
                        if (cleared_count < 4) cleared_indices[cleared_count] = static_cast<uint8_t>(y);
                        cleared_count++;
                        for (int x = 0; x < GRID_W; ++x) {
                            if (s.grid[y][x] == static_cast<uint8_t>(PieceType::Garbage)) {
                                if (cleared_garbage_mass < 255) cleared_garbage_mass++;
                            }
                        }
                        for (int ny = y; ny < GRID_H - 1; ++ny) s.grid[ny] = s.grid[ny + 1];
                        s.grid[GRID_H - 1].fill(0);
                        y--;
                    }
                }
                if (cleared_count > 0) {
                    result.events.push_back({
                        .type = MatrixEventType::LINES_CLEARED,
                        .lines_cleared_count = cleared_count,
                        .cleared_rows = { cleared_indices[0], cleared_indices[1], cleared_indices[2], cleared_indices[3] },
                        .world_position = glm::vec3(0.0f, (float)cleared_indices[0] + 0.5f, 0.0f),
                        .garbage_cells = cleared_garbage_mass
                    });
                }
            } else if (const auto* gr = std::get_if<AddGarbageRowsIntent>(&cmd)) {
                // L5 garbage rain: shift the stack up, fill the bottom rows
                // with Garbage (one hole each), lift the active piece above
                // any new overlap. Raw facts in, raw facts out — the volley
                // size/holes were decided upstream (encounter overseer).
                const int rows = std::min<int>(gr->rows, AddGarbageRowsIntent::MAX_ROWS);
                if (rows > 0) {
                    for (int r = 0; r < rows; ++r) {
                        for (int ny = 0; ny < GRID_H - 1; ++ny) s.grid[ny] = s.grid[ny + 1];
                        s.grid[GRID_H - 1].fill(static_cast<uint8_t>(PieceType::Garbage));
                        s.grid[GRID_H - 1][gr->hole_x[r] % GRID_W] = 0;
                    }
                    if (s.active.type != PieceType::None) {
                        const auto blocks = get_piece_blocks(s.active.type, s.active.rotation);
                        for (int lift = 0; lift < GRID_H; ++lift) {
                            bool collides = false;
                            for (const auto& b : blocks) {
                                const int gx = s.active.pos.x + b.x;
                                const int gy = s.active.pos.y + b.y;
                                if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H
                                    && s.grid[gy][gx] != 0) { collides = true; break; }
                            }
                            if (!collides) break;
                            s.active.pos.y -= 1;
                        }
                    }
                    result.events.push_back({
                        .type = MatrixEventType::GARBAGE_RAINED,
                        .world_position = glm::vec3(0.0f, 0.5f, 0.0f),
                        .rain_rows = static_cast<uint8_t>(rows)
                    });
                }
            }
        }

        TetrisCommandFrame input = reduce_tetris_commands(commands);

        // Restart (restores a stamped initial board when one exists)
        if (input.reset_pressed) {
            const CellGrid saved_initial   = prev.initial_grid;
            const bool     had_initial     = prev.has_initial_board;
            s = MatrixSnapshot();
            if (had_initial) {
                s.initial_grid      = saved_initial;
                s.has_initial_board = true;
                s.grid              = saved_initial;
            }
            s.active.type = pull_next_piece(s.rng_state, s.next_queue);
            if (s.pending_special != 0) {   // L4: scripted special override
                s.active.type = static_cast<PieceType>(s.pending_special);
                s.pending_special = 0;
            }
            s.active.pos  = { 4, 19 };
            result.events.push_back({ MatrixEventType::PIECE_SPAWNED });
            return result;
        }

        if (s.game_over) return result;

        s.game_time += dt;

        // Initialize first piece if empty.
        // NOTE: this pull must happen exactly once per run; double-pulls from
        // multiple init paths desync the 7-bag stream across runs.
        if (s.active.type == PieceType::None) {
            s.active.type = pull_next_piece(s.rng_state, s.next_queue);
            if (s.pending_special != 0) {
                s.active.type = static_cast<PieceType>(s.pending_special);
                s.pending_special = 0;
            }
            s.active.pos  = { 4, 19 };
            s.active.rotation = 0;
            result.events.push_back({ MatrixEventType::PIECE_SPAWNED });
        }

        // 1. HOLD PIECE
        if (input.hold_pressed && !s.hold_locked) {
            PieceType current = s.active.type;
            if (s.hold_piece == PieceType::None) {
                s.hold_piece  = current;
                s.active.type = pull_next_piece(s.rng_state, s.next_queue);
                if (s.pending_special != 0) {
                    s.active.type = static_cast<PieceType>(s.pending_special);
                    s.pending_special = 0;
                }
            } else {
                s.active.type = s.hold_piece;
                s.hold_piece  = current;
            }
            s.active.pos        = { 4, 19 };
            s.active.rotation   = 0;
            s.active.lock_timer = 0.0f;
            s.hold_locked       = true;
            result.events.push_back({ MatrixEventType::HOLD_SWAPPED });
        }

        // 2. HORIZONTAL MOVEMENT
        if (input.move_x != 0) {
            glm::ivec2 target_pos = { s.active.pos.x + input.move_x, s.active.pos.y };
            if (is_valid_position(s.grid, s.active.type, target_pos, s.active.rotation)) {
                s.active.pos = target_pos;
                result.events.push_back({ MatrixEventType::PIECE_MOVED });
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
                    result.events.push_back({ MatrixEventType::PIECE_ROTATED });
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

            // Immediate lock
            s.active.lock_timer = 1.0f;
            result.events.push_back({
                .type = MatrixEventType::HARD_DROP_SLAM,
                .world_position = glm::vec3((float)s.active.pos.x - 4.5f, (float)s.active.pos.y + 0.5f, 0.0f),
                .cells = dropped_cells,
                .garbage_cells = touches_garbage(s.grid, s.active)
            });
        }

        // 5. GRAVITY STEP (L4: FreezeGravityIntent pauses the fall entirely —
        // movement/rotation stay live, the lock timer does not accumulate)
        // Part 6 input feel: soft drop is a HELD STATE (continuous), not an
        // OS-repeat stream. The frame's soft_drop_held comes from the input
        // edge each tick.
        const bool soft_held  = input.soft_drop_held || input.soft_drop;
        float current_interval = soft_held ? (s.drop_interval * 0.12f) : s.drop_interval;
        const bool frozen = (s.gravity_freeze > 0.0f);
        if (frozen) {
            s.gravity_freeze -= dt;
        } else {
            s.gravity_timer += dt;
        }

        if (!frozen && s.gravity_timer >= current_interval) {
            s.gravity_timer = 0.0f;
            glm::ivec2 down_pos = { s.active.pos.x, s.active.pos.y - 1 };

            if (is_valid_position(s.grid, s.active.type, down_pos, s.active.rotation)) {
                s.active.pos = down_pos;
                if (input.soft_drop) result.events.push_back({ .type = MatrixEventType::SOFT_DROP, .cells = 1 });
            } else {
                s.active.lock_timer += current_interval;
            }
        }

        // Check if resting on surface (frozen pieces never lock by timer)
        bool on_ground = !is_valid_position(s.grid, s.active.type, { s.active.pos.x, s.active.pos.y - 1 }, s.active.rotation);
        if (on_ground && !frozen) {
            s.active.lock_timer += dt;
        }

        // 6. PIECE LOCKING & LINE CLEARING
        if (on_ground && !frozen && s.active.lock_timer >= 0.5f) {
            // L4: specials never write the grid — they detonate. The raw fact
            // (type + anchor cell) goes out; the scripted ruling decides the
            // mutation, which arrives next frame as ClearCells/Freeze commands.
            if (is_special_piece(s.active.type)) {
                result.events.push_back({
                    .type = MatrixEventType::SPECIAL_LOCKED,
                    .world_position = glm::vec3((float)s.active.pos.x - 4.5f, (float)s.active.pos.y + 0.5f, 0.0f),
                    .special_type = static_cast<uint8_t>(s.active.type),
                    .lock_x = static_cast<int16_t>(s.active.pos.x),
                    .lock_y = static_cast<int16_t>(s.active.pos.y)
                });
                s.active.type       = pull_next_piece(s.rng_state, s.next_queue);
                if (s.pending_special != 0) {
                    s.active.type = static_cast<PieceType>(s.pending_special);
                    s.pending_special = 0;
                }
                s.active.pos        = { 4, 19 };
                s.active.rotation   = 0;
                s.active.lock_timer = 0.0f;
                s.active.lock_resets= 0;
                s.hold_locked       = false;
                if (!is_valid_position(s.grid, s.active.type, s.active.pos, s.active.rotation)) {
                    s.game_over = true;
                    result.events.push_back({ MatrixEventType::GAME_OVER });
                }
                return result;
            }
            auto blocks = get_piece_blocks(s.active.type, s.active.rotation);
            for (const auto& b : blocks) {
                int gx = s.active.pos.x + b.x;
                int gy = s.active.pos.y + b.y;
                if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H) {
                    s.grid[gy][gx] = static_cast<uint8_t>(s.active.type);
                }
            }

            result.events.push_back({
                .type = MatrixEventType::PIECE_LOCK_IMPACT,
                .world_position = glm::vec3((float)s.active.pos.x - 4.5f, (float)s.active.pos.y + 0.5f, 0.0f),
                .garbage_cells = touches_garbage(s.grid, s.active)
            });

            // Find cleared lines
            uint8_t cleared_count = 0;
            uint8_t cleared_indices[4]{ 0 };
            uint8_t cleared_garbage_mass = 0;

            for (int y = 0; y < GRID_H; ++y) {
                bool full = true;
                for (int x = 0; x < GRID_W; ++x) {
                    if (s.grid[y][x] == 0) { full = false; break; }
                }
                if (full) {
                    if (cleared_count < 4) cleared_indices[cleared_count] = static_cast<uint8_t>(y);
                    cleared_count++;
                    for (int x = 0; x < GRID_W; ++x) {
                        if (s.grid[y][x] == static_cast<uint8_t>(PieceType::Garbage)) {
                            if (cleared_garbage_mass < 255) cleared_garbage_mass++;
                        }
                    }

                    // Shift down
                    for (int ny = y; ny < GRID_H - 1; ++ny) {
                        s.grid[ny] = s.grid[ny + 1];
                    }
                    s.grid[GRID_H - 1].fill(0);
                    y--; // Re-check shifted row
                }
            }

            if (cleared_count > 0) {



                result.events.push_back({
                    .type = MatrixEventType::LINES_CLEARED,
                    .lines_cleared_count = cleared_count,
                    .cleared_rows = { cleared_indices[0], cleared_indices[1], cleared_indices[2], cleared_indices[3] },
                    .world_position = glm::vec3(0.0f, (float)cleared_indices[0] + 0.5f, 0.0f),
                    .garbage_cells = cleared_garbage_mass
                });
            } else {
            }

            // Spawn next piece
            s.active.type       = pull_next_piece(s.rng_state, s.next_queue);
            if (s.pending_special != 0) {
                s.active.type = static_cast<PieceType>(s.pending_special);
                s.pending_special = 0;
            }
            s.active.pos        = { 4, 19 };
            s.active.rotation   = 0;
            s.active.lock_timer = 0.0f;
            s.active.lock_resets= 0;
            s.hold_locked       = false;

            // Top-out Game Over check
            if (!is_valid_position(s.grid, s.active.type, s.active.pos, s.active.rotation)) {
                s.game_over = true;
                result.events.push_back({ MatrixEventType::GAME_OVER });
            }
        }


        return result;
    }
} // namespace tetris::matrix
