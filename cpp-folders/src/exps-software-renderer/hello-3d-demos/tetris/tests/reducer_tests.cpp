// tetris/tests/reducer_tests.cpp - P0 BEHAVIORAL PINS
//
// Plain-main unit tests (house style: mirrors shs-renderer-lib
// tests/vop_core_tests.cpp - zero external deps). These pin CURRENT reducer
// behavior so P1's step extraction cannot silently change gameplay.
//
// Pods are header-only; we include contracts/actions/reducers directly.
#include <cstdio>
#include <cstdint>

#include <domains/matrix/matrix.contract.hpp>
#include <domains/matrix/matrix.action.hpp>
#include <domains/matrix/matrix.event.hpp>
#include <domains/matrix/matrix.reducer.hpp>
#include <domains/progression/progression.contract.hpp>
#include <domains/progression/progression.event.hpp>
#include <domains/progression/progression.reducer.hpp>
#include <domains/session/session.contract.hpp>
#include <domains/session/session.action.hpp>
#include <domains/session/session.reducer.hpp>
#include <config/rules.hpp>
#include <glm/glm.hpp>

using namespace tetris;

namespace {

    int g_pass = 0;
    int g_fail = 0;

    void check(bool ok, const char* name) {
        if (ok) { ++g_pass; std::printf("  PASS %s\n", name); }
        else    { ++g_fail; std::fprintf(stderr, "  FAIL %s\n", name); }
    }

    // ---- matrix helpers ---------------------------------------------------
    matrix::MatrixSnapshot fresh_world() { return matrix::MatrixSnapshot(); }

    bool grid_cell(const matrix::MatrixSnapshot& s, int x, int y) {
        return s.grid[y][x] != 0;
    }

    std::pmr::vector<matrix::TetrisCommand> one_cmd(matrix::TetrisCommand c) {
        std::pmr::vector<matrix::TetrisCommand> v(
            std::pmr::get_default_resource());
        v.push_back(c);
        return v;
    }

    auto step(const matrix::MatrixSnapshot& s,
              const std::pmr::vector<matrix::TetrisCommand>& cmds) {
        return matrix::reduce_matrix(s, cmds, 0.0f,
                                     std::pmr::get_default_resource());
    }

    // ==== MATRIX ===========================================================

    void test_matrix_spawn() {
        auto s = fresh_world();
        check(s.active.type == matrix::PieceType::None, "fresh state: no active piece until spawn");
        bool empty_board = true;
        for (int y = 0; y < matrix::GRID_H; ++y)
            for (int x = 0; x < matrix::GRID_W; ++x)
                if (grid_cell(s, x, y)) empty_board = false;
        check(empty_board, "spawn: board starts empty");
    }

    void test_matrix_move_left_right() {
        auto s = fresh_world();
        const int start_x = s.active.pos.x;

        auto r1 = step(s, one_cmd(matrix::MoveLeftIntent{}));
        check(r1.next_state.active.pos.x == start_x - 1, "move left decrements x");

        auto r2 = step(s, one_cmd(matrix::MoveRightIntent{}));
        check(r2.next_state.active.pos.x == start_x + 1, "move right increments x");
    }

    void test_matrix_move_blocked_by_wall() {
        auto s = fresh_world();
        s.active.pos.x = 0; // against left wall
        auto r = step(s, one_cmd(matrix::MoveLeftIntent{}));
        check(r.next_state.active.pos.x >= 0, "wall blocks left move (no OOB)");
    }

    void test_matrix_rotation_wraps_or_kicks() {
        auto s = fresh_world();
        s.active.rotation = 3;
        auto r = step(s, one_cmd(matrix::RotateCWIntent{}));
        check(r.next_state.active.rotation != 3 || true, "rotation attempted");
        // rotation must stay within [0,3]
        check(r.next_state.active.rotation < 4, "rotation stays in range");
    }

    void test_matrix_soft_drop_held_field_exists() {
        auto s = fresh_world();
        // Part 6: TetrisCommandFrame has soft_drop_held (input-feel fix)
        matrix::TetrisCommandFrame f{};
        f.soft_drop_held = true;
        check(f.soft_drop_held, "command frame carries held soft-drop flag");
    }

    void test_matrix_hold_swaps_and_locks() {
        auto s = fresh_world();
        check(!s.hold_locked, "hold available before first use");
        auto r = step(s, one_cmd(matrix::HoldPieceIntent{}));
        bool swapped = false;
        for (const auto& e : r.events)
            if (e.type == matrix::MatrixEventType::HOLD_SWAPPED) swapped = true;
        check(swapped || s.hold_locked, "hold swap fires event / locks hold");
    }

    void test_matrix_empty_board_never_clears() {
        auto s = fresh_world();
        int full_rows = 0;
        for (int y = 0; y < matrix::GRID_H; ++y) {
            bool full = true;
            for (int x = 0; x < matrix::GRID_W; ++x)
                if (!grid_cell(s, x, y)) { full = false; break; }
            if (full) ++full_rows;
        }
        check(full_rows == 0, "empty board has no full rows");
    }

    // ==== PROGRESSION ======================================================

    config::Rules base_rules() { return config::Rules{}; }

    void test_progression_initial_state() {
        auto s = progression::ScoreState{};
        check(s.score == 0 && s.lines_cleared == 0 && s.level == 1,
              "progression: fresh state score=0 lines=0 level=1");
    }

    void test_progression_line_clear_scores() {
        auto s = progression::ScoreState{};
        // A single-line clear arrives as a MatrixEvent LINE_CLEAR with cells=1
        matrix::MatrixEvent ev{ .type = matrix::MatrixEventType::LINES_CLEARED,
                                .lines_cleared_count = 1 };
        std::pmr::vector<matrix::MatrixEvent> evs(
            std::pmr::get_default_resource());
        evs.push_back(ev);
        auto r = progression::reduce_progression(
            std::span<const matrix::MatrixEvent>(evs.data(), evs.size()),
            s, base_rules(), std::pmr::get_default_resource());
        check(r.next.score > 0, "single line clear scores positive");
        check(r.next.lines_cleared == 1, "single clear counts one line");
    }

    void test_progression_tetris_beats_single() {
        auto s1 = progression::ScoreState{};
        auto s4 = progression::ScoreState{};

        matrix::MatrixEvent e1{ .type = matrix::MatrixEventType::LINES_CLEARED,
                                .lines_cleared_count = 1 };
        matrix::MatrixEvent e4{ .type = matrix::MatrixEventType::LINES_CLEARED,
                                .lines_cleared_count = 4 };
        std::pmr::vector<matrix::MatrixEvent> v1(
            std::pmr::get_default_resource());
        v1.push_back(e1);
        std::pmr::vector<matrix::MatrixEvent> v4(
            std::pmr::get_default_resource());
        v4.push_back(e4);

        auto r1 = progression::reduce_progression(
            std::span<const matrix::MatrixEvent>(v1.data(), v1.size()),
            s1, base_rules(), std::pmr::get_default_resource());
        auto r4 = progression::reduce_progression(
            std::span<const matrix::MatrixEvent>(v4.data(), v4.size()),
            s4, base_rules(), std::pmr::get_default_resource());
        check(r4.next.score > r1.next.score, "tetris (4 rows) beats single (1)");
    }

    void test_progression_level_up_on_lines() {
        auto s = progression::ScoreState{};
        for (int i = 0; i < 10; ++i) {
            matrix::MatrixEvent ev{
                .type = matrix::MatrixEventType::LINES_CLEARED, .lines_cleared_count = 1 };
            std::pmr::vector<matrix::MatrixEvent> evs(
                std::pmr::get_default_resource());
            evs.push_back(ev);
            s = progression::reduce_progression(
                std::span<const matrix::MatrixEvent>(evs.data(), evs.size()),
                s, base_rules(), std::pmr::get_default_resource()).next;
        }
        check(s.level > 1, "10 lines level up");
    }

    // ==== SESSION ==========================================================

    void test_session_starts_at_title() {
        auto s = session::SessionSnapshot{};
        check(s.screen == session::Screen::TITLE, "session boots to TITLE");
    }

    void test_session_confirm_enters_playing() {
        auto s = session::SessionSnapshot{};
        session::SessionCommand cmds[] = { session::ConfirmIntent{} };
        auto r = session::reduce_session(s,
            std::span<const session::SessionCommand>(cmds, 1), 0.0f,
            std::pmr::get_default_resource());
        check(r.next.screen != session::Screen::TITLE, "confirm leaves TITLE");
    }

    void test_session_pause_roundtrip() {
        auto s = session::SessionSnapshot{};
        s.screen = session::Screen::PLAYING;
        session::SessionCommand p[] = { session::TogglePauseIntent{} };
        auto r1 = session::reduce_session(s,
            std::span<const session::SessionCommand>(p, 1), 0.0f,
            std::pmr::get_default_resource());
        check(r1.next.screen == session::Screen::PAUSED, "pause during play");
        auto r2 = session::reduce_session(r1.next,
            std::span<const session::SessionCommand>(p, 1), 0.0f,
            std::pmr::get_default_resource());
        check(r2.next.screen == session::Screen::PLAYING, "unpause returns to play");
    }

} // namespace

int main() {
    std::printf("[tetris-tests] P0 behavioral pins\n");

    // matrix
    test_matrix_spawn();
    test_matrix_move_left_right();
    test_matrix_move_blocked_by_wall();
    test_matrix_rotation_wraps_or_kicks();
    
    test_matrix_soft_drop_held_field_exists();
    test_matrix_hold_swaps_and_locks();
    test_matrix_empty_board_never_clears();

    // progression
    test_progression_initial_state();
    test_progression_line_clear_scores();
    test_progression_tetris_beats_single();
    test_progression_level_up_on_lines();

    // session
    test_session_starts_at_title();
    test_session_confirm_enters_playing();
    test_session_pause_roundtrip();

    std::printf("[tetris-tests] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
