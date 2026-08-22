#pragma once
// tetris/domains/powerups/powerups.reducer.hpp — PURE TRANSITION (tetris::powerups)
// State_{t+1}, {mutations[], events[]} = f(State_t, matrix_facts, rulings, dt).
// Listens to matrix raw facts only (Rule 8.1); the grid is never touched here —
// mutations are emitted as plain matrix commands for main to inject.
#include <algorithm>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <vector>

#include <domains/matrix/matrix.action.hpp>
#include <domains/matrix/matrix.contract.hpp>
#include <domains/matrix/matrix.event.hpp>
#include <domains/powerups/powerups.action.hpp>
#include <domains/powerups/powerups.contract.hpp>
#include <domains/powerups/powerups.event.hpp>

namespace tetris::powerups {
using tetris::matrix::MatrixEvent;
using tetris::matrix::MatrixEventType;
using tetris::matrix::TetrisCommand;
using tetris::matrix::QueueSpecialIntent;
using tetris::matrix::ClearCellsIntent;
using tetris::matrix::FreezeGravityIntent;

    struct PowerupStep {
        PowerupSnapshot                    next;
        std::pmr::vector<PowerupEvent>     events;
        std::pmr::vector<TetrisCommand>    mutations;

        explicit PowerupStep(std::pmr::memory_resource* mr)
            : events(mr), mutations(mr) {}
    };

    // every_n / freeze_seconds come from config::Rules (plain numbers in).
    static inline PowerupStep reduce_powerups(
        const PowerupSnapshot&            prev,
        std::span<const MatrixEvent>      matrix_events,
        std::span<const ApplyRulingIntent> rulings,
        int                               every_n,
        float                             freeze_seconds,
        std::pmr::memory_resource*        frame_mr
    ) {
        PowerupStep out(frame_mr);
        out.next = prev;

        // --- Scripted rulings: convert to raw mutation commands --------------
        for (const auto& r : rulings) {
            if (!r.ruling.valid) continue;
            ClearCellsIntent cc{};
            const int n = std::min<int>(r.ruling.clear_count, ClearCellsIntent::MAX_CELLS);
            for (int i = 0; i < n; ++i) {
                cc.cells[cc.count++] = { r.ruling.clear_x[i], r.ruling.clear_y[i] };
            }
            if (cc.count > 0) out.mutations.push_back(cc);
            if (r.ruling.freeze_seconds > 0.0f) {
                out.mutations.push_back(FreezeGravityIntent{ r.ruling.freeze_seconds });
                out.next.freeze_left = std::max(out.next.freeze_left, r.ruling.freeze_seconds);
            }
            out.events.push_back({ PowerupEventType::POWERUP_TRIGGERED,
                                   static_cast<uint8_t>(out.next.armed_next) });
            out.next.blasts_fired += (r.ruling.fx_id == 1) ? 1 : 0;
            out.next.lasers_fired += (r.ruling.fx_id == 2) ? 1 : 0;
            out.next.freezes_used += (r.ruling.fx_id == 3) ? 1 : 0;
        }

        // --- Matrix raw facts -------------------------------------------------
        for (const auto& ev : matrix_events) {
            switch (ev.type) {
            case MatrixEventType::PIECE_SPAWNED:
                if (out.next.request_open) break;   // a special is already queued
                out.next.pieces_since_special++;
                if (every_n > 0 && out.next.pieces_since_special >= every_n) {
                    out.next.request_open = true;   // latched until main fulfills
                    out.events.push_back({ PowerupEventType::SPAWN_SPECIAL_REQUESTED,
                                           static_cast<uint8_t>(out.next.armed_next) });
                }
                break;
            case MatrixEventType::SPECIAL_LOCKED:
                // The special spawned and detonated: advance the cycle, reset
                // the cadence counter, clear the latch.
                out.next.armed_next =
                    (out.next.armed_next == PowerupType::Bomb)   ? PowerupType::Laser :
                    (out.next.armed_next == PowerupType::Laser)  ? PowerupType::Freeze :
                                                                   PowerupType::Bomb;
                out.next.pieces_since_special = 0;
                out.next.request_open         = false;
                break;
            default:
                break;
            }
        }

        return out;
    }

} // namespace tetris::powerups