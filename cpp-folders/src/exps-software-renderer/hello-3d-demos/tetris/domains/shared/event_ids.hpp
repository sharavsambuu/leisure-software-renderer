#pragma once
// domains/shared/event_ids.hpp - P4 EVENT REGISTRY (single source of truth)
//
// Every fact that can flow through the event log is registered here with its
// producing pod. Reducers keep their own typed event structs; this registry
// is the cross-pod VOCABULARY used by:
//   - scripts/generate-event-flow.mjs  (docs/pods/EVENT_FLOW.md generation)
//   - future debug event-log dumps / mission scripting
//
// Law: a fact MUST have exactly one PRODUCER. Adding an emission requires
// adding a registry row in the same change.
#pragma once
#include <array>
#include <cstdint>
#include <string_view>

namespace tetris::facts {

    enum class FactId : uint16_t {
        // matrix pod
        PIECE_SPAWNED,
        PIECE_MOVED,
        PIECE_ROTATED,
        PIECE_LOCK_IMPACT,
        HARD_DROP_SLAM,
        SOFT_DROP,
        LINES_CLEARED,
        HOLD_SWAPPED,
        GAME_OVER,
        SPECIAL_LOCKED,
        GARBAGE_RAINED,
        // progression pod
        SCORE_CHANGED,
        COMBO_STREAK,
        LEVEL_UP,
        OBJECTIVE_COMPLETED,
        CLOCK_TICK,
        TIME_BONUS,
        TIME_UP,
        // powerups pod
        POWERUP_TRIGGERED,
        SPAWN_SPECIAL_REQUESTED,
        SPECIAL_ARMED,
        // environment pod
        PHASE_CHANGED,
        RAIN_DUE,
        // count sentinel - KEEP LAST
        COUNT
    };

    struct FactInfo {
        FactId      id;
        const char* name;
        const char* producer;   // owning pod
    };

    inline constexpr std::array<FactInfo, static_cast<size_t>(FactId::COUNT)>
        FACT_REGISTRY{{
            { FactId::PIECE_SPAWNED,       "PIECE_SPAWNED",       "matrix"      },
            { FactId::PIECE_MOVED,         "PIECE_MOVED",         "matrix"      },
            { FactId::PIECE_ROTATED,       "PIECE_ROTATED",       "matrix"      },
            { FactId::PIECE_LOCK_IMPACT,   "PIECE_LOCK_IMPACT",   "matrix"      },
            { FactId::HARD_DROP_SLAM,      "HARD_DROP_SLAM",      "matrix"      },
            { FactId::SOFT_DROP,           "SOFT_DROP",           "matrix"      },
            { FactId::LINES_CLEARED,       "LINES_CLEARED",       "matrix"      },
            { FactId::HOLD_SWAPPED,        "HOLD_SWAPPED",        "matrix"      },
            { FactId::GAME_OVER,           "GAME_OVER",           "matrix"      },
            { FactId::SPECIAL_LOCKED,      "SPECIAL_LOCKED",      "matrix"      },
            { FactId::GARBAGE_RAINED,      "GARBAGE_RAINED",      "matrix"      },
            { FactId::SCORE_CHANGED,       "SCORE_CHANGED",       "progression" },
            { FactId::COMBO_STREAK,        "COMBO_STREAK",        "progression" },
            { FactId::LEVEL_UP,            "LEVEL_UP",            "progression" },
            { FactId::OBJECTIVE_COMPLETED, "OBJECTIVE_COMPLETED", "progression" },
            { FactId::CLOCK_TICK,          "CLOCK_TICK",          "progression" },
            { FactId::TIME_BONUS,          "TIME_BONUS",          "progression" },
            { FactId::TIME_UP,             "TIME_UP",             "progression" },
            { FactId::POWERUP_TRIGGERED,   "POWERUP_TRIGGERED",   "powerups"    },
            { FactId::SPAWN_SPECIAL_REQUESTED, "SPAWN_SPECIAL_REQUESTED", "powerups" },
            { FactId::SPECIAL_ARMED,       "SPECIAL_ARMED",       "powerups"    },
            { FactId::PHASE_CHANGED,       "PHASE_CHANGED",       "environment" },
            { FactId::RAIN_DUE,            "RAIN_DUE",            "environment" },
        }};

    inline constexpr const char* fact_name(FactId id) {
        return FACT_REGISTRY[static_cast<size_t>(id)].name;
    }

    inline constexpr const char* fact_producer(FactId id) {
        return FACT_REGISTRY[static_cast<size_t>(id)].producer;
    }

} // namespace tetris::facts
