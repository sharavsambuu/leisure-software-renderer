#pragma once
// domains/mission/mission.contract.hpp - MISSION POD CONTRACT (Part 7 G2)
//
// The mission pod owns ONLY mission-specific truth: which goal is active,
// per-goal progress, completed/failed ids. It NEVER duplicates game truth
// (score/lines live in progression; the grid lives in matrix) - goals READ
// facts and snapshots.
//
// Two goal styles (SCRIPTING.md section 2):
//   cumulative counter: relevant(ev) + target  - progress accumulates across
//     ticks; complete when progress >= target
//   batch predicate:    test(events, snapshot) - true in one tick completes
//
// Sequencing is fact-chained: completing goal[i] advances to goal[i+1];
// finishing the last sets stage = complete. Timed goals expire to failed.
#include <cstdint>
#include <string>
#include <vector>

namespace tetris::mission {

    // Opaque per-tick event view. The engine converts MatrixEvent /
    // ProgressionEvent / PowerupEvent into these plain rows so the pod (and,
    // later, Lua goal scripts via IScriptHost) can scan them uniformly.
    struct MissionEvent {
        std::string type;      // "LINES_CLEARED", "ENTITY_KILLED", ...
        int         a = 0;     // primary payload (count / rows / team ...)
        int         b = 0;     // secondary payload
    };

    struct MissionSnapshot {
        int   score       = 0;
        int   lines       = 0;
        int   level       = 1;
        bool  overdrive   = false;   // freeze/overdrive window flag
        float stack_ratio = 0.0f;    // 0..1 danger measure
    };

    enum class Stage : uint8_t { ACTIVE, COMPLETE, FAILED };

    struct GoalDef {
        std::string id;
        std::string hint;

        // --- cumulative style ---
        // relevant(ev): true if this event counts toward progress.
        // target: completion threshold.
        // weight(ev): optional per-event contribution (default 1).
        // Implemented host-side via a small enum + params instead of
        // std::function so the pod stays serializable and Lua-definable.
        enum class Kind : uint8_t {
            EVENT_COUNT,      // count events where type==event_type (+filter_a)
            LINES_AT_LEAST,   // any single clear with a.count >= target
            SCORE_REACHED,    // snapshot.score >= target
            ALWAYS_TRUE       // immediate completion (testing / tutorial)
        };
        Kind kind          = Kind::EVENT_COUNT;
        std::string event_type;          // for EVENT_COUNT
        int         filter_a = 0;        // e.g. min lines_cleared_count
        bool        use_filter_a = false;
        int         target   = 1;
        float       time_limit = 0.0f;   // 0 = untimed
    };

    struct MissionState {
        std::vector<GoalDef> defs;
        size_t               active_index = 0;
        Stage                stage        = Stage::COMPLETE;  // empty => done
        std::vector<int64_t> progress;                        // per goal
        std::vector<std::string> completed;
        std::vector<std::string> failed;
        float                time_left = 0.0f;              // active timed goal

        static MissionState create(const std::vector<GoalDef>& defs) {
            MissionState s;
            s.defs          = defs;
            s.active_index  = defs.empty() ? 0 : 0;
            s.stage         = defs.empty() ? Stage::COMPLETE : Stage::ACTIVE;
            s.progress.assign(defs.size(), 0);
            s.time_left     = defs.empty()
                                  ? 0.0f
                                  : defs[0].time_limit;
            return s;
        }
    };

    // Non-owning per-event view used for script marshaling (G1). The host
    // builds these from the tick's typed events; goal scripts consume them
    // as plain tables.
    struct MissionEventView {
        const char* type;
        int         a = 0;
        int         b = 0;
    };

} // namespace tetris::mission
