#pragma once
// domains/mission/mission.reducer.hpp - GENERIC GOAL INTERPRETER (G2)
//
// Pure: (state, events) -> state' + MissionEvents out.
// Evaluates the ACTIVE goal only; completion fact-chains to the next goal
// (last completion => stage COMPLETE). Timed goals decay by dt; expiry =>
// FAILED. The pod announces MISSION_COMPLETE / MISSION_FAILED - owners of
// other data react, never the reverse.
#include <algorithm>
#include <memory_resource>
#include <span>

#include <domains/mission/mission.contract.hpp>

namespace tetris::mission {

    enum class MissionEventType : uint8_t { MISSION_COMPLETE, MISSION_FAILED };

    struct MissionEventOut {
        MissionEventType type;
        std::string      id;
    };

    struct MissionStep {
        MissionState           next;
        std::pmr::vector<MissionEventOut> events;

        explicit MissionStep(std::pmr::memory_resource* mr) : events(mr) {}
    };

    // Does this event count toward the active cumulative goal?
    inline bool relevant(const GoalDef& g, const MissionEvent& ev) {
        if (g.kind != GoalDef::Kind::EVENT_COUNT) return false;
        return ev.type == g.event_type
            && (!g.use_filter_a || ev.a >= g.filter_a);
    }

    inline MissionStep reduce_mission(
        const MissionState& prev,
        std::span<const MissionEvent> events,
        float dt)
    {
        MissionStep result(std::pmr::get_default_resource());
        MissionState s = prev;
        result.next = s;

        if (s.stage != Stage::ACTIVE || s.defs.empty()) return result;

        const size_t i   = s.active_index;
        const GoalDef& g = s.defs[i];
        auto& progress   = result.next.progress[i];
        bool  done       = false;

        switch (g.kind) {
        case GoalDef::Kind::EVENT_COUNT:
            for (const auto& ev : events) {
                if (relevant(g, ev)) {
                    ++progress;
                    if (progress >= g.target) break;
                }
            }
            done = progress >= g.target;
            break;

        case GoalDef::Kind::LINES_AT_LEAST:
            for (const auto& ev : events) {
                if (ev.type == "LINES_CLEARED" && ev.a >= g.target) {
                    progress = g.target;
                    break;
                }
            }
            done = progress >= g.target;
            break;

        case GoalDef::Kind::ALWAYS_TRUE:
            progress = g.target;
            done     = true;
            break;

        case GoalDef::Kind::SCORE_REACHED:
            // snapshot-driven: handled by the caller setting progress
            // directly when score crosses target (kept for completeness).
            done = progress >= g.target;
            break;
        }

        if (done) {
            result.next.completed.push_back(g.id);
            result.events.push_back({ MissionEventType::MISSION_COMPLETE,
                                      g.id });
            if (i + 1 < result.next.defs.size()) {
                result.next.active_index = i + 1;
                result.next.stage        = Stage::ACTIVE;
                result.next.time_left =
                    result.next.defs[i + 1].time_limit;
            } else {
                result.next.stage = Stage::COMPLETE;
            }
            return result;
        }

        // timed expiry
        if (g.time_limit > 0.0f) {
            result.next.time_left = (s.time_left > 0.0f ? s.time_left
                                                        : g.time_limit) - dt;
            if (result.next.time_left <= 0.0f) {
                result.next.failed.push_back(g.id);
                result.next.stage = Stage::FAILED;
                result.events.push_back({ MissionEventType::MISSION_FAILED,
                                          g.id });
            }
        }
        return result;
    }

} // namespace tetris::mission
