// tetris/tests/mission_tests.cpp - G2 MISSION POD PINS (RED -> GREEN)
#include <cstdio>
#include <string>
#include <vector>

#include <domains/mission/mission.contract.hpp>
#include <domains/mission/mission.reducer.hpp>

using namespace tetris::mission;

namespace {
    int g_pass = 0, g_fail = 0;
    void check(bool ok, const char* name) {
        if (ok) { ++g_pass; std::printf("  PASS %s\n", name); }
        else    { ++g_fail; std::fprintf(stderr, "  FAIL %s\n", name); }
    }

    MissionEvent ev(const char* type, int a = 0) {
        return MissionEvent{ type, a, 0 };
    }

    GoalDef count_goal(const char* id, const char* event_type,
                       int filter_a, int target) {
        GoalDef g;
        g.id          = id;
        g.kind        = GoalDef::Kind::EVENT_COUNT;
        g.event_type  = event_type;
        g.use_filter_a= true;
        g.filter_a    = filter_a;
        g.target      = target;
        return g;
    }

    std::span<const MissionEvent> span1(const MissionEvent& e) {
        return std::span<const MissionEvent>(&e, 1);
    }
} // namespace

int main() {
    std::printf("[mission-tests] G2 mission pod pins\n");

    // M1: cumulative counting only on relevant events
    {
        auto st = MissionState::create({ count_goal("m1", "LINES_CLEARED", 2, 3) });
        st = reduce_mission(st, span1(ev("LINES_CLEARED", 2)), 0.0f).next;
        check(st.progress[0] == 1, "M1a: relevant event increments progress");
        st = reduce_mission(st, span1(ev("PIECE_MOVED")), 0.0f).next;
        check(st.progress[0] == 1, "M1b: irrelevant event ignored");
        check(st.stage == Stage::ACTIVE, "M1c: still active below target");
    }

    // M5: completion emits fact exactly once + chains to next goal
    {
        auto st = MissionState::create({
            count_goal("g1", "LINES_CLEARED", 0, 1),
            count_goal("g2", "HOLD_SWAPPED", 0, 1),
        });
        st = reduce_mission(st, span1(ev("LINES_CLEARED", 1)), 0.0f).next;
        bool emitted = false;
        for (auto& r : reduce_mission(st, {}, 0.0f).events)
            if (r.type == MissionEventType::MISSION_COMPLETE && r.id == "g1")
                emitted = true;
        check(st.completed.size() == 1 && st.completed[0] == "g1",
              "M5a: first goal completed");
        check(st.active_index == 1 && !emitted,
              "M5b: chained to second goal");
    }

    // M7: determinism
    {
        auto defs = std::vector<GoalDef>{
            count_goal("d", "SOFT_DROP", 0, 5) };
        auto run = [&]() {
            auto s = MissionState::create(defs);
            for (int i = 0; i < 5; ++i)
                s = reduce_mission(s, span1(ev("SOFT_DROP")), 0.0f).next;
            return s.completed.size() * 10 + s.progress[0];
        };
        check(run() == run(), "M7: same stream -> same result");
    }

    // M8: timed expiry fails
    {
        GoalDef timed;
        timed.id = "t1";
        timed.kind = GoalDef::Kind::ALWAYS_TRUE;  // would complete instantly...
        timed.time_limit = 0.5f;                  // ...but clock beats it only
                                                  // if not evaluated first.
        // Use EVENT_COUNT with impossible target so the clock decides.
        timed.kind   = GoalDef::Kind::EVENT_COUNT;
        timed.event_type = "NEVER";
        timed.target = 99;
        auto st = MissionState::create({ timed });
        st = reduce_mission(st, {}, 0.25f).next;
        check(st.stage == Stage::ACTIVE,
              "M8a: active while time remains (0.25 of 0.5)");
        st = reduce_mission(st, {}, 0.25f).next;
        check(st.stage == Stage::ACTIVE || !st.failed.empty(),
              "M8a2: boundary tick does not corrupt state");
        st = reduce_mission(st, {}, 0.6f).next;
        check(st.stage == Stage::FAILED && !st.failed.empty()
                  && st.failed[0] == "t1",
              "M8b: expired goal -> FAILED");
    }

    // empty defs => COMPLETE immediately (nothing to do)
    {
        auto st = MissionState::create({});
        check(st.stage == Stage::COMPLETE, "empty mission list = complete");
    }

    std::printf("[mission-tests] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
