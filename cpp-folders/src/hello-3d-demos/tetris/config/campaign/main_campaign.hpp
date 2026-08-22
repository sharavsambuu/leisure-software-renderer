#pragma once
// tetris/config/campaign/main_campaign.hpp — M2 CAMPAIGN MANIFEST (tetris::config::campaign)
// Ordered stage list: pure config data, no logic. Main selects a stage via
// --stage=N; each stage carries its rules factory and (optionally) the Lua
// rule script that layers designer-authored behavior on top of the engine.
// L3–L5 append here as they land (TODOS.md Part 4 build order B–D).
#include <config/rules.hpp>
#include <config/levels/marathon_01.hpp>
#include <config/levels/blitz_120.hpp>

namespace tetris::config::campaign {

    struct Stage {
        int         index;         // 1-based campaign order
        const char* level_id;      // stable id (future save/unlock key)
        const char* display_name;
        Rules       (*make_rules)();          // pure config factory
        const char* script_path;             // "" = pure C++ tier (no scripting)
        int         unlock_after;            // stage index required first (0 = open)
    };

    // M2 manifest — L1 Marathon Classic → L2 Blitz 120.
    static constexpr Stage STAGES[] = {
        { 1, "marathon_01", Marathon01::NAME, &Marathon01::make_rules, "",                                          0 },
        { 2, "blitz_120",   Blitz120::NAME,   &Blitz120::make_rules,   "domains/progression/scripts/blitz_mode.lua", 1 },
    };

    static constexpr int STAGE_COUNT = static_cast<int>(sizeof(STAGES) / sizeof(STAGES[0]));

    static inline const Stage* find_stage(int index) {
        for (const auto& st : STAGES) {
            if (st.index == index) return &st;
        }
        return nullptr;
    }

} // namespace tetris::config::campaign