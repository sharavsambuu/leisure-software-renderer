// tetris/tests/stage_loader_tests.cpp - P3 DATA-DRIVEN CAMPAIGN PINS
#include <cstdio>
#include <string>

#include <domains/matrix/matrix.action.hpp>
#include <domains/powerups/powerups.contract.hpp>
#include <domains/environment/environment.contract.hpp>
#include <game/stage.hpp>

using namespace tetris;

namespace {
    int g_pass = 0, g_fail = 0;
    void check(bool ok, const char* name) {
        if (ok) { ++g_pass; std::printf("  PASS %s\n", name); }
        else    { ++g_fail; std::fprintf(stderr, "  FAIL %s\n", name); }
    }
}

int main() {
#ifndef TETRIS_LUA_ENABLED
    std::printf("[stage-tests] built without Lua - fallback path only\n");
    auto fb = game::load_campaign(TETRIS_SOURCE_ROOT);
    check(fb.used_fallback && fb.stages.size() == 1
              && fb.stages[0].id == "marathon_01",
          "no-Lua build: marathon fallback, no crash");
#else
    std::printf("[stage-tests] P3 data-driven campaign\n");

    auto result = game::load_campaign(TETRIS_SOURCE_ROOT);

    check(result.stages.size() == 5, "campaign.lua: 5 stages loaded");
    if (result.stages.size() == 5) {
        check(result.stages[0].id == "marathon_01", "stage 1 id = marathon_01");
        check(result.stages[1].id == "blitz_120",   "stage 2 id = blitz_120");
        check(result.stages[2].name == "GARBAGE CANYON", "stage 3 name");

        check(result.stages[1].rules.mode_id == config::MODE_BLITZ_120,
              "blitz: mode_id override applied");
        check(result.stages[1].rules.time_limit == 120.0f,
              "blitz: time_limit override applied");
        check(result.stages[1].rules.target_score == 20000,
              "blitz: target_score override applied");
        check(result.stages[1].rules.camera.fov_deg == 55.0f,
              "blitz: camera fov override applied");
        check(result.stages[2].rules.target_lines == 20,
              "canyon: target_lines override applied");
        std::fprintf(stderr, "[stage-debug] canyon eye=(%g, %g, %g)\n",
            result.stages[2].rules.camera.eye.x,
            result.stages[2].rules.camera.eye.y,
            result.stages[2].rules.camera.eye.z);
        check(result.stages[2].rules.camera.eye.y == 13.5f,
              "canyon: camera eye override applied");

        check(result.stages[4].unlock_after == 4,
              "encore: unlock_after carried through");

        check(result.stages[1].script_path.find("blitz_mode.lua")
                  != std::string::npos,
              "blitz script path carried through");
        check(result.stages[0].script_path.empty(),
              "marathon is pure C++ tier (no script)");
    }

    check(!result.used_fallback, "real campaign.lua loads without fallback");

    // Fallback law: bad root => marathon default, no crash.
    auto fb = game::load_campaign("/nonexistent/xyz");
    check(fb.used_fallback && fb.stages.size() == 1
              && fb.stages[0].id == "marathon_01",
          "fallback: missing campaign -> marathon defaults, no crash");
#endif

    std::printf("[stage-tests] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
