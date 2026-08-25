// tetris/tests/goal_bridge_tests.cpp - G1 SCRIPTED GOAL PINS
//
// Proves the full chain: C++ events -> Lua tables -> Goals.test predicate ->
// boolean back. Uses the REAL StatelessLuaEvaluator + the same marshaling
// logic as main's LuaScriptHost.
#include <cstdio>
#include <string>
#include <vector>

#include <domains/mission/mission.contract.hpp>

#ifdef TETRIS_LUA_ENABLED
#include <domains/matrix/matrix.action.hpp>
#include <domains/powerups/powerups.contract.hpp>
#include <domains/environment/environment.contract.hpp>
#include <edges/lua/lua.edge.hpp>
#endif

using namespace tetris::mission;

namespace {
    int g_pass = 0, g_fail = 0;
    void check(bool ok, const char* name) {
        if (ok) { ++g_pass; std::printf("  PASS %s\n", name); }
        else    { ++g_fail; std::fprintf(stderr, "  FAIL %s\n", name); }
    }
}

#ifdef TETRIS_LUA_ENABLED
class GoalHost {
public:
    explicit GoalHost(tetris::lua_edge::StatelessLuaEvaluator* e) : ev_(e) {}
    bool load(const std::string& path) {
        return ev_->valid() && ev_->load_script_file(path.c_str());
    }
    bool test(const char* table, const char* goal_id,
              const std::vector<MissionEventView>& events,
              const MissionSnapshot& snap) const {
        lua_State* L = ev_->raw();
        lua_getglobal(L, table);
        if (!lua_istable(L, -1)) return false;
        lua_getfield(L, -1, "test");
        if (!lua_isfunction(L, -1)) { lua_pop(L, 2); return false; }
        lua_pushstring(L, goal_id);   // arg 1: goal id
        lua_createtable(L, (int)events.size(), 0);
        for (size_t i = 0; i < events.size(); ++i) {
            lua_createtable(L, 0, 3);
            lua_pushstring(L, events[i].type);  lua_setfield(L, -2, "type");
            lua_pushinteger(L, events[i].a);    lua_setfield(L, -2, "a");
            lua_pushinteger(L, events[i].b);    lua_setfield(L, -2, "b");
            lua_rawseti(L, -2, (int)(i + 1));
        }
        lua_createtable(L, 0, 5);
        lua_pushinteger(L, snap.score);      lua_setfield(L, -2, "score");
        lua_pushinteger(L, snap.lines);      lua_setfield(L, -2, "lines");
        lua_pushinteger(L, snap.level);      lua_setfield(L, -2, "level");
        lua_pushboolean(L, snap.overdrive ? 1 : 0);
        lua_setfield(L, -2, "overdrive");
        lua_pushnumber(L, snap.stack_ratio); lua_setfield(L, -2, "stack_ratio");
        if (lua_pcall(L, 3, 1, 0) != LUA_OK) { lua_pop(L, 1); return false; }
        bool ok = lua_toboolean(L, -1) != 0;
        lua_pop(L, 2);
        return ok;
    }
private:
    tetris::lua_edge::StatelessLuaEvaluator* ev_;
};
#endif

int main() {
#ifndef TETRIS_LUA_ENABLED
    std::printf("[goal-bridge] built without Lua - nothing to prove\n");
    return 0;
#else
    std::printf("[goal-bridge] G1 scripted goal pins\n");

    tetris::lua_edge::StatelessLuaEvaluator eval;
    GoalHost host(&eval);

    const std::string root = TETRIS_SOURCE_ROOT;
    // goals.lua dofile()s the DSL via this global; set it in the sandbox.
    lua_State* L0 = eval.raw();
    lua_pushstring(L0, root.c_str());
    lua_setglobal(L0, "TETRIS_SOURCE_ROOT");
    check(host.load(root + "/assets/levels/cyber_storm/goals.lua"),
          "bridge: goals.lua loads in sandbox");

    MissionSnapshot snap{};
    snap.score = 15000; snap.lines = 12; snap.level = 2;
    snap.overdrive = true;

    std::vector<MissionEventView> evs{
        { "LINES_CLEARED", 4, 0 },
        { "PIECE_MOVED", 0, 0 },
    };

    check(host.test("Goals", "storm_2", evs, snap),
          "storm_2: overdrive+clear+score completes");

    std::vector<MissionEventView> no_clear{ { "PIECE_MOVED", 0, 0 } };
    check(!host.test("Goals", "storm_2", no_clear, snap),
          "storm_2: needs an actual clear this tick");

    snap.overdrive = false;
    check(!host.test("Goals", "storm_2", evs, snap),
          "storm_2: requires overdrive window");
    snap.overdrive = true;

    check(!host.test("Goals", "nonexistent", evs, snap),
          "unknown goal id is safe-false");

    // storm_1 gate: lines >= 10 via snapshot
    snap.lines = 9;
    std::vector<MissionEventView> none{};
    check(!host.test("Goals", "storm_1", none, snap),
          "storm_1: below line threshold stays incomplete");
    snap.lines = 12;
    check(host.test("Goals", "storm_1", none, snap),
          "storm_1: snapshot-driven completion works");

    std::printf("[goal-bridge] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
#endif
}
