#pragma once
// tetris/edges/lua/lua.edge.hpp — STATELESS LUA EVALUATOR EDGE (tetris::lua_edge)
// The ONLY file in this demo that includes Lua headers (ARCHITECTURE.md §4.1).
//
// Constitution II Rule 8.2: scripts are PURE STATELESS REDUCERS. Plain-value
// snapshot in → explicit value patch out. No pointers cross the boundary in
// either direction; scripts never hold references to C++ objects.
//
// Sandboxing (determinism): the evaluator opens ONLY base/table/math and then
// strips math.random/randomseed (non-deterministic) and print. os/io/package/
// debug are never opened. Same script + same inputs ⇒ identical outputs.
//
// Build wiring: compiled only when CMake finds Lua (TETRIS_LUA_ENABLED);
// otherwise this header is empty and pods run their native C++ rules.

#if defined(TETRIS_LUA_ENABLED)

#include <lua.hpp>   // C API wrapped in extern "C" (plain <lua.h> would
                     // C++-mangle every reference and fail to link)

#include <config/rules.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace tetris::lua_edge {

    // Plain-value results (mirrors of the script's return tables).
    struct ScoreRuleResult {
        int   score_added  = 0;
        bool  level_up     = false;
        bool  danger_alert = false;
        float time_bonus   = 0.0f;   // seconds granted back (blitz economy)
    };

    struct ClockRuleResult {
        bool danger_alert = false;
        bool hurry        = false;
    };

    // Owns one sandboxed lua_State; scripts load once at boot, then every call
    // is a pure value-in/value-out evaluation ("stateless" = no C++ pointers
    // ever enter the state; results depend only on the inputs).
    class StatelessLuaEvaluator {
    public:
        StatelessLuaEvaluator() {
            L_ = luaL_newstate();
            if (!L_) return;
            luaL_requiref(L_, "_G",            luaopen_base,  1);
            luaL_requiref(L_, LUA_TABLIBNAME,  luaopen_table, 1);
            luaL_requiref(L_, LUA_MATHLIBNAME, luaopen_math,  1);
            lua_pop(L_, 3);
            sandbox_strip();
        }

        ~StatelessLuaEvaluator() {
            if (L_) lua_close(L_);
        }

        StatelessLuaEvaluator(const StatelessLuaEvaluator&)            = delete;
        StatelessLuaEvaluator& operator=(const StatelessLuaEvaluator&) = delete;

        bool valid()     const noexcept { return L_ != nullptr; }
        bool has_error() const noexcept { return error_; }

        // Load (run) a script chunk; the chunk must define its rule table
        // (e.g. BlitzRules). Returns false on any syntax/runtime error.
        bool load_script_text(const char* chunk_name, const char* text) {
            if (!L_ || !text) return false;
            if (luaL_loadbuffer(L_, text, std::strlen(text), chunk_name) != LUA_OK) {
                report_error();
                return false;
            }
            if (lua_pcall(L_, 0, 0, 0) != LUA_OK) {
                report_error();
                return false;
            }
            return true;
        }

        bool load_script_file(const char* path) {
            if (!L_ || !path) return false;
            std::FILE* f = std::fopen(path, "rb");
            if (!f) return false;
            std::string contents;
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) contents.append(buf, n);
            const bool ok = std::ferror(f) == 0;
            std::fclose(f);
            if (!ok) return false;
            return load_script_text(path, contents.c_str());
        }

        bool has_function(const char* table, const char* func) const {
            if (!L_) return false;
            lua_getglobal(L_, table);                       // [table]
            bool ok = lua_istable(L_, -1) != 0;
            if (ok) {
                lua_getfield(L_, -1, func);                 // [table, func]
                ok = lua_isfunction(L_, -1) != 0;
                lua_pop(L_, 1);
            }
            lua_pop(L_, 1);
            return ok;
        }

        // BlitzRules.calculate_score(level, lines, combo, is_tspin) -> ruling
        ScoreRuleResult call_calculate_score(int level, int lines, int combo, bool is_tspin) {
            ScoreRuleResult out;
            if (!begin_call("BlitzRules", "calculate_score", 4)) return out;
            lua_pushinteger(L_, static_cast<lua_Integer>(level));
            lua_pushinteger(L_, static_cast<lua_Integer>(lines));
            lua_pushinteger(L_, static_cast<lua_Integer>(combo));
            lua_pushboolean(L_, is_tspin ? 1 : 0);
            if (!finish_call(4)) return out;
            out.score_added  = field_int("score_added");
            out.level_up     = field_bool("level_up");
            out.danger_alert = field_bool("danger_alert");
            out.time_bonus   = field_float("time_bonus");
            lua_pop(L_, 1);                                 // pop result table
            return out;
        }

        // BlitzRules.evaluate_clock(time_left, stack_height) -> urgency flags
        ClockRuleResult call_evaluate_clock(float time_left, int stack_height) {
            ClockRuleResult out;
            if (!begin_call("BlitzRules", "evaluate_clock", 2)) return out;
            lua_pushnumber(L_, static_cast<lua_Number>(time_left));
            lua_pushinteger(L_, static_cast<lua_Integer>(stack_height));
            if (!finish_call(2)) return out;
            out.danger_alert = field_bool("danger_alert");
            out.hurry        = field_bool("hurry");
            lua_pop(L_, 1);
            return out;
        }

        // Merge BlitzRules.get_config() known keys into a Rules instance
        // (Lua as an authoring format for plain config values — §4.2).
        void apply_config_overrides(config::Rules& rules) {
            if (!L_) return;
            if (!begin_call("BlitzRules", "get_config", 0)) return;
            if (!finish_call(0)) return;
            rules.mode_id      = field_int("mode_id",      rules.mode_id);
            rules.target_score = field_int("target_score", rules.target_score);
            rules.time_limit   = field_float("time_limit", rules.time_limit);
            lua_pop(L_, 1);
        }

    private:
        // Push table.func above any already-pushed args: [args...] → [func, args...]
        bool begin_call(const char* table, const char* func, int nargs) {
            lua_getglobal(L_, table);                       // [args..., table]
            if (!lua_istable(L_, -1)) { lua_pop(L_, nargs + 1); return fail(); }
            lua_getfield(L_, -1, func);                     // [args..., table, func]
            if (!lua_isfunction(L_, -1)) { lua_pop(L_, nargs + 2); return fail(); }
            lua_rotate(L_, -(nargs + 2), 1);                // [func, args..., table]
            lua_pop(L_, 1);                                 // [func, args...]
            return true;
        }

        // pcall the prepared call; leaves the result table on top on success.
        bool finish_call(int nargs) {
            if (lua_pcall(L_, nargs, 1, 0) != LUA_OK) {     // [result] or [errmsg]
                report_error();
                return false;
            }
            if (!lua_istable(L_, -1)) { lua_pop(L_, 1); return fail(); }
            return true;
        }

        // Field readers — operate on the table at the top of the stack.
        int field_int(const char* key, int def = 0) {
            lua_getfield(L_, -1, key);
            int v = lua_isnumber(L_, -1) ? static_cast<int>(lua_tointeger(L_, -1)) : def;
            lua_pop(L_, 1);
            return v;
        }
        float field_float(const char* key, float def = 0.0f) {
            lua_getfield(L_, -1, key);
            float v = lua_isnumber(L_, -1) ? static_cast<float>(lua_tonumber(L_, -1)) : def;
            lua_pop(L_, 1);
            return v;
        }
        bool field_bool(const char* key, bool def = false) {
            lua_getfield(L_, -1, key);
            bool v = lua_isboolean(L_, -1) ? (lua_toboolean(L_, -1) != 0) : def;
            lua_pop(L_, 1);
            return v;
        }

        // Determinism sandbox: strip non-deterministic / side-effecting globals.
        void sandbox_strip() {
            lua_getglobal(L_, "math");                      // math.random is C-seeded
            if (lua_istable(L_, -1)) {
                lua_pushnil(L_); lua_setfield(L_, -2, "random");
                lua_pushnil(L_); lua_setfield(L_, -2, "randomseed");
            }
            lua_pop(L_, 1);
            lua_pushnil(L_); lua_setglobal(L_, "print");    // keep stdout clean
        }

        void report_error() {
            if (lua_isstring(L_, -1)) {
                std::fprintf(stderr, "[lua.edge] %s\n", lua_tostring(L_, -1));
            }
            lua_pop(L_, 1);
            error_ = true;
        }

        bool fail() { error_ = true; return false; }

        lua_State* L_ = nullptr;
        bool       error_ = false;
    };

} // namespace tetris::lua_edge

#endif // TETRIS_LUA_ENABLED