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
#include <domains/powerups/powerups.contract.hpp>
#include <domains/environment/environment.contract.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <glm/glm.hpp>

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

    // L3 generator result (plain-value mirror of CanyonGen.generate's table).
    // Rows are BOTTOM-UP strings ('X' garbage block, '.' hole); main maps them
    // into the matrix stamp payload. Fixed caps keep this domain-agnostic.
    static constexpr int GEN_MAX_ROWS = 24;
    static constexpr int GEN_MAX_COLS = 16;

    struct GenerationResult {
        bool valid     = false;
        int  row_count = 0;
        char rows[GEN_MAX_ROWS][GEN_MAX_COLS + 1]{};
        int  target_lines = 0;
        float time_limit  = 0.0f;
        int  mode_id      = 0;
        int  target_score = 0;
        int  seed_tag     = 0;
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

        bool has_table(const char* table) const {
            if (!L_) return false;
            lua_getglobal(L_, table);                       // [table]
            const bool ok = lua_istable(L_, -1) != 0;
            lua_pop(L_, 1);
            return ok;
        }

        // ---- P3 generic data access (stage/level loaders) ------------------
        // campaign[i] / Level tables: plain data reads, no behavior invoked.

        int table_length(const char* table) {
            if (!L_) return 0;
            lua_getglobal(L_, table);
            int n = 0;
            if (lua_istable(L_, -1)) n = static_cast<int>(lua_rawlen(L_, -1));
            lua_pop(L_, 1);
            return n;
        }

        // table[i][key] as string ("" when absent)
        std::string table_string(const char* table, int i, const char* key) {
            std::string out;
            if (!L_) return out;
            lua_getglobal(L_, table);                       // [table]
            if (lua_istable(L_, -1)) {
                lua_rawgeti(L_, -1, i);                     // [table, entry]
                if (lua_istable(L_, -1)) {
                    lua_getfield(L_, -1, key);              // [.., value]
                    if (lua_isstring(L_, -1)) out = lua_tostring(L_, -1);
                    lua_pop(L_, 1);
                }
                lua_pop(L_, 1);
            }
            lua_pop(L_, 1);
            return out;
        }

        // table[i][key] as integer (0 when absent)
        int table_int(const char* table, int i, const char* key) {
            if (!L_) return 0;
            int out = 0;
            lua_getglobal(L_, table);
            if (lua_istable(L_, -1)) {
                lua_rawgeti(L_, -1, i);
                if (lua_istable(L_, -1)) {
                    lua_getfield(L_, -1, key);
                    if (lua_isnumber(L_, -1)) out = static_cast<int>(lua_tointeger(L_, -1));
                    lua_pop(L_, 1);
                }
                lua_pop(L_, 1);
            }
            lua_pop(L_, 1);
            return out;
        }

        // table[key] as integer at depth 1 (Level.mode_id etc.)
        int global_table_int(const char* table, const char* key) {
            if (!L_) return 0;
            int out = 0;
            lua_getglobal(L_, table);
            if (lua_istable(L_, -1)) {
                lua_getfield(L_, -1, key);
                if (lua_isnumber(L_, -1)) out = static_cast<int>(lua_tointeger(L_, -1));
                lua_pop(L_, 1);
            }
            lua_pop(L_, 1);
            return out;
        }

        float global_table_float(const char* table, const char* key) {
            if (!L_) return 0.0f;
            float out = 0.0f;
            lua_getglobal(L_, table);
            if (lua_istable(L_, -1)) {
                lua_getfield(L_, -1, key);
                if (lua_isnumber(L_, -1)) out = static_cast<float>(lua_tonumber(L_, -1));
                lua_pop(L_, 1);
            }
            lua_pop(L_, 1);
            return out;
        }

        // Does table[key] exist at all (any type)?
        bool has_field(const char* table, int i, const char* key) {
            if (!L_) return false;
            bool ok = false;
            lua_getglobal(L_, table);
            if (lua_istable(L_, -1)) {
                if (i > 0) lua_rawgeti(L_, -1, i);          // indexed entry
                else       lua_getfield(L_, -1, key);       // named field path
                if (lua_istable(L_, -1) || i == 0) {
                    if (i > 0) { lua_getfield(L_, -1, key); }
                    ok = !lua_isnil(L_, -1);
                    lua_pop(L_, 1);
                }
                lua_pop(L_, 1);
            }
            lua_pop(L_, 1);
            return ok;
        }

        float table_float_field(const char* table, const char* sub,
                                const char* key) {
            if (!L_) return 0.0f;
            float out = 0.0f;
            lua_getglobal(L_, table);                       // [table]
            if (lua_istable(L_, -1)) {
                lua_getfield(L_, -1, sub);                  // [table, sub]
                if (lua_istable(L_, -1)) {
                    lua_getfield(L_, -1, key);              // [sub, value]
                    if (lua_isnumber(L_, -1)) out = static_cast<float>(lua_tonumber(L_, -1));
                    lua_pop(L_, 1);
                }
                lua_pop(L_, 1);
            }
            lua_pop(L_, 1);
            return out;
        }

        float table_int_field(const char* table, int i, const char* key) {
            return static_cast<float>(table_int(table, i, key));
        }

        bool has_number(const char* table, const char* sub, const char* key) {
            if (!L_) return false;
            bool ok = false;
            lua_getglobal(L_, table);
            if (lua_istable(L_, -1)) {
                lua_getfield(L_, -1, sub);
                if (lua_istable(L_, -1)) {
                    lua_getfield(L_, -1, key);
                    ok = lua_isnumber(L_, -1) != 0;
                    lua_pop(L_, 1);
                }
                lua_pop(L_, 1);
            }
            lua_pop(L_, 1);
            return ok;
        }

        glm::vec3 table_vec3_field(const char* table, const char* sub,
                                   const char* key) {
            if (!L_) return glm::vec3{};
            glm::vec3 out{};
            lua_getglobal(L_, table);
            if (lua_istable(L_, -1)) {
                lua_getfield(L_, -1, sub);
                if (lua_istable(L_, -1)) {
                    lua_getfield(L_, -1, key);              // [sub, vec]
                    if (lua_istable(L_, -1)) {
                        lua_rawgeti(L_, -1, 1);
                        out.x = lua_isnumber(L_, -1) ? (float)lua_tonumber(L_, -1) : 0.f;
                        lua_pop(L_, 1);
                        lua_rawgeti(L_, -1, 2);
                        out.y = lua_isnumber(L_, -1) ? (float)lua_tonumber(L_, -1) : 0.f;
                        lua_pop(L_, 1);
                        lua_rawgeti(L_, -1, 3);
                        out.z = lua_isnumber(L_, -1) ? (float)lua_tonumber(L_, -1) : 0.f;
                        lua_pop(L_, 1);
                    }
                    lua_pop(L_, 1);
                }
                lua_pop(L_, 1);
            }
            lua_pop(L_, 1);
            return out;
        }

        // <table>.generate(difficulty, seed) -> plain-value generation result.
        GenerationResult call_generate(const char* table, int difficulty, long long seed) {
            GenerationResult out;
            if (!begin_call(table, "generate", 2)) return out;
            lua_pushinteger(L_, static_cast<lua_Integer>(difficulty));
            lua_pushinteger(L_, static_cast<lua_Integer>(seed));
            if (!finish_call(2)) return out;

            lua_getfield(L_, -1, "rows");                   // [result, rows]
            if (lua_istable(L_, -1)) {
                int n = static_cast<int>(lua_rawlen(L_, -1));
                if (n > GEN_MAX_ROWS) n = GEN_MAX_ROWS;
                out.row_count = n;
                for (int i = 1; i <= n; ++i) {
                    lua_rawgeti(L_, -1, i);                 // [result, rows, row_i]
                    if (lua_isstring(L_, -1)) {
                        const char* srow = lua_tostring(L_, -1);
                        std::snprintf(out.rows[i - 1], GEN_MAX_COLS + 1, "%s", srow);
                    }
                    lua_pop(L_, 1);
                }
            }
            lua_pop(L_, 1);                                 // pop rows

            out.valid        = out.row_count > 0;
            out.target_lines = field_int("target_lines");
            out.time_limit   = field_float("time_limit");
            out.mode_id      = field_int("mode_id");
            out.target_score = field_int("target_score");
            out.seed_tag     = field_int("seed_tag");
            lua_pop(L_, 1);                                 // pop result table
            return out;
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

        // Merge <table>.get_config() known keys into a Rules instance
        // (Lua as an authoring format for plain config values — §4.2).
        // Table name is caller-chosen: "BlitzRules" (L2 economy),
        // "CanyonGen" (L3 board generator), or "CyberRules" (L4 mechanics),
        // per what the script defines.
        void apply_config_overrides(config::Rules& rules, const char* table = "BlitzRules") {
            if (!L_) return;
            if (!begin_call(table, "get_config", 0)) return;
            if (!finish_call(0)) return;
            rules.mode_id      = field_int("mode_id",      rules.mode_id);
            rules.target_score = field_int("target_score", rules.target_score);
            rules.target_lines = field_int("target_lines", rules.target_lines);
            rules.time_limit   = field_float("time_limit", rules.time_limit);
            rules.special_every_n = field_int("special_every_n", rules.special_every_n);
            rules.freeze_seconds  = field_float("freeze_seconds", rules.freeze_seconds);
            lua_pop(L_, 1);
        }

        // L4: <table>.decide_spawn(pieces_since_special, armed_index) →
        // { type = N } where N is a matrix PieceType value (9..11), or
        // { type = 0 } to skip. Pure function of the two counters.
        powerups::SpawnDecision call_decide_spawn(const char* table,
                                                  int pieces_since, int armed_index) {
            powerups::SpawnDecision out;
            if (!L_) return out;
            if (!begin_call(table, "decide_spawn", 2)) return out;
            lua_pushinteger(L_, pieces_since);              // [func, n1]
            lua_pushinteger(L_, armed_index);               // [func, n1, n2]
            if (!finish_call(2)) return out;
            const int t = field_int("type", 0);
            lua_pop(L_, 1);
            out.valid        = (t >= 9 && t <= 11);
            out.special_type = static_cast<uint8_t>(out.valid ? t : 0);
            return out;
        }

        // L4: <table>.on_special_lock(type, gx, gy, grid_flat) → ruling.
        // grid_flat is the row-major CellGrid flattened bottom-up as ints
        // (220 entries; 0 empty / piece id). The script returns
        // { clear_count, cx[], cy[], freeze_seconds, fx_id } — plain values.
        powerups::SpecialRuling call_on_special_lock(const char* table,
                                                     int special_type,
                                                     int gx, int gy,
                                                     const matrix::CellGrid& grid) {
            powerups::SpecialRuling out;
            if (!L_) return out;
            if (!begin_call(table, "on_special_lock", 4)) return out;
            lua_pushinteger(L_, special_type);              // [func, n1..n4, tbl]
            lua_pushinteger(L_, gx);
            lua_pushinteger(L_, gy);
            lua_createtable(L_, matrix::GRID_W * matrix::GRID_H, 0);
            for (int y = 0; y < matrix::GRID_H; ++y) {
                for (int x = 0; x < matrix::GRID_W; ++x) {
                    lua_pushinteger(L_, grid[y][x]);
                    lua_rawseti(L_, -2, y * matrix::GRID_W + x + 1);
                }
            }
            if (!finish_call(4)) return out;
            out.valid          = true;
            out.clear_count    = static_cast<uint8_t>(field_int("clear_count", 0));
            out.freeze_seconds = field_float("freeze_seconds", 0.0f);
            out.fx_id          = static_cast<uint8_t>(field_int("fx_id", 0));
            lua_getfield(L_, -1, "cx");                     // [result, cx]
            if (lua_istable(L_, -1)) {
                for (int i = 0; i < powerups::RULING_MAX_CELLS; ++i) {
                    lua_rawgeti(L_, -1, i + 1);
                    if (lua_isnumber(L_, -1)) out.clear_x[i] = static_cast<int16_t>(lua_tointeger(L_, -1));
                    lua_pop(L_, 1);
                }
            }
            lua_pop(L_, 1);
            lua_getfield(L_, -1, "cy");
            if (lua_istable(L_, -1)) {
                for (int i = 0; i < powerups::RULING_MAX_CELLS; ++i) {
                    lua_rawgeti(L_, -1, i + 1);
                    if (lua_isnumber(L_, -1)) out.clear_y[i] = static_cast<int16_t>(lua_tointeger(L_, -1));
                    lua_pop(L_, 1);
                }
            }
            lua_pop(L_, 1);
            lua_pop(L_, 1);
            return out;
        }

        // L5: <table>.get_config() encounter numbers (phase_count / rain_every
        // / rain_rows). Missing keys keep the struct defaults.
        environment::EncounterConfig call_encounter_config(const char* table) {
            environment::EncounterConfig out;
            if (!L_) return out;
            if (!begin_call(table, "get_config", 0)) return out;
            if (!finish_call(0)) return out;
            out.phase_count = field_int("phase_count", out.phase_count);
            out.rain_every  = field_float("rain_every", out.rain_every);
            out.rain_rows   = field_int("rain_rows", out.rain_rows);
            lua_pop(L_, 1);
            out.valid = true;
            return out;
        }

        // L5: <table>.decide_phase(phase, phase_time, lines, danger) →
        // { new_phase, mood_target }. Pure function of plain values; called
        // every frame by main (cheap, stateless).
        environment::OverseerRuling call_decide_phase(const char* table,
                                                      int phase, float phase_time,
                                                      int lines, bool danger) {
            environment::OverseerRuling out;
            if (!L_) return out;
            if (!begin_call(table, "decide_phase", 4)) return out;
            lua_pushinteger(L_, phase);                     // [func, n1..n4]
            lua_pushnumber(L_, phase_time);
            lua_pushinteger(L_, lines);
            lua_pushboolean(L_, danger ? 1 : 0);
            if (!finish_call(4)) return out;
            out.valid       = true;
            out.new_phase   = field_int("new_phase", 0);
            out.mood_target = field_float("mood_target", -1.0f);
            lua_pop(L_, 1);
            return out;
        }

        // L5: <table>.on_event(type, value) → { crowd_pulse }.
        // type 1 = LINES_CLEARED (value = line count), type 2 = VICTORY.
        environment::CrowdPulse call_on_event(const char* table,
                                              int event_type, int value) {
            environment::CrowdPulse out;
            if (!L_) return out;
            if (!begin_call(table, "on_event", 2)) return out;
            lua_pushinteger(L_, event_type);                // [func, n1, n2]
            lua_pushinteger(L_, value);
            if (!finish_call(2)) return out;
            out.valid = true;
            out.kick  = field_float("crowd_pulse", 0.0f);
            lua_pop(L_, 1);
            return out;
        }

    private:
        // Resolve table.func and leave JUST the function on the stack:
        // [] → [func]. Callers push their args afterwards, then finish_call
        // pcalls with the same nargs (keeps every index in-range — the old
        // rotate-based version indexed below the stack bottom whenever
        // nargs > 0, corrupting the Lua/C stack).
        bool begin_call(const char* table, const char* func, int nargs) {
            (void)nargs;   // kept in the signature for call-site readability
            lua_getglobal(L_, table);                       // [table]
            if (!lua_istable(L_, -1)) { lua_pop(L_, 1); return fail(); }
            lua_getfield(L_, -1, func);                     // [table, func]
            if (!lua_isfunction(L_, -1)) { lua_pop(L_, 2); return fail(); }
            lua_remove(L_, -2);                             // [func]
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

    public:
        // P3: raw state access for the data-driven campaign loader
        // (game/stage.hpp). Read-only walks only; callers must balance stack.
        lua_State* raw() { return L_; }
        lua_State* L_ = nullptr;
        bool       error_ = false;
    };

} // namespace tetris::lua_edge

#endif // TETRIS_LUA_ENABLED