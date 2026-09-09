#pragma once
// game/stage.hpp - P3 DATA-DRIVEN STAGE LOADER
//
// Loads assets/campaign/campaign.lua (pure data) through the sandboxed Lua
// edge and produces StageDef records. Replaces config/levels/*.hpp +
// config/campaign/main_campaign.hpp as the source of truth for campaign
// CONTENT.
//
// Fallback law: missing/corrupt file or any load error => marathon defaults
// for the whole campaign + error flag. The game never crashes on bad content.
//
// Purity: parsing lives here (game layer may use edges/lua); the returned
// StageDefs are plain structs the rest of the engine consumes without
// knowing Lua exists.
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <config/rules.hpp>

#ifdef TETRIS_LUA_ENABLED
#include <edges/lua/lua.edge.hpp>
#include <lua.hpp>
#endif

namespace tetris::game {

    struct StageDef {
        std::string id;
        std::string name;
        int         unlock_after = 0;
        config::Rules rules{};          // defaults merged with overrides
        std::string script_path;         // "" = pure C++ tier
    };

    struct CampaignLoadResult {
        std::vector<StageDef> stages;
        bool used_fallback = false;       // true => campaign.lua failed to load
    };

    // Apply one rules-override table onto a Rules struct (shared by campaign
    // and per-stage tables). Unknown keys are IGNORED (forward compat).
#ifdef TETRIS_LUA_ENABLED
    inline void apply_overrides(lua_State* L, config::Rules& r) {
        // expects override table at top of stack
        auto set_f = [&](const char* k, float& dst) {
            lua_getfield(L, -1, k);
            if (lua_isnumber(L, -1)) dst = (float)lua_tonumber(L, -1);
            lua_pop(L, 1);
        };
        auto set_i = [&](const char* k, int& dst) {
            lua_getfield(L, -1, k);
            if (lua_isnumber(L, -1)) dst = (int)lua_tointeger(L, -1);
            lua_pop(L, 1);
        };
        set_i("mode_id",         r.mode_id);
        set_f("time_limit",      r.time_limit);
        set_i("target_score",    r.target_score);
        set_i("target_lines",    r.target_lines);
        set_i("special_every_n", r.special_every_n);
        set_f("freeze_seconds",  r.freeze_seconds);
        set_f("initial_drop_interval", r.initial_drop_interval);
        set_f("gravity_decay",   r.gravity_decay);

        // camera triple: nested tables camera_eye = {x, y, z}
        auto set_vec3 = [&](const char* k, auto& dst) {
            lua_getfield(L, -1, k);
            if (lua_istable(L, -1)) {
                // positional form { x, y, z }
                auto idx = [&](int i) -> float {
                    lua_rawgeti(L, -1, i);
                    float v = lua_isnumber(L, -1) ? (float)lua_tonumber(L, -1)
                                                  : dst.x; // harmless fallback
                    lua_pop(L, 1);
                    return v;
                };
                dst.x = idx(1);
                dst.y = idx(2);
                dst.z = idx(3);
            }
            lua_pop(L, 1);
        };
        float fov = r.camera.fov_deg;
        set_vec3("camera_eye",    r.camera.eye);
        set_vec3("camera_target", r.camera.target);
        set_f("camera_fov", fov);
        r.camera.fov_deg = fov;
    }
#endif

    // Load campaign. Without TETRIS_LUA_ENABLED, or on any error, returns the
    // hardcoded fallback (marathon-first linear unlock).
    inline CampaignLoadResult load_campaign(const std::string& source_root) {
        CampaignLoadResult out;

#ifndef TETRIS_LUA_ENABLED
        out.used_fallback = true;
        StageDef m; m.id = "marathon_01"; m.name = "MARATHON";
        out.stages.push_back(m);
        return out;
#else
        lua_edge::StatelessLuaEvaluator eval;
        const std::string path = source_root + "/assets/campaign/campaign.lua";
        if (!eval.valid() || !eval.load_script_file(path.c_str())) {
            out.used_fallback = true;
            StageDef m; m.id = "marathon_01"; m.name = "MARATHON";
            out.stages.push_back(m);
            return out;
        }

        lua_State* L = eval.raw();
        lua_getglobal(L, "campaign");
        if (!lua_istable(L, -1)) { lua_pop(L, 1); out.used_fallback = true;
            StageDef m; m.id = "marathon_01"; m.name = "MARATHON";
            out.stages.push_back(m); return out; }
        lua_getfield(L, -1, "stages");
        if (!lua_istable(L, -1)) { lua_pop(L, 2); out.used_fallback = true;
            StageDef m; m.id = "marathon_01"; m.name = "MARATHON";
            out.stages.push_back(m); return out; }

        const int n = (int)lua_rawlen(L, -1);
        for (int i = 1; i <= n; ++i) {
            lua_rawgeti(L, -1, i);
            if (!lua_istable(L, -1)) { lua_pop(L, 1); continue; }

            StageDef def;
            lua_getfield(L, -1, "id");
            if (lua_isstring(L, -1)) def.id = lua_tostring(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, -1, "name");
            if (lua_isstring(L, -1)) def.name = lua_tostring(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, -1, "unlock_after");
            if (lua_isnumber(L, -1)) def.unlock_after = (int)lua_tointeger(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, -1, "script");
            if (lua_isstring(L, -1)) def.script_path = lua_tostring(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, -1, "rules");
            if (lua_istable(L, -1)) apply_overrides(L, def.rules);
            lua_pop(L, 1);

            if (!def.id.empty()) out.stages.push_back(def);
            lua_pop(L, 1);  // stage table
        }
        lua_pop(L, 2);  // stages, campaign

        if (out.stages.empty()) {
            out.used_fallback = true;
            StageDef m; m.id = "marathon_01"; m.name = "MARATHON";
            out.stages.push_back(m);
        }
        return out;
#endif
    }


    // ---- P3.5: per-level level.lua -----------------------------------------
    // Loads assets/levels/<level_id>/level.lua (Level table with optional
    // name + rules_overrides) and merges onto `rules`. Missing file or any
    // error is NOT fatal - the campaign-provided rules stand.
#ifdef TETRIS_LUA_ENABLED
    inline bool load_level(const std::string& source_root,
                           const std::string& level_id,
                           std::string& name_out,
                           config::Rules& rules) {
        lua_edge::StatelessLuaEvaluator eval;
        const std::string path =
            source_root + "/assets/levels/" + level_id + "/level.lua";
        if (!eval.valid() || !eval.load_script_file(path.c_str()))
            return false;
        lua_State* L = eval.raw();
        lua_getglobal(L, "Level");
        if (!lua_istable(L, -1)) { lua_pop(L, 1); return false; }

        lua_getfield(L, -1, "name");
        if (lua_isstring(L, -1)) name_out = lua_tostring(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, -1, "rules_overrides");
        if (lua_istable(L, -1)) apply_overrides(L, rules);
        lua_pop(L, 1);

        lua_pop(L, 1);  // Level
        return true;
    }
#endif

} // namespace tetris::game
