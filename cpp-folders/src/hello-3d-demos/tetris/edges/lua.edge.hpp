#pragma once

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
#include <string_view>
#include <span>
#include "shs_renderer.hpp"

namespace tetris {
    namespace lua_edge {

        // ============================================================================
        // EDGE: Stateless Lua Scripting Engine (Constitution II Rule 8.2)
        // ───────────────────────────────────────────────────────────────────────────
        //
        // Rules for all Lua scripts inside domain pods:
        //   • No mutable references to C++ objects — only read values from stack.
        //   → Return explicit value structs via the table on top of the stack.
        //
        // Architecture:
        //   ┌─────────────────────────────┐          ┌──────────────────────────┐          ┌─────────────────────────┐
        //   │ Immutable C++ Snapshot       ├─────────►│ Stateless Lua Script     │          │ C++ Value Patch Result  │
        //   │ (state, dt, commands, etc.)  │          │ (pure function)          │          │ (score_delta, events)    │
        //   └─────────────────────────────┘          └──────────────────────────┘          └─────────────────────────┘
        // ============================================================================

        /// A single Lua state instance that stays alive across frames.
        /// Each thread gets its own state (no global registry).
        struct LuaStateWrapper {
            lua_State* L = nullptr;

            explicit LuaStateWrapper() {
                L = luaL_newstate();
                if (!L) throw std::runtime_error("luaL_newstate failed");
                luaL_openlibs(L);
            }

            ~LuaStateWrapper() {
                if (L) lua_close(L);
            }

            /// Pushes a C++ struct onto the Lua stack and returns its field accessors.
            static inline void push_struct(
                lua_State* L, const std::string& name, const shs::Color& col,
                float score_delta = 0.0f, int combo = 0, bool danger_alert = false
            ) {
                lua_createtable(L, 0, 4);
                lua_pushfloat(L, static_cast<float>(col.r)); lua_setfield(L, -2, "r");
                lua_pushfloat(L, static_cast<float>(col.g)); lua_setfield(L, -2, "g");
                lua_pushfloat(L, static_cast<float>(col.b)); lua_setfield(L, -2, "b");
                lua_pushfloat(L, score_delta);     lua_setfield(L, -2, "score_delta");
                lua_pushinteger(L, combo);         lua_setfield(L, -2, "combo");
                lua_pushboolean(L, danger_alert);  lua_setfield(L, -2, "danger_alert");
                lua_setglobal(L, name.c_str());
            }

            static inline int push_nil(lua_State* L) { return 0; } // No args.
        };

        /// Evaluates a pure Lua scoring function that receives immutable C++ inputs.
        /// Returns an explicit value struct packed into the result table.
        template <typename ScriptResult>
        static inline void eval_script(
            lua_State* L, const char* script_name,
            float score_delta = 0.0f, int combo = 0, bool danger_alert = false
        ) {
            // Push input arguments onto the stack: (score_delta, combo, danger_alert)
            lua_pushfloat(L, score_delta);
            lua_pushinteger(L, combo);
            lua_pushboolean(L, danger_alert);

            // Call the user-defined pure function and pop result table
            if (lua_getglobal(L, script_name.c_str()) != LUA_TFUNCTION) {
                // Fallback default values if script missing or error
                LuaStateWrapper::push_struct(L, "BlitzRules", shs::Color{255,100,80}, 1.0f, 0, false);
            } else {
                int err = lua_pcall(L, 3, 1, 0); // 3 inputs, 1 output table
                if (err != LUA_OK) {
                    const char* err_str = lua_tostring(L, -1);
                    LuaStateWrapper::push_struct(L, "BlitzRules", shs::Color{255,100,80}, 1.0f, 0, false);
                } else if (lua_isnil(L, -1)) {
                    // Fallback on nil return
                    LuaStateWrapper::push_struct(L, "BlitzRules", shs::Color{255,100,80}, 1.0f, 0, false);
                } else {
                    // Stack top is now the result table — extract fields into struct
                    ScriptResult out{};
                    lua_getfield(L, -1, "score_delta");
                    out.score_added = static_cast<int>(lua_tonumber(L, -1));
                    lua_pop(L, 1);

                    lua_getfield(L, -1, "level_up");
                    out.level_up = lua_toboolean(L, -1) != 0;
                    lua_pop(L, 1);

                    lua_getfield(L, -1, "danger_alert");
                    out.danger_alert = lua_toboolean(L, -1) != 0;
                    lua_pop(L, 1);

                    // Push back the populated struct table for caller consumption
                    lua_createtable(L, 0, 3);
                    lua_setfloat(L, static_cast<float>(out.score_added)); lua_setfield(L, -2, "score_delta");
                    lua_pushboolean(L, out.level_up);                   lua_setfield(L, -2, "level_up");
                    lua_pushboolean(L, out.danger_alert);              lua_setfield(L, -2, "danger_alert");

                    // Replace top of stack with the populated result table
                    lua_replace(L, -3); // replace function call result slot with structured output
                }
            }
        }

    } // namespace lua_edge
} // namespace tetris
