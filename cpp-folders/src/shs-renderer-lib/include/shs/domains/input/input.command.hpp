#pragma once

/*
    SHS RENDERER SAN

    FILE: input.command.hpp
    MODULE: domains/input
    PURPOSE: CORE 2. COMMAND — the closed input command vocabulary as a PURE
             closed variant (K2.1, Run B): the variant IS the tag — no
             parallel kind enum, no payload fishing (renderpath shape). The
             retired {RuntimeCommandKind, RuntimeCommandPayload} envelope was
             a second, desync-prone dispatch key.
*/

#include <variant>

#include <glm/glm.hpp>

namespace shs
{
    struct MoveLocalIntent
    {
        glm::vec3 local_dir{0.0f};
        float     meters_per_sec = 0.0f;

        bool operator==(const MoveLocalIntent&) const = default;
    };

    struct LookIntent
    {
        float dx           = 0.0f;
        float dy           = 0.0f;
        float sensitivity  = 0.0f;

        bool operator==(const LookIntent&) const = default;
    };

    struct ToggleLightShaftsIntent
    {
        bool operator==(const ToggleLightShaftsIntent&) const = default;
    };

    struct ToggleBotIntent
    {
        bool operator==(const ToggleBotIntent&) const = default;
    };

    struct QuitIntent
    {
        bool operator==(const QuitIntent&) const = default;
    };

    // Closed command vocabulary for the input pod (Constitution §6.1).
    using RuntimeCommand = std::variant<
        MoveLocalIntent,
        LookIntent,
        ToggleLightShaftsIntent,
        ToggleBotIntent,
        QuitIntent>;

    inline RuntimeCommand make_move_local_intent(glm::vec3 local_dir, float meters_per_sec)
    {
        return RuntimeCommand{MoveLocalIntent{local_dir, meters_per_sec}};
    }

    inline RuntimeCommand make_look_intent(float dx, float dy, float sensitivity)
    {
        return RuntimeCommand{LookIntent{dx, dy, sensitivity}};
    }

    inline RuntimeCommand make_toggle_light_shafts_intent()
    {
        return RuntimeCommand{ToggleLightShaftsIntent{}};
    }

    inline RuntimeCommand make_toggle_bot_intent()
    {
        return RuntimeCommand{ToggleBotIntent{}};
    }

    inline RuntimeCommand make_quit_intent()
    {
        return RuntimeCommand{QuitIntent{}};
    }
} // namespace shs

namespace shs::input
{
    // The pod's closed command vocabulary, named explicitly (Constitution §6.1).
    using InputCommand = shs::RuntimeCommand;

    static_assert(std::variant_size_v<shs::RuntimeCommand> == 5,
        "input command vocabulary changed: update gateway + event pins");
} // namespace shs::input
