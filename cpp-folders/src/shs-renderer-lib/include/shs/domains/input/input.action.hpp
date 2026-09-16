#pragma once

/*
    SHS RENDERER SAN

    FILE: input.action.hpp
    MODULE: domains/input
    PURPOSE: CORE 2. COMMAND — the closed input command vocabulary.
             Home of the RuntimeAction variant (moved from value_actions.hpp
             in R3 so the vocabulary lives in its Core 4 file; value_actions
             re-exports these names, zero breakage for existing consumers).
*/

#include <cstdint>
#include <variant>

#include <glm/glm.hpp>

namespace shs
{
    struct MoveLocalAction
    {
        glm::vec3 local_dir{0.0f};
        float     meters_per_sec = 0.0f;

        bool operator==(const MoveLocalAction&) const = default;
    };

    struct LookAction
    {
        float dx           = 0.0f;
        float dy           = 0.0f;
        float sensitivity  = 0.0f;

        bool operator==(const LookAction&) const = default;
    };

    struct ToggleFlagAction
    {
        bool value = false;

        bool operator==(const ToggleFlagAction&) const = default;
    };

    enum class RuntimeActionType : uint8_t
    {
        MoveLocal          = 0,
        Look               = 1,
        ToggleLightShafts  = 2,
        ToggleBot          = 3,
        Quit               = 4
    };

    using RuntimeActionPayload = std::variant<std::monostate, MoveLocalAction, LookAction, ToggleFlagAction>;

    struct RuntimeAction
    {
        RuntimeActionType    type    = RuntimeActionType::MoveLocal;
        RuntimeActionPayload payload{};

        bool operator==(const RuntimeAction&) const = default;
    };

    inline RuntimeAction make_move_local_action(glm::vec3 local_dir, float meters_per_sec)
    {
        RuntimeAction out{};
        out.type    = RuntimeActionType::MoveLocal;
        out.payload = MoveLocalAction{local_dir, meters_per_sec};
        return out;
    }

    inline RuntimeAction make_look_action(float dx, float dy, float sensitivity)
    {
        RuntimeAction out{};
        out.type    = RuntimeActionType::Look;
        out.payload = LookAction{dx, dy, sensitivity};
        return out;
    }

    inline RuntimeAction make_toggle_light_shafts_action()
    {
        RuntimeAction out{};
        out.type    = RuntimeActionType::ToggleLightShafts;
        out.payload = ToggleFlagAction{};
        return out;
    }

    inline RuntimeAction make_toggle_bot_action()
    {
        RuntimeAction out{};
        out.type    = RuntimeActionType::ToggleBot;
        out.payload = ToggleFlagAction{};
        return out;
    }

    inline RuntimeAction make_quit_action()
    {
        RuntimeAction out{};
        out.type    = RuntimeActionType::Quit;
        out.payload = ToggleFlagAction{};
        return out;
    }
} // namespace shs

namespace shs::input
{
    // The pod's closed command vocabulary, named explicitly (Constitution §6.1).
    using InputAction = shs::RuntimeAction;

    static_assert(std::variant_size_v<shs::RuntimeActionPayload> == 4,
        "input command vocabulary changed: update reducer + event pins");
} // namespace shs::input
