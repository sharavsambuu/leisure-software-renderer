#pragma once

/*
    SHS RENDERER SAN

    FILE: input.command.hpp
    MODULE: domains/input
    PURPOSE: CORE 2. COMMAND — the closed input command vocabulary.
             Home of the RuntimeCommand variant (moved from value_commands.hpp
             in R3 so the vocabulary lives in its Core 4 file; value_commands
             re-exports these names, zero breakage for existing consumers).
*/

#include <cstdint>
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

    struct ToggleFlagIntent
    {
        bool value = false;

        bool operator==(const ToggleFlagIntent&) const = default;
    };

    enum class RuntimeCommandKind : uint8_t
    {
        MoveLocal          = 0,
        Look               = 1,
        ToggleLightShafts  = 2,
        ToggleBot          = 3,
        Quit               = 4
    };

    using RuntimeCommandPayload = std::variant<std::monostate, MoveLocalIntent, LookIntent, ToggleFlagIntent>;

    struct RuntimeCommand
    {
        RuntimeCommandKind    type    = RuntimeCommandKind::MoveLocal;
        RuntimeCommandPayload payload{};

        bool operator==(const RuntimeCommand&) const = default;
    };

    inline RuntimeCommand make_move_local_intent(glm::vec3 local_dir, float meters_per_sec)
    {
        RuntimeCommand out{};
        out.type    = RuntimeCommandKind::MoveLocal;
        out.payload = MoveLocalIntent{local_dir, meters_per_sec};
        return out;
    }

    inline RuntimeCommand make_look_intent(float dx, float dy, float sensitivity)
    {
        RuntimeCommand out{};
        out.type    = RuntimeCommandKind::Look;
        out.payload = LookIntent{dx, dy, sensitivity};
        return out;
    }

    inline RuntimeCommand make_toggle_light_shafts_intent()
    {
        RuntimeCommand out{};
        out.type    = RuntimeCommandKind::ToggleLightShafts;
        out.payload = ToggleFlagIntent{};
        return out;
    }

    inline RuntimeCommand make_toggle_bot_intent()
    {
        RuntimeCommand out{};
        out.type    = RuntimeCommandKind::ToggleBot;
        out.payload = ToggleFlagIntent{};
        return out;
    }

    inline RuntimeCommand make_quit_intent()
    {
        RuntimeCommand out{};
        out.type    = RuntimeCommandKind::Quit;
        out.payload = ToggleFlagIntent{};
        return out;
    }
} // namespace shs

namespace shs::input
{
    // The pod's closed command vocabulary, named explicitly (Constitution §6.1).
    using InputCommand = shs::RuntimeCommand;

    static_assert(std::variant_size_v<shs::RuntimeCommandPayload> == 4,
        "input command vocabulary changed: update gateway + event pins");
} // namespace shs::input
