#pragma once

/*
    SHS RENDERER SAN

    FILE: input.event.hpp
    MODULE: domains/input
    PURPOSE: CORE 3. EVENT — raw facts emitted by input_gateway (R3).
             One fact per applied command: what moved, what toggled, whether
             quit was requested. No downstream interpretation lives here.
*/

#include <cstdint>
#include <variant>

#include <glm/glm.hpp>

namespace shs::input
{
    enum class RuntimeFlagId : uint8_t
    {
        LightShafts  = 0,
        Bot          = 1
    };

    struct CameraTranslatedEvent
    {
        glm::vec3 world_delta{0.0f};    // applied position change (already scaled by speed*dt)

        bool operator==(const CameraTranslatedEvent&) const = default;
    };

    struct CameraRotatedEvent
    {
        float yaw_delta    = 0.0f;      // applied yaw change (post-clamp pitch basis)
        float pitch_delta  = 0.0f;      // applied pitch change (post-clamp)

        bool operator==(const CameraRotatedEvent&) const = default;
    };

    struct RuntimeFlagToggledEvent
    {
        RuntimeFlagId id     = RuntimeFlagId::LightShafts;
        bool          value  = false;   // post-toggle value (raw fact)

        bool operator==(const RuntimeFlagToggledEvent&) const = default;
    };

    struct QuitRequestedEvent
    {
        bool operator==(const QuitRequestedEvent&) const = default;
    };

    using InputEvent = std::variant<
        CameraTranslatedEvent,
        CameraRotatedEvent,
        RuntimeFlagToggledEvent,
        QuitRequestedEvent>;

    inline const char* input_event_name(const InputEvent& ev)
    {
        if (std::holds_alternative<CameraTranslatedEvent>(ev))   return "camera_translated";
        if (std::holds_alternative<CameraRotatedEvent>(ev))      return "camera_rotated";
        if (std::holds_alternative<RuntimeFlagToggledEvent>(ev)) return "runtime_flag_toggled";
        return "quit_requested";
    }

    static_assert(std::variant_size_v<InputEvent> == 4,
        "input event vocabulary changed: update name table + gateway pins");
} // namespace shs::input
