#pragma once

/*
    SHS RENDERER SAN

    FILE: frame.event.hpp
    MODULE: domains/frame
    PURPOSE: CORE 3. EVENT — explicitly empty (R3, Constitution §6.1).
             The identity gateway emits nothing; the voyage is the log.
*/

#include <string_view>
#include <variant>

namespace shs::frame
{
    using FrameEvent = std::variant<std::monostate>;

    inline constexpr std::size_t k_frame_event_count = 0;

    static_assert(k_frame_event_count == 0 && std::variant_size_v<FrameEvent> == 1,
        "frame event vocabulary changed: update gateway pins");
} // namespace shs::frame
