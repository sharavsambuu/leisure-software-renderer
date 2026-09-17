#pragma once

/*
    SHS RENDERER SAN

    FILE: lighting.event.hpp
    MODULE: domains/lighting
    PURPOSE: CORE 3. EVENT — explicitly empty (R4, Constitution §6.1).
*/

#include <variant>

namespace shs::lighting
{
    using LightingEvent = std::variant<std::monostate>;

    static_assert(std::variant_size_v<LightingEvent> == 1,
        "lighting event vocabulary changed: it is empty by law (§6.1); update the pod pins");
} // namespace shs::lighting
