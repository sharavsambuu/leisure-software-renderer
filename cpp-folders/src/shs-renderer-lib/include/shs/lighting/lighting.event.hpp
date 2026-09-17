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
} // namespace shs::lighting
