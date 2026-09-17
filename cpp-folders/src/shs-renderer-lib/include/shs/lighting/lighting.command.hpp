#pragma once

/*
    SHS RENDERER SAN

    FILE: lighting.command.hpp
    MODULE: domains/lighting
    PURPOSE: CORE 2. COMMAND — explicitly empty (R4, Constitution §6.1).
*/

#include <variant>

namespace shs::lighting
{
    using LightingCommand = std::variant<std::monostate>;

    static_assert(std::variant_size_v<LightingCommand> == 1,
        "lighting command vocabulary changed: it is empty by law (§6.1); "
        "land a new intent as a named apply_* arrow behind a real gateway first");
} // namespace shs::lighting
