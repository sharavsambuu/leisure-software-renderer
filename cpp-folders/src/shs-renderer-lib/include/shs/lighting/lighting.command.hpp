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
} // namespace shs::lighting
