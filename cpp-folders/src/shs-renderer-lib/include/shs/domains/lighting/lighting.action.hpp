#pragma once

/*
    SHS RENDERER SAN

    FILE: lighting.action.hpp
    MODULE: domains/lighting
    PURPOSE: CORE 2. COMMAND — explicitly empty (R4, Constitution §6.1).
*/

#include <variant>

namespace shs::lighting
{
    using LightingAction = std::variant<std::monostate>;
} // namespace shs::lighting
