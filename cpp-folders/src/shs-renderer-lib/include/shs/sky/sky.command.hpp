#pragma once

/*
    SHS RENDERER SAN

    FILE: sky.command.hpp
    MODULE: domains/sky
    PURPOSE: CORE 2. COMMAND — explicitly empty (R5a, Constitution §6.1).
*/

#include <variant>

namespace shs::sky
{
    using SkyCommand = std::variant<std::monostate>;
} // namespace shs::sky
