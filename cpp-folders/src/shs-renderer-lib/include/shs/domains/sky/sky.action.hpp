#pragma once

/*
    SHS RENDERER SAN

    FILE: sky.action.hpp
    MODULE: domains/sky
    PURPOSE: CORE 2. COMMAND — explicitly empty (R5a, Constitution §6.1).
*/

#include <variant>

namespace shs::sky
{
    using SkyAction = std::variant<std::monostate>;
} // namespace shs::sky
