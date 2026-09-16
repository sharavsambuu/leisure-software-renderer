#pragma once

/*
    SHS RENDERER SAN

    FILE: sky.event.hpp
    MODULE: domains/sky
    PURPOSE: CORE 3. EVENT — explicitly empty (R5a, Constitution §6.1).
*/

#include <variant>

namespace shs::sky
{
    using SkyEvent = std::variant<std::monostate>;
} // namespace shs::sky
