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

    static_assert(std::variant_size_v<SkyCommand> == 1,
        "sky command vocabulary changed: it is empty by law (§6.1); "
        "land a new intent as a named apply_* arrow behind a real gateway first");
} // namespace shs::sky
