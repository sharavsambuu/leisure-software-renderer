#pragma once

/*
    SHS RENDERER SAN

    FILE: gfx.command.hpp
    MODULE: domains/gfx
    PURPOSE: CORE 2. COMMAND — explicitly empty (R5b, Constitution §6.1).
             Alloc/command intents arrive with the registry edge migration.
*/

#include <variant>

namespace shs::gfx
{
    using GfxCommand = std::variant<std::monostate>;

    static_assert(std::variant_size_v<GfxCommand> == 1,
        "gfx command vocabulary changed: it is empty by law (§6.1); "
        "land a new intent as a named apply_* arrow behind a real gateway first");
} // namespace shs::gfx
