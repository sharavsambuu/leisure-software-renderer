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
} // namespace shs::gfx
