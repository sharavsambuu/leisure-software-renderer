#pragma once

/*
    SHS RENDERER SAN

    FILE: gfx.event.hpp
    MODULE: domains/gfx
    PURPOSE: CORE 3. EVENT — explicitly empty (R5b, Constitution §6.1).
*/

#include <variant>

namespace shs::gfx
{
    using GfxEvent = std::variant<std::monostate>;
} // namespace shs::gfx
