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

    static_assert(std::variant_size_v<GfxEvent> == 1,
        "gfx event vocabulary changed: it is empty by law (§6.1); update the pod pins");
} // namespace shs::gfx
