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

    static_assert(std::variant_size_v<SkyEvent> == 1,
        "sky event vocabulary changed: it is empty by law (§6.1); update the pod pins");
} // namespace shs::sky
