#pragma once

/*
    SHS RENDERER SAN

    FILE: geometry.event.hpp
    MODULE: domains/geometry
    PURPOSE: CORE 3. EVENT — explicitly empty (R4, Constitution §6.1).
*/

#include <variant>

namespace shs::geometry
{
    using GeometryEvent = std::variant<std::monostate>;

    static_assert(std::variant_size_v<GeometryEvent> == 1,
        "geometry event vocabulary changed: it is empty by law (§6.1); update the pod pins");
} // namespace shs::geometry
