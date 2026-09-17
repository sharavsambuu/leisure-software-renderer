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
} // namespace shs::geometry
