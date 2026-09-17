#pragma once

/*
    SHS RENDERER SAN

    FILE: geometry.command.hpp
    MODULE: domains/geometry
    PURPOSE: CORE 2. COMMAND — explicitly empty (R4, Constitution §6.1).
             Shapes and adapters carry no transition vocabulary; culling
             runtime commands arrive with the R5 culling migration.
*/

#include <variant>

namespace shs::geometry
{
    using GeometryCommand = std::variant<std::monostate>;

    static_assert(std::variant_size_v<GeometryCommand> == 1,
        "geometry command vocabulary changed: it is empty by law (§6.1); "
        "land a new intent as a named apply_* arrow behind a real gateway first");
} // namespace shs::geometry
