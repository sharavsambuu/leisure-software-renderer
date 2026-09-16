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
} // namespace shs::geometry
