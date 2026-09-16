#pragma once

/*
    SHS RENDERER SAN

    FILE: geometry.action.hpp
    MODULE: domains/geometry
    PURPOSE: CORE 2. COMMAND — explicitly empty (R4, Constitution §6.1).
             Shapes and adapters carry no transition vocabulary; culling
             runtime commands arrive with the R5 culling migration.
*/

#include <variant>

namespace shs::geometry
{
    using GeometryAction = std::variant<std::monostate>;
} // namespace shs::geometry
