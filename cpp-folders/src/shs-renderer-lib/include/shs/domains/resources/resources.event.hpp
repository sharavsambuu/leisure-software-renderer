#pragma once

/*
    SHS RENDERER SAN

    FILE: resources.event.hpp
    MODULE: domains/resources
    PURPOSE: CORE 3. EVENT — explicitly empty (R5a, Constitution §6.1).
*/

#include <variant>

namespace shs::resources
{
    using ResourcesEvent = std::variant<std::monostate>;
} // namespace shs::resources
