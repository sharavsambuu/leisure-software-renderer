#pragma once

/*
    SHS RENDERER SAN

    FILE: resources.command.hpp
    MODULE: domains/resources
    PURPOSE: CORE 2. COMMAND — explicitly empty (R5a, Constitution §6.1).
             Registration flows through registry methods today (edge-candidate
             API, see contract); Register* intents arrive with the R5b edge
             migration if an edge needs them.
*/

#include <variant>

namespace shs::resources
{
    using ResourcesCommand = std::variant<std::monostate>;
} // namespace shs::resources
