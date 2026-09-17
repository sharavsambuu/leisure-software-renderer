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

    static_assert(std::variant_size_v<ResourcesCommand> == 1,
        "resources command vocabulary changed: it is empty by law (§6.1); "
        "land a new intent as a named apply_* arrow behind a real gateway first");
} // namespace shs::resources
