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

    static_assert(std::variant_size_v<ResourcesEvent> == 1,
        "resources event vocabulary changed: it is empty by law (§6.1); update the pod pins");
} // namespace shs::resources
