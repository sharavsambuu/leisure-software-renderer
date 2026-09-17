#pragma once

/*
    SHS RENDERER SAN

    FILE: scene.command.hpp
    MODULE: domains/scene
    PURPOSE: CORE 2. COMMAND — explicitly empty (R5b, Constitution §6.1).
             Spawn/destroy intents arrive with the store edge migration.
*/

#include <variant>

namespace shs::scene
{
    using SceneCommand = std::variant<std::monostate>;

    static_assert(std::variant_size_v<SceneCommand> == 1,
        "scene command vocabulary changed: it is empty by law (§6.1); "
        "land a new intent as a named apply_* arrow behind a real gateway first");
} // namespace shs::scene
