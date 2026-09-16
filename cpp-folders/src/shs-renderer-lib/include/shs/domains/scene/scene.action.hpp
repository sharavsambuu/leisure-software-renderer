#pragma once

/*
    SHS RENDERER SAN

    FILE: scene.action.hpp
    MODULE: domains/scene
    PURPOSE: CORE 2. COMMAND — explicitly empty (R5b, Constitution §6.1).
             Spawn/destroy intents arrive with the store edge migration.
*/

#include <variant>

namespace shs::scene
{
    using SceneAction = std::variant<std::monostate>;
} // namespace shs::scene
