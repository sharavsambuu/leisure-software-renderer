#pragma once

/*
    SHS RENDERER SAN

    FILE: scene.event.hpp
    MODULE: domains/scene
    PURPOSE: CORE 3. EVENT — explicitly empty (R5b, Constitution §6.1).
*/

#include <variant>

namespace shs::scene
{
    using SceneEvent = std::variant<std::monostate>;
} // namespace shs::scene
