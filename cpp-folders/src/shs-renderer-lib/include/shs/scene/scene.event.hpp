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

    static_assert(std::variant_size_v<SceneEvent> == 1,
        "scene event vocabulary changed: it is empty by law (§6.1); update the pod pins");
} // namespace shs::scene
