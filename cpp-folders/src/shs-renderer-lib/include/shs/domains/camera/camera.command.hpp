#pragma once

/*
    SHS RENDERER SAN

    FILE: camera.command.hpp
    MODULE: domains/camera
    PURPOSE: CORE 2. COMMAND — explicitly empty (R5a, Constitution §6.1).
             Camera motion arrives as input-pod commands applied elsewhere;
             follow/light builders are pure functions of their arguments.
*/

#include <variant>

namespace shs::camera
{
    using CameraCommand = std::variant<std::monostate>;
} // namespace shs::camera
