#pragma once

/*
    SHS RENDERER SAN

    FILE: camera.event.hpp
    MODULE: domains/camera
    PURPOSE: CORE 3. EVENT — explicitly empty (R5a, Constitution §6.1).
*/

#include <variant>

namespace shs::camera
{
    using CameraEvent = std::variant<std::monostate>;
} // namespace shs::camera
