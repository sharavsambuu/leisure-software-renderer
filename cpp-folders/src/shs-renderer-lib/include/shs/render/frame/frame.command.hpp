#pragma once

/*
    SHS RENDERER SAN

    FILE: frame.command.hpp
    MODULE: domains/frame
    PURPOSE: CORE 2. COMMAND — explicitly empty (R3, Constitution §6.1).
             Frame configuration has no transition vocabulary today: planners
             rebuild FrameParams per frame instead of reducing it. The empty
             closed type keeps that fact greppable; the first real knob
             (exposure, debug view) replaces monostate with intents.
*/

#include <variant>

namespace shs::frame
{
    using FrameCommand = std::variant<std::monostate>;
} // namespace shs::frame
