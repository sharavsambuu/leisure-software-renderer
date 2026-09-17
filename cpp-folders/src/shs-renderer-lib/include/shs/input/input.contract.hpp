#pragma once

/*
    SHS RENDERER SAN

    FILE: input.contract.hpp
    MODULE: domains/input
    PURPOSE: CORE 1. TYPES — the input pod's discoverability seam (R3).
             Re-exports the pod vocabulary; owns no logic. Command emitters
             (ICommand hierarchy, CommandProcessor) are cold edge-side queueing
             (P5 classifies them); the value vocabulary below is the pod truth.
*/

#include "shs/input/bot_controller.hpp"
#include "shs/input/storage/camera_commands.hpp"
#include "shs/input/storage/command.hpp"
#include "shs/input/storage/command_processor.hpp"
#include "shs/input/human_controller.hpp"
#include "shs/input/input.command.hpp"
#include "shs/input/input.event.hpp"
#include "shs/input/input.gateway.hpp"
#include "shs/input/input_state.hpp"
#include "shs/input/value_commands.hpp"
#include "shs/input/value_input_latch.hpp"

namespace shs::input
{
    // --- state vocabulary ---
    using shs::InputState;
    using shs::RuntimeState;
    using shs::RuntimeInputLatch;

    // --- command vocabulary (closed; see input.command.hpp) ---
    using shs::RuntimeCommand;
    using shs::input::InputCommand;
    using shs::MoveLocalIntent;
    using shs::LookIntent;
    using shs::ToggleLightShaftsIntent;
    using shs::ToggleBotIntent;
    using shs::QuitIntent;

    // --- latch event vocabulary (edge-tokenized OS events) ---
    using shs::RuntimeInputEvent;
    using shs::RuntimeInputEventType;

    // --- pod event vocabulary (gateway-emitted facts) ---
    // (InputEvent + members live in shs::input already.)

    // --- pure helpers ---
    using shs::emit_human_commands;
    using shs::emit_orbit_bot_commands;
    using shs::emit_human_runtime_commands;
    using shs::emit_orbit_bot_runtime_commands;
    using shs::input_latch_gateway;
} // namespace shs::input
