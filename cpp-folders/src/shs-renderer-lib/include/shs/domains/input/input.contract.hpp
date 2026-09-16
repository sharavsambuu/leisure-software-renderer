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

#include "shs/domains/input/bot_controller.hpp"
#include "shs/domains/input/edge/camera_commands.hpp"
#include "shs/domains/input/edge/command.hpp"
#include "shs/domains/input/edge/command_processor.hpp"
#include "shs/domains/input/human_controller.hpp"
#include "shs/domains/input/input.action.hpp"
#include "shs/domains/input/input.event.hpp"
#include "shs/domains/input/input.reducer.hpp"
#include "shs/domains/input/input_state.hpp"
#include "shs/domains/input/value_actions.hpp"
#include "shs/domains/input/value_input_latch.hpp"

namespace shs::input
{
    // --- state vocabulary ---
    using shs::InputState;
    using shs::RuntimeState;
    using shs::RuntimeInputLatch;

    // --- command vocabulary (closed; see input.action.hpp) ---
    using shs::RuntimeAction;
    using shs::RuntimeActionType;
    using shs::MoveLocalAction;
    using shs::LookAction;
    using shs::ToggleFlagAction;

    // --- latch event vocabulary (edge-tokenized OS events) ---
    using shs::RuntimeInputEvent;
    using shs::RuntimeInputEventType;

    // --- pod event vocabulary (reducer-emitted facts) ---
    // (InputEvent + members live in shs::input already.)

    // --- pure helpers ---
    using shs::emit_human_actions;
    using shs::emit_orbit_bot_actions;
    using shs::emit_human_runtime_actions;
    using shs::emit_orbit_bot_runtime_actions;
    using shs::reduce_runtime_input_latch;
} // namespace shs::input
