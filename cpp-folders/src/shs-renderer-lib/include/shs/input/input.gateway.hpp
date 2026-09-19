#pragma once

/*
    SHS RENDERER SAN

    FILE: input.gateway.hpp
    MODULE: domains/input
    PURPOSE: CORE 4. GATEWAY — translation-support vocabulary. Step 4.1
             (docs/outdated/engine_domain_separation_migration.md) split
             input translation from application: the state-mutating
             camera/render/session application arrows moved to the explicit
             app orchestrator, shs::app::session_orchestrate (supersedes the
             K1.4 interim note that parked the camera rig in this pod's
             state aggregate). This pod now owns TRANSLATION only: raw
             input -> latched intents (value_input_latch.hpp) and latch ->
             command batches (value_commands.hpp). The context and tally
             types below are the shared vocabulary of that seam.
*/

#include <cstdint>

namespace shs::input
{
    // Application context carried with a command batch (per-frame dt).
    struct InputContext
    {
        float dt = 0.0f;
    };

    // Batch outcome summary (house shape per
    // docs/outdated/kdba_kleisli_migration_plan.md; the rim is infallible —
    // every intent is valid for this pod).
    struct InputStep
    {
        uint32_t commands_applied = 0;  // commands that mutated pod state

        bool operator==(const InputStep&) const = default;
    };
} // namespace shs::input
