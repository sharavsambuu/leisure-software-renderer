#pragma once

/*
    SHS RENDERER SAN

    FILE: sky.reducer.hpp
    MODULE: domains/sky
    PURPOSE: CORE 4. REDUCER — the identity transition (R5a, P4.1 house shape).
             Sky models are caller-owned values sampled by edges/passes.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/sky/sky.action.hpp"
#include "shs/domains/sky/sky.contract.hpp"
#include "shs/domains/sky/sky.event.hpp"

namespace shs::sky
{
    struct SkyState
    {
        bool operator==(const SkyState&) const = default;
    };

    struct SkyReduceInputs
    {
    };

    inline void reduce_sky(
        SkyState&                   state,
        std::span<const SkyAction>  actions,
        const SkyReduceInputs&      inputs,
        std::pmr::vector<SkyEvent>& events)
    {
        (void)state;
        (void)actions;
        (void)inputs;
        (void)events;
    }
} // namespace shs::sky
