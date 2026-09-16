#pragma once

/*
    SHS RENDERER SAN

    FILE: gfx.reducer.hpp
    MODULE: domains/gfx
    PURPOSE: CORE 4. REDUCER — the identity transition (R5b, P4.1 house shape).
             Handles and pixel buffers are values; allocation lifecycles
             reduce through commands only after the registry edge migration.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/gfx/gfx.action.hpp"
#include "shs/domains/gfx/gfx.contract.hpp"
#include "shs/domains/gfx/gfx.event.hpp"

namespace shs::gfx
{
    struct GfxState
    {
        bool operator==(const GfxState&) const = default;
    };

    struct GfxReduceInputs
    {
    };

    inline void reduce_gfx(
        GfxState&                   state,
        std::span<const GfxAction>  actions,
        const GfxReduceInputs&      inputs,
        std::pmr::vector<GfxEvent>& events)
    {
        (void)state;
        (void)actions;
        (void)inputs;
        (void)events;
    }
} // namespace shs::gfx
