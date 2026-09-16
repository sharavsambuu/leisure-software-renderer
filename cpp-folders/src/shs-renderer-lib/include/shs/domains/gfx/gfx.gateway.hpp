#pragma once

/*
    SHS RENDERER SAN

    FILE: gfx.gateway.hpp
    MODULE: domains/gfx
    PURPOSE: CORE 4. GATEWAY — the identity transition (R5b, P4.1 house shape).
             Handles and pixel buffers are values; allocation lifecycles
             reduce through commands only after the registry edge migration.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/gfx/gfx.command.hpp"
#include "shs/domains/gfx/gfx.contract.hpp"
#include "shs/domains/gfx/gfx.event.hpp"

namespace shs::gfx
{
    struct GfxState
    {
        bool operator==(const GfxState&) const = default;
    };

    struct GfxContext
    {
    };

    inline void gfx_gateway(
        GfxState&                   state,
        std::span<const GfxCommand>  commands,
        const GfxContext&      context,
        std::pmr::vector<GfxEvent>& events)
    {
        (void)state;
        (void)commands;
        (void)context;
        (void)events;
    }
} // namespace shs::gfx
