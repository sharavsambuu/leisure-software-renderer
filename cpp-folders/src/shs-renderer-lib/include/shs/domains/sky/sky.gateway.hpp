#pragma once

/*
    SHS RENDERER SAN

    FILE: sky.gateway.hpp
    MODULE: domains/sky
    PURPOSE: CORE 4. GATEWAY — the identity transition (R5a, P4.1 house shape).
             Sky models are caller-owned values sampled by edges/passes.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/sky/sky.command.hpp"
#include "shs/domains/sky/sky.contract.hpp"
#include "shs/domains/sky/sky.event.hpp"

namespace shs::sky
{
    struct SkyState
    {
        bool operator==(const SkyState&) const = default;
    };

    struct SkyContext
    {
    };

    inline void sky_gateway(
        SkyState&                   state,
        std::span<const SkyCommand>  commands,
        const SkyContext&      context,
        std::pmr::vector<SkyEvent>& events)
    {
        (void)state;
        (void)commands;
        (void)context;
        (void)events;
    }
} // namespace shs::sky
