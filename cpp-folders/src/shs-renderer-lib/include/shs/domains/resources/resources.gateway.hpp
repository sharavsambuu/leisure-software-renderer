#pragma once

/*
    SHS RENDERER SAN

    FILE: resources.gateway.hpp
    MODULE: domains/resources
    PURPOSE: CORE 4. GATEWAY — the identity transition (R5a, P4.1 house shape).
             Asset data are values; registry mutation still flows through
             methods (flagged edge API until R5b), so the pod reduces nothing.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/resources/resources.command.hpp"
#include "shs/domains/resources/resources.contract.hpp"
#include "shs/domains/resources/resources.event.hpp"

namespace shs::resources
{
    struct ResourcesState
    {
        bool operator==(const ResourcesState&) const = default;
    };

    struct ResourcesContext
    {
    };

    inline void resources_gateway(
        ResourcesState&                   state,
        std::span<const ResourcesCommand>  commands,
        const ResourcesContext&      context,
        std::pmr::vector<ResourcesEvent>& events)
    {
        (void)state;
        (void)commands;
        (void)context;
        (void)events;
    }
} // namespace shs::resources
