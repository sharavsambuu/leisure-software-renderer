#pragma once

/*
    SHS RENDERER SAN

    FILE: resources.reducer.hpp
    MODULE: domains/resources
    PURPOSE: CORE 4. REDUCER — the identity transition (R5a, P4.1 house shape).
             Asset data are values; registry mutation still flows through
             methods (flagged edge API until R5b), so the pod reduces nothing.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/resources/resources.action.hpp"
#include "shs/domains/resources/resources.contract.hpp"
#include "shs/domains/resources/resources.event.hpp"

namespace shs::resources
{
    struct ResourcesState
    {
        bool operator==(const ResourcesState&) const = default;
    };

    struct ResourcesReduceInputs
    {
    };

    inline void reduce_resources(
        ResourcesState&                   state,
        std::span<const ResourcesAction>  actions,
        const ResourcesReduceInputs&      inputs,
        std::pmr::vector<ResourcesEvent>& events)
    {
        (void)state;
        (void)actions;
        (void)inputs;
        (void)events;
    }
} // namespace shs::resources
