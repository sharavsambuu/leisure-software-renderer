#pragma once

/*
    SHS RENDERER SAN

    FILE: geometry.gateway.hpp
    MODULE: domains/geometry
    PURPOSE: CORE 4. GATEWAY — the identity transition (R4, P4.1 house shape).
             The pod owns no mutable state yet (shapes/adapters are values;
             culling runtime migrates in R5), so reduction is stability by
             construction, pinned by vop_geometry_tests.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/geometry/geometry.command.hpp"
#include "shs/domains/geometry/geometry.contract.hpp"
#include "shs/domains/geometry/geometry.event.hpp"

namespace shs::geometry
{
    struct GeometryState
    {
        bool operator==(const GeometryState&) const = default;
    };

    struct GeometryContext
    {
    };

    inline void geometry_gateway(
        GeometryState&                   state,
        std::span<const GeometryCommand>  commands,
        const GeometryContext&      context,
        std::pmr::vector<GeometryEvent>& events)
    {
        (void)state;
        (void)commands;
        (void)context;
        (void)events;
    }
} // namespace shs::geometry
