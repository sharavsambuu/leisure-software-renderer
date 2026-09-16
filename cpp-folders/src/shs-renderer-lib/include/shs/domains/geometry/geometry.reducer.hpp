#pragma once

/*
    SHS RENDERER SAN

    FILE: geometry.reducer.hpp
    MODULE: domains/geometry
    PURPOSE: CORE 4. REDUCER — the identity transition (R4, P4.1 house shape).
             The pod owns no mutable state yet (shapes/adapters are values;
             culling runtime migrates in R5), so reduction is stability by
             construction, pinned by vop_geometry_tests.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/geometry/geometry.action.hpp"
#include "shs/domains/geometry/geometry.contract.hpp"
#include "shs/domains/geometry/geometry.event.hpp"

namespace shs::geometry
{
    struct GeometryState
    {
        bool operator==(const GeometryState&) const = default;
    };

    struct GeometryReduceInputs
    {
    };

    inline void reduce_geometry(
        GeometryState&                   state,
        std::span<const GeometryAction>  actions,
        const GeometryReduceInputs&      inputs,
        std::pmr::vector<GeometryEvent>& events)
    {
        (void)state;
        (void)actions;
        (void)inputs;
        (void)events;
    }
} // namespace shs::geometry
