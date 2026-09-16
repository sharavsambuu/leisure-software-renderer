#pragma once

/*
    SHS RENDERER SAN

    FILE: frame.reducer.hpp
    MODULE: domains/frame
    PURPOSE: CORE 4. REDUCER — the identity transition (R3, P4.1 house shape).
             Proves frame configuration is transition-free: any command log
             (today only the empty log typechecks) leaves state bit-identical
             and emits nothing. Pinned by vop_frame_tests, not by convention.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/frame/frame.action.hpp"
#include "shs/domains/frame/frame.contract.hpp"
#include "shs/domains/frame/frame.event.hpp"

namespace shs::frame
{
    struct FrameReduceInputs
    {
    };

    inline void reduce_frame(
        FrameParams&                     state,
        std::span<const FrameAction>     actions,
        const FrameReduceInputs&         inputs,
        std::pmr::vector<FrameEvent>&    events)
    {
        (void)state;
        (void)actions;
        (void)inputs;
        (void)events;
    }
} // namespace shs::frame
