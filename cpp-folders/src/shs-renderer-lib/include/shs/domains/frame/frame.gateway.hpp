#pragma once

/*
    SHS RENDERER SAN

    FILE: frame.gateway.hpp
    MODULE: domains/frame
    PURPOSE: CORE 4. GATEWAY — the identity transition (R3, P4.1 house shape).
             Proves frame configuration is transition-free: any command log
             (today only the empty log typechecks) leaves state bit-identical
             and emits nothing. Pinned by vop_frame_tests, not by convention.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/frame/frame.command.hpp"
#include "shs/domains/frame/frame.contract.hpp"
#include "shs/domains/frame/frame.event.hpp"

namespace shs::frame
{
    struct FrameContext
    {
    };

    inline void frame_gateway(
        FrameParams&                     state,
        std::span<const FrameCommand>     commands,
        const FrameContext&         context,
        std::pmr::vector<FrameEvent>&    events)
    {
        (void)state;
        (void)commands;
        (void)context;
        (void)events;
    }
} // namespace shs::frame
