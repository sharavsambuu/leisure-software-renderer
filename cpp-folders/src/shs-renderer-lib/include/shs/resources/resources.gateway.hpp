#pragma once

/*
    SHS RENDERER SAN

    FILE: resources.gateway.hpp
    MODULE: domains/resources
    PURPOSE: CORE 4. GATEWAY — the pinned identity transition, Kleisli house
             shape (Run C, K1.5): (State, span<Commands>, Context, arena)
             -> ResourcesStep. Asset data are values; registry mutation still
             flows through edge methods, so the pod reduces nothing yet; the
             empty-vocabulary static_assert makes the silence provable (§6.1).
             Pinned by resources_tests.
*/

#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

#include "shs/resources/resources.command.hpp"
#include "shs/resources/resources.contract.hpp"
#include "shs/resources/resources.event.hpp"

namespace shs::resources
{
    struct ResourcesState
    {
        bool operator==(const ResourcesState&) const = default;
    };

    struct ResourcesContext
    {
    };

    // Batch outcome summary (house shape per kdba_kleisli_migration_plan.md).
    // An identity pod applies nothing — it counts what passed the seam.
    struct ResourcesStep
    {
        uint32_t commands_observed = 0;

        bool operator==(const ResourcesStep&) const = default;
    };

    inline ResourcesStep resources_gateway(
        ResourcesState&                   state,
        std::span<const ResourcesCommand>  commands,
        const ResourcesContext&      context,
        std::pmr::vector<ResourcesEvent>& events)
    {
        (void)state;
        (void)context;
        (void)events;
        ResourcesStep step{};
        for (const ResourcesCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                static_assert(std::is_same_v<T, std::monostate>,
                    "resources command vocabulary is empty by law (§6.1): "
                    "land a new intent as a named apply_* arrow first");
                (void)cmd;
            }, command);
            step.commands_observed += 1;
        }
        return step;
    }
} // namespace shs::resources
