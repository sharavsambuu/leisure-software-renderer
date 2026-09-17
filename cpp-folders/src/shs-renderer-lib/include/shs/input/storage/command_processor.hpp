#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: command_processor.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <memory_resource>
#include <span>
#include <utility>
#include <vector>

#include "shs/input/input.command.hpp"
#include "shs/input/storage/command.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace input
    {
    class CommandProcessor
    {
    public:
        void enqueue(CommandPtr cmd)
        {
            if (cmd) queue_.push_back(std::move(cmd));
        }

        template<typename TCmd, typename... Args>
        void emplace(Args&&... args)
        {
            queue_.push_back(std::make_unique<TCmd>(std::forward<Args>(args)...));
        }

        std::vector<RuntimeCommand> collect_runtime_commands()
        {
            std::vector<RuntimeCommand> commands{};
            commands.reserve(queue_.size());

            for (auto& c : queue_)
            {
                if (!c) continue;
                commands.push_back(c->to_runtime_action());
            }
            queue_.clear();
            return commands;
        }

        // Step 4.1 (domain separation): application moved OUT of the input
        // pod. The former apply_commands() edge convenience is retired —
        // cold queueing + command translation (above) is this class's whole
        // job; the host applies the collected batch through the explicit
        // app orchestrator, shs::app::session_orchestrate, which also
        // preserves the fact log this method used to drop.

    private:
        std::vector<CommandPtr> queue_{};
    };

    } // inline namespace input
}
