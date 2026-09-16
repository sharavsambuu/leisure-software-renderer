#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: command_processor.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <utility>
#include <vector>

#include "shs/domains/input/edge/command.hpp"

namespace shs
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

        RuntimeState apply_commands(RuntimeState state, float dt)
        {
            const std::vector<RuntimeCommand> commands = collect_runtime_commands();
            if (!commands.empty())
            {
                state = runtime_state_gateway(state, commands, dt);
            }
            return state;
        }

    private:
        std::vector<CommandPtr> queue_{};
    };
}
