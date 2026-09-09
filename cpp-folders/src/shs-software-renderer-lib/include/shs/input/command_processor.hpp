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

#include "shs/input/command.hpp"

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

        std::vector<RuntimeAction> collect_runtime_actions()
        {
            std::vector<RuntimeAction> actions{};
            actions.reserve(queue_.size());

            for (auto& c : queue_)
            {
                if (!c) continue;
                actions.push_back(c->to_runtime_action());
            }
            queue_.clear();
            return actions;
        }

        RuntimeState reduce_all(RuntimeState state, float dt)
        {
            const std::vector<RuntimeAction> actions = collect_runtime_actions();
            if (!actions.empty())
            {
                state = reduce_runtime_state(state, actions, dt);
            }
            return state;
        }

    private:
        std::vector<CommandPtr> queue_{};
    };
}
