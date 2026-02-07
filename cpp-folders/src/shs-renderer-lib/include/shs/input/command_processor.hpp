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

        void execute_all(CommandContext& ctx)
        {
            for (auto& c : queue_) c->execute(ctx);
            queue_.clear();
        }

    private:
        std::vector<CommandPtr> queue_{};
    };
}

