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

#include "shs/domains/input/input.command.hpp"
#include "shs/domains/input/input.gateway.hpp"
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
            if (commands.empty()) return state;

            // K5.1 (Run B): the pod's single public Kleisli gateway, driven
            // directly from the edge; this edge convenience drops the fact
            // log (a frame-arena sink would be the replay-ready form).
            std::pmr::monotonic_buffer_resource arena{1024};
            std::pmr::vector<shs::input::InputEvent> events{&arena};
            shs::input::input_gateway(state,
                std::span<const RuntimeCommand>{commands.data(), commands.size()},
                shs::input::InputContext{dt}, events);
            return state;
        }

    private:
        std::vector<CommandPtr> queue_{};
    };
}
