#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: command.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <memory>

#include "shs/app/runtime_state.hpp"

namespace shs
{
    struct CommandContext
    {
        RuntimeState& state;
        float dt = 0.0f;
    };

    struct ICommand
    {
        virtual ~ICommand() = default;
        virtual void execute(CommandContext& ctx) = 0;
    };

    using CommandPtr = std::unique_ptr<ICommand>;
}

