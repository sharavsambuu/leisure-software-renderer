#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: command.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <memory>

#include "shs/input/value_commands.hpp"

namespace shs
{
    struct ICommand
    {
        virtual ~ICommand() = default;
        // VOP-first command contract: commands must emit an equivalent value command.
        virtual RuntimeCommand to_runtime_action() const = 0;
    };

    using CommandPtr = std::unique_ptr<ICommand>;
}
