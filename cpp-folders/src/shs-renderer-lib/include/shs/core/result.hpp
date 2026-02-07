#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: result.hpp
    МОДУЛЬ: core
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн core модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>
#include <utility>

namespace shs
{
    template<typename T>
    struct Result
    {
        bool ok = false;
        T value{};
        std::string error{};

        static Result<T> success(T v)
        {
            return Result<T>{true, std::move(v), {}};
        }

        static Result<T> failure(std::string e)
        {
            return Result<T>{false, T{}, std::move(e)};
        }
    };
}
