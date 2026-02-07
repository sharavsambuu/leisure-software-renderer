#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: program.hpp
    МОДУЛЬ: shader
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн shader модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <functional>

#include "shs/shader/types.hpp"

namespace shs
{
    using VertexShaderFn = std::function<VertexOut(const ShaderVertex&, const ShaderUniforms&)>;
    using FragmentShaderFn = std::function<FragmentOut(const FragmentIn&, const ShaderUniforms&)>;

    struct ShaderProgram
    {
        VertexShaderFn vs{};
        FragmentShaderFn fs{};

        bool valid() const
        {
            return (bool)vs && (bool)fs;
        }
    };
}

