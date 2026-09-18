#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: program.hpp
    МОДУЛЬ: shader
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн shader модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <functional>

#include "shs/render/shader/types.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace render
    {
    using VertexShaderFn = std::function<VertexOut(const ShaderVertex&, const ShaderUniforms&)>;
    using FragmentShaderFn = std::function<FragmentOut(const FragmentIn&, const ShaderUniforms&)>;

    // R1 (renderer-lib review 2026-09-18): concrete, non-erased shader pair.
    // The software rasterizer is templated on the program type; when callers
    // pass a ShaderProgramFn the per-pixel fragment invocation is a direct
    // (inlinable, vectorizable) call. std::function-based ShaderProgram
    // remains only at host seams where type erasure is genuinely needed.
    template <typename VsFn, typename FsFn>
    struct ShaderProgramFn
    {
        VsFn vs{};
        FsFn fs{};

        constexpr bool valid() const { return true; }
    };

    struct ShaderProgram
    {
        VertexShaderFn vs{};
        FragmentShaderFn fs{};

        bool valid() const
        {
            return (bool)vs && (bool)fs;
        }
    };

    } // inline namespace render
}

