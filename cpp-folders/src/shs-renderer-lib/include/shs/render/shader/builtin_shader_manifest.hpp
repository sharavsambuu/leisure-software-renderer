#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: builtin_shader_manifest.hpp
    МОДУЛЬ: shader
    ЗОРИЛГО: Value-tier builtin shader-уудын identity манифест (Slang plan
            P1.5). Эдгээр нь бүгд ЗӨВХӨН software realization-тай: авторлагдсан
            GPU модуль хараахан байхгүй. Үүнийг манифест өгөгдлөөрөө хэлнэ —
            ингэснээр dual-realization-ийн зөрүү (drift) нуугдахгүй, харагдана.

    Анхаар: shaders/vulkan доторх frozen GLSL era файлууд ЗОРИУД бүртгэгдээгүй.
    Тэдгээр нь Slang руу шилжих P3 хүртэл realization-гүй төлөвт байгаа бөгөөд
    манифестэд орох нь байхгүй realization-ийг зарлах гэсэн үг.
*/

#include "shs/render/shader/builtin_shaders.hpp"
#include "shs/render/shader/shader_identity.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace render
    {
    // Type-erased CPU realizations. Never restate a shader body here: each one
    // wraps the concrete factory, so the hot path and the identity layer cannot
    // drift.
    inline ShaderProgram erased_blinn_phong_program()
    {
        return make_erased_program(make_blinn_phong_program());
    }

    inline ShaderProgram erased_pbr_mr_program()
    {
        return make_erased_program(make_pbr_mr_program());
    }

    inline ShaderProgram erased_debug_view_albedo_program()
    {
        return make_erased_program(make_debug_view_shader_program(DebugViewMode::Albedo));
    }

    inline ShaderProgram erased_debug_view_normal_program()
    {
        return make_erased_program(make_debug_view_shader_program(DebugViewMode::Normal));
    }

    inline ShaderProgram erased_debug_view_depth_program()
    {
        return make_erased_program(make_debug_view_shader_program(DebugViewMode::Depth));
    }

    // Caller-owned by design: build one where you need it. There is no global
    // manifest, so no hidden state can decide which shader resolves.
    [[nodiscard]] inline ShaderManifest builtin_value_shader_manifest()
    {
        ShaderManifest m{};

        const auto reg = [&m](ShaderId id, ShaderCppImplFn impl) {
            ShaderDesc d{};
            d.name = shader_id_builtin_name(id);
            d.entries = ShaderEntryPoints{};
            d.module = {};
            d.realization_mask = kShaderRealizationSoftware;
            d.cpp_impl = impl;
            // Verified pairing: the canonical name is used for both sides, so a
            // registration can only fail here if the vocabulary itself drifts.
            (void)m.register_shader(id, shader_id_builtin_name(id), d);
        };

        reg(ShaderId::BlinnPhong, &erased_blinn_phong_program);
        reg(ShaderId::PbrMetallicRoughness, &erased_pbr_mr_program);
        reg(ShaderId::LitDefault, &erased_pbr_mr_program);
        reg(ShaderId::DebugViewAlbedo, &erased_debug_view_albedo_program);
        reg(ShaderId::DebugViewNormal, &erased_debug_view_normal_program);
        reg(ShaderId::DebugViewDepth, &erased_debug_view_depth_program);

        return m;
    }
    } // inline namespace render
} // namespace shs
