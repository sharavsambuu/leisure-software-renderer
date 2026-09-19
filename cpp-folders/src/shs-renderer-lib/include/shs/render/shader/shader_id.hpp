#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: shader_id.hpp
    МОДУЛЬ: shader
    ЗОРИЛГО: Shader identity-ийн id ба range law (Constitution I §7 — No User
            Lock-In; arch/render_path_architecture.md §4 graduation req 6).

            `ShaderId` нь жижиг хаалттай builtin vocabulary; consumer/demo-ийн
            shader-ууд open *registered* range-д орж, core-д гар хүрэхгүй.
            Open id-ууд зөвхөн `ShaderIdRegistry`
            (`shs/render/shader/shader_id_registry.hpp`)-ээр нэрээсээ
            content-addressed байдлаар үүснэ — call site дээр нэр хэшлэж
            БОЛОХГҮЙ, учир нь id оноолт нь hash/iteration order-оос хамаарах
            ёсгүй (replay/determinism хууль).

    Энэ файл нь `renderpath/planning/pass_id.hpp`-ийн адил shape: law энд,
    registry нь тусдаа файлд. `shader_identity.hpp` хоёуланг нь нэгтгэн
    дахин экспортолдог тул хуучин consumer-ууд хөндөгдөхгүй.
*/

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace render
    {
    // ---------------------------------------------------------------------
    // Closed builtin vocabulary.
    // Only identities with at least one *real* realization are enumerated;
    // the frozen GLSL-era modules under shaders/vulkan are deliberately NOT
    // listed until they migrate (Slang plan P3), so the manifest can never
    // claim a realization that does not exist.
    // ---------------------------------------------------------------------
    enum class ShaderId : uint16_t
    {
        Unknown = 0,

        // Value-tier builtin programs. Software realization today; no GPU
        // realization is wired, and the manifest says so in data.
        BlinnPhong = 1,
        PbrMetallicRoughness = 2,
        LitDefault = 3,
        DebugViewAlbedo = 4,
        DebugViewNormal = 5,
        DebugViewDepth = 6,

        // The one identity with BOTH realizations today: the authored
        // tests/shaders/offscreen_pipeline.slang module and its CPU counterpart
        // (rhi/software/sw_offscreen.hpp). Exercised by four CTest gates.
        OffscreenPipeline = 7,

        Count = 8 // sentinel, never registerable
    };

    // --- Shader-id range law (Constitution I §7; arch §4 graduation req 6) ---
    // Builtin identity-ууд нь дээрх жижиг хаалттай enum; consumer/demo-ийн
    // shader-ууд open *registered* range-д орно. Open id-ууд нь
    // `ShaderIdRegistry`-гээр нэрээсээ content-addressed байдлаар үүснэ.
    inline constexpr uint16_t kShaderIdUnknown = 0u;
    inline constexpr uint16_t kShaderIdBuiltinMin = 1u;
    inline constexpr uint16_t kShaderIdBuiltinMax = 1023u;
    inline constexpr uint16_t kShaderIdOpenBase = 1024u;
    inline constexpr uint16_t kShaderIdOpenMax = 65534u;
    inline constexpr uint16_t kShaderIdReserved = 65535u;

    // Builtin identity-ийн тоо. `ShaderId::Count` нь sentinel хэвээр — тэр нь
    // manifest-ийн builtin массив extent ба builtin/open хил хоёуланг тодорхойлно.
    inline constexpr std::size_t kShaderIdBuiltinCount = static_cast<std::size_t>(ShaderId::Count);

    // Legacy spelling — behavior-identical, and kept so existing call sites
    // (the manifest array extent) and any external user do not change.
    inline constexpr std::size_t kShaderIdCount = kShaderIdBuiltinCount;

    // Vocabulary pin: builtin enum нь open range руу чимээгй өсөж, consumer
    // shader-тай alias болохгүй; null id нь null хэвээр.
    static_assert(static_cast<uint16_t>(ShaderId::Unknown) == kShaderIdUnknown,
                  "ShaderId::Unknown must stay the null id");
    static_assert(static_cast<uint16_t>(ShaderId::OffscreenPipeline) <= kShaderIdBuiltinMax,
                  "builtin ShaderId vocabulary overflowed into the open registered range");
    static_assert(kShaderIdBuiltinCount <= static_cast<std::size_t>(kShaderIdBuiltinMax) + 1u,
                  "the builtin ShaderId count left no room below the open range");
    static_assert(kShaderIdOpenBase > kShaderIdBuiltinMax,
                  "the open shader-id range must start above the builtin ceiling");
    static_assert(kShaderIdOpenMax < kShaderIdReserved,
                  "the reserved top slot must stay outside the open range");

    inline constexpr bool shader_id_is_builtin(ShaderId id)
    {
        const uint16_t raw = static_cast<uint16_t>(id);
        return raw >= kShaderIdBuiltinMin && raw < static_cast<uint16_t>(kShaderIdBuiltinCount);
    }

    inline constexpr bool shader_id_is_open(ShaderId id)
    {
        const uint16_t raw = static_cast<uint16_t>(id);
        return raw >= kShaderIdOpenBase && raw <= kShaderIdOpenMax;
    }

    // A plan or a registry-д хууль ёсоор агуулагдаж болох typed id: builtin
    // эсвэл minted. `Unknown`, `Count` sentinel, builtin ceiling ба open base
    // хоорондын зай (зориуд нөөцлөгдсөн), ба reserved top slot нь орохгүй.
    inline constexpr bool shader_id_in_valid_range(ShaderId id)
    {
        return shader_id_is_builtin(id) || shader_id_is_open(id);
    }

    // Honest, deterministic spelling for an open-range id that carries no
    // resolvable registry here. Энэ нь ЗОРИУД registration key БИШ: жинхэнэ
    // нэрийг эзэмшигч registry (ShaderIdRegistry::try_name) мэднэ — түүгээр
    // resolve хийнэ. Үүнийг нэр болгон intern хийхийг registry татгалзана.
    inline constexpr const char* kShaderIdOpenSpelling = "open_shader";

    // Canonical name of a builtin identity. Empty for Unknown, which is never
    // registerable and therefore never resolvable.
    [[nodiscard]] constexpr std::string_view shader_id_builtin_name(ShaderId id)
    {
        switch (id)
        {
        case ShaderId::BlinnPhong: return "blinn_phong";
        case ShaderId::PbrMetallicRoughness: return "pbr_metallic_roughness";
        case ShaderId::LitDefault: return "lit_default";
        case ShaderId::DebugViewAlbedo: return "debug_view_albedo";
        case ShaderId::DebugViewNormal: return "debug_view_normal";
        case ShaderId::DebugViewDepth: return "debug_view_depth";
        case ShaderId::OffscreenPipeline: return "offscreen_pipeline";
        case ShaderId::Unknown:
        case ShaderId::Count:
        default: return {};
        }
    }

    // Static name for a builtin id, nullptr when the name lives in a registry.
    [[nodiscard]] inline const char* shader_id_name_or_null(ShaderId id)
    {
        const std::string_view n = shader_id_builtin_name(id);
        return n.empty() ? nullptr : n.data();
    }

    [[nodiscard]] inline ShaderId parse_shader_id(std::string_view id)
    {
        if (id == "blinn_phong") return ShaderId::BlinnPhong;
        if (id == "pbr_metallic_roughness") return ShaderId::PbrMetallicRoughness;
        if (id == "lit_default") return ShaderId::LitDefault;
        if (id == "debug_view_albedo") return ShaderId::DebugViewAlbedo;
        if (id == "debug_view_normal") return ShaderId::DebugViewNormal;
        if (id == "debug_view_depth") return ShaderId::DebugViewDepth;
        if (id == "offscreen_pipeline") return ShaderId::OffscreenPipeline;
        return ShaderId::Unknown;
    }

    // A typed id the manifest may carry: a builtin identity, or an open id
    // minted by `ShaderIdRegistry`. `Unknown`, the `Count` sentinel and the
    // reserved top slot are never registerable.
    //
    // Back-compat law: for every value the *old* closed range could hold this is
    // character-for-character the previous predicate (`id != Unknown &&
    // raw < Count`); the only new acceptance is the open registered range.
    [[nodiscard]] constexpr bool shader_id_is_registerable(ShaderId id)
    {
        return shader_id_in_valid_range(id);
    }

    } // inline namespace render
} // namespace shs
