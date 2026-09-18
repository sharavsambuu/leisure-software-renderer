#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: shader_identity.hpp
    МОДУЛЬ: shader
    ЗОРИЛГО: Shader identity as data (Slang plan P1.5). Нэг ShaderId нь нэг
            authored shader-ийг нэрлэнэ; backend бүр тэр identity-г өөрийн
            realization руу задална — software: C++ ShaderProgram factory,
            Vulkan: Slang module + entry point-ууд. Ингэснээр recipe нь
            backend-blind хэвээр үлдэж, realization-гүй backend чимээгүй
            ойролцоолохгүй, харин шууд татгалзана.

    Хамрах хүрээ (энэ slice): builtin identity + backend-blind resolution +
    realization-гүй тохиолдлын шууд refusal. Consumer/open shader id-ууд
    ЗОРИУД орхигдсон: энэ нь PassIdRegistry-ийн shape-ийг дахин ашиглах
    rule-of-two дараагийн алхам.
*/

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string_view>

#include "shs/render/frame/backend_type.hpp"
#include "shs/render/shader/program.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace render
    {
    // ---------------------------------------------------------------------
    // Canonical entry-point names — single source of truth.
    // The software realization and the authored Slang module both reference
    // these constants instead of repeating string literals, so "same entry"
    // is a name that can only be written once.
    // ---------------------------------------------------------------------
    inline constexpr std::string_view kShaderEntryVsMain = "vs_main";
    inline constexpr std::string_view kShaderEntryFsMain = "fs_main";
    inline constexpr std::string_view kShaderEntryVsUploaded = "vs_uploaded";

    struct ShaderEntryPoints
    {
        std::string_view vs{};
        std::string_view fs{};
        std::string_view cs{};

        [[nodiscard]] constexpr bool empty() const
        {
            return vs.empty() && fs.empty() && cs.empty();
        }
    };

    constexpr bool operator==(const ShaderEntryPoints& a, const ShaderEntryPoints& b)
    {
        return a.vs == b.vs && a.fs == b.fs && a.cs == b.cs;
    }

    constexpr bool operator!=(const ShaderEntryPoints& a, const ShaderEntryPoints& b)
    {
        return !(a == b);
    }

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

    inline constexpr std::size_t kShaderIdCount = static_cast<std::size_t>(ShaderId::Count);

    static_assert(static_cast<uint16_t>(ShaderId::Unknown) == 0, "Unknown must be 0");
    static_assert(static_cast<uint16_t>(ShaderId::OffscreenPipeline) <
                  static_cast<uint16_t>(ShaderId::Count), "builtin ids must stay below Count");

    [[nodiscard]] constexpr bool shader_id_is_registerable(ShaderId id)
    {
        return id != ShaderId::Unknown && static_cast<uint16_t>(id) < kShaderIdCount;
    }

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

    // ---------------------------------------------------------------------
    // Realization masks: which backends this identity is realized on.
    // A bit that is absent is a hard refusal, never a fallback. This is where
    // the OpenGL seat tells the truth: today it has zero realizations, so an
    // OpenGL resolve reports BackendNotRealized instead of silently running
    // the software program.
    // ---------------------------------------------------------------------
    [[nodiscard]] constexpr uint8_t shader_backend_bit(RenderBackendType backend)
    {
        switch (backend)
        {
        case RenderBackendType::Software: return 1u << 0;
        case RenderBackendType::Vulkan: return 1u << 1;
        case RenderBackendType::OpenGL: return 1u << 2;
        default: return 0u;
        }
    }

    inline constexpr uint8_t kShaderRealizationNone = 0u;
    inline constexpr uint8_t kShaderRealizationSoftware = 1u << 0;
    inline constexpr uint8_t kShaderRealizationVulkan = 1u << 1;
    inline constexpr uint8_t kShaderRealizationOpenGL = 1u << 2;

    // The CPU realization of a shader: a factory for the *type-erased* program.
    // Hot paths keep using the concrete (non-erased) ShaderProgramFn factories
    // so the software rasterizer can still inline the per-pixel fragment call
    // (R1); this seam exists so identity, resolution and tests have one handle.
    using ShaderCppImplFn = ShaderProgram (*)();

    // Wraps a concrete ShaderProgramFn as a ShaderProgram without restating any
    // shader body: the two forms cannot drift because one is built from the
    // other.
    template <typename Fn>
    [[nodiscard]] inline ShaderProgram make_erased_program(const Fn& fn)
    {
        return ShaderProgram{fn.vs, fn.fs};
    }

    struct ShaderDesc
    {
        std::string_view name{};            // canonical identity name
        ShaderEntryPoints entries{};        // authored entry points
        std::string_view module{};          // authored module stem; empty = no GPU realization
        uint8_t realization_mask = kShaderRealizationNone;
        ShaderCppImplFn cpp_impl = nullptr; // required iff the software bit is set
    };

    [[nodiscard]] constexpr bool shader_desc_realizes(const ShaderDesc& d, RenderBackendType backend)
    {
        const uint8_t bit = shader_backend_bit(backend);
        return bit != 0u && (d.realization_mask & bit) != 0u;
    }

    constexpr bool operator==(const ShaderDesc& a, const ShaderDesc& b)
    {
        return a.name == b.name && a.entries == b.entries && a.module == b.module &&
               a.realization_mask == b.realization_mask && a.cpp_impl == b.cpp_impl;
    }

    // ---------------------------------------------------------------------
    // Closed error vocabulary. Every failure mode is named, so a refusal can
    // never be mistaken for a silent approximation.
    // ---------------------------------------------------------------------
    enum class ShaderIdentityError : uint8_t
    {
        None = 0,
        UnknownShader,      // id was never registered (foreign id, or Unknown)
        NameMismatch,       // the (id, name) registration pair contradicts the descriptor
        AlreadyRegistered,  // duplicate registration refused
        NoRealization,      // descriptor declares no backend at all
        BackendNotRealized, // backend has no realization for this identity (never approximated)
        EntryPointMismatch, // caller declared entry points that differ from the registered ones
        MissingCppImpl,     // software realization declared without a program factory
        MissingModule,      // GPU realization declared without an authored module
        MissingEntryPoints, // declared realization without entry points
    };

    [[nodiscard]] constexpr std::string_view shader_identity_error_name(ShaderIdentityError e)
    {
        switch (e)
        {
        case ShaderIdentityError::None: return "none";
        case ShaderIdentityError::UnknownShader: return "unknown_shader";
        case ShaderIdentityError::NameMismatch: return "name_mismatch";
        case ShaderIdentityError::AlreadyRegistered: return "already_registered";
        case ShaderIdentityError::NoRealization: return "no_realization";
        case ShaderIdentityError::BackendNotRealized: return "backend_not_realized";
        case ShaderIdentityError::EntryPointMismatch: return "entry_point_mismatch";
        case ShaderIdentityError::MissingCppImpl: return "missing_cpp_impl";
        case ShaderIdentityError::MissingModule: return "missing_module";
        case ShaderIdentityError::MissingEntryPoints: return "missing_entry_points";
        default: return "unknown_error";
        }
    }

    // A resolved identity for one backend. Exactly one side is populated:
    //   software -> program (has_program), no module
    //   GPU      -> module + entries, no program
    // Callers cannot accidentally use the wrong side: has_program is the switch.
    struct ShaderBinding
    {
        ShaderId id = ShaderId::Unknown;
        RenderBackendType backend = RenderBackendType::Software;
        const ShaderDesc* desc = nullptr;
        std::string_view module{};
        ShaderEntryPoints entries{};
        ShaderProgram program{};
        bool has_program = false;
    };

    constexpr bool operator!=(const ShaderDesc& a, const ShaderDesc& b)
    {
        return !(a == b);
    }

    // ---------------------------------------------------------------------
    // ShaderManifest — caller-owned, explicit, copyable, comparable.
    // Same shape as the pass-id registry: no ambient global state (the
    // determinism gates forbid hidden registries), every refusal loud, and
    // registration is *verified pairing* — an id and a name must agree with
    // the descriptor they are registered under, so a foreign or colliding
    // identity can never map onto a builtin slot.
    // ---------------------------------------------------------------------
    class ShaderManifest
    {
    public:
        struct Slot
        {
            bool registered = false;
            ShaderDesc desc{};
        };

        [[nodiscard]] std::expected<void, ShaderIdentityError> register_shader(
            ShaderId id, std::string_view name, const ShaderDesc& desc)
        {
            if (!shader_id_is_registerable(id)) return std::unexpected(ShaderIdentityError::UnknownShader);
            if (name.empty() || desc.name != name) return std::unexpected(ShaderIdentityError::NameMismatch);

            Slot& slot = slots_[static_cast<std::size_t>(id)];
            if (slot.registered) return std::unexpected(ShaderIdentityError::AlreadyRegistered);
            if (desc.realization_mask == kShaderRealizationNone)
                return std::unexpected(ShaderIdentityError::NoRealization);
            if ((desc.realization_mask & kShaderRealizationSoftware) != 0u && desc.cpp_impl == nullptr)
                return std::unexpected(ShaderIdentityError::MissingCppImpl);

            const uint8_t gpu_bits = static_cast<uint8_t>(kShaderRealizationVulkan | kShaderRealizationOpenGL);
            if ((desc.realization_mask & gpu_bits) != 0u)
            {
                if (desc.module.empty()) return std::unexpected(ShaderIdentityError::MissingModule);
                if (desc.entries.empty()) return std::unexpected(ShaderIdentityError::MissingEntryPoints);
            }

            slot.registered = true;
            slot.desc = desc;
            return {};
        }

        [[nodiscard]] bool has(ShaderId id) const
        {
            return shader_id_is_registerable(id) && slots_[static_cast<std::size_t>(id)].registered;
        }

        [[nodiscard]] const ShaderDesc* get(ShaderId id) const
        {
            return has(id) ? &slots_[static_cast<std::size_t>(id)].desc : nullptr;
        }

        [[nodiscard]] std::size_t size() const
        {
            std::size_t n = 0;
            for (const Slot& s : slots_) n += s.registered ? 1u : 0u;
            return n;
        }

        void clear() { slots_ = {}; }

        [[nodiscard]] const Slot* slots() const { return slots_.data(); }

        // Backend-blind resolution. Declared entry points, when given, must
        // match the registered ones exactly — that is what preserves the
        // software realization's existing law: bound by entry name, and an
        // unknown name is a rejection, never a silent approximation.
        [[nodiscard]] std::expected<ShaderBinding, ShaderIdentityError> resolve(
            ShaderId id, RenderBackendType backend, const ShaderEntryPoints& declared = {}) const
        {
            if (!has(id)) return std::unexpected(ShaderIdentityError::UnknownShader);
            const ShaderDesc& d = slots_[static_cast<std::size_t>(id)].desc;

            // An absent realization bit is a refusal, never a fallback: this is
            // exactly where an OpenGL resolve stops instead of quietly running
            // the software program.
            if (!shader_desc_realizes(d, backend))
                return std::unexpected(ShaderIdentityError::BackendNotRealized);
            if (!declared.empty() && declared != d.entries)
                return std::unexpected(ShaderIdentityError::EntryPointMismatch);

            ShaderBinding b{};
            b.id = id;
            b.backend = backend;
            b.desc = &d;
            if (backend == RenderBackendType::Software)
            {
                b.program = d.cpp_impl();
                b.has_program = true;
            }
            else
            {
                b.module = d.module;
                b.entries = d.entries;
            }
            return b;
        }

        [[nodiscard]] static constexpr std::size_t capacity() { return kShaderIdCount; }

        constexpr bool operator==(const ShaderManifest& other) const
        {
            for (std::size_t i = 0; i < kShaderIdCount; ++i)
            {
                if (slots_[i].registered != other.slots_[i].registered) return false;
                if (slots_[i].registered && slots_[i].desc != other.slots_[i].desc) return false;
            }
            return true;
        }

        constexpr bool operator!=(const ShaderManifest& other) const { return !(*this == other); }

    private:
        std::array<Slot, kShaderIdCount> slots_{};
    };
    } // inline namespace render
} // namespace shs

