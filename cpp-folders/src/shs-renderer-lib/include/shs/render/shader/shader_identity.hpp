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

    Хамрах хүрээ: builtin identity + backend-blind resolution + realization-гүй
    тохиолдлын шууд refusal, БА open/consumer shader id (2026-09-18-нд
    нээгдсэн): `ShaderId` нь builtin диапазон + open registered диапазонтой
    болж, consumer нь `ShaderManifest::intern_shader` /
    `register_named_shader`-ээр НЭРээсээ id үүсгэнэ — core-ийн enum-д гар
    хүрэхгүй.

    Энэ файл нь umbrella: id ба range law нь `shader_id.hpp`, mint registry нь
    `shader_id_registry.hpp` (хоёулаа `renderpath/planning/pass_id.hpp` +
    `execution/pass_id_registry.hpp`-ийн shape); manifest нь тэднийг
    агуулна. Хуучин consumer-ууд энэ header-ээр бүгдийг харсаар байна.
*/

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <optional>
#include <string_view>

#include "shs/render/frame/backend_type.hpp"
#include "shs/render/shader/program.hpp"
#include "shs/render/shader/shader_id.hpp"
#include "shs/render/shader/shader_id_registry.hpp"

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
    // ShaderId, its range law and the canonical builtin names now live in
    // `shader_id.hpp` (mirroring planning/pass_id.hpp), with the open-range
    // mint registry in `shader_id_registry.hpp` (mirroring
    // execution/pass_id_registry.hpp). Both are included above, so every name
    // this header used to provide — `ShaderId`, `kShaderIdCount`,
    // `shader_id_is_registerable`, `shader_id_builtin_name` — is still visible
    // here, unchanged, for existing consumers.
    // ---------------------------------------------------------------------

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

        // --- open registered range (Constitution I §7; arch §4 req 6) --------
        UnregisteredOpenId, // an open-range id this manifest never minted (foreign or stale)
        NameCollision,      // two distinct names hashed to the same open slot; both refused
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
        case ShaderIdentityError::UnregisteredOpenId: return "unregistered_open_id";
        case ShaderIdentityError::NameCollision: return "name_collision";
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
    //
    // Open ids: the manifest OWNS a `ShaderIdRegistry` — the same shape as
    // `PassIdRegistry`, which `PassFactoryRegistry` owns — so a consumer mints
    // a shader identity from a name and registers a descriptor against it with
    // no core edit. Storage therefore has two halves: the fixed builtin array
    // (unchanged, so every builtin path keeps its exact previous behavior) and
    // the minted open slots.
    // ---------------------------------------------------------------------
    class ShaderManifest
    {
    public:
        struct Slot
        {
            bool registered = false;
            ShaderDesc desc{};
        };

        // A minted (open-range) identity slot, keyed by its content-addressed
        // offset. Lookup is a scan of typically a handful of entries, which is
        // what keeps the manifest copyable and comparable.
        //
        // Held in a std::deque on purpose: `ShaderBinding::desc` hands out a
        // POINTER into this storage, and deque guarantees that pushing further
        // slots never invalidates references to existing elements. A vector
        // would silently dangle an earlier binding when it reallocates.
        struct OpenSlot
        {
            uint16_t offset = 0u;
            ShaderDesc desc{};
        };

        // The fixed builtin array extent.
        [[nodiscard]] static constexpr std::size_t builtin_capacity() { return kShaderIdBuiltinCount; }

        // Total identity slots this manifest can hold: builtin + open range.
        [[nodiscard]] static constexpr std::size_t capacity()
        {
            return kShaderIdBuiltinCount + ShaderIdRegistry::capacity();
        }

        // Mint a consumer/demo-owned shader identity from its name. Total: a
        // builtin name resolves to its builtin id (a consumer can never shadow
        // a core shader), a repeat is idempotent, and an empty name, the
        // reserved open spelling or a real collision returns nullopt — see
        // `ShaderIdRegistry` for why the id is a pure function of the name.
        [[nodiscard]] std::optional<ShaderId> intern_shader(std::string_view name)
        {
            return ids_.intern(name);
        }

        // The owned mint registry (name resolution, enumeration, equality).
        [[nodiscard]] const ShaderIdRegistry& ids() const { return ids_; }

        // One-call consumer path: mint the name, then register its descriptor.
        // Named refusals: an empty name is NameMismatch, a real hash collision
        // is NameCollision, and everything `register_shader` refuses passes
        // through unchanged.
        [[nodiscard]] std::expected<ShaderId, ShaderIdentityError> register_named_shader(
            std::string_view name, const ShaderDesc& desc)
        {
            if (name.empty()) return std::unexpected(ShaderIdentityError::NameMismatch);

            const std::optional<ShaderId> id = ids_.intern(name);
            if (!id.has_value()) return std::unexpected(ShaderIdentityError::NameCollision);

            const std::expected<void, ShaderIdentityError> r = register_shader(*id, name, desc);
            if (!r) return std::unexpected(r.error());
            return *id;
        }

        [[nodiscard]] std::expected<void, ShaderIdentityError> register_shader(
            ShaderId id, std::string_view name, const ShaderDesc& desc)
        {
            // The id must be a real identity slot before anything else can be
            // judged about it: Unknown, the Count sentinel, the reserved gap
            // below the open range and the reserved top slot are hard misses,
            // exactly as before.
            if (!shader_id_in_valid_range(id)) return std::unexpected(ShaderIdentityError::UnknownShader);
            if (name.empty() || desc.name != name) return std::unexpected(ShaderIdentityError::NameMismatch);

            if (shader_id_is_open(id))
            {
                // Verified pairing against the OWNED registry: an open id this
                // manifest never minted — a foreign manifest's id, or a stale
                // one — is a hard miss, so a foreign id can never fill a slot
                // here (the same anti-aliasing property the pass-id registry
                // gets from `try_name`).
                const std::optional<std::string_view> minted = ids_.try_name(id);
                if (!minted.has_value() || *minted != name)
                    return std::unexpected(ShaderIdentityError::UnregisteredOpenId);

                const uint16_t offset = open_offset_of(id);
                for (const OpenSlot& s : open_slots_)
                {
                    if (s.offset == offset) return std::unexpected(ShaderIdentityError::AlreadyRegistered);
                }

                const std::expected<void, ShaderIdentityError> v = validate_desc(desc);
                if (!v) return v;

                open_slots_.push_back(OpenSlot{offset, desc});
                return {};
            }

            Slot& slot = slots_[static_cast<std::size_t>(id)];
            if (slot.registered) return std::unexpected(ShaderIdentityError::AlreadyRegistered);

            const std::expected<void, ShaderIdentityError> v = validate_desc(desc);
            if (!v) return v;

            slot.registered = true;
            slot.desc = desc;
            return {};
        }

        [[nodiscard]] bool has(ShaderId id) const { return desc_of(id) != nullptr; }

        [[nodiscard]] const ShaderDesc* get(ShaderId id) const { return desc_of(id); }

        // Name of a registered identity: the builtin table, or the name an open
        // id was minted from. nullopt for anything this manifest cannot resolve
        // — a foreign open id included.
        [[nodiscard]] std::optional<std::string_view> shader_name(ShaderId id) const
        {
            return ids_.try_name(id);
        }

        [[nodiscard]] std::size_t size() const
        {
            std::size_t n = open_slots_.size();
            for (const Slot& s : slots_) n += s.registered ? 1u : 0u;
            return n;
        }

        // Registered open identities (the builtin identities are the fixed
        // census `slots()` reports).
        [[nodiscard]] std::size_t open_count() const { return open_slots_.size(); }

        void clear()
        {
            slots_ = {};
            open_slots_.clear();
            ids_.clear();
        }

        [[nodiscard]] const Slot* slots() const { return slots_.data(); }

        // Backend-blind resolution. Declared entry points, when given, must
        // match the registered ones exactly — that is what preserves the
        // software realization's existing law: bound by entry name, and an
        // unknown name is a rejection, never a silent approximation.
        [[nodiscard]] std::expected<ShaderBinding, ShaderIdentityError> resolve(
            ShaderId id, RenderBackendType backend, const ShaderEntryPoints& declared = {}) const
        {
            const ShaderDesc* const p = desc_of(id);
            if (p == nullptr) return std::unexpected(ShaderIdentityError::UnknownShader);
            const ShaderDesc& d = *p;

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

        // Value semantics. Two manifests are equal iff they hold the same
        // registered descriptors AND the same minted names — a minted-but-
        // unregistered identity is observable through `shader_name`, so it
        // counts. Open slots compare by content-addressed offset, so the order
        // in which names were registered can never make two equal manifests
        // differ.
        //
        // Not constexpr, deliberately: the open half lives in
        // std::deque / std::string, so no invocation could constant-evaluate
        // this; claiming constexpr would be a lie the compiler may accept
        // silently.
        [[nodiscard]] bool operator==(const ShaderManifest& other) const
        {
            if (ids_ != other.ids_) return false;

            for (std::size_t i = 0; i < kShaderIdBuiltinCount; ++i)
            {
                if (slots_[i].registered != other.slots_[i].registered) return false;
                if (slots_[i].registered && slots_[i].desc != other.slots_[i].desc) return false;
            }

            if (open_slots_.size() != other.open_slots_.size()) return false;
            for (const OpenSlot& s : open_slots_)
            {
                bool found = false;
                for (const OpenSlot& t : other.open_slots_)
                {
                    if (t.offset != s.offset) continue;
                    found = (t.desc == s.desc);
                    break;
                }
                if (!found) return false;
            }
            return true;
        }

        [[nodiscard]] bool operator!=(const ShaderManifest& other) const { return !(*this == other); }

    private:
        // Descriptor invariants, shared by the builtin and the open
        // registration paths so the two cannot drift apart.
        [[nodiscard]] static std::expected<void, ShaderIdentityError> validate_desc(const ShaderDesc& desc)
        {
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
            return {};
        }

        // Offset of an open id inside the open range — the inverse of the
        // registry's `open_offset`, used to key the open slots.
        [[nodiscard]] static constexpr uint16_t open_offset_of(ShaderId id)
        {
            return static_cast<uint16_t>(static_cast<uint16_t>(id) - static_cast<uint16_t>(kShaderIdOpenBase));
        }

        // The one way to reach a descriptor: builtin array, or open slot, or
        // nullptr. Everything public that asks "is this registered" goes
        // through here, so neither half can be forgotten.
        [[nodiscard]] const ShaderDesc* desc_of(ShaderId id) const
        {
            if (shader_id_is_open(id))
            {
                const uint16_t offset = open_offset_of(id);
                for (const OpenSlot& s : open_slots_)
                {
                    if (s.offset == offset) return &s.desc;
                }
                return nullptr;
            }
            if (!shader_id_is_builtin(id)) return nullptr;

            const Slot& slot = slots_[static_cast<std::size_t>(id)];
            return slot.registered ? &slot.desc : nullptr;
        }

        std::array<Slot, kShaderIdBuiltinCount> slots_{};
        std::deque<OpenSlot> open_slots_{};
        ShaderIdRegistry ids_{};
    };
    } // inline namespace render
} // namespace shs

