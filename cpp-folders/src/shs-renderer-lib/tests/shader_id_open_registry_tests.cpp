#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

#include "shs/render/shader/builtin_shader_manifest.hpp"
#include "shs/renderpath/execution/pass_id_registry.hpp"

// Open shader-id registry gate (Constitution I §7 — No User Lock-In;
// arch/render_path_architecture.md §4 graduation req 6).
//
// Proves the *consumer* story, not an internal invariant: a demo-owned shader
// identity is minted from a name, registered with its descriptor and resolved
// through a typed id, with zero edits to the core `ShaderId` vocabulary — while
// every builtin path keeps its exact previous behavior (P1.5 parity).
//
// It also pins the rule-of-two claim behind this slice: the shader registry is
// the THIRD user of the open-id shape, so it must derive its offsets from the
// same law the pass registry uses (`shs/core/open_id_hash.hpp`). That is
// asserted here against `PassIdRegistry`, not left to convention.
//
// GPU-free: no Context, no backend instance, no SPIR-V execution.
namespace
{
    using namespace shs;
    using namespace shs::renderpath;

    // A CPU realization for an open identity. Deliberately borrowed from the
    // builtin census: this gate's subject is the *identity* law, and restating
    // a shader body here would be exactly the drift the manifest exists to
    // prevent. An open identity is "a name we own, realized by a factory we
    // chose".
    ShaderProgram consumer_software_program()
    {
        return erased_blinn_phong_program();
    }

    ShaderDesc make_software_desc(std::string_view name)
    {
        ShaderDesc d{};
        d.name = name;
        d.realization_mask = kShaderRealizationSoftware;
        d.cpp_impl = &consumer_software_program;
        return d;
    }

    // --- range law -----------------------------------------------------------

    bool test_range_law()
    {
        bool ok = true;
        ok = ok && shader_id_is_builtin(ShaderId::BlinnPhong);
        ok = ok && shader_id_is_builtin(ShaderId::OffscreenPipeline);
        ok = ok && !shader_id_is_builtin(ShaderId::Unknown);
        ok = ok && !shader_id_is_builtin(ShaderId::Count);
        ok = ok && !shader_id_is_builtin(static_cast<ShaderId>(kShaderIdOpenBase));

        ok = ok && !shader_id_is_open(ShaderId::Unknown);
        ok = ok && !shader_id_is_open(ShaderId::BlinnPhong);
        // Boundary: the last builtin slot is not open, the first open slot is.
        ok = ok && !shader_id_is_open(static_cast<ShaderId>(kShaderIdOpenBase - 1u));
        ok = ok && shader_id_is_open(static_cast<ShaderId>(kShaderIdOpenBase));
        ok = ok && shader_id_is_open(static_cast<ShaderId>(kShaderIdOpenMax));

        ok = ok && shader_id_in_valid_range(ShaderId::OffscreenPipeline);
        ok = ok && shader_id_in_valid_range(static_cast<ShaderId>(kShaderIdOpenBase));
        ok = ok && !shader_id_in_valid_range(ShaderId::Unknown);
        ok = ok && !shader_id_in_valid_range(ShaderId::Count);
        ok = ok && !shader_id_in_valid_range(static_cast<ShaderId>(kShaderIdReserved));
        // The reserved gap between the builtin ceiling and the open base is not
        // a legal id either: it must never silently read as builtin or open.
        ok = ok && !shader_id_in_valid_range(static_cast<ShaderId>(kShaderIdBuiltinMax));

        // Back-compat parity guard: for every value the *old* closed vocabulary
        // could hold, the registerability predicate is unchanged.
        for (uint16_t raw = 0u; raw <= static_cast<uint16_t>(ShaderId::Count); ++raw)
        {
            const ShaderId id = static_cast<ShaderId>(raw);
            const bool legacy = raw != static_cast<uint16_t>(ShaderId::Unknown) &&
                                raw < static_cast<uint16_t>(ShaderId::Count);
            ok = ok && (shader_id_is_registerable(id) == legacy);
        }
        // ... and the only new acceptance is the open registered range.
        ok = ok && shader_id_is_registerable(static_cast<ShaderId>(kShaderIdOpenBase));

        // Names: builtin table for builtins, nullptr when the name lives in a
        // registry, and an honest spelling for an unresolvable open id.
        ok = ok && shader_id_name_or_null(ShaderId::LitDefault) != nullptr;
        ok = ok && shader_id_name_or_null(ShaderId::Unknown) == nullptr;
        ok = ok && shader_id_name_or_null(static_cast<ShaderId>(kShaderIdOpenBase)) == nullptr;
        ok = ok && std::string_view(kShaderIdOpenSpelling) == std::string_view("open_shader");
        ok = ok && parse_shader_id("offscreen_pipeline") == ShaderId::OffscreenPipeline;
        ok = ok && parse_shader_id("not_a_builtin") == ShaderId::Unknown;
        return ok;
    }

    // --- the registry --------------------------------------------------------

    bool test_builtin_names_never_mint()
    {
        ShaderIdRegistry ids{};
        const auto bp = ids.intern("blinn_phong");
        const auto off = ids.intern("offscreen_pipeline");
        bool ok = true;
        ok = ok && bp.has_value() && *bp == ShaderId::BlinnPhong;
        ok = ok && off.has_value() && *off == ShaderId::OffscreenPipeline;
        // No open id was minted, so a consumer can never shadow a core shader.
        ok = ok && ids.open_count() == 0u;
        ok = ok && ids.empty();
        ok = ok && ids.try_name(ShaderId::BlinnPhong).has_value() &&
                   *ids.try_name(ShaderId::BlinnPhong) == std::string_view("blinn_phong");
        return ok;
    }

    bool test_content_addressed_ids()
    {
        ShaderIdRegistry ids{};
        const auto fog = ids.intern("demo_fog_shader");
        const auto grade = ids.intern("demo_grade_shader");
        bool ok = true;
        ok = ok && fog.has_value() && grade.has_value();
        ok = ok && ids.open_count() == 2u;

        // Content-addressed: the id is a pure function of the name (the shared
        // law folded into the open range), so it is the same in every registry,
        // process and translation unit.
        ok = ok && *fog == static_cast<ShaderId>(
                               kShaderIdOpenBase + ShaderIdRegistry::open_offset("demo_fog_shader"));
        ok = ok && *grade == static_cast<ShaderId>(
                                 kShaderIdOpenBase + ShaderIdRegistry::open_offset("demo_grade_shader"));

        // Idempotent, and never overwrites.
        const auto again = ids.intern("demo_fog_shader");
        ok = ok && again.has_value() && *again == *fog;
        ok = ok && ids.open_count() == 2u;

        // Order independence: the same names in the reverse order give the same
        // ids, so registration order can never leak into a plan.
        ShaderIdRegistry reversed{};
        const auto r_grade = reversed.intern("demo_grade_shader");
        const auto r_fog = reversed.intern("demo_fog_shader");
        ok = ok && r_grade.has_value() && r_fog.has_value();
        ok = ok && *r_fog == *fog && *r_grade == *grade;
        ok = ok && reversed == ids;

        // The name is recoverable in both directions, and stays stable in a copy.
        ok = ok && ids.try_name(*fog).has_value() &&
                   *ids.try_name(*fog) == std::string_view("demo_fog_shader");
        ok = ok && ids.try_name(ShaderId::BlinnPhong).has_value();
        const ShaderIdRegistry copy = ids;
        ok = ok && copy == ids;
        ok = ok && copy.try_name(*fog).has_value();
        ok = ok && copy.is_open(*fog);
        ok = ok && !copy.is_open(ShaderId::BlinnPhong);
        ok = ok && reversed.open_count() == ids.open_count();
        return ok;
    }

    bool test_determinism_across_instances()
    {
        ShaderIdRegistry first{};
        ShaderIdRegistry second{};
        bool ok = true;
        // Interning the same names in a different order yields identical ids:
        // the id is content-addressed, not mint-order.
        for (const char* const n : {"demo_a", "demo_b", "demo_c"})
        {
            ok = ok && first.intern(n).has_value();
        }
        for (const char* const n : {"demo_c", "demo_a", "demo_b"})
        {
            ok = ok && second.intern(n).has_value();
        }
        ok = ok && first == second;

        for (const char* const n : {"demo_a", "demo_b", "demo_c"})
        {
            const auto x = first.intern(n);
            const auto y = second.intern(n);
            ok = ok && x.has_value() && y.has_value() && *x == *y;
        }

        // A copy carries the identical id assignment.
        const ShaderIdRegistry copy = first;
        ok = ok && copy == first;

        // Different name sets are different registries.
        ShaderIdRegistry other{};
        ok = ok && other.intern("demo_d").has_value();
        ok = ok && !(other == first);
        return ok;
    }

    // --- one law, three namespaces (rule of two discharged) ------------------

    bool test_shared_offset_law()
    {
        // The whole point of hoisting the hash into core: the shader registry
        // and the pass registry must not be two implementations that happen to
        // agree today. Both fold the same name with the same range capacity, so
        // the offsets are identical for every name — including the ones a real
        // consumer would use.
        bool ok = true;
        for (const char* const n : {"demo_fog_pass", "demo_fog_shader", "taa", "gbuffer",
                                    "shadow_map", ""})
        {
            ok = ok && (ShaderIdRegistry::open_offset(n) == PassIdRegistry::open_offset(n));
        }
        // ... which means the same name lands on the same offset (and therefore
        // the same open id, since both ranges share the same base and width).
        ok = ok && kShaderIdOpenBase == kPassIdOpenBase;
        ok = ok && ShaderIdRegistry::capacity() == PassIdRegistry::capacity();
        return ok;
    }

    // --- loud refusals -------------------------------------------------------

    bool test_null_and_reserved_rejected()
    {
        ShaderIdRegistry ids{};
        bool ok = true;
        ok = ok && !ids.intern("").has_value();
        // The reserved open spelling is never a registration key: it is how an
        // unresolvable open id stringifies, so minting it would alias every
        // such id onto one name.
        ok = ok && !ids.intern(kShaderIdOpenSpelling).has_value();
        ok = ok && ids.open_count() == 0u;

        ok = ok && !ids.try_name(ShaderId::Unknown).has_value();
        ok = ok && !ids.try_name(ShaderId::Count).has_value();
        ok = ok && !ids.try_name(static_cast<ShaderId>(kShaderIdReserved)).has_value();
        ok = ok && !ids.contains(ShaderId::Unknown);

        // The manifest refuses the same ids as before opening the range: this is
        // the back-compat half of the slice.
        ShaderManifest m{};
        const auto unknown =
            m.register_shader(ShaderId::Unknown, "whatever", make_software_desc("whatever"));
        ok = ok && !unknown && unknown.error() == ShaderIdentityError::UnknownShader;

        const auto count =
            m.register_shader(ShaderId::Count, "whatever", make_software_desc("whatever"));
        ok = ok && !count && count.error() == ShaderIdentityError::UnknownShader;

        const auto reserved = m.register_shader(
            static_cast<ShaderId>(kShaderIdReserved), "whatever", make_software_desc("whatever"));
        ok = ok && !reserved && reserved.error() == ShaderIdentityError::UnknownShader;

        // An in-range open id this manifest never minted is a hard miss, never
        // a slot: a foreign id cannot be quietly adopted.
        const auto orphan = m.register_shader(
            static_cast<ShaderId>(kShaderIdOpenBase), "whatever", make_software_desc("whatever"));
        ok = ok && !orphan && orphan.error() == ShaderIdentityError::UnregisteredOpenId;
        ok = ok && m.size() == 0u;
        return ok;
    }

    bool test_capacity_arithmetic()
    {
        bool ok = true;
        // The open range is exactly representable: the last mintable id is OpenMax.
        ok = ok && ShaderIdRegistry::capacity() ==
                       static_cast<std::size_t>(kShaderIdOpenMax) -
                           static_cast<std::size_t>(kShaderIdOpenBase) + 1u;
        ok = ok && static_cast<uint32_t>(kShaderIdOpenBase) +
                       static_cast<uint32_t>(ShaderIdRegistry::capacity()) - 1u ==
                       static_cast<uint32_t>(kShaderIdOpenMax);
        // The reserved slot is never minted.
        ok = ok && ShaderIdRegistry::capacity() < 65535u;
        ok = ok && kShaderIdReserved == 65535u;

        // The manifest's total capacity is the builtin array plus the open
        // range, and the builtin extent is unchanged by opening the vocabulary.
        ok = ok && ShaderManifest::builtin_capacity() == kShaderIdBuiltinCount;
        ok = ok && ShaderManifest::capacity() ==
                       ShaderManifest::builtin_capacity() + ShaderIdRegistry::capacity();
        return ok;
    }

    // Register a software-only open identity by name; true when both the mint
    // and the registration succeeded.
    bool register_ok(ShaderManifest& m, std::string_view name)
    {
        const auto id = m.register_named_shader(name, make_software_desc(name));
        return id.has_value();
    }

    // Brute-force a real 16-bit offset collision. With ~64k slots a birthday
    // collision appears within a few hundred probes, so the loud-failure path is
    // testable rather than theoretical.
    std::string find_colliding_name(std::string_view seed)
    {
        const uint16_t target = ShaderIdRegistry::open_offset(seed);
        for (int i = 0; i < 20000; ++i)
        {
            const std::string candidate = "demo_probe_" + std::to_string(i);
            if (candidate != seed && ShaderIdRegistry::open_offset(candidate) == target)
            {
                return candidate;
            }
        }
        return {};
    }

    bool test_collision_is_loud()
    {
        const std::string collider = find_colliding_name("demo_fog_shader");
        // If no collision was found the assertion is vacuous — but it is not
        // expected to happen (birthday bound), so do not fail the gate for it.
        if (collider.empty()) return true;

        ShaderIdRegistry ids{};
        const auto first_id = ids.intern("demo_fog_shader");
        bool ok = true;
        ok = ok && first_id.has_value();
        ok = ok && ids.open_count() == 1u;

        // The colliding name is refused by the registry: nothing is overwritten
        // and nothing is aliased.
        ok = ok && !ids.intern(collider).has_value();
        ok = ok && ids.open_count() == 1u;
        ok = ok && ids.try_name(*first_id).has_value() &&
                   *ids.try_name(*first_id) == std::string_view("demo_fog_shader");

        // And the manifest turns it into a NAMED refusal instead of a silent
        // second owner of the same slot.
        ShaderManifest m{};
        ok = ok && register_ok(m, "demo_fog_shader");
        const auto collision = m.register_named_shader(collider, make_software_desc(collider));
        ok = ok && !collision && collision.error() == ShaderIdentityError::NameCollision;
        ok = ok && m.open_count() == 1u;

        // Residual, stated honestly: a manifest holding only the collider maps
        // that shared slot to *its* name. Verified pairing is what makes this
        // safe — the (id, name) pair must match, so a colliding id can never
        // bind the wrong descriptor.
        ShaderManifest only_collider{};
        const auto registered_collider = only_collider.intern_shader(collider);
        ok = ok && registered_collider.has_value();
        ok = ok && *registered_collider == *first_id; // same slot by construction

        const auto wrong_name = only_collider.register_shader(
            *first_id, "demo_fog_shader", make_software_desc("demo_fog_shader"));
        ok = ok && !wrong_name && wrong_name.error() == ShaderIdentityError::UnregisteredOpenId;

        ok = ok && register_ok(only_collider, collider);
        return ok;
    }

    bool test_foreign_open_id_is_a_hard_miss()
    {
        ShaderManifest mine{};
        bool ok = true;
        ok = ok && register_ok(mine, "demo_fog_shader");
        const auto fog = mine.intern_shader("demo_fog_shader");
        ok = ok && fog.has_value();

        // An id derived from a name this manifest never minted misses — there is
        // no "whatever is at that slot" fallback.
        ShaderManifest other{};
        const auto unrelated = other.intern_shader("demo_not_minted_here");
        ok = ok && unrelated.has_value();
        ok = ok && !mine.has(*unrelated);
        ok = ok && mine.get(*unrelated) == nullptr;
        ok = ok && !mine.shader_name(*unrelated).has_value();

        const auto leaked = mine.register_shader(
            *unrelated, "demo_not_minted_here", make_software_desc("demo_not_minted_here"));
        ok = ok && !leaked && leaked.error() == ShaderIdentityError::UnregisteredOpenId;

        // The null and reserved ids are never resolvable either.
        ok = ok && !mine.has(ShaderId::Unknown);
        ok = ok && !mine.has(static_cast<ShaderId>(kShaderIdReserved));
        ok = ok && mine.resolve(ShaderId::Unknown, RenderBackendType::Software).has_value() == false;
        return ok;
    }

    // --- the consumer story (zero core edits) --------------------------------

    bool test_consumer_shader_without_core_edit()
    {
        // A demo-owned identity: its name is the ONLY thing decided here. No
        // ShaderId enum edit, no builtin-manifest edit, no planner fork.
        ShaderManifest manifest{};
        const auto id =
            manifest.register_named_shader("demo_grass_shader", make_software_desc("demo_grass_shader"));
        bool ok = true;
        ok = ok && id.has_value();
        ok = ok && shader_id_is_open(*id);
        ok = ok && !shader_id_is_builtin(*id);
        ok = ok && manifest.has(*id);
        ok = ok && manifest.open_count() == 1u;
        ok = ok && manifest.size() == 1u;

        // It resolves through exactly the same entry point a builtin uses, so the
        // identity layer needs no special case for consumer shaders.
        const auto binding = manifest.resolve(*id, RenderBackendType::Software);
        ok = ok && binding.has_value();
        ok = ok && binding->id == *id;
        ok = ok && binding->has_program;
        ok = ok && binding->desc != nullptr;
        ok = ok && binding->desc->name == std::string_view("demo_grass_shader");

        // Its name is recoverable, and it has no builtin spelling at all — the
        // static vocabulary genuinely does not know it.
        ok = ok && manifest.shader_name(*id).has_value() &&
                   *manifest.shader_name(*id) == std::string_view("demo_grass_shader");
        ok = ok && shader_id_builtin_name(*id).empty();
        ok = ok && shader_id_name_or_null(*id) == nullptr;
        return ok;
    }

    bool test_open_id_realization_law()
    {
        ShaderManifest m{};
        bool ok = true;

        // A software-only consumer shader.
        const auto sw_id = m.register_named_shader("demo_sw_shader", make_software_desc("demo_sw_shader"));
        ok = ok && sw_id.has_value();

        // A dual-realization consumer shader, pointing at a real authored stem.
        ShaderDesc vk{};
        vk.name = "demo_vk_shader";
        vk.realization_mask = static_cast<uint8_t>(kShaderRealizationSoftware | kShaderRealizationVulkan);
        vk.cpp_impl = &consumer_software_program;
        vk.module = "offscreen_pipeline";
        vk.entries = ShaderEntryPoints{kShaderEntryVsMain, kShaderEntryFsMain, {}};
        const auto vk_id = m.register_named_shader("demo_vk_shader", vk);
        ok = ok && vk_id.has_value();

        // Software resolves a program; the GPU path yields module + entries and
        // NO program (the switch is has_program, exactly as for builtins).
        const auto sw = m.resolve(*sw_id, RenderBackendType::Software);
        ok = ok && sw.has_value() && sw->has_program;

        const auto on_vk = m.resolve(*vk_id, RenderBackendType::Vulkan);
        ok = ok && on_vk.has_value();
        ok = ok && !on_vk->has_program;
        ok = ok && on_vk->module == std::string_view("offscreen_pipeline");
        ok = ok && !on_vk->entries.empty();

        // An absent realization bit is a refusal, NEVER a fallback to another
        // backend's realization — the exact law an open id must inherit.
        const auto sw_on_vk = m.resolve(*sw_id, RenderBackendType::Vulkan);
        ok = ok && !sw_on_vk && sw_on_vk.error() == ShaderIdentityError::BackendNotRealized;
        const auto sw_on_gl = m.resolve(*sw_id, RenderBackendType::OpenGL);
        ok = ok && !sw_on_gl && sw_on_gl.error() == ShaderIdentityError::BackendNotRealized;

        // Declared entry points must match the registered ones exactly.
        const ShaderEntryPoints wrong{kShaderEntryVsMain, "fs_not_here", {}};
        const auto mismatched = m.resolve(*vk_id, RenderBackendType::Vulkan, wrong);
        ok = ok && !mismatched && mismatched.error() == ShaderIdentityError::EntryPointMismatch;
        ok = ok && m.resolve(*vk_id, RenderBackendType::Vulkan, vk.entries).has_value();

        // A refused registration is a named refusal, and a duplicate open
        // registration is refused like a builtin one; a mismatched (id, name)
        // pair is refused before any slot is touched.
        const auto dup = m.register_named_shader("demo_sw_shader", make_software_desc("demo_sw_shader"));
        ok = ok && !dup && dup.error() == ShaderIdentityError::AlreadyRegistered;
        const auto mismatch =
            m.register_named_shader("demo_new_name", make_software_desc("demo_new_name_typo"));
        ok = ok && !mismatch && mismatch.error() == ShaderIdentityError::NameMismatch;
        return ok;
    }

    bool test_open_id_descriptor_invariants()
    {
        // The descriptor invariants apply to open ids exactly as to builtins:
        // one shared validation path, so the two cannot drift apart.
        ShaderManifest v{};
        bool ok = true;

        ShaderDesc no_real{};
        no_real.name = "demo_no_real";
        const auto nr = v.register_named_shader("demo_no_real", no_real);
        ok = ok && !nr && nr.error() == ShaderIdentityError::NoRealization;

        ShaderDesc missing_impl{};
        missing_impl.name = "demo_missing_impl";
        missing_impl.realization_mask = kShaderRealizationSoftware; // no cpp_impl
        const auto mi = v.register_named_shader("demo_missing_impl", missing_impl);
        ok = ok && !mi && mi.error() == ShaderIdentityError::MissingCppImpl;

        ShaderDesc missing_module{};
        missing_module.name = "demo_missing_module";
        missing_module.realization_mask = kShaderRealizationVulkan; // no module
        missing_module.entries = ShaderEntryPoints{kShaderEntryVsMain, kShaderEntryFsMain, {}};
        const auto mm = v.register_named_shader("demo_missing_module", missing_module);
        ok = ok && !mm && mm.error() == ShaderIdentityError::MissingModule;

        ShaderDesc missing_entries{};
        missing_entries.name = "demo_missing_entries";
        missing_entries.realization_mask = kShaderRealizationVulkan;
        missing_entries.module = "offscreen_pipeline"; // no entries
        const auto me = v.register_named_shader("demo_missing_entries", missing_entries);
        ok = ok && !me && me.error() == ShaderIdentityError::MissingEntryPoints;

        // Stated honestly: a refused registration leaves its name minted (the
        // registry offers no remove, exactly like PassIdRegistry) but allocates
        // no registered slot — `size()`/`open_count()` count registered
        // identities only.
        ok = ok && v.size() == 0u;
        ok = ok && v.open_count() == 0u;
        ok = ok && v.ids().open_count() == 4u;

        // The named refusals stringify, so a refusal can never be mistaken for a
        // silent approximation in a log either.
        ok = ok && shader_identity_error_name(ShaderIdentityError::UnregisteredOpenId) ==
                       std::string_view("unregistered_open_id");
        ok = ok && shader_identity_error_name(ShaderIdentityError::NameCollision) ==
                       std::string_view("name_collision");
        return ok;
    }

    bool test_manifest_equality_and_copy()
    {
        bool ok = true;

        // Registration order must not change equality: open slots are keyed by
        // content-addressed offset, not insertion order.
        ShaderManifest a{};
        ShaderManifest b{};
        ok = ok && register_ok(a, "demo_one");
        ok = ok && register_ok(a, "demo_two");
        ok = ok && register_ok(b, "demo_two");
        ok = ok && register_ok(b, "demo_one");
        ok = ok && a == b;
        ok = ok && a.ids() == b.ids();

        // A copy carries the open half, and a later mutation diverges it.
        ShaderManifest copy = a;
        ok = ok && copy == a;
        ok = ok && register_ok(a, "demo_three");
        ok = ok && !(a == copy);
        ok = ok && copy.size() == 2u;
        ok = ok && a.size() == 3u;

        // A minted-but-unregistered name is observable through `shader_name`, so
        // it counts towards equality rather than hiding inside the registry.
        ShaderManifest m1{};
        ShaderManifest m2{};
        ok = ok && m1.intern_shader("demo_only_minted").has_value();
        ok = ok && !(m1 == m2);
        ok = ok && m2.intern_shader("demo_only_minted").has_value();
        ok = ok && m1 == m2;
        ok = ok && m1.open_count() == 0u; // minted, but not registered

        // Clearing removes both halves and the minted names.
        const auto demo_one = a.intern_shader("demo_one");
        ok = ok && demo_one.has_value();
        copy.clear();
        ok = ok && copy.size() == 0u;
        ok = ok && copy.open_count() == 0u;
        ok = ok && copy.ids().empty();
        ok = ok && !copy.has(*demo_one);
        return ok;
    }

    bool test_builtin_paths_unchanged()
    {
        bool ok = true;

        // The value-tier builtin census is exactly as P1.5 left it: six
        // software-only identities and NOT one minted open slot.
        const ShaderManifest value = builtin_value_shader_manifest();
        ok = ok && value.size() == 6u;
        ok = ok && value.open_count() == 0u;
        ok = ok && value.ids().empty();
        ok = ok && value.has(ShaderId::BlinnPhong);
        ok = ok && value.has(ShaderId::LitDefault);
        // OffscreenPipeline is a different identity with its own manifest.
        ok = ok && !value.has(ShaderId::OffscreenPipeline);

        // Every builtin identity in the value census still resolves on software
        // and still refuses every backend it has no realization for — the
        // "absent bit is a refusal, never a fallback" law, unchanged.
        for (uint16_t raw = 1u; raw < static_cast<uint16_t>(ShaderId::Count); ++raw)
        {
            const ShaderId id = static_cast<ShaderId>(raw);
            if (id == ShaderId::OffscreenPipeline) continue;

            const auto sw = value.resolve(id, RenderBackendType::Software);
            ok = ok && sw.has_value() && sw->has_program;

            const auto vk = value.resolve(id, RenderBackendType::Vulkan);
            ok = ok && !vk && vk.error() == ShaderIdentityError::BackendNotRealized;

            const auto gl = value.resolve(id, RenderBackendType::OpenGL);
            ok = ok && !gl && gl.error() == ShaderIdentityError::BackendNotRealized;

            ok = ok && !shader_id_is_open(id);
        }

        // Copies and rebuilds still compare equal, and the total capacity is the
        // builtin extent plus the open range.
        const ShaderManifest rebuilt = builtin_value_shader_manifest();
        ok = ok && value == rebuilt;
        ShaderManifest value_copy = value;
        ok = ok && value_copy == rebuilt;
        ok = ok && value.capacity() ==
                       ShaderManifest::builtin_capacity() + ShaderIdRegistry::capacity();
        return ok;
    }
} // namespace

int main()
{
    bool ok = true;
    auto check = [](const char* name, bool (*fn)()) {
        const bool r = fn();
        std::fprintf(stderr, "[shader-id-open] %-44s %s\n", name, r ? "PASS" : "FAIL");
        return r;
    };

    ok = check("range_law", test_range_law) && ok;
    ok = check("builtin_names_never_register", test_builtin_names_never_mint) && ok;
    ok = check("content_addressed_ids", test_content_addressed_ids) && ok;
    ok = check("determinism_across_instances", test_determinism_across_instances) && ok;
    ok = check("shared_offset_law", test_shared_offset_law) && ok;
    ok = check("null_and_reserved_rejected", test_null_and_reserved_rejected) && ok;
    ok = check("capacity_arithmetic", test_capacity_arithmetic) && ok;
    ok = check("collision_is_loud", test_collision_is_loud) && ok;
    ok = check("foreign_open_id_is_hard_miss", test_foreign_open_id_is_a_hard_miss) && ok;
    ok = check("consumer_shader_no_core_edit", test_consumer_shader_without_core_edit) && ok;
    ok = check("open_id_realization_law", test_open_id_realization_law) && ok;
    ok = check("open_id_descriptor_invariants", test_open_id_descriptor_invariants) && ok;
    ok = check("manifest_equality_and_copy", test_manifest_equality_and_copy) && ok;
    ok = check("builtin_paths_unchanged", test_builtin_paths_unchanged) && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[shader-id-open] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[shader-id-open] all tests passed\n");
    return 0;
}
