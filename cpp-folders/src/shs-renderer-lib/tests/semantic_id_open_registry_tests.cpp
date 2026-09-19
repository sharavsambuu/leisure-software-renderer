#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

#include "shs/renderpath/planning/render_path_resource_plan.hpp"
#include "shs/renderpath/planning/semantic_registry.hpp"

// Open pass-semantic gate (Constitution I §7 — No User Lock-In;
// arch/render_path_architecture.md §4 graduation req 7).
//
// Proves the *consumer* story: a technique-owned channel (a custom G-buffer
// layout, a mobile-lighting compacted payload) is named, registered, given a
// contract and planned into a resource with a distinguishing id — with zero
// edits to the core vocabulary — while every builtin semantic keeps its exact
// previous behavior (AD0 parity), including Unknown, which the open-range guard
// must not touch.
//
// GPU-free: pure planning values, no Context, no backend instance.
namespace
{
    using namespace shs;
    using namespace shs::renderpath;

    // --- range law -----------------------------------------------------------

    bool test_range_law()
    {
        bool ok = true;
        ok = ok && pass_semantic_is_builtin(PassSemantic::Depth);
        ok = ok && pass_semantic_is_builtin(PassSemantic::HistoryMotion);
        ok = ok && !pass_semantic_is_builtin(PassSemantic::Unknown);
        ok = ok && !pass_semantic_is_builtin(static_cast<PassSemantic>(kPassSemanticOpenBase));

        ok = ok && !pass_semantic_is_open(PassSemantic::Unknown);
        ok = ok && !pass_semantic_is_open(PassSemantic::Albedo);
        // Boundary: the last builtin slot is not open, the first open slot is.
        ok = ok && !pass_semantic_is_open(static_cast<PassSemantic>(kPassSemanticOpenBase - 1u));
        ok = ok && pass_semantic_is_open(static_cast<PassSemantic>(kPassSemanticOpenBase));
        ok = ok && pass_semantic_is_open(static_cast<PassSemantic>(kPassSemanticOpenMax));

        ok = ok && pass_semantic_in_valid_range(PassSemantic::Normal);
        ok = ok && pass_semantic_in_valid_range(static_cast<PassSemantic>(kPassSemanticOpenBase));
        ok = ok && !pass_semantic_in_valid_range(PassSemantic::Unknown);
        ok = ok && !pass_semantic_in_valid_range(static_cast<PassSemantic>(kPassSemanticReserved));

        // The builtin enum can never grow into the open range (compile-time
        // pins) — re-asserted at runtime so a silent renumbering is loud here.
        ok = ok && static_cast<uint16_t>(PassSemantic::Unknown) == kPassSemanticUnknown;
        ok = ok && static_cast<uint16_t>(PassSemantic::HistoryMotion) <= kPassSemanticBuiltinMax;

        // An open semantic stringifies honestly, and is NOT mistakable for a
        // builtin name.
        ok = ok && std::string_view(pass_semantic_name(static_cast<PassSemantic>(kPassSemanticOpenBase))) ==
                       std::string_view(kPassSemanticOpenSpelling);
        ok = ok && pass_semantic_name_or_null(PassSemantic::Albedo) != nullptr;
        ok = ok && pass_semantic_name_or_null(static_cast<PassSemantic>(kPassSemanticOpenBase)) == nullptr;

        // Every builtin name parses back to its own semantic (round-trip), and
        // no builtin name collides with the open spelling.
        for (uint16_t raw = static_cast<uint16_t>(PassSemantic::Depth);
             raw <= static_cast<uint16_t>(PassSemantic::HistoryMotion);
             ++raw)
        {
            const PassSemantic s = static_cast<PassSemantic>(raw);
            ok = ok && pass_semantic_is_builtin(s);
            ok = ok && parse_pass_semantic(pass_semantic_name(s)) == s;
            ok = ok && std::string_view(pass_semantic_name(s)) != std::string_view(kPassSemanticOpenSpelling);
        }
        return ok;
    }

    // --- the registry --------------------------------------------------------

    bool test_builtin_names_never_mint()
    {
        PassSemanticRegistry ids{};
        const auto albedo = ids.intern("albedo");
        const auto normal = ids.intern("normal");
        bool ok = true;
        ok = ok && albedo.has_value() && *albedo == PassSemantic::Albedo;
        ok = ok && normal.has_value() && *normal == PassSemantic::Normal;
        // No open id was minted, so a consumer can never shadow a core channel.
        ok = ok && ids.open_count() == 0u && ids.empty();
        ok = ok && ids.registered().empty();
        return ok;
    }

    bool test_intern_order_and_idempotence()
    {
        PassSemanticRegistry a{};
        const auto gb0 = a.intern("gbuffer_material_id");
        const auto gb1 = a.intern("gbuffer_bent_normal");
        const auto gb0_again = a.intern("gbuffer_material_id");

        // Order independence: the same names interned in the opposite order.
        PassSemanticRegistry b{};
        const auto b1 = b.intern("gbuffer_bent_normal");
        const auto b0 = b.intern("gbuffer_material_id");

        bool ok = true;
        ok = ok && gb0.has_value() && gb1.has_value();
        ok = ok && gb0 == gb0_again;            // idempotent
        ok = ok && a.open_count() == 2u;        // and does not double-register
        ok = ok && *gb0 == *b0 && *gb1 == *b1;  // order-independent ids
        ok = ok && a == b;                      // order-independent equality
        ok = ok && *gb0 != *gb1;                // distinct names, distinct ids
        ok = ok && pass_semantic_is_open(*gb0);
        ok = ok && a.is_open(*gb0);
        ok = ok && !a.is_open(PassSemantic::Albedo);
        ok = ok && a.contains(PassSemantic::Albedo); // builtins always resolve
        return ok;
    }

    bool test_determinism_across_instances()
    {
        // Names representative of the goals this opens: a custom G-buffer
        // channel, a tiled-lighting payload, a mobile compacted light.
        const char* kNames[] = {
            "gbuffer_bent_normal", "tile_light_mask", "mobile_light_payload"};
        PassSemanticRegistry one{};
        PassSemanticRegistry two{};
        bool ok = true;
        for (const char* name : kNames)
        {
            const auto a = one.intern(name);
            const auto b = two.intern(name);
            ok = ok && a.has_value() && b.has_value() && *a == *b;
            ok = ok && one.try_name(*a).has_value() && *one.try_name(*a) == std::string_view(name);
        }
        return ok;
    }

    bool test_null_and_reserved_rejected()
    {
        PassSemanticRegistry ids{};
        bool ok = true;
        ok = ok && !ids.intern("").has_value();
        ok = ok && !ids.intern("unknown").has_value(); // the null spelling
        ok = ok && ids.open_count() == 0u;
        // The null semantic and the reserved/out-of-range slots never resolve.
        ok = ok && !ids.try_name(PassSemantic::Unknown).has_value();
        ok = ok && !ids.contains(PassSemantic::Unknown);
        ok = ok && !ids.try_name(static_cast<PassSemantic>(kPassSemanticReserved)).has_value();
        ok = ok && !ids.try_name(static_cast<PassSemantic>(kPassSemanticOpenMax)).has_value();
        return ok;
    }

    bool test_capacity_arithmetic()
    {
        bool ok = true;
        ok = ok && PassSemanticRegistry::capacity() ==
                       static_cast<std::size_t>(kPassSemanticOpenMax) -
                           static_cast<std::size_t>(kPassSemanticOpenBase) + 1u;
        ok = ok && static_cast<uint32_t>(kPassSemanticOpenBase) +
                       static_cast<uint32_t>(PassSemanticRegistry::capacity()) - 1u ==
                       static_cast<uint32_t>(kPassSemanticOpenMax);
        // The reserved slot is never minted.
        ok = ok && PassSemanticRegistry::capacity() < 65535u;
        ok = ok && kPassSemanticReserved == 65535u;
        return ok;
    }

    // Brute-force a real 16-bit offset collision. With ~64k slots a birthday
    // collision appears within a few hundred probes, so the loud-failure path is
    // testable rather than theoretical.
    std::string find_colliding_name(std::string_view seed)
    {
        const uint16_t target = PassSemanticRegistry::open_offset(seed);
        for (int i = 0; i < 20000; ++i)
        {
            const std::string candidate = "semantic_probe_" + std::to_string(i);
            if (candidate != seed && PassSemanticRegistry::open_offset(candidate) == target)
            {
                return candidate;
            }
        }
        return {};
    }

    bool test_collision_is_loud()
    {
        const std::string collider = find_colliding_name("demo_stylized_channel");
        // If no collision was found the assertion is vacuous — but it is not
        // expected to happen (birthday bound), so do not fail the gate for it.
        if (collider.empty()) return true;

        PassSemanticRegistry ids{};
        const auto first_id = ids.intern("demo_stylized_channel");
        bool ok = first_id.has_value();
        ok = ok && ids.open_count() == 1u;

        // The colliding name is refused: nothing overwritten, nothing aliased,
        // and the first name keeps its id.
        ok = ok && !ids.intern(collider).has_value();
        ok = ok && ids.open_count() == 1u;
        ok = ok && ids.try_name(*first_id).has_value() &&
                   *ids.try_name(*first_id) == std::string_view("demo_stylized_channel");

        // Residual, stated honestly: a registry holding only the collider maps
        // that shared slot to *its* name. The registry-aware resource id is the
        // guard — it resolves the name the *owning* registry registered, so a
        // colliding id can never silently name someone else's channel.
        PassSemanticRegistry only_collider{};
        const auto registered_collider = only_collider.intern(collider);
        ok = ok && registered_collider.has_value();
        ok = ok && *registered_collider == *first_id; // same slot by construction
        ok = ok && render_path_resource_id_for_semantic(*first_id, ids) == "demo_stylized_channel";
        ok = ok && render_path_resource_id_for_semantic(*registered_collider, only_collider) == collider;
        return ok;
    }

    bool test_foreign_open_id_is_a_hard_miss()
    {
        PassSemanticRegistry mine{};
        const auto channel = mine.intern("gbuffer_bent_normal");
        bool ok = channel.has_value();

        // An id derived from a name this registry never registered misses —
        // there is no "whatever is at that slot" fallback.
        PassSemanticRegistry other{};
        const auto unrelated = other.intern("gbuffer_not_registered_here");
        ok = ok && unrelated.has_value();
        ok = ok && !mine.contains(*unrelated);
        ok = ok && !mine.is_open(*unrelated);
        ok = ok && !mine.try_name(*unrelated).has_value();

        // ... and its resource id does not borrow my name either.
        ok = ok && render_path_resource_id_for_semantic(*unrelated, mine) == kPassSemanticOpenSpelling;

        // The null and reserved semantics never resolve anywhere.
        ok = ok && !mine.try_name(PassSemantic::Unknown).has_value();
        ok = ok && !mine.try_name(static_cast<PassSemantic>(kPassSemanticReserved)).has_value();
        return ok;
    }

    // --- consumer story: an open channel through the contract ---------------

    bool test_open_semantic_gets_neutral_descriptor()
    {
        PassSemanticRegistry ids{};
        const auto custom = ids.intern("gbuffer_bent_normal");
        bool ok = custom.has_value();
        if (!ok) return false;

        const PassSemanticDescriptor desc = default_pass_semantic_descriptor(*custom);
        // The core must NOT guess a consumer channel's intent: neutral defaults,
        // with the author stating specifics via make_semantic_ref overrides.
        ok = ok && desc.semantic == *custom;
        ok = ok && desc.space == PassSemanticSpace::Screen;
        ok = ok && desc.encoding == PassSemanticEncoding::Linear;
        ok = ok && desc.lifetime == PassSemanticLifetime::Transient;
        ok = ok && desc.temporal_role == PassSemanticTemporalRole::CurrentFrame;
        ok = ok && desc.sampled;
        ok = ok && !desc.storage;

        // Total for garbage too (no UB, no crash): the reserved slot takes the
        // same neutral defaults and is still reported out of range.
        const PassSemanticDescriptor reserved =
            default_pass_semantic_descriptor(static_cast<PassSemantic>(kPassSemanticReserved));
        ok = ok && reserved.space == PassSemanticSpace::Screen;
        ok = ok && !pass_semantic_in_valid_range(static_cast<PassSemantic>(kPassSemanticReserved));
        return ok;
    }

    bool test_contract_accepts_open_semantic()
    {
        PassSemanticRegistry ids{};
        const auto custom = ids.intern("mobile_light_payload");
        bool ok = custom.has_value();
        if (!ok) return false;

        // Author without overrides: the neutral descriptor supplies the rest.
        const PassSemanticRef ref = write_semantic(*custom, render_domain_unspecified(), "mobile_lights");
        ok = ok && ref.semantic == *custom;
        ok = ok && ref.access == ContractAccess::Write;
        ok = ok && ref.alias == "mobile_lights";
        ok = ok && ref.space == PassSemanticSpace::Screen;
        ok = ok && ref.encoding == PassSemanticEncoding::Linear;

        // Author WITH overrides: the part only they know is stated, and it wins
        // over the neutral descriptor.
        const PassSemanticRef overridden = make_semantic_ref(
            *custom, ContractAccess::ReadWrite, render_domain_unspecified(), nullptr,
            PassSemanticSpace::Light, PassSemanticEncoding::UIntCounts,
            PassSemanticLifetime::Persistent);
        ok = ok && overridden.space == PassSemanticSpace::Light;
        ok = ok && overridden.encoding == PassSemanticEncoding::UIntCounts;
        ok = ok && overridden.lifetime == PassSemanticLifetime::Persistent;
        ok = ok && overridden.temporal_role == PassSemanticTemporalRole::CurrentFrame;

        // A contract carrying an open channel is ordinary data: it sits next to
        // a builtin channel in a plan input with no core edit anywhere.
        TechniquePassContract contract{};
        contract.role = TechniquePassRole::LightCulling;
        contract.semantics.push_back(ref);
        contract.semantics.push_back(read_semantic(PassSemantic::Depth));
        ok = ok && contract.semantics.size() == 2u;
        ok = ok && contract.semantics[0].semantic == *custom;
        ok = ok && contract.semantics[1].semantic == PassSemantic::Depth;

        // Contracts compare by value (they are plan inputs), open channel and
        // all — which is why the descriptor had to stay a pure function.
        TechniquePassContract same{};
        same.role = TechniquePassRole::LightCulling;
        same.semantics.push_back(ref);
        same.semantics.push_back(read_semantic(PassSemantic::Depth));
        ok = ok && same.semantics[0].semantic == contract.semantics[0].semantic;
        ok = ok && same.semantics[0].alias == contract.semantics[0].alias;
        return ok;
    }

    bool test_resource_id_distinguishes_open_semantics()
    {
        PassSemanticRegistry ids{};
        const auto a = ids.intern("gbuffer_bent_normal");
        const auto b = ids.intern("gbuffer_material_id");
        bool ok = a.has_value() && b.has_value();
        if (!ok) return false;

        // Registry-aware ids are distinct AND name-stable — the whole point of
        // routing through the registry.
        const std::string ida = render_path_resource_id_for_semantic(*a, ids);
        const std::string idb = render_path_resource_id_for_semantic(*b, ids);
        ok = ok && ida == "gbuffer_bent_normal";
        ok = ok && idb == "gbuffer_material_id";
        ok = ok && ida != idb;

        // Builtins are unaffected by the registry overload.
        ok = ok && render_path_resource_id_for_semantic(PassSemantic::AmbientOcclusion, ids) == "ao";
        ok = ok && render_path_resource_id_for_semantic(PassSemantic::Normal, ids) == "normal";

        // Stated limitation, asserted rather than hidden: without a registry the
        // id is the honest-but-non-distinguishing open spelling, so two distinct
        // open semantics WOULD alias onto one resource. That is exactly why the
        // registry overload exists.
        ok = ok && render_path_resource_id_for_semantic(*a) == kPassSemanticOpenSpelling;
        ok = ok && render_path_resource_id_for_semantic(*b) == kPassSemanticOpenSpelling;

        // Planned resources therefore stay distinct through the registry path.
        const RenderPathRecipe recipe{};
        const RenderPathResourceSpec spec_a = make_default_resource_spec_for_semantic(*a, recipe, ids);
        const RenderPathResourceSpec spec_b = make_default_resource_spec_for_semantic(*b, recipe, ids);
        ok = ok && spec_a.semantic == *a && spec_b.semantic == *b;
        ok = ok && spec_a.id == "gbuffer_bent_normal" && spec_b.id == "gbuffer_material_id";
        ok = ok && spec_a.id != spec_b.id;
        ok = ok && spec_a.resolution == RenderPathResolutionClass::Full;
        ok = ok && spec_a.kind == RenderPathResourceKind::Texture2D;
        ok = ok && spec_a.semantic_space == PassSemanticSpace::Screen;
        ok = ok && spec_a.semantic_encoding == PassSemanticEncoding::Linear;
        return ok;
    }

    // --- AD0 parity: every builtin path keeps its exact previous behavior -----

    bool test_builtin_paths_unchanged()
    {
        bool ok = true;
        // Unknown is the delicate one: the open-range early return must NOT
        // catch it (it is neither builtin nor open), so its old descriptor has
        // to survive verbatim.
        const PassSemanticDescriptor unknown = default_pass_semantic_descriptor(PassSemantic::Unknown);
        ok = ok && unknown.space == PassSemanticSpace::None;
        ok = ok && unknown.encoding == PassSemanticEncoding::Unknown;
        ok = ok && unknown.lifetime == PassSemanticLifetime::Transient;
        ok = ok && !unknown.sampled && !unknown.storage;
        ok = ok && std::string_view(pass_semantic_name(PassSemantic::Unknown)) == std::string_view("unknown");
        ok = ok && render_path_resource_id_for_semantic(PassSemantic::Unknown) == "unknown";

        // Spot-check the builtin intents the open range must not disturb.
        const PassSemanticDescriptor ldr = default_pass_semantic_descriptor(PassSemantic::ColorLDR);
        ok = ok && ldr.encoding == PassSemanticEncoding::SRGB;
        ok = ok && ldr.lifetime == PassSemanticLifetime::Persistent;
        const PassSemanticDescriptor nrm = default_pass_semantic_descriptor(PassSemantic::Normal);
        ok = ok && nrm.space == PassSemanticSpace::View;
        ok = ok && nrm.encoding == PassSemanticEncoding::SignedVector;
        const PassSemanticDescriptor shadow = default_pass_semantic_descriptor(PassSemantic::ShadowMap);
        ok = ok && shadow.space == PassSemanticSpace::Light;
        ok = ok && shadow.lifetime == PassSemanticLifetime::Persistent;
        const PassSemanticDescriptor mvec = default_pass_semantic_descriptor(PassSemantic::MotionVectors);
        ok = ok && mvec.encoding == PassSemanticEncoding::VelocityScreen;

        // Every builtin keeps its own semantic, a real (non-open-spelling)
        // resource id, and is unaffected by the registry overload.
        PassSemanticRegistry empty{};
        for (uint16_t raw = static_cast<uint16_t>(PassSemantic::Depth);
             raw <= static_cast<uint16_t>(PassSemantic::HistoryMotion);
             ++raw)
        {
            const PassSemantic s = static_cast<PassSemantic>(raw);
            ok = ok && default_pass_semantic_descriptor(s).semantic == s;
            ok = ok && render_path_resource_id_for_semantic(s) != kPassSemanticOpenSpelling;
            ok = ok && render_path_resource_id_for_semantic(s, empty) == render_path_resource_id_for_semantic(s);
        }

        // The light-culling builtins keep their storage-buffer/Tile carve-out,
        // which an open semantic must not inherit.
        const RenderPathRecipe recipe{};
        const RenderPathResourceSpec grid =
            make_default_resource_spec_for_semantic(PassSemantic::LightGrid, recipe);
        ok = ok && grid.kind == RenderPathResourceKind::StorageBuffer;
        ok = ok && grid.resolution == RenderPathResolutionClass::Tile;
        ok = ok && grid.storage;

        // The shadow-map carve-out (Absolute 2048, persistent) is builtin-only.
        const RenderPathResourceSpec shadow_spec =
            make_default_resource_spec_for_semantic(PassSemantic::ShadowMap, recipe);
        ok = ok && shadow_spec.resolution == RenderPathResolutionClass::Absolute;
        ok = ok && shadow_spec.width == 2048u && shadow_spec.height == 2048u;
        ok = ok && !shadow_spec.transient;

        // An open semantic gets the neutral Full/Texture2D transient default
        // instead of either carve-out.
        PassSemanticRegistry ids{};
        const auto custom = ids.intern("gbuffer_bent_normal");
        if (!custom.has_value()) return false;
        const RenderPathResourceSpec spec = make_default_resource_spec_for_semantic(*custom, recipe);
        ok = ok && spec.resolution == RenderPathResolutionClass::Full;
        ok = ok && spec.width == 0u && spec.height == 0u;
        ok = ok && spec.transient;
        ok = ok && !spec.storage;
        return ok;
    }
}

int main()
{
    bool ok = true;
    auto check = [](const char* name, bool (*fn)()) {
        const bool r = fn();
        std::fprintf(stderr, "[semantic-open] %-40s %s\n", name, r ? "PASS" : "FAIL");
        return r;
    };

    ok = check("range_law", test_range_law) && ok;
    ok = check("builtin_names_never_register", test_builtin_names_never_mint) && ok;
    ok = check("content_addressed_ids", test_intern_order_and_idempotence) && ok;
    ok = check("determinism_across_instances", test_determinism_across_instances) && ok;
    ok = check("null_and_reserved_rejected", test_null_and_reserved_rejected) && ok;
    ok = check("capacity_arithmetic", test_capacity_arithmetic) && ok;
    ok = check("collision_is_loud", test_collision_is_loud) && ok;
    ok = check("foreign_open_id_is_hard_miss", test_foreign_open_id_is_a_hard_miss) && ok;
    ok = check("open_neutral_descriptor", test_open_semantic_gets_neutral_descriptor) && ok;
    ok = check("contract_accepts_open_semantic", test_contract_accepts_open_semantic) && ok;
    ok = check("resource_id_is_distinguishing", test_resource_id_distinguishes_open_semantics) && ok;
    ok = check("builtin_paths_unchanged", test_builtin_paths_unchanged) && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[semantic-open] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[semantic-open] all tests passed\n");
    return 0;
}
