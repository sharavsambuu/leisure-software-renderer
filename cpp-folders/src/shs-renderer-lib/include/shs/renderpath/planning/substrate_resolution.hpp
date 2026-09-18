#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: substrate_resolution.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Substrate-ийг authoring биш, compile (plan) үед policy-оор шийдэх
            (resolution) цэвэр логик. Recipe intent хэлнэ, compiler шийднэ.
*/


#include <cstdint>

#include "shs/renderpath/planning/pass_contract.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
    {
    // ---------------------------------------------------------------------
    // RP-1 (graduation req 4; owner sign-off 2026-09-18).
    //
    // Before: `render_path_recipe.hpp` forked BOTH the pass chain and the
    // technique mode on `backend`, so "forward-plus on Vulkan" and "forward-lit
    // on software" were two authored recipes rather than one recipe resolved two
    // ways. Now the recipe states *intent*, and the substrate a pass runs on is
    // a RESOLUTION OUTPUT of the compiler.
    //
    // This header is the pure half: it knows nothing about contexts, backends or
    // registries. Callers hand it masks and an intent; it hands back a substrate
    // or "unresolved" — never a guess.
    // ---------------------------------------------------------------------
    enum class SubstratePolicy : uint8_t
    {
        // Never substitute: every pass runs on the recipe's declared substrate.
        // This is the historical behaviour AND the default, so a recipe that
        // states no policy resolves exactly where it resolved before.
        ExactMatch = 0,
        // Prefer device substrates; host only where no device substrate can
        // realize the pass.
        DevicePreferred = 1,
        // Prefer the host rasterizer; device only where the host cannot realize
        // the pass.
        HostPreferred = 2,
        // Fewest crossings: reuse the predecessor's substrate where it is
        // admissible. Deliberately a *crossing count*, not a hand-kept cost
        // table — no honest per-pass cost data exists today, and inventing one
        // would be a fabricated measurement. Greedy over the authored chain
        // order (stated limitation, not a claim of global optimality).
        Cheapest = 3
    };

    inline const char* substrate_policy_name(SubstratePolicy p)
    {
        switch (p)
        {
            case SubstratePolicy::ExactMatch: return "exact_match";
            case SubstratePolicy::DevicePreferred: return "device_preferred";
            case SubstratePolicy::HostPreferred: return "host_preferred";
            case SubstratePolicy::Cheapest: return "cheapest";
        }
        return "unknown";
    }

    // --- substrate masks ---------------------------------------------------
    // A mask is over `Substrate` ordinals. `PassFactoryRegistry::backend_bit`
    // is the same encoding over `RenderBackendType` ordinals; the two are
    // interchangeable by the declared 1:1 identity (`substrate_of_backend`),
    // which is why there is no conversion helper here — only a naming one.

    inline constexpr uint32_t substrate_bit(Substrate s)
    {
        return 1u << static_cast<uint32_t>(s);
    }

    inline constexpr uint32_t substrate_mask_none()
    {
        return 0u;
    }

    inline constexpr uint32_t substrate_mask_all()
    {
        return substrate_bit(Substrate::SoftwareRaster) |
               substrate_bit(Substrate::OpenGL) |
               substrate_bit(Substrate::Vulkan);
    }

    inline constexpr unsigned substrate_count = 3u;

    // The closed substrate set, in ordinal order. Kept as one table so the
    // preference ladders below and `substrate_mask_of` cannot drift apart.
    inline constexpr Substrate substrate_ordinal[substrate_count] = {
        Substrate::SoftwareRaster,
        Substrate::OpenGL,
        Substrate::Vulkan
    };

    // Set of substrates that execute on a given unit — DERIVED from
    // `execution_unit_of`, so it cannot disagree with the axis relation.
    inline constexpr uint32_t substrate_mask_of(ExecutionUnit unit)
    {
        uint32_t mask = substrate_mask_none();
        for (unsigned i = 0; i < substrate_count; ++i)
        {
            if (execution_unit_of(substrate_ordinal[i]) == unit) mask |= substrate_bit(substrate_ordinal[i]);
        }
        return mask;
    }

    // Policy preference ladders. Deterministic by construction: the first
    // admissible entry wins, so the same inputs always resolve the same way
    // (a plan is data, never ambient state — the snapshot contract).
    inline constexpr Substrate device_first_order[substrate_count] = {
        Substrate::Vulkan,
        Substrate::OpenGL,
        Substrate::SoftwareRaster
    };

    inline constexpr Substrate host_first_order[substrate_count] = {
        Substrate::SoftwareRaster,
        Substrate::OpenGL,
        Substrate::Vulkan
    };

    inline constexpr Substrate ordinal_order[substrate_count] = {
        Substrate::SoftwareRaster,
        Substrate::OpenGL,
        Substrate::Vulkan
    };

    // Everything the resolver is allowed to look at. No globals, no ambient
    // reads: the same request always yields the same answer.
    struct SubstrateResolutionRequest
    {
        // What the author said the pass needs. `Unspecified` and `Any` narrow
        // nothing.
        RenderDomain intent = render_domain_unspecified();

        // Substrates that CAN realize this pass (pass registry `backend_mask`).
        // `realized_known == false` means "no registry said" — treated as "any
        // admissible substrate", never as "none".
        uint32_t realized_mask = 0u;
        bool realized_known = false;

        // Substrates this host can actually drive. An EMPTY mask is not
        // "nothing available" — it means "unknown / single-substrate host", and
        // resolves as the declared substrate only. Declaring more than one is
        // the explicit opt-in that makes a hybrid resolution possible at all.
        uint32_t available_mask = 0u;

        SubstratePolicy policy = SubstratePolicy::ExactMatch;

        // The recipe's declared substrate: the `ExactMatch` target, and the
        // fallback whenever the host's available set is unknown.
        Substrate declared = Substrate::SoftwareRaster;

        // Set for every pass but the first; `Cheapest` measures crossings
        // against it.
        bool has_predecessor = false;
        Substrate predecessor = Substrate::SoftwareRaster;
    };

    struct SubstrateResolution
    {
        Substrate substrate = Substrate::SoftwareRaster;
        // False means "no admissible substrate" — a rejection at the call site,
        // never a silent default.
        bool resolved = false;

        bool operator==(const SubstrateResolution&) const = default;
    };

    inline SubstrateResolution resolve_substrate(const SubstrateResolutionRequest& in)
    {
        SubstrateResolution out{};

        // Admissible = what the host can drive, narrowed by what the pass can
        // realize. An empty available mask is the single-substrate case.
        uint32_t admissible = (in.available_mask != substrate_mask_none())
            ? in.available_mask
            : substrate_bit(in.declared);
        if (in.realized_known) admissible &= in.realized_mask;

        // Intent narrows further. A pinned substrate is one bit; a pinned unit
        // is that unit's whole set (which is how "any device substrate" is
        // expressed); Any/Unspecified narrow nothing.
        if (in.intent.kind == RenderDomainKind::Pinned)
        {
            admissible &= in.intent.substrate_pinned
                ? substrate_bit(in.intent.substrate)
                : substrate_mask_of(in.intent.unit);
        }

        if (admissible == substrate_mask_none()) return out;

        if (in.policy == SubstratePolicy::ExactMatch)
        {
            if ((admissible & substrate_bit(in.declared)) != 0u)
            {
                out.substrate = in.declared;
                out.resolved = true;
            }
            return out;
        }

        if (in.policy == SubstratePolicy::Cheapest && in.has_predecessor)
        {
            if ((admissible & substrate_bit(in.predecessor)) != 0u)
            {
                out.substrate = in.predecessor;
                out.resolved = true;
            }
            return out;
        }

        const Substrate* order = ordinal_order;
        if (in.policy == SubstratePolicy::DevicePreferred) order = device_first_order;
        else if (in.policy == SubstratePolicy::HostPreferred) order = host_first_order;

        for (unsigned i = 0; i < substrate_count; ++i)
        {
            if ((admissible & substrate_bit(order[i])) != 0u)
            {
                out.substrate = order[i];
                out.resolved = true;
                return out;
            }
        }
        return out;
    }

    // Two resolved passes CROSS when they do not run on the same substrate.
    // Since `execution_unit_of` is a function of the substrate, this covers both
    // halves of the RP-2 rule ("crossing execution units OR substrates") with
    // one comparison — a different unit always implies a different substrate.
    inline constexpr bool substrates_cross(Substrate a, Substrate b)
    {
        return a != b;
    }

    } // inline namespace renderpath
}
