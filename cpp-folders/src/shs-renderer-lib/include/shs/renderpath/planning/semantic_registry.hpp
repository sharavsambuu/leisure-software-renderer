#pragma once

/*
    SHS RENDERER SAN

    FILE: semantic_registry.hpp
    MODULE: renderpath/planning
    PURPOSE: Open registered range for pass semantics (Constitution I §7 — No
             User Lock-In; arch/render_path_architecture.md §4 graduation req 7).

             `PassSemantic` (planning/pass_contract.hpp) is a small closed
             builtin vocabulary; consumer-owned channels — a tiled/deferred
             G-buffer's own layouts, a technique's private intermediate, a
             mobile lighting path's compacted payload — are added through this
             registry instead of by editing the core.

             This is the third namespace to open, and it is deliberately the
             SAME mechanism as the other two (`PassIdRegistry`,
             `ShaderIdRegistry`) rather than a fourth spelling of it: ids are
             CONTENT-ADDRESSED from the registered *name* through the shared
             `core/open_id_hash.hpp` law. Three properties follow, and they are
             why the choice is repeated rather than re-derived:

             1. Stability across processes and translation units — a saved
                recipe, a replay log or a resource plan that names a consumer
                semantic stays valid, because the same name always yields the
                same id.
             2. Order independence — interning the same names in any order
                produces identical ids, so registration order cannot leak into
                a plan.
             3. No cross-registry aliasing — `try_name(foreign_id)` can only
                return the name that id was derived from, or nullopt. A
                registry can never hand back a *different* semantic for someone
                else's id.

             Collisions are therefore the one failure mode, and they are LOUD:
             two distinct names landing on the same slot mean the second
             `intern` returns nullopt and nothing is overwritten. Storage is a
             flat vector scanned linearly (typically < 100 consumer semantics),
             so the registry stays copyable, comparable and cheap — no hashing
             at lookup time and no ambient global (one instance per owner).

             Why a registry is required rather than optional: a resource spec's
             `id` is derived from the semantic, so two distinct open semantics
             that were never registered would both fall back to
             `kPassSemanticOpenSpelling` and silently MERGE into one planned
             resource. `render_path_resource_id_for_semantic(semantic, registry)`
             is the path that keeps them distinct.
*/

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "shs/core/open_id_hash.hpp"
#include "shs/renderpath/planning/pass_contract.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
    {
    class PassSemanticRegistry
    {
    public:
        // How many distinct consumer-owned semantics fit in the open range.
        static constexpr std::size_t capacity()
        {
            return static_cast<std::size_t>(kPassSemanticOpenMax) -
                   static_cast<std::size_t>(kPassSemanticOpenBase) + 1u;
        }

        // Deterministic offset of a name inside the open range (FNV-1a 32 with a
        // final mix, folded to the range). Pure function of the name: no state,
        // no iteration order, no ambient seed.
        static constexpr uint16_t open_offset(std::string_view name)
        {
            // Shared law (shs/core/open_id_hash.hpp), not a local copy: the
            // pass-id and shader-id registries derive their offsets the same
            // way, so a divergence cannot silently change which id a name maps
            // to.
            return core::open_id_offset(name, capacity());
        }

        // Resolve a name to a typed semantic. Total (no throw, no partial
        // state):
        //   - a builtin name ("albedo") resolves to its builtin semantic and
        //     registers nothing, so a consumer can never shadow a core channel;
        //   - an empty name or the null spelling resolves to nullopt;
        //   - any other name is minted into the open range, or resolves to the
        //     already-registered id if the same name was interned before.
        // Returns nullopt on a collision — never a wrong id, never an overwrite.
        std::optional<PassSemantic> intern(std::string_view name)
        {
            if (name.empty() || name == pass_semantic_name(PassSemantic::Unknown)) return std::nullopt;

            const PassSemantic builtin = parse_pass_semantic(name);
            if (pass_semantic_is_builtin(builtin)) return builtin;

            const uint16_t offset = open_offset(name);
            const PassSemantic id =
                static_cast<PassSemantic>(static_cast<uint16_t>(kPassSemanticOpenBase + offset));

            for (const auto& slot : slots_)
            {
                if (slot.first == offset)
                {
                    // Same name => same id. Different name => collision: refuse.
                    return slot.second == name ? std::optional<PassSemantic>(id) : std::nullopt;
                }
            }

            slots_.emplace_back(offset, std::string(name));
            return id;
        }

        // Name of a typed semantic: the builtin table, or the name that id was
        // derived from (which this registry registered). nullopt for the null
        // semantic (Unknown), out-of-range ids, and ids this registry never
        // registered — importantly, a foreign id can only miss, never alias
        // another semantic.
        std::optional<std::string_view> try_name(PassSemantic id) const
        {
            if (pass_semantic_is_builtin(id)) return std::string_view(pass_semantic_name(id));
            if (!pass_semantic_is_open(id)) return std::nullopt;

            const uint16_t offset = static_cast<uint16_t>(
                static_cast<uint16_t>(id) - static_cast<uint16_t>(kPassSemanticOpenBase));
            for (const auto& slot : slots_)
            {
                if (slot.first == offset) return std::string_view(slot.second);
            }
            return std::nullopt;
        }

        // True for a builtin semantic or an open semantic this registry
        // registered.
        bool contains(PassSemantic id) const { return try_name(id).has_value(); }

        // True only for an open semantic this registry registered.
        bool is_open(PassSemantic id) const
        {
            return pass_semantic_is_open(id) && contains(id);
        }

        // Registered pairs in registration order (deterministic; the *ids*
        // themselves are order-independent, see `open_offset`).
        const std::vector<std::pair<uint16_t, std::string>>& registered() const { return slots_; }

        std::size_t open_count() const { return slots_.size(); }
        bool empty() const { return slots_.empty(); }
        void clear() { slots_.clear(); }

        // Value semantics, order-independent: two registries are equal iff they
        // registered the same names (hence the same content-addressed ids).
        bool operator==(const PassSemanticRegistry& other) const
        {
            if (slots_.size() != other.slots_.size()) return false;
            for (const auto& slot : slots_)
            {
                const PassSemantic id = static_cast<PassSemantic>(
                    static_cast<uint16_t>(kPassSemanticOpenBase + slot.first));
                const std::optional<std::string_view> in_other = other.try_name(id);
                if (!in_other.has_value() || *in_other != std::string_view(slot.second)) return false;
            }
            return true;
        }

    private:
        // (open-range offset, registered name). Flat and linear: typically a
        // handful of entries, and it keeps the registry copyable and comparable.
        std::vector<std::pair<uint16_t, std::string>> slots_{};
    };

    } // inline namespace renderpath
}
