#pragma once

/*
    SHS RENDERER SAN

    FILE: pass_id_registry.hpp
    MODULE: renderpath/execution
    PURPOSE: Open registered range for pass ids (Constitution I §7 — No User
             Lock-In; arch/render_path_architecture.md §4 graduation req 1).

             `PassId` (planning/pass_id.hpp) is a small closed builtin
             vocabulary; consumer/demo-owned passes are added through this
             registry instead of by editing the core.

             Ids are CONTENT-ADDRESSED, not mint-order: an open id is a
             deterministic function of the registered *name*
             (`open_offset`), landing in the open range. Three properties
             follow, and they are the reason for the choice:

             1. Stability across processes and translation units — a saved
                recipe, a replay log or a barrier table that names a consumer
                pass stays valid, because the same name always yields the same
                id.
             2. Order independence — interning the same names in any order
                produces identical ids, so registration order can never leak
                into a plan.
             3. No cross-registry aliasing — `try_name(foreign_id)` can only
                return the name that id was derived from, or nullopt. A
                registry can never hand back a *different* pass for someone
                else's id. (Mint-order ids fail this: id `base+0` means
                "whatever was registered first here", which silently resolves
                to the wrong pass in another registry.)

             Collisions are therefore the one failure mode, and they are LOUD:
             two distinct names landing on the same slot mean the second
             `intern` returns nullopt and nothing is overwritten. Storage is a
             flat vector scanned linearly (typically < 100 consumer passes), so
             the registry stays copyable, comparable and cheap — no hashing at
             lookup time and no ambient global (one instance per owner).
*/

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "shs/renderpath/planning/pass_id.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
    {
    class PassIdRegistry
    {
    public:
        // How many distinct consumer-owned passes fit in the open range.
        static constexpr std::size_t capacity()
        {
            return static_cast<std::size_t>(kPassIdOpenMax) -
                   static_cast<std::size_t>(kPassIdOpenBase) + 1u;
        }

        // Deterministic offset of a name inside the open range (FNV-1a 32 with a
        // final mix, folded to the range). Pure function of the name: no state,
        // no iteration order, no ambient seed.
        static constexpr uint16_t open_offset(std::string_view name)
        {
            uint32_t h = 2166136261u;
            for (const char c : name)
            {
                h ^= static_cast<uint32_t>(static_cast<unsigned char>(c));
                h *= 16777619u;
            }
            h ^= h >> 16;
            h *= 0x7feb352du;
            h ^= h >> 15;
            return static_cast<uint16_t>(h % static_cast<uint32_t>(capacity()));
        }

        // Resolve a name to a typed pass id. Total (no throw, no partial state):
        //   - a builtin name ("taa") resolves to its builtin id and registers
        //     nothing, so a consumer can never shadow a core pass;
        //   - an already-registered name returns the same id (idempotent);
        //   - a new name is registered at its content-addressed slot;
        //   - empty, "unknown", out-of-range, or a slot already taken by a
        //     *different* name returns nullopt (loud collision, no overwrite).
        std::optional<PassId> intern(std::string_view name)
        {
            if (name.empty() || name == pass_id_name(PassId::Unknown)) return std::nullopt;

            const PassId builtin = parse_pass_id(name);
            if (pass_id_is_builtin(builtin)) return builtin;

            const uint16_t offset = open_offset(name);
            const PassId id = static_cast<PassId>(static_cast<uint16_t>(kPassIdOpenBase + offset));

            for (const auto& slot : slots_)
            {
                if (slot.first == offset)
                {
                    // Same name => same id. Different name => collision: refuse.
                    return slot.second == name ? std::optional<PassId>(id) : std::nullopt;
                }
            }

            slots_.emplace_back(offset, std::string(name));
            return id;
        }

        // Name of a typed id: the builtin table, or the name that id was
        // derived from (which this registry registered). nullopt for the null id
        // (Unknown), out-of-range ids, and ids this registry never registered —
        // importantly, a foreign id can only miss, never alias another pass.
        std::optional<std::string_view> try_name(PassId id) const
        {
            if (pass_id_is_builtin(id)) return std::string_view(pass_id_name(id));
            if (!pass_id_is_open(id)) return std::nullopt;

            const uint16_t offset = static_cast<uint16_t>(
                static_cast<uint16_t>(id) - static_cast<uint16_t>(kPassIdOpenBase));
            for (const auto& slot : slots_)
            {
                if (slot.first == offset) return std::string_view(slot.second);
            }
            return std::nullopt;
        }

        // True for a builtin id or an open id this registry registered.
        bool contains(PassId id) const { return try_name(id).has_value(); }

        // True only for an open id this registry registered.
        bool is_open(PassId id) const
        {
            return pass_id_is_open(id) && contains(id);
        }

        // Registered pairs in registration order (deterministic; the *ids*
        // themselves are order-independent, see `open_offset`).
        const std::vector<std::pair<uint16_t, std::string>>& registered() const { return slots_; }

        std::size_t open_count() const { return slots_.size(); }
        bool empty() const { return slots_.empty(); }
        void clear() { slots_.clear(); }

        // Value semantics, order-independent: two registries are equal iff they
        // registered the same names (hence the same content-addressed ids).
        bool operator==(const PassIdRegistry& other) const
        {
            if (slots_.size() != other.slots_.size()) return false;
            for (const auto& slot : slots_)
            {
                const PassId id =
                    static_cast<PassId>(static_cast<uint16_t>(kPassIdOpenBase + slot.first));
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
