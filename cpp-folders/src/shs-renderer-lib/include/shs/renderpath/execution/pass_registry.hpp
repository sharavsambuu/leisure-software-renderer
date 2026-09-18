#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_registry.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Pass-уудыг id-аар бүртгэж, runtime дээр үйлдвэр (factory)-ээр үүсгэх
            registry abstraction өгнө.
*/


#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "shs/renderpath/planning/pass_id.hpp"
#include "shs/renderpath/planning/pass_contract.hpp"
#include "shs/renderpath/execution/pass_id_registry.hpp"
#include "shs/renderpath/execution/render_pass.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
    {
    struct PassFactoryDescriptor
    {
        TechniquePassContract contract{};
        uint32_t backend_mask = 0u;
        bool has_contract = false;
        bool backend_mask_known = false;
    };

    class PassFactoryRegistry
    {
    public:
        using Factory = std::function<std::unique_ptr<IRenderPass>()>;

        static constexpr uint32_t backend_bit(RenderBackendType t)
        {
            return 1u << static_cast<uint32_t>(t);
        }

        static constexpr uint32_t backend_mask_all()
        {
            return backend_bit(RenderBackendType::Software) |
                   backend_bit(RenderBackendType::OpenGL) |
                   backend_bit(RenderBackendType::Vulkan);
        }

        bool register_factory(const std::string& id, Factory factory)
        {
            if (id.empty() || !factory) return false;
            factories_[id] = std::move(factory);
            return true;
        }

        // Typed overload: a builtin id, or an open id this registry minted
        // (`intern_pass_id`). An unresolvable id is rejected outright — the key
        // is never guessed, so a mistyped consumer id cannot alias a core pass.
        bool register_factory(PassId id, Factory factory)
        {
            const std::string key = typed_key(id);
            if (key.empty()) return false;
            return register_factory(key, std::move(factory));
        }

        // Verified typed registration: the id must resolve to exactly this name
        // here. A foreign or colliding id therefore cannot bind a consumer pass
        // to the wrong factory — the pairing (id, name) is what is registered,
        // and the name is what execution keys on.
        bool register_factory(PassId id, std::string_view expected_name, Factory factory)
        {
            const std::optional<std::string_view> name = pass_ids_.try_name(id);
            if (!name.has_value() || *name != expected_name) return false;
            return register_factory(std::string(*name), std::move(factory));
        }

        // Verified typed query, same rule as above.
        bool has(PassId id, std::string_view expected_name) const
        {
            const std::optional<std::string_view> name = pass_ids_.try_name(id);
            if (!name.has_value() || *name != expected_name) return false;
            return has(std::string(*name));
        }

        bool has(const std::string& id) const
        {
            return factories_.find(id) != factories_.end();
        }

        bool has(PassId id) const
        {
            const std::string key = typed_key(id);
            if (key.empty()) return false;
            return has(key);
        }

        std::unique_ptr<IRenderPass> create(const std::string& id) const
        {
            auto it = factories_.find(id);
            if (it == factories_.end() || !it->second) return nullptr;
            return it->second();
        }

        std::unique_ptr<IRenderPass> create(PassId id) const
        {
            const std::string key = typed_key(id);
            if (key.empty()) return nullptr;
            return create(key);
        }

        std::vector<std::string> ids() const
        {
            std::vector<std::string> out{};
            out.reserve(factories_.size());
            for (const auto& kv : factories_) out.push_back(kv.first);
            return out;
        }

        bool register_descriptor(
            const std::string& id,
            const TechniquePassContract& contract,
            uint32_t backend_mask = backend_mask_all(),
            bool backend_mask_known = true)
        {
            if (id.empty()) return false;
            PassFactoryDescriptor d{};
            d.contract = contract;
            d.backend_mask = backend_mask;
            d.has_contract = true;
            d.backend_mask_known = backend_mask_known;
            descriptors_[id] = std::move(d);
            return true;
        }

        bool register_descriptor(
            PassId id,
            const TechniquePassContract& contract,
            uint32_t backend_mask = backend_mask_all(),
            bool backend_mask_known = true)
        {
            const std::string key = typed_key(id);
            if (key.empty()) return false;
            return register_descriptor(key, contract, backend_mask, backend_mask_known);
        }

        bool try_get_descriptor(std::string_view id, PassFactoryDescriptor& out) const
        {
            const auto it = descriptors_.find(std::string(id));
            if (it == descriptors_.end()) return false;
            out = it->second;
            return true;
        }

        bool try_get_descriptor(PassId id, PassFactoryDescriptor& out) const
        {
            const std::string key = typed_key(id);
            if (key.empty()) return false;
            return try_get_descriptor(key, out);
        }

        bool try_get_contract_hint(std::string_view id, TechniquePassContract& out) const
        {
            PassFactoryDescriptor d{};
            if (!try_get_descriptor(id, d)) return false;
            if (!d.has_contract) return false;
            out = d.contract;
            return true;
        }

        bool try_get_contract_hint(PassId id, TechniquePassContract& out) const
        {
            const std::string key = typed_key(id);
            if (key.empty()) return false;
            return try_get_contract_hint(key, out);
        }

        std::optional<bool> supports_backend_hint(std::string_view id, RenderBackendType backend) const
        {
            PassFactoryDescriptor d{};
            if (!try_get_descriptor(id, d)) return std::nullopt;
            if (!d.backend_mask_known) return std::nullopt;
            return (d.backend_mask & backend_bit(backend)) != 0u;
        }

        std::optional<bool> supports_backend_hint(PassId id, RenderBackendType backend) const
        {
            const std::string key = typed_key(id);
            if (key.empty()) return std::nullopt;
            return supports_backend_hint(key, backend);
        }

        std::optional<bool> supports_technique_mode_hint(std::string_view id, TechniqueMode mode) const
        {
            TechniquePassContract c{};
            if (!try_get_contract_hint(id, c)) return std::nullopt;
            return technique_mode_in_mask(c.supported_modes_mask, mode);
        }

        std::optional<bool> supports_technique_mode_hint(PassId id, TechniqueMode mode) const
        {
            const std::string key = typed_key(id);
            if (key.empty()) return std::nullopt;
            return supports_technique_mode_hint(key, mode);
        }

        // --- open registered range (Constitution I §7 — No User Lock-In) -----
        // Explicit, caller-owned: no ambient global, so a plan and its replay
        // stay deterministic. Builtin names always resolve to their builtin id,
        // so a consumer can never shadow a core pass.

        PassIdRegistry& pass_ids() noexcept { return pass_ids_; }
        const PassIdRegistry& pass_ids() const noexcept { return pass_ids_; }

        // Mint (or resolve) a consumer-owned pass id from its registered name.
        std::optional<PassId> intern_pass_id(std::string_view name)
        {
            return pass_ids_.intern(name);
        }

        // Registered name of a typed id (builtin table or minted open id).
        std::optional<std::string_view> pass_id_registered_name(PassId id) const
        {
            return pass_ids_.try_name(id);
        }

    private:
        // Factory/descriptor key of a typed id: its builtin name, or the name
        // minted for it here. Empty means "not resolvable in this registry" —
        // every typed overload treats that as a hard miss, and no key is ever
        // guessed from an id that has no name.
        std::string typed_key(PassId id) const
        {
            const std::optional<std::string_view> name = pass_ids_.try_name(id);
            return name.has_value() ? std::string(*name) : std::string{};
        }

        PassIdRegistry pass_ids_{};
        std::unordered_map<std::string, Factory> factories_{};
        std::unordered_map<std::string, PassFactoryDescriptor> descriptors_{};
    };

    } // inline namespace renderpath
}
