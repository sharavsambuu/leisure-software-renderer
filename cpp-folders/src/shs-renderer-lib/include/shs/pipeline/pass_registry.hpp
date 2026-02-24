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

#include "shs/pipeline/pass_id.hpp"
#include "shs/pipeline/pass_contract.hpp"
#include "shs/pipeline/render_pass.hpp"

namespace shs
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

        bool register_factory(PassId id, Factory factory)
        {
            if (!pass_id_is_standard(id)) return false;
            return register_factory(pass_id_name(id), std::move(factory));
        }

        bool has(const std::string& id) const
        {
            return factories_.find(id) != factories_.end();
        }

        bool has(PassId id) const
        {
            if (!pass_id_is_standard(id)) return false;
            return has(pass_id_string(id));
        }

        std::unique_ptr<IRenderPass> create(const std::string& id) const
        {
            auto it = factories_.find(id);
            if (it == factories_.end() || !it->second) return nullptr;
            return it->second();
        }

        std::unique_ptr<IRenderPass> create(PassId id) const
        {
            if (!pass_id_is_standard(id)) return nullptr;
            return create(pass_id_string(id));
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
            if (!pass_id_is_standard(id)) return false;
            return register_descriptor(pass_id_string(id), contract, backend_mask, backend_mask_known);
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
            if (!pass_id_is_standard(id)) return false;
            return try_get_descriptor(pass_id_string(id), out);
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
            if (!pass_id_is_standard(id)) return false;
            return try_get_contract_hint(pass_id_string(id), out);
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
            if (!pass_id_is_standard(id)) return std::nullopt;
            return supports_backend_hint(pass_id_string(id), backend);
        }

        std::optional<bool> supports_technique_mode_hint(std::string_view id, TechniqueMode mode) const
        {
            TechniquePassContract c{};
            if (!try_get_contract_hint(id, c)) return std::nullopt;
            return technique_mode_in_mask(c.supported_modes_mask, mode);
        }

        std::optional<bool> supports_technique_mode_hint(PassId id, TechniqueMode mode) const
        {
            if (!pass_id_is_standard(id)) return std::nullopt;
            return supports_technique_mode_hint(pass_id_string(id), mode);
        }

    private:
        std::unordered_map<std::string, Factory> factories_{};
        std::unordered_map<std::string, PassFactoryDescriptor> descriptors_{};
    };
}
