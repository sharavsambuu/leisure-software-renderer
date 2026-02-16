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
#include <string>
#include <unordered_map>
#include <vector>

#include "shs/pipeline/pass_id.hpp"
#include "shs/pipeline/render_pass.hpp"

namespace shs
{
    class PassFactoryRegistry
    {
    public:
        using Factory = std::function<std::unique_ptr<IRenderPass>()>;

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

    private:
        std::unordered_map<std::string, Factory> factories_{};
    };
}
