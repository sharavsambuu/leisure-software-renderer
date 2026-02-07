#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pluggable_pipeline.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн pipeline модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "shs/pipeline/render_pass.hpp"

namespace shs
{
    class PluggablePipeline
    {
    public:
        template<typename TPass, typename... Args>
        TPass& add_pass(Args&&... args)
        {
            auto p = std::make_unique<TPass>(std::forward<Args>(args)...);
            TPass& ref = *p;
            passes_.push_back(std::move(p));
            return ref;
        }

        IRenderPass* find(const std::string& pass_id)
        {
            for (auto& p : passes_)
            {
                if (pass_id == p->id()) return p.get();
            }
            return nullptr;
        }

        const IRenderPass* find(const std::string& pass_id) const
        {
            for (const auto& p : passes_)
            {
                if (pass_id == p->id()) return p.get();
            }
            return nullptr;
        }

        bool set_enabled(const std::string& pass_id, bool enabled)
        {
            if (auto* p = find(pass_id))
            {
                p->set_enabled(enabled);
                return true;
            }
            return false;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr)
        {
            for (auto& p : passes_)
            {
                if (p->enabled()) p->execute(ctx, scene, fp, rtr);
            }
        }

    private:
        std::vector<std::unique_ptr<IRenderPass>> passes_{};
    };
}
