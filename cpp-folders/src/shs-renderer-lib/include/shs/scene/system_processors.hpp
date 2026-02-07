#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: system_processors.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн scene модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <memory>
#include <utility>
#include <vector>

#include "shs/core/context.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_registry.hpp"
#include "shs/pipeline/pluggable_pipeline.hpp"
#include "shs/scene/scene_objects.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs
{
    struct LogicSystemContext
    {
        float dt = 0.0f;
        float time = 0.0f;
        SceneObjectSet* objects = nullptr;
        Scene* scene = nullptr;
        FrameParams* frame = nullptr;
    };

    class ILogicSystem
    {
    public:
        virtual ~ILogicSystem() = default;
        virtual void tick(LogicSystemContext& ctx) = 0;
    };

    class LogicSystemProcessor
    {
    public:
        template <typename TSystem, typename... Args>
        TSystem& add_system(Args&&... args)
        {
            auto s = std::make_unique<TSystem>(std::forward<Args>(args)...);
            TSystem& ref = *s;
            systems_.push_back(std::move(s));
            return ref;
        }

        void tick(LogicSystemContext& ctx)
        {
            for (auto& s : systems_) s->tick(ctx);
        }

    private:
        std::vector<std::unique_ptr<ILogicSystem>> systems_{};
    };

    struct RenderSystemContext
    {
        Context* ctx = nullptr;
        Scene* scene = nullptr;
        FrameParams* frame = nullptr;
        RTRegistry* rtr = nullptr;
    };

    class IRenderSystem
    {
    public:
        virtual ~IRenderSystem() = default;
        virtual void render(RenderSystemContext& ctx) = 0;
    };

    class RenderSystemProcessor
    {
    public:
        template <typename TSystem, typename... Args>
        TSystem& add_system(Args&&... args)
        {
            auto s = std::make_unique<TSystem>(std::forward<Args>(args)...);
            TSystem& ref = *s;
            systems_.push_back(std::move(s));
            return ref;
        }

        void render(RenderSystemContext& ctx)
        {
            for (auto& s : systems_) s->render(ctx);
        }

    private:
        std::vector<std::unique_ptr<IRenderSystem>> systems_{};
    };

    class PipelineRenderSystem final : public IRenderSystem
    {
    public:
        explicit PipelineRenderSystem(PluggablePipeline* pipeline) : pipeline_(pipeline) {}

        void render(RenderSystemContext& ctx) override
        {
            if (!pipeline_ || !ctx.ctx || !ctx.scene || !ctx.frame || !ctx.rtr) return;
            pipeline_->execute(*ctx.ctx, *ctx.scene, *ctx.frame, *ctx.rtr);
        }

    private:
        PluggablePipeline* pipeline_ = nullptr;
    };
}
