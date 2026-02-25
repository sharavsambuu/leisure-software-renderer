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
#include <functional>

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

    // VOP: Logic systems are just functions that mutate context/external state.
    // No "ISystem" inheritance. State should be managed by the caller (closures/structs).
    using LogicSystemTickFn = std::function<void(LogicSystemContext&)>;

    struct LogicSystem
    {
        std::string name;
        LogicSystemTickFn tick;
    };

    class LogicSystemProcessor
    {
    public:
        // Use add("name", [&state](auto& ctx) { state.tick(ctx); }) pattern.
        void add(std::string name, LogicSystemTickFn fn)
        {
            systems_.push_back({std::move(name), std::move(fn)});
        }

        void tick(LogicSystemContext& ctx)
        {
            for (auto& s : systems_) s.tick(ctx);
        }

    private:
        std::vector<LogicSystem> systems_{};
    };

    struct RenderSystemContext
    {
        Context* ctx = nullptr;
        Scene* scene = nullptr;
        FrameParams* frame = nullptr;
        RTRegistry* rtr = nullptr;
    };

    // VOP: Render systems are just functions.
    using RenderSystemDrawFn = std::function<void(RenderSystemContext&)>;

    struct RenderSystem
    {
        std::string name;
        RenderSystemDrawFn render;
    };

    class RenderSystemProcessor
    {
    public:
        void add(std::string name, RenderSystemDrawFn fn)
        {
            systems_.push_back({std::move(name), std::move(fn)});
        }

        void render(RenderSystemContext& ctx)
        {
            for (auto& s : systems_) s.render(ctx);
        }

    private:
        std::vector<RenderSystem> systems_{};
    };
}
