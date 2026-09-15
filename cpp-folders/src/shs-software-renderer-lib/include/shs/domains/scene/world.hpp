#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: world.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн scene модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <memory>
#include <vector>

#include "shs/scene/system.hpp"

namespace shs
{
    class World
    {
    public:
        template<typename TSystem, typename... Args>
        TSystem& add_system(Args&&... args)
        {
            auto s = std::make_unique<TSystem>(std::forward<Args>(args)...);
            TSystem& ref = *s;
            systems_.push_back(std::move(s));
            return ref;
        }

        void tick(float dt)
        {
            for (auto& s : systems_) s->tick(dt);
        }

    private:
        std::vector<std::unique_ptr<ISystem>> systems_{};
    };
}

