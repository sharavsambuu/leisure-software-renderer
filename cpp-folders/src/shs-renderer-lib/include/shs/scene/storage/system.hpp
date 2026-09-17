#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: system.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн scene модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace scene
    {
    class ISystem
    {
    public:
        virtual ~ISystem() = default;
        virtual void tick(float dt) = 0;
    };

    } // inline namespace scene
}

