#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: device.hpp
    МОДУЛЬ: render
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн render модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <functional>

#include "shs/render/draw_packet.hpp"
#include "shs/render/target.hpp"

namespace shs
{
    class IRenderDevice
    {
    public:
        virtual ~IRenderDevice() = default;

        virtual RenderTargetHandle create_target(const RenderTargetDesc& desc) = 0;
        virtual void destroy_target(RenderTargetHandle h) = 0;
        virtual void begin_frame() = 0;
        virtual void end_frame() = 0;
        virtual void submit_draw(const DrawPacket& packet) = 0;
    };
}

