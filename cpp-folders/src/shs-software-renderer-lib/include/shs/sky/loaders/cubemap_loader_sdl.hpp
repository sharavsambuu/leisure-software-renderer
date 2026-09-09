#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: cubemap_loader_sdl.hpp
    МОДУЛЬ: sky
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн sky модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>

#include "shs/resources/loaders/texture_loader_sdl.hpp"
#include "shs/sky/cubemap_sky.hpp"

namespace shs
{
    inline CubemapData load_cubemap_sdl_faces(
        const std::string& right,
        const std::string& left,
        const std::string& top,
        const std::string& bottom,
        const std::string& front,
        const std::string& back,
        bool flip_y = true
    )
    {
        CubemapData cm{};
        cm.face[0] = load_texture2d_sdl_image(right, flip_y);
        cm.face[1] = load_texture2d_sdl_image(left, flip_y);
        cm.face[2] = load_texture2d_sdl_image(top, flip_y);
        cm.face[3] = load_texture2d_sdl_image(bottom, flip_y);
        cm.face[4] = load_texture2d_sdl_image(front, flip_y);
        cm.face[5] = load_texture2d_sdl_image(back, flip_y);
        return cm;
    }

    inline CubemapData load_cubemap_sdl_folder(const std::string& folder, bool flip_y = true)
    {
        return load_cubemap_sdl_faces(
            folder + "/right.png",
            folder + "/left.png",
            folder + "/top.png",
            folder + "/bottom.png",
            folder + "/front.png",
            folder + "/back.png",
            flip_y
        );
    }
}

