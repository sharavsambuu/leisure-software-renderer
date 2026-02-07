#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: material.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>
#include <string>

#include <glm/glm.hpp>

#include "shs/resources/texture.hpp"

namespace shs
{
    using MaterialAssetHandle = uint32_t;

    struct MaterialData
    {
        std::string name{};

        glm::vec3 base_color{1.0f, 1.0f, 1.0f};
        float metallic = 0.0f;
        float roughness = 0.6f;
        float ao = 1.0f;

        glm::vec3 emissive_color{0.0f, 0.0f, 0.0f};
        float emissive_intensity = 0.0f;

        TextureAssetHandle base_color_tex = 0;
        TextureAssetHandle normal_tex = 0;
        TextureAssetHandle orm_tex = 0;
        TextureAssetHandle emissive_tex = 0;
    };
}

