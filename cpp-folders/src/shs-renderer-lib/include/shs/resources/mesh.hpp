#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: mesh.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace shs
{
    using MeshAssetHandle = uint32_t;

    struct MeshData
    {
        std::string source_path{};
        std::vector<glm::vec3> positions{};
        std::vector<glm::vec3> normals{};
        std::vector<glm::vec2> uvs{};
        std::vector<uint32_t> indices{};

        bool empty() const
        {
            return positions.empty() || indices.empty();
        }

        void clear()
        {
            positions.clear();
            normals.clear();
            uvs.clear();
            indices.clear();
        }
    };
}

