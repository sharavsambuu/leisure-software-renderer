#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: aabb.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Тэнхлэгтэй зэрэгцээ хязгаарын хайрцаг (AABB) тодорхойлох ба түүнийг өргөтгөх, 
            төв болон хэмжээг тооцоолох бүтэц.
*/

#include <glm/glm.hpp>
#include <algorithm>

namespace shs {

struct AABB {
    glm::vec3 minv{  1e30f };
    glm::vec3 maxv{ -1e30f };

    inline void expand(const glm::vec3& p) {
        minv = glm::min(minv, p);
        maxv = glm::max(maxv, p);
    }

    inline glm::vec3 center() const { return 0.5f * (minv + maxv); }
    inline glm::vec3 extent() const { return 0.5f * (maxv - minv); }
};

} // namespace shs
