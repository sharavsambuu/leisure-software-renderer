#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: scene_instance.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Standard rendering instance definitions for demo culling loops.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <cstdint>
#include <glm/glm.hpp>
#include "shs/geometry/scene_shape.hpp"
#include "shs/geometry/jolt_adapter.hpp"

namespace shs
{
    struct AnimationState
    {
        glm::vec3 base_pos{0.0f};
        glm::vec3 base_rot{0.0f};
        glm::vec3 angular_vel{0.0f};
        bool animated = true;
    };

    struct SceneInstance
    {
        SceneShape geometry;
        uint32_t user_index = 0;
        glm::vec3 tint_color{1.0f};
        
        AnimationState anim{};

        bool visible = true;
        bool frustum_visible = true;
        bool occluded = false;
        bool casts_shadow = true;
    };
}

#endif // SHS_HAS_JOLT
