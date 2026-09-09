#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: free_camera.hpp
    МОДУЛЬ: camera
    ЗОРИЛГО: Demo-уудад зориулсан чөлөөт камер (Free Camera) controller.
            WASD + Mouse look + Speed boost дэмжинэ.
            LH (+Z forward) систем ашиглана.
*/

#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "shs/camera/camera_math.hpp"
#include "shs/camera/convention.hpp"
#include "shs/platform/platform_input.hpp"

namespace shs
{
    /**
     * @brief PlatformInputState-ээр удирдагддаг чөлөөт камер.
     */
    struct FreeCamera
    {
        glm::vec3 pos{0.0f, 14.0f, -28.0f};
        float yaw = glm::half_pi<float>();
        float pitch = -0.25f;

        float move_speed = 20.0f;
        float look_speed = 0.003f;

        // WSL2/Remote-д гарч болох mouse spike-ийг шүүх утгууд
        static constexpr float kMouseSpikeThreshold = 180.0f;
        static constexpr float kMouseDeltaClamp = 70.0f;

        /**
         * @brief Оролтын төлөвөөр камерын байрлал болон өнцгийг шинэчилнэ.
         */
        void update(const PlatformInputState& input, float dt)
            const glm::vec3 right = right_from_forward(fwd);
            const glm::vec3 up{0.0f, 1.0f, 0.0f};

            const float current_speed = move_speed * (input.boost ? 2.0f : 1.0f);

            if (input.forward)  pos += fwd * current_speed * dt;
            if (input.backward) pos -= fwd * current_speed * dt;
            
            if (input.left)     pos -= right * current_speed * dt;
            if (input.right)    pos += right * current_speed * dt;
            
            if (input.ascend)   pos += up * current_speed * dt;
            if (input.descend)  pos -= up * current_speed * dt;
        }

        /**
         * @brief Камерын харах матрицыг буцаана (LH).
         */
        glm::mat4 get_view_matrix() const
        {
            return look_at_lh(pos, pos + forward_from_yaw_pitch(yaw, pitch), glm::vec3(0.0f, 1.0f, 0.0f));
        }

        glm::vec3 forward_vector() const
        {
            return forward_from_yaw_pitch(yaw, pitch);
        }

        glm::vec3 right_vector() const
        {
            return right_from_forward(forward_vector());
        }
    };
}
