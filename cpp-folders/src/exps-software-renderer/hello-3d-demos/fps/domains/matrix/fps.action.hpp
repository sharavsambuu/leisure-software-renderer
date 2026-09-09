#pragma once

// ============================================================================
// fps/domains/matrix/fps.action.hpp — reduce_user_commands (pure)
// Folds a frame's worth of UserCommands into one PlayerCommandFrame.
// ============================================================================

#include <span>
#include <type_traits>

#include <glm/glm.hpp>

#include "fps.contract.hpp"

namespace fps::matrix {

    inline PlayerCommandFrame reduce_user_commands(std::span<const UserCommand> commands) {
        PlayerCommandFrame out{};
        for (const auto& cmd : commands) {
            std::visit([&out](auto&& c) {
                using T = std::decay_t<decltype(c)>;
                if constexpr (std::is_same_v<T, MoveIntent>) {
                    out.move_dir += c.direction_xz;
                } else if constexpr (std::is_same_v<T, LookIntent>) {
                    out.delta_yaw   += c.delta_yaw;
                    out.delta_pitch += c.delta_pitch;
                } else if constexpr (std::is_same_v<T, JumpIntent>) {
                    out.jump_pressed = true;
                } else if constexpr (std::is_same_v<T, FireIntent>) {
                    out.fire_pressed = true;
                } else if constexpr (std::is_same_v<T, ResetIntent>) {
                    out.reset_pressed = true;
                }
            }, cmd);
        }
        if (glm::length(out.move_dir) > 0.01f) {
            out.move_dir = glm::normalize(out.move_dir);
        }
        return out;
    }

} // namespace fps::matrix