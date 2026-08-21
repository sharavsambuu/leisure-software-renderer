#pragma once

#include <algorithm>
#include <glm/glm.hpp>

// environment pod — reactive mood color. Listens to progression score/length; never mutates it.

namespace snake::environment {

    struct MoodState {
        glm::vec3 base_color = {0.08f, 0.09f, 0.12f};   // ambient arena tint (matrix::ArenaState)
        glm::vec3 mood_color = {0.0f, 0.0f, 0.0f};       // reactive pulse (plan pod blends this in)

        static MoodState fresh() { return {}; }
    };

    inline MoodState reduce_environment(float score, int length) {
        MoodState m;
        // Pulse hue from calm blue toward energetic magenta as the game intensifies.
        float t = std::min(1.0f, (float)(score + length * 5) / 200.0f);
        glm::vec3 calm(0.10f, 0.20f, 0.45f);   // blue
        glm::vec3 hot(0.65f, 0.15f, 0.75f);    // magenta
        m.mood_color = calm * (1.0f - t) + hot * t;
        return m;
    }

} // namespace snake::environment
