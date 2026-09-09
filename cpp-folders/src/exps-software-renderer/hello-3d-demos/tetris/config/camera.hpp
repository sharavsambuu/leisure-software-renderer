#pragma once
// tetris/config/camera.hpp — PER-LEVEL CAMERA DEFINITION (tetris::config::CameraConfig)
// Plain-data camera preset carried inside Rules: every level configures its own
// perspective (eye position, look-at target, vertical FOV, clip planes). The
// planner consumes it as read-only facts; FX dynamics (shake/pulse) are applied
// ON TOP of the preset as small offsets, never replacing it.
//
// Framing math (vertical fit at the target plane):
//   visible_height = 2 * distance * tan(fov/2)
// The board is 22 world-units tall and 10 wide, centered near y≈11. The old
// hardcoded shot (fov 60°, distance ≈18.4 → ~21.3 units of vertical coverage)
// cropped the stack and let HUD panels overlap pieces; the default below pulls
// back to ~26 units of coverage so the full board fits with clearance.
#include <glm/glm.hpp>

namespace tetris::config {

    struct CameraConfig {
        glm::vec3 eye    { 0.0f, 12.0f, -25.0f };   // camera position (LH space)
        glm::vec3 target { 0.0f, 10.5f,   0.0f };   // look-at point (board center-ish)
        float     fov_deg = 55.0f;                  // vertical FOV
        float     near_z  = 0.15f;
        float     far_z   = 150.0f;
    };

} // namespace tetris::config