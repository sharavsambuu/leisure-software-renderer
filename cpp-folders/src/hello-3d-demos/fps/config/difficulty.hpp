#pragma once

// ============================================================================
// fps/config/difficulty.hpp — fps::config::Difficulty
// Pure gameplay-tuning data. No SDL, no GLM math beyond vec3 storage.
// All values default to the original single-file demo's tuned constants.
// ============================================================================

#include <cstdint>
#include <glm/glm.hpp>

namespace fps::config {

    struct Difficulty {
        // --- Player ---
        float player_speed        = 7.0f;    // m/s horizontal
        float mouse_sensitivity   = 0.0035f; // rad per mouse count
        float key_look_rate_yaw   = 2.4f;    // rad/s via arrow keys
        float key_look_rate_pitch = 2.0f;    // rad/s via arrow keys
        float gravity             = 24.0f;   // m/s^2
        float jump_velocity       = 8.5f;    // initial upward velocity
        float max_pitch           = 1.4835f; // ~85 degrees

        // --- Weapon ---
        float fire_cooldown    = 0.18f;
        float fire_damage      = 35.0f;
        float recoil_offset    = 0.08f;
        float muzzle_flash_time = 0.05f;
        float hitmarker_time   = 0.15f;

        // --- Bots ---
        float bot_speed            = 3.8f;
        float chase_range          = 18.0f;
        float attack_range         = 9.0f;
        float attack_cooldown_base = 1.25f;
        float attack_cooldown_jitter = 0.40f; // + rand*JITTER
        float respawn_delay        = 4.0f;
        int16_t bot_max_hp         = 100;

        // --- Projectiles ---
        float projectile_speed    = 18.0f;
        float projectile_damage   = 15.0f;
        float projectile_lifetime = 3.0f;

        // --- Determinism ---
        uint32_t rng_seed = 0x853c49e6u; // LCG seed for bot AI (replaces rand())
    };

} // namespace fps::config