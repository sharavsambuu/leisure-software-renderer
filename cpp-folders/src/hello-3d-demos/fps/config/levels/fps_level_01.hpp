#pragma once

// ============================================================================
// fps/config/levels/fps_level_01.hpp — fps::config::FpsLevel01
// Arena dimensions, spawn data, sun direction, and the color palette.
// Pure data; consumed by the spatial_fx mesh builders and the matrix reducer
// (arena bounds for clamping).
// ============================================================================

#include <glm/glm.hpp>

#include "shs_renderer.hpp"

namespace fps::config {

    struct BotSpawn {
        glm::vec3 position;
        glm::vec3 waypoint;
    };

    struct FpsLevel01 {
        // --- Arena geometry ---
        float arena_half_size = 16.0f;
        float wall_height     = 4.5f;
        int   floor_tiles     = 16;

        // Player movement clamp margin (inside the walls)
        float player_clamp_margin = 1.5f;  // clamp to +- (half - margin)
        float bot_clamp_margin    = 2.0f;

        // --- Spawns ---
        glm::vec3 player_spawn_position{ 0.0f, 1.70f, -8.0f };
        float     player_spawn_yaw   = 0.0f;
        float     player_spawn_pitch = 0.0f;
        float     player_eye_height  = 1.70f;

        static constexpr int BOT_COUNT = 4;
        BotSpawn bot_spawns[BOT_COUNT] = {
            { glm::vec3(-7.0f, 0.0f,  6.0f), glm::vec3(-7.0f, 0.0f,  6.0f) },
            { glm::vec3( 7.0f, 0.0f,  6.0f), glm::vec3( 7.0f, 0.0f,  6.0f) },
            { glm::vec3(-6.0f, 0.0f, -4.0f), glm::vec3(-6.0f, 0.0f, -4.0f) },
            { glm::vec3( 6.0f, 0.0f, -4.0f), glm::vec3( 6.0f, 0.0f, -4.0f) },
        };

        // --- Lighting ---
        glm::vec3 sun_dir_world = glm::normalize(glm::vec3(0.45f, -0.85f, 0.35f));

        // --- Palette ---
        shs::Color floor_dark   { 45, 52, 60, 255 };
        shs::Color floor_light  { 65, 75, 88, 255 };
        shs::Color wall_base    { 95, 105, 118, 255 };
        shs::Color wall_trim    { 130, 140, 155, 255 };
        shs::Color platform_top { 180, 140, 80, 255 };
        shs::Color platform_side{ 120, 95, 60, 255 };
        shs::Color pillar       { 140, 145, 155, 255 };
        shs::Color crate_wood   { 165, 110, 60, 255 };
        shs::Color crate_dark   { 120, 75, 40, 255 };

        // Derived helpers
        float player_clamp_bound() const { return arena_half_size - player_clamp_margin; }
        float bot_clamp_bound() const { return arena_half_size - bot_clamp_margin; }
    };

} // namespace fps::config