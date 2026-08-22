#pragma once

// ============================================================================
// fps/domains/spatial_fx/fps.meshes.hpp — CONTENT BUILDERS (pure)
// Startup-time mesh construction: arena from level data, actor meshes
// (bot normal/hit-flash, viewmodel gun, muzzle flash, projectile bolt).
// ============================================================================

#include <vector>

#include <glm/glm.hpp>

#include "spatial_fx.contract.hpp"

#include <config/levels/fps_level_01.hpp>

namespace fps::spatial_fx {

    // --- Arena: floor, walls, trim, platform, pillars, crates ---------------
    inline std::vector<LowPolyTriangle> build_arena_mesh(const config::FpsLevel01& lvl) {
        std::vector<LowPolyTriangle> tris;

        const float S   = lvl.arena_half_size;
        const float H   = lvl.wall_height;
        const int   T   = lvl.floor_tiles;
        const float TSZ = (S * 2.0f) / static_cast<float>(T);

        for (int iz = 0; iz < T; ++iz) {
            const float z0 = -S + static_cast<float>(iz) * TSZ;
            const float z1 = z0 + TSZ;
            for (int ix = 0; ix < T; ++ix) {
                const float x0 = -S + static_cast<float>(ix) * TSZ;
                const float x1 = x0 + TSZ;

                const shs::Color c = ((ix + iz) % 2 == 0) ? lvl.floor_dark : lvl.floor_light;

                const glm::vec3 p00(x0, 0.0f, z0);
                const glm::vec3 p10(x1, 0.0f, z0);
                const glm::vec3 p11(x1, 0.0f, z1);
                const glm::vec3 p01(x0, 0.0f, z1);

                MeshBuilder::add_quad(tris, p00, p01, p11, p10, c);
            }
        }

        MeshBuilder::add_quad(tris, glm::vec3(-S, 0, -S), glm::vec3(-S, H, -S), glm::vec3( S, H, -S), glm::vec3( S, 0, -S), lvl.wall_base);
        MeshBuilder::add_quad(tris, glm::vec3( S, 0,  S), glm::vec3( S, H,  S), glm::vec3(-S, H,  S), glm::vec3(-S, 0,  S), lvl.wall_base);
        MeshBuilder::add_quad(tris, glm::vec3( S, 0, -S), glm::vec3( S, 0,  S), glm::vec3( S, H,  S), glm::vec3( S, H, -S), lvl.wall_base);
        MeshBuilder::add_quad(tris, glm::vec3(-S, 0,  S), glm::vec3(-S, 0, -S), glm::vec3(-S, H, -S), glm::vec3(-S, H,  S), lvl.wall_base);

        MeshBuilder::add_box(tris, glm::vec3( 0, H + 0.15f, -S), glm::vec3(S * 2.0f, 0.3f, 0.6f), lvl.wall_trim, lvl.wall_trim, lvl.wall_trim);
        MeshBuilder::add_box(tris, glm::vec3( 0, H + 0.15f,  S), glm::vec3(S * 2.0f, 0.3f, 0.6f), lvl.wall_trim, lvl.wall_trim, lvl.wall_trim);
        MeshBuilder::add_box(tris, glm::vec3( S, H + 0.15f,  0), glm::vec3(0.6f, 0.3f, S * 2.0f), lvl.wall_trim, lvl.wall_trim, lvl.wall_trim);
        MeshBuilder::add_box(tris, glm::vec3(-S, H + 0.15f,  0), glm::vec3(0.6f, 0.3f, S * 2.0f), lvl.wall_trim, lvl.wall_trim, lvl.wall_trim);

        MeshBuilder::add_box(tris, glm::vec3(0, 0.25f, 0), glm::vec3(7.0f, 0.5f, 7.0f), lvl.platform_top, lvl.platform_side, lvl.platform_side);

        constexpr float P_OFF = 8.5f;
        MeshBuilder::add_cylinder(tris, glm::vec3(-P_OFF, 0, -P_OFF), 1.1f, H, 8, lvl.pillar);
        MeshBuilder::add_cylinder(tris, glm::vec3( P_OFF, 0, -P_OFF), 1.1f, H, 8, lvl.pillar);
        MeshBuilder::add_cylinder(tris, glm::vec3(-P_OFF, 0,  P_OFF), 1.1f, H, 8, lvl.pillar);
        MeshBuilder::add_cylinder(tris, glm::vec3( P_OFF, 0,  P_OFF), 1.1f, H, 8, lvl.pillar);

        MeshBuilder::add_box(tris, glm::vec3(-4.5f, 0.6f, 3.0f), glm::vec3(1.2f, 1.2f, 1.2f), lvl.crate_wood, lvl.crate_dark, lvl.crate_dark);
        MeshBuilder::add_box(tris, glm::vec3( 4.5f, 0.6f, -3.5f), glm::vec3(1.2f, 1.2f, 1.2f), lvl.crate_wood, lvl.crate_dark, lvl.crate_dark);
        MeshBuilder::add_box(tris, glm::vec3( 5.5f, 0.5f,  6.0f), glm::vec3(1.0f, 1.0f, 1.0f), lvl.crate_wood, lvl.crate_dark, lvl.crate_dark);

        return tris;
    }

    // --- Bot: hit_flash variant swaps all colors to white --------------------
    inline std::vector<LowPolyTriangle> build_bot_mesh(bool hit_flash) {
        std::vector<LowPolyTriangle> tris;

        const shs::Color armor  = hit_flash ? shs::Color{ 255, 255, 255, 255 } : shs::Color{ 60 , 120, 190, 255 };
        const shs::Color joints = hit_flash ? shs::Color{ 255, 200, 200, 255 } : shs::Color{ 40 , 45 , 50 , 255 };
        const shs::Color visor  = hit_flash ? shs::Color{ 255, 255, 255, 255 } : shs::Color{ 240, 60 , 50 , 255 };
        const shs::Color metal  = hit_flash ? shs::Color{ 255, 255, 255, 255 } : shs::Color{ 160, 170, 180, 255 };

        MeshBuilder::add_box(tris, glm::vec3( 0    , 0.95f,  0    ), glm::vec3(0.65f, 0.70f, 0.38f), armor, armor, joints);
        MeshBuilder::add_box(tris, glm::vec3( 0    , 1.55f,  0    ), glm::vec3(0.42f, 0.42f, 0.42f), armor, metal, joints);
        MeshBuilder::add_box(tris, glm::vec3( 0    , 1.58f,  0.22f), glm::vec3(0.32f, 0.14f, 0.06f), visor, visor, visor, -0.001f);
        MeshBuilder::add_box(tris, glm::vec3( 0    , 1.05f, -0.24f), glm::vec3(0.35f, 0.45f, 0.15f), visor, metal, joints);

        MeshBuilder::add_box(tris, glm::vec3(-0.48f, 0.95f,  0.0f ), glm::vec3(0.20f, 0.65f, 0.20f), metal, joints, metal);
        MeshBuilder::add_box(tris, glm::vec3( 0.48f, 0.95f,  0.0f ), glm::vec3(0.20f, 0.65f, 0.20f), metal, joints, metal);

        MeshBuilder::add_box(tris, glm::vec3(-0.20f, 0.32f,  0.0f ), glm::vec3(0.22f, 0.65f, 0.22f), metal, joints, joints);
        MeshBuilder::add_box(tris, glm::vec3( 0.20f, 0.32f,  0.0f ), glm::vec3(0.22f, 0.65f, 0.22f), metal, joints, joints);

        return tris;
    }

    // --- Viewmodel gun --------------------------------------------------------
    inline std::vector<LowPolyTriangle> build_gun_mesh() {
        std::vector<LowPolyTriangle> tris;

        const shs::Color metal_dark = shs::Color{ 45 , 48 , 55 , 255 };
        const shs::Color metal_body = shs::Color{ 80 , 85 , 95 , 255 };
        const shs::Color grip_wood  = shs::Color{ 130, 75 , 45 , 255 };
        const shs::Color glow_cyan  = shs::Color{ 40 , 220, 240, 255 };

        MeshBuilder::add_box(tris, glm::vec3(0, -0.15f, -0.05f), glm::vec3(0.08f, 0.25f, 0.12f), grip_wood , grip_wood , grip_wood );
        MeshBuilder::add_box(tris, glm::vec3(0,  0.02f,  0.08f), glm::vec3(0.10f, 0.12f, 0.35f), metal_body, metal_dark, metal_dark);
        MeshBuilder::add_box(tris, glm::vec3(0,  0.04f,  0.32f), glm::vec3(0.06f, 0.06f, 0.25f), metal_dark, metal_dark, metal_dark);
        MeshBuilder::add_box(tris, glm::vec3(0,  0.10f,  0.05f), glm::vec3(0.04f, 0.04f, 0.22f), glow_cyan , metal_dark, metal_dark);

        return tris;
    }

    // --- Muzzle flash star -----------------------------------------------------
    inline std::vector<LowPolyTriangle> build_muzzle_flash() {
        std::vector<LowPolyTriangle> tris;
        const shs::Color c_bright = shs::Color{ 255, 240, 150, 255 };
        const shs::Color c_orange = shs::Color{ 255, 120, 30, 255 };

        auto add_spike = [&](glm::vec3 dir, float len, float w) {
            const glm::vec3 side = glm::normalize(glm::cross(dir, glm::vec3(0, 1, 0))) * w;
            const glm::vec3 tip  = dir * len;
            tris.emplace_back(-side, side, tip, c_bright, -0.002f);
            tris.emplace_back(side, -side, tip, c_orange, -0.002f);
        };

        add_spike(glm::vec3( 0   ,     0, 1.0f), 0.35f, 0.08f);
        add_spike(glm::vec3( 0.7f,  0.3f, 0.6f), 0.25f, 0.06f);
        add_spike(glm::vec3(-0.7f,  0.3f, 0.6f), 0.25f, 0.06f);
        add_spike(glm::vec3( 0.0f,  0.8f, 0.5f), 0.22f, 0.06f);
        add_spike(glm::vec3( 0.0f, -0.6f, 0.5f), 0.22f, 0.06f);

        return tris;
    }

    // --- Projectile bolt ---------------------------------------------------------
    inline std::vector<LowPolyTriangle> build_projectile_mesh() {
        std::vector<LowPolyTriangle> tris;
        const shs::Color plasma_core   = shs::Color{ 255, 60, 40, 255 };
        const shs::Color plasma_orange = shs::Color{ 255, 180, 50, 255 };
        MeshBuilder::add_box(tris, glm::vec3(0), glm::vec3(0.20f, 0.20f, 0.35f), plasma_orange, plasma_core, plasma_core);
        return tris;
    }

} // namespace fps::spatial_fx