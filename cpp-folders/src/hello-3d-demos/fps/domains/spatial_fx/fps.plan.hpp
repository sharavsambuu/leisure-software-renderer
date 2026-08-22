#pragma once

// ============================================================================
// fps/domains/spatial_fx/fps.plan.hpp — plan_fps_scene (pure)
// (WorldSnapshot, meshes, canvas size) -> PipelineExecutionPlan.
// Builds the LH view/proj from the player's eye, transforms + lights every
// mesh batch into render-ready ProcessedTriangles.
//
// NOTE on the view matrix: this demo keeps glm::lookAtLH. Unlike snake's
// front-facing camera, the first-person base heading (+Z forward, yaw=0)
// yields side = cross(up, f) = (+1,0,0), which matches screen-right — so the
// known lookAtLH mirror quirk does NOT bite here. If you ever change the
// camera convention, re-verify with an autodrive screenshot FIRST.
// ============================================================================

#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "spatial_fx.contract.hpp"
#include "fps.meshes.hpp"

#include <domains/matrix/fps.contract.hpp>

namespace fps::spatial_fx {

    // Bundle of startup-built meshes consumed by the planner each frame.
    struct SceneMeshes {
        const std::vector<LowPolyTriangle>* arena        = nullptr;
        const std::vector<LowPolyTriangle>* bot_normal   = nullptr;
        const std::vector<LowPolyTriangle>* bot_flash    = nullptr;
        const std::vector<LowPolyTriangle>* gun          = nullptr;
        const std::vector<LowPolyTriangle>* muzzle_flash = nullptr;
        const std::vector<LowPolyTriangle>* bolt         = nullptr;
    };

    inline PipelineExecutionPlan plan_fps_scene(
        const matrix::WorldSnapshot& world,
        const SceneMeshes&           meshes,
        const glm::vec3&             sun_dir_world,
        float                        fov_degrees,
        float                        z_near,
        float                        z_far,
        int                          canvas_w,
        int                          canvas_h,
        std::pmr::memory_resource*   frame_arena
    ) {
        PipelineExecutionPlan plan(frame_arena);
        const size_t reserve =
            (meshes.arena ? meshes.arena->size() : 0)
            + world.bots.size() * 120
            + world.projectiles.size() * 12
            + 200;
        plan.triangles.reserve(reserve);

        const glm::vec3 eye = world.player.position;
        const glm::vec3 fwd = world.player.get_forward();

        plan.view_matrix = glm::lookAtLH(eye, eye + fwd, glm::vec3(0, 1, 0));
        plan.proj_matrix = glm::perspectiveLH_NO(glm::radians(fov_degrees),
                                                 static_cast<float>(canvas_w) / static_cast<float>(canvas_h),
                                                 z_near, z_far);
        plan.vp_matrix   = plan.proj_matrix * plan.view_matrix;

        const glm::vec3 L = -sun_dir_world;

        auto process_batch = [&](const std::vector<LowPolyTriangle>& batch, const glm::mat4& model) {
            const glm::mat4 mvp = plan.vp_matrix * model;

            for (const auto& tri : batch) {
                const glm::vec3 w0 = glm::vec3(model * glm::vec4(tri.p0, 1.0f));
                const glm::vec3 w1 = glm::vec3(model * glm::vec4(tri.p1, 1.0f));
                const glm::vec3 w2 = glm::vec3(model * glm::vec4(tri.p2, 1.0f));

                glm::vec3 N = glm::cross(w1 - w0, w2 - w0);
                const float len = glm::length(N);
                if (len < 1e-6f) continue;
                N /= len;

                const float diffuse = std::max(0.0f, glm::dot(N, L)) * 0.75f + 0.25f;
                const float ambient = std::max(0.0f, N.y) * 0.20f + 0.15f;

                const glm::vec3 base_col = glm::vec3(tri.color.r, tri.color.g, tri.color.b) / 255.0f;
                const glm::vec3 lit_rgb  = base_col * (diffuse * glm::vec3(1.0f, 0.98f, 0.92f)
                                                     + ambient * glm::vec3(0.50f, 0.70f, 1.0f));

                plan.triangles.push_back({
                    mvp * glm::vec4(tri.p0, 1.0f),
                    mvp * glm::vec4(tri.p1, 1.0f),
                    mvp * glm::vec4(tri.p2, 1.0f),
                    shs::rgb01_to_color(lit_rgb),
                    tri.depth_bias
                });
            }
        };

        if (meshes.arena) process_batch(*meshes.arena, glm::mat4(1.0f));

        for (size_t i = 0; i < world.bots.size(); ++i) {
            if (!meshes.bot_normal || !meshes.bot_flash) break;

            if (world.bots.state[i] == matrix::BotState::DEAD) {
                const glm::mat4 m = glm::translate(glm::mat4(1.0f), world.bots.position[i] + glm::vec3(0, 0.2f, 0))
                                  * glm::rotate(glm::mat4(1.0f), world.bots.yaw[i], glm::vec3(0, 1, 0))
                                  * glm::rotate(glm::mat4(1.0f), glm::radians(-80.0f), glm::vec3(1, 0, 0));
                process_batch(*meshes.bot_normal, m);
            } else {
                const float hover_y = std::sin(world.bots.bob_phase[i]) * 0.05f;
                const glm::mat4 m = glm::translate(glm::mat4(1.0f), world.bots.position[i] + glm::vec3(0, hover_y, 0))
                                  * glm::rotate(glm::mat4(1.0f), world.bots.yaw[i], glm::vec3(0, 1, 0));
                process_batch(world.bots.hit_flash_time[i] > 0.0f ? *meshes.bot_flash : *meshes.bot_normal, m);
            }
        }

        if (meshes.bolt) {
            for (size_t i = 0; i < world.projectiles.size(); ++i) {
                const glm::mat4 m = glm::translate(glm::mat4(1.0f), world.projectiles.position[i])
                                  * glm::scale(glm::mat4(1.0f), glm::vec3(1.2f));
                process_batch(*meshes.bolt, m);
            }
        }

        if (meshes.gun) {
            const glm::vec3 gun_offset(0.22f, -0.22f, 0.45f - world.player.recoil_offset);
            const glm::mat4 gun_rot = glm::rotate(glm::mat4(1.0f), world.player.yaw, glm::vec3(0, 1, 0))
                                    * glm::rotate(glm::mat4(1.0f), world.player.pitch, glm::vec3(-1, 0, 0));
            const glm::vec3 gun_world_pos = eye + glm::vec3(gun_rot * glm::vec4(gun_offset, 1.0f));
            const glm::mat4 gun_model = glm::translate(glm::mat4(1.0f), gun_world_pos)
                                      * gun_rot
                                      * glm::scale(glm::mat4(1.0f), glm::vec3(0.9f));
            process_batch(*meshes.gun, gun_model);

            if (world.player.muzzle_flash > 0.0f && meshes.muzzle_flash) {
                const glm::mat4 flash_model = gun_model * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.04f, 0.46f));
                process_batch(*meshes.muzzle_flash, flash_model);
            }
        }

        return plan;
    }

} // namespace fps::spatial_fx