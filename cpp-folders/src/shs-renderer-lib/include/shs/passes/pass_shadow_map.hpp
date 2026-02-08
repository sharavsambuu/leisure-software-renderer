#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_shadow_map.hpp
    МОДУЛЬ: passes
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн passes модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/passes/pass_context.hpp"
#include "shs/scene/scene_types.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"
#include "shs/gfx/rt_shadow.hpp"
#include "shs/geometry/aabb.hpp"
#include "shs/camera/light_camera.hpp"
#include "shs/resources/resource_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#include <glm/gtc/matrix_transform.hpp>

namespace shs
{
    struct Context;

    class PassShadowMap
    {
    public:
        struct Inputs
        {
            const Scene* scene = nullptr;
            const FrameParams* fp = nullptr;
            RTRegistry* rtr = nullptr;

            RT_Shadow rt_shadow{};
        };

        const LightCamera& last_light_camera() const { return light_cam_; }

        void execute(Context& ctx, const Inputs& in)
        {
            ctx.shadow.reset();
            if (!in.scene || !in.fp || !in.rtr) return;
            if (!in.rt_shadow.valid()) return;
            if (!in.fp->pass.shadow.enable) return;

            auto* shadow = static_cast<RT_ShadowDepth*>(in.rtr->get(in.rt_shadow));
            if (!shadow || shadow->w <= 0 || shadow->h <= 0) return;

            shadow->clear(1.0f);

            auto make_model = [](const RenderItem& item) {
                glm::mat4 model(1.0f);
                model = glm::translate(model, item.tr.pos);
                model = glm::rotate(model, item.tr.rot_euler.x, glm::vec3(1.0f, 0.0f, 0.0f));
                model = glm::rotate(model, item.tr.rot_euler.y, glm::vec3(0.0f, 1.0f, 0.0f));
                model = glm::rotate(model, item.tr.rot_euler.z, glm::vec3(0.0f, 0.0f, 1.0f));
                model = glm::scale(model, item.tr.scl);
                return model;
            };

            auto barycentric_2d = [](const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
                const glm::vec2 v0 = b - a;
                const glm::vec2 v1 = c - a;
                const glm::vec2 v2 = p - a;
                const float den = v0.x * v1.y - v1.x * v0.y;
                if (std::abs(den) < 1e-8f) return glm::vec3(-1.0f);
                const float inv_den = 1.0f / den;
                const float v = (v2.x * v1.y - v1.x * v2.y) * inv_den;
                const float w = (v0.x * v2.y - v2.x * v0.y) * inv_den;
                const float u = 1.0f - v - w;
                return glm::vec3(u, v, w);
            };

            // Shadow camera фрустум таслахгүйн тулд world AABB-г консерватив байдлаар цуглуулна.
            AABB scene_aabb{};
            bool has_any_shadow_caster = false;
            using BoundsPair = std::pair<glm::vec3, glm::vec3>;
            static std::unordered_map<const MeshData*, BoundsPair> mesh_bounds_cache{};
            for (const auto& item : in.scene->items)
            {
                if (!item.visible || !item.casts_shadow) continue;
                const MeshData* mesh = (in.scene->resources) ? in.scene->resources->get_mesh((MeshAssetHandle)item.mesh) : nullptr;
                if (mesh && !mesh->positions.empty())
                {
                    const glm::mat4 model = make_model(item);
                    glm::vec3 bmin(std::numeric_limits<float>::max());
                    glm::vec3 bmax(std::numeric_limits<float>::lowest());
                    auto it = mesh_bounds_cache.find(mesh);
                    if (it == mesh_bounds_cache.end())
                    {
                        for (const glm::vec3& p : mesh->positions)
                        {
                            bmin = glm::min(bmin, p);
                            bmax = glm::max(bmax, p);
                        }
                        mesh_bounds_cache.emplace(mesh, BoundsPair{bmin, bmax});
                    }
                    else
                    {
                        bmin = it->second.first;
                        bmax = it->second.second;
                    }

                    const glm::vec3 c[8] = {
                        {bmin.x, bmin.y, bmin.z}, {bmax.x, bmin.y, bmin.z},
                        {bmin.x, bmax.y, bmin.z}, {bmax.x, bmax.y, bmin.z},
                        {bmin.x, bmin.y, bmax.z}, {bmax.x, bmin.y, bmax.z},
                        {bmin.x, bmax.y, bmax.z}, {bmax.x, bmax.y, bmax.z}
                    };
                    for (const glm::vec3& lc : c)
                    {
                        scene_aabb.expand(glm::vec3(model * glm::vec4(lc, 1.0f)));
                    }
                }
                else
                {
                    scene_aabb.expand(item.tr.pos);
                }
                has_any_shadow_caster = true;
            }

            if (!has_any_shadow_caster)
            {
                scene_aabb.expand(glm::vec3(-1.0f));
                scene_aabb.expand(glm::vec3(1.0f));
            }

            light_cam_ = build_dir_light_camera_aabb(
                in.scene->sun.dir_ws,
                scene_aabb,
                10.0f,
                static_cast<uint32_t>(std::max(shadow->w, 1)));
            // Энэ frame-ийн shadow sampling-д хэрэгтэй runtime төлөвийг context-д хадгална.
            ctx.shadow.map = shadow;
            ctx.shadow.light_viewproj = light_cam_.viewproj;
            ctx.shadow.valid = true;

            // Shadow map нь depth-only буфер тул зөвхөн хамгийн ойрын z01-ийг үлдээнэ.
            for (const auto& item : in.scene->items)
            {
                if (!item.visible || !item.casts_shadow) continue;
                if (!in.scene->resources) continue;
                const MeshData* mesh = in.scene->resources->get_mesh((MeshAssetHandle)item.mesh);
                if (!mesh || mesh->positions.empty()) continue;

                const glm::mat4 model = make_model(item);
                const bool indexed = !mesh->indices.empty();
                const size_t tri_count = indexed ? (mesh->indices.size() / 3) : (mesh->positions.size() / 3);

                for (size_t ti = 0; ti < tri_count; ++ti)
                {
                    uint32_t i0 = indexed ? mesh->indices[ti * 3 + 0] : (uint32_t)(ti * 3 + 0);
                    uint32_t i1 = indexed ? mesh->indices[ti * 3 + 1] : (uint32_t)(ti * 3 + 1);
                    uint32_t i2 = indexed ? mesh->indices[ti * 3 + 2] : (uint32_t)(ti * 3 + 2);
                    if (i0 >= mesh->positions.size() || i1 >= mesh->positions.size() || i2 >= mesh->positions.size()) continue;

                    const glm::vec3 p0 = glm::vec3(model * glm::vec4(mesh->positions[i0], 1.0f));
                    const glm::vec3 p1 = glm::vec3(model * glm::vec4(mesh->positions[i1], 1.0f));
                    const glm::vec3 p2 = glm::vec3(model * glm::vec4(mesh->positions[i2], 1.0f));
                    const glm::vec4 c0 = light_cam_.viewproj * glm::vec4(p0, 1.0f);
                    const glm::vec4 c1 = light_cam_.viewproj * glm::vec4(p1, 1.0f);
                    const glm::vec4 c2 = light_cam_.viewproj * glm::vec4(p2, 1.0f);
                    if (std::abs(c0.w) < 1e-8f || std::abs(c1.w) < 1e-8f || std::abs(c2.w) < 1e-8f) continue;

                    const glm::vec3 n0 = glm::vec3(c0) / c0.w;
                    const glm::vec3 n1 = glm::vec3(c1) / c1.w;
                    const glm::vec3 n2 = glm::vec3(c2) / c2.w;

                    // Бүх орой нэг талаараа NDC-гээс гарсан бол early reject.
                    if ((n0.x < -1.0f && n1.x < -1.0f && n2.x < -1.0f) || (n0.x > 1.0f && n1.x > 1.0f && n2.x > 1.0f)) continue;
                    if ((n0.y < -1.0f && n1.y < -1.0f && n2.y < -1.0f) || (n0.y > 1.0f && n1.y > 1.0f && n2.y > 1.0f)) continue;
                    if ((n0.z < -1.0f && n1.z < -1.0f && n2.z < -1.0f) || (n0.z > 1.0f && n1.z > 1.0f && n2.z > 1.0f)) continue;

                    const glm::vec2 s0{(n0.x * 0.5f + 0.5f) * (float)(shadow->w - 1), (n0.y * 0.5f + 0.5f) * (float)(shadow->h - 1)};
                    const glm::vec2 s1{(n1.x * 0.5f + 0.5f) * (float)(shadow->w - 1), (n1.y * 0.5f + 0.5f) * (float)(shadow->h - 1)};
                    const glm::vec2 s2{(n2.x * 0.5f + 0.5f) * (float)(shadow->w - 1), (n2.y * 0.5f + 0.5f) * (float)(shadow->h - 1)};

                    const int minx = std::max(0, (int)std::floor(std::min({s0.x, s1.x, s2.x})));
                    const int maxx = std::min(shadow->w - 1, (int)std::ceil(std::max({s0.x, s1.x, s2.x})));
                    const int miny = std::max(0, (int)std::floor(std::min({s0.y, s1.y, s2.y})));
                    const int maxy = std::min(shadow->h - 1, (int)std::ceil(std::max({s0.y, s1.y, s2.y})));
                    if (minx > maxx || miny > maxy) continue;

                    for (int y = miny; y <= maxy; ++y)
                    {
                        for (int x = minx; x <= maxx; ++x)
                        {
                            const glm::vec2 p{(float)x + 0.5f, (float)y + 0.5f};
                            const glm::vec3 bc = barycentric_2d(p, s0, s1, s2);
                            if (bc.x < 0.0f || bc.y < 0.0f || bc.z < 0.0f) continue;

                            const float z_ndc = bc.x * n0.z + bc.y * n1.z + bc.z * n2.z;
                            const float z01 = std::clamp(z_ndc * 0.5f + 0.5f, 0.0f, 1.0f);
                            float& zbuf = shadow->at(x, y);
                            if (z01 < zbuf) zbuf = z01;
                        }
                    }
                }
            }
        }

    private:
        LightCamera light_cam_{};
    };
}
