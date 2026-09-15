#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: rasterizer.hpp
    МОДУЛЬ: render
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн render модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>

#include "shs/job/parallel_for.hpp"
#include "shs/resources/mesh.hpp"
#include "shs/shader/program.hpp"

namespace shs
{
    enum class RasterizerCullMode
    {
        None = 0,
        Back = 1,
        Front = 2
    };

    struct RasterizerConfig
    {
        RasterizerCullMode cull_mode = RasterizerCullMode::Back;
        bool front_face_ccw = true;
        IJobSystem* job_system = nullptr;
        int parallel_min_rows = 8;
        int parallel_min_pixels = 128 * 128;
    };

    struct RasterizerTarget
    {
        RT_ColorHDR* hdr = nullptr;
        RT_ColorDepthMotion* depth_motion = nullptr;
    };

    struct RasterizerStats
    {
        uint64_t tri_input = 0;
        uint64_t tri_after_clip = 0;
        uint64_t tri_raster = 0;
    };

    namespace detail
    {
        struct RasterVertex
        {
            glm::vec4 clip{0.0f, 0.0f, 0.0f, 1.0f};
            std::array<glm::vec4, SHS_MAX_VARYINGS> varyings{};
            uint32_t varying_mask = 0u;

            // Raster шатанд шууд хэрэглэгдэх world/normal/uv өгөгдөл.
            glm::vec3 world_pos{0.0f};
            glm::vec3 normal_ws{0.0f, 1.0f, 0.0f};
            glm::vec2 uv{0.0f};
        };

        inline RasterVertex lerp_rv(const RasterVertex& a, const RasterVertex& b, float t)
        {
            RasterVertex o{};
            o.clip = glm::mix(a.clip, b.clip, t);
            o.varying_mask = a.varying_mask | b.varying_mask;
            for (uint32_t i = 0; i < SHS_MAX_VARYINGS; ++i) o.varyings[i] = glm::mix(a.varyings[i], b.varyings[i], t);
            o.world_pos = glm::mix(a.world_pos, b.world_pos, t);
            o.normal_ws = glm::normalize(glm::mix(a.normal_ws, b.normal_ws, t));
            o.uv = glm::mix(a.uv, b.uv, t);
            return o;
        }

        inline float plane_dist_left(const RasterVertex& v)
        {
            return v.clip.x + v.clip.w;
        }

        inline float plane_dist_right(const RasterVertex& v)
        {
            return v.clip.w - v.clip.x;
        }

        inline float plane_dist_bottom(const RasterVertex& v)
        {
            return v.clip.y + v.clip.w;
        }

        inline float plane_dist_top(const RasterVertex& v)
        {
            return v.clip.w - v.clip.y;
        }

        inline float plane_dist_near(const RasterVertex& v)
        {
            return v.clip.z + v.clip.w;
        }

        inline float plane_dist_far(const RasterVertex& v)
        {
            return v.clip.w - v.clip.z;
        }

        template <typename PlaneDistFn>
        inline std::vector<RasterVertex> clip_polygon_plane(const std::vector<RasterVertex>& in_poly, PlaneDistFn plane_dist_fn)
        {
            std::vector<RasterVertex> out{};
            if (in_poly.empty()) return out;

            out.reserve(in_poly.size() + 2);
            for (size_t i = 0; i < in_poly.size(); ++i)
            {
                const RasterVertex& cur = in_poly[i];
                const RasterVertex& nxt = in_poly[(i + 1) % in_poly.size()];
                const float da = plane_dist_fn(cur);
                const float db = plane_dist_fn(nxt);
                const bool cur_in = da >= 0.0f;
                const bool nxt_in = db >= 0.0f;

                if (cur_in && nxt_in)
                {
                    out.push_back(nxt);
                }
                else if (cur_in && !nxt_in)
                {
                    const float denom = da - db;
                    if (std::abs(denom) > 1e-8f)
                    {
                        const float t = da / denom;
                        out.push_back(lerp_rv(cur, nxt, t));
                    }
                }
                else if (!cur_in && nxt_in)
                {
                    const float denom = da - db;
                    if (std::abs(denom) > 1e-8f)
                    {
                        const float t = da / denom;
                        out.push_back(lerp_rv(cur, nxt, t));
                    }
                    out.push_back(nxt);
                }
            }
            return out;
        }

        inline std::vector<RasterVertex> clip_polygon_frustum(const std::vector<RasterVertex>& in_poly)
        {
            std::vector<RasterVertex> poly = in_poly;
            poly = clip_polygon_plane(poly, plane_dist_left);
            poly = clip_polygon_plane(poly, plane_dist_right);
            poly = clip_polygon_plane(poly, plane_dist_bottom);
            poly = clip_polygon_plane(poly, plane_dist_top);
            poly = clip_polygon_plane(poly, plane_dist_near);
            poly = clip_polygon_plane(poly, plane_dist_far);
            return poly;
        }
    }

    inline glm::vec3 barycentric_2d(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c)
    {
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
    }

    inline RasterizerStats rasterize_mesh(
        const MeshData& mesh,
        const ShaderProgram& program,
        const ShaderUniforms& uniforms,
        RasterizerTarget target,
        const RasterizerConfig& config = {}
    )
    {
        RasterizerStats stats{};
        if (!target.hdr || !program.valid()) return stats;
        if (mesh.positions.empty()) return stats;
        const int W = target.hdr->w;
        const int H = target.hdr->h;
        if (W <= 0 || H <= 0) return stats;

        auto read_v = [&](uint32_t idx) -> ShaderVertex {
            ShaderVertex v{};
            v.position = mesh.positions[(size_t)idx];
            if (idx < mesh.normals.size()) v.normal = mesh.normals[(size_t)idx];
            if (idx < mesh.uvs.size()) v.uv = mesh.uvs[(size_t)idx];
            return v;
        };

        const bool indexed = !mesh.indices.empty();
        const size_t tri_count = indexed ? (mesh.indices.size() / 3) : (mesh.positions.size() / 3);
        for (size_t ti = 0; ti < tri_count; ++ti)
        {
            stats.tri_input++;
            uint32_t i0 = 0, i1 = 0, i2 = 0;
            if (indexed)
            {
                i0 = mesh.indices[ti * 3 + 0];
                i1 = mesh.indices[ti * 3 + 1];
                i2 = mesh.indices[ti * 3 + 2];
            }
            else
            {
                i0 = (uint32_t)(ti * 3 + 0);
                i1 = (uint32_t)(ti * 3 + 1);
                i2 = (uint32_t)(ti * 3 + 2);
            }
            if (i0 >= mesh.positions.size() || i1 >= mesh.positions.size() || i2 >= mesh.positions.size()) continue;

            const VertexOut v0 = program.vs(read_v(i0), uniforms);
            const VertexOut v1 = program.vs(read_v(i1), uniforms);
            const VertexOut v2 = program.vs(read_v(i2), uniforms);

            const detail::RasterVertex rv0{v0.clip, v0.varyings, v0.varying_mask, v0.world_pos, v0.normal_ws, v0.uv};
            const detail::RasterVertex rv1{v1.clip, v1.varyings, v1.varying_mask, v1.world_pos, v1.normal_ws, v1.uv};
            const detail::RasterVertex rv2{v2.clip, v2.varyings, v2.varying_mask, v2.world_pos, v2.normal_ws, v2.uv};

            const auto fully_inside_clip = [](const detail::RasterVertex& rv) -> bool
            {
                const glm::vec4 c = rv.clip;
                if (!(c.w > 0.0f)) return false;
                return
                    (c.x >= -c.w && c.x <= c.w) &&
                    (c.y >= -c.w && c.y <= c.w) &&
                    (c.z >= -c.w && c.z <= c.w);
            };

            std::vector<detail::RasterVertex> poly = {
                rv0, rv1, rv2
            };
            // Ихэнх кадарт харагдаж буй трианглууд clip volume дотор байдаг тул clip-ийг алгасна.
            if (!(fully_inside_clip(rv0) && fully_inside_clip(rv1) && fully_inside_clip(rv2)))
            {
                poly = detail::clip_polygon_frustum(poly);
            }
            if (poly.size() < 3) continue;

            // Клип хийсний дараах олон өнцөгтийг fan аргаар гурвалжилна.
            for (size_t k = 1; k + 1 < poly.size(); ++k)
            {
                stats.tri_after_clip++;
                const detail::RasterVertex rv0 = poly[0];
                const detail::RasterVertex rv1 = poly[k];
                const detail::RasterVertex rv2 = poly[k + 1];

                const glm::vec3 n0 = glm::vec3(rv0.clip) / rv0.clip.w;
                const glm::vec3 n1 = glm::vec3(rv1.clip) / rv1.clip.w;
                const glm::vec3 n2 = glm::vec3(rv2.clip) / rv2.clip.w;
                if (!std::isfinite(n0.x) || !std::isfinite(n0.y) || !std::isfinite(n0.z)) continue;
                if (!std::isfinite(n1.x) || !std::isfinite(n1.y) || !std::isfinite(n1.z)) continue;
                if (!std::isfinite(n2.x) || !std::isfinite(n2.y) || !std::isfinite(n2.z)) continue;

                const glm::vec2 s0{(n0.x * 0.5f + 0.5f) * (float)(W - 1), (n0.y * 0.5f + 0.5f) * (float)(H - 1)};
                const glm::vec2 s1{(n1.x * 0.5f + 0.5f) * (float)(W - 1), (n1.y * 0.5f + 0.5f) * (float)(H - 1)};
                const glm::vec2 s2{(n2.x * 0.5f + 0.5f) * (float)(W - 1), (n2.y * 0.5f + 0.5f) * (float)(H - 1)};

                const glm::vec2 e0 = s1 - s0;
                const glm::vec2 e1 = s2 - s0;
                const float signed_area2 = e0.x * e1.y - e0.y * e1.x;
                if (std::abs(signed_area2) < 1e-10f) continue;
                const bool tri_ccw = signed_area2 > 0.0f;
                const bool is_front = (tri_ccw == config.front_face_ccw);
                if (config.cull_mode == RasterizerCullMode::Back && !is_front) continue;
                if (config.cull_mode == RasterizerCullMode::Front && is_front) continue;

                const float minx_f = std::min({s0.x, s1.x, s2.x});
                const float maxx_f = std::max({s0.x, s1.x, s2.x});
                const float miny_f = std::min({s0.y, s1.y, s2.y});
                const float maxy_f = std::max({s0.y, s1.y, s2.y});

                const int minx = std::max(0, (int)std::floor(minx_f));
                const int maxx = std::min(W - 1, (int)std::ceil(maxx_f));
                const int miny = std::max(0, (int)std::floor(miny_f));
                const int maxy = std::min(H - 1, (int)std::ceil(maxy_f));
                if (minx > maxx || miny > maxy) continue;
                stats.tri_raster++;

                const float invw0 = 1.0f / rv0.clip.w;
                const float invw1 = 1.0f / rv1.clip.w;
                const float invw2 = 1.0f / rv2.clip.w;
                const bool write_motion = (target.depth_motion != nullptr) && uniforms.enable_motion_vectors;
                glm::mat4 curr_to_prev_model{1.0f};
                if (write_motion)
                {
                    const float det_model = glm::determinant(uniforms.model);
                    if (std::abs(det_model) > 1e-10f)
                    {
                        curr_to_prev_model = uniforms.prev_model * glm::inverse(uniforms.model);
                    }
                    else
                    {
                        curr_to_prev_model = glm::mat4(1.0f);
                    }
                }
                const uint32_t varying_mask = rv0.varying_mask | rv1.varying_mask | rv2.varying_mask;
                const glm::vec3 wpw0 = rv0.world_pos * invw0;
                const glm::vec3 wpw1 = rv1.world_pos * invw1;
                const glm::vec3 wpw2 = rv2.world_pos * invw2;
                const glm::vec3 npw0 = rv0.normal_ws * invw0;
                const glm::vec3 npw1 = rv1.normal_ws * invw1;
                const glm::vec3 npw2 = rv2.normal_ws * invw2;
                const glm::vec2 uvw0 = rv0.uv * invw0;
                const glm::vec2 uvw1 = rv1.uv * invw1;
                const glm::vec2 uvw2 = rv2.uv * invw2;
                std::array<glm::vec4, SHS_MAX_VARYINGS> varw0{};
                std::array<glm::vec4, SHS_MAX_VARYINGS> varw1{};
                std::array<glm::vec4, SHS_MAX_VARYINGS> varw2{};
                for (uint32_t i = 0; i < SHS_MAX_VARYINGS; ++i)
                {
                    if ((varying_mask & varying_bit(i)) == 0u) continue;
                    varw0[i] = rv0.varyings[i] * invw0;
                    varw1[i] = rv1.varyings[i] * invw1;
                    varw2[i] = rv2.varyings[i] * invw2;
                }

                auto raster_rows = [&](int yb, int ye)
                {
                    for (int y = yb; y < ye; ++y)
                    {
                        for (int x = minx; x <= maxx; ++x)
                        {
                            const glm::vec2 p{(float)x + 0.5f, (float)y + 0.5f};
                            const glm::vec3 bc = barycentric_2d(p, s0, s1, s2);
                            if (bc.x < 0.0f || bc.y < 0.0f || bc.z < 0.0f) continue;

                            // 1/w interpolation: perspective-correct varying/position/uv тооцоо.
                            const float denom = bc.x * invw0 + bc.y * invw1 + bc.z * invw2;
                            if (denom <= 1e-10f) continue;
                            const float inv_denom = 1.0f / denom;

                            const float z_clip = bc.x * (rv0.clip.z * invw0) + bc.y * (rv1.clip.z * invw1) + bc.z * (rv2.clip.z * invw2);
                            const float z_ndc = z_clip * inv_denom;
                            float z01 = glm::clamp(z_ndc * 0.5f + 0.5f, 0.0f, 1.0f);
                            if (target.depth_motion)
                            {
                                // Perspective projection үед clip.w-аас view-space z сэргээж depth-ийг тогтвортой болгоно.
                                const float view_z = 1.0f / denom;
                                const float zn = target.depth_motion->zn;
                                const float zf = target.depth_motion->zf;
                                if (zf > zn + 1e-6f)
                                {
                                    z01 = glm::clamp((view_z - zn) / (zf - zn), 0.0f, 1.0f);
                                }
                                float& zbuf = target.depth_motion->depth.at(x, y);
                                if (z01 >= zbuf) continue;
                                zbuf = z01;
                            }

                            FragmentIn fin{};
                            fin.varying_mask = varying_mask;
                            for (uint32_t i = 0; i < SHS_MAX_VARYINGS; ++i)
                            {
                                if ((varying_mask & varying_bit(i)) == 0u) continue;
                                fin.varyings[i] = (bc.x * varw0[i] + bc.y * varw1[i] + bc.z * varw2[i]) * inv_denom;
                            }

                            fin.world_pos = (bc.x * wpw0 + bc.y * wpw1 + bc.z * wpw2) * inv_denom;
                            fin.normal_ws = glm::normalize((bc.x * npw0 + bc.y * npw1 + bc.z * npw2) * inv_denom);
                            fin.uv = (bc.x * uvw0 + bc.y * uvw1 + bc.z * uvw2) * inv_denom;
                            // Shader өөрийн semantic varying гаргасан бол түүнд давуу эрх өгнө.
                            if ((fin.varying_mask & varying_bit((uint32_t)VaryingSemantic::WorldPos)) != 0u)
                            {
                                fin.world_pos = glm::vec3(get_varying(fin, VaryingSemantic::WorldPos));
                            }
                            if ((fin.varying_mask & varying_bit((uint32_t)VaryingSemantic::NormalWS)) != 0u)
                            {
                                fin.normal_ws = glm::normalize(glm::vec3(get_varying(fin, VaryingSemantic::NormalWS)));
                            }
                            if ((fin.varying_mask & varying_bit((uint32_t)VaryingSemantic::UV0)) != 0u)
                            {
                                const glm::vec4 uv0 = get_varying(fin, VaryingSemantic::UV0);
                                fin.uv = glm::vec2(uv0.x, uv0.y);
                            }
                            if (write_motion)
                            {
                                const glm::vec4 curr_world = glm::vec4(fin.world_pos, 1.0f);
                                const glm::vec4 prev_world = curr_to_prev_model * curr_world;
                                const glm::vec4 curr_clip = uniforms.viewproj * curr_world;
                                const glm::vec4 prev_clip = uniforms.prev_viewproj * prev_world;
                                if (std::abs(curr_clip.w) > 1e-8f && std::abs(prev_clip.w) > 1e-8f)
                                {
                                    const glm::vec2 curr_ndc = glm::vec2(curr_clip) / curr_clip.w;
                                    const glm::vec2 prev_ndc = glm::vec2(prev_clip) / prev_clip.w;
                                    glm::vec2 vel = (curr_ndc - prev_ndc) * 0.5f * glm::vec2((float)W, (float)H);
                                    const float len = glm::length(vel);
                                    const float max_vel = 96.0f;
                                    if (len > max_vel && len > 1e-6f)
                                    {
                                        vel *= (max_vel / len);
                                    }
                                    target.depth_motion->motion.at(x, y) = Motion2f{vel.x, vel.y};
                                }
                                else
                                {
                                    target.depth_motion->motion.at(x, y) = Motion2f{};
                                }
                            }
                            fin.depth01 = z01;
                            fin.px = x;
                            fin.py = y;

                            const FragmentOut fout = program.fs(fin, uniforms);
                            if (fout.discard) continue;

                            target.hdr->color.at(x, y) = fout.color;
                        }
                    }
                };

                const int bbox_rows = maxy - miny + 1;
                const int bbox_pixels = (maxx - minx + 1) * bbox_rows;
                // Том bbox дээр л parallel замыг асааж scheduling overhead-оос зайлсхийж байна.
                const bool use_parallel =
                    config.job_system &&
                    bbox_rows >= std::max(1, config.parallel_min_rows) &&
                    bbox_pixels >= std::max(1, config.parallel_min_pixels);
                if (use_parallel)
                {
                    parallel_for_1d(config.job_system, miny, maxy + 1, std::max(1, config.parallel_min_rows), raster_rows);
                }
                else
                {
                    raster_rows(miny, maxy + 1);
                }
            }
        }
        return stats;
    }
}
