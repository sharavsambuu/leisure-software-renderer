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

#include "shs/task/parallel_for.hpp"
#include "shs/resources/mesh.hpp"
#include "shs/render/shader/program.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace render
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
        // R3 (renderer-lib review 2026-09-18): coarse tile size for the
        // job-system rasterization path (one wait-free job per tile).
        int tile_size = 32;
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

        // G1.1 (governance review 2026-09-18): fixed-capacity clip polygon.
        // Sutherland-Hodgman clipping of one triangle against the 6 frustum
        // half-spaces produces at most 3 + 6 = 9 vertices; 16 gives headroom.
        // This removes ALL per-triangle heap allocations from the clip path
        // (previously: one std::vector per plane pass, re-assigned 6x, plus
        // the initial {rv0, rv1, rv2} vector) — enforced by
        // shs_renderer_frame_allocator_interception_tests.
        inline constexpr size_t kClipPolyMaxVertices = 16;

        struct ClipPolygon
        {
            std::array<RasterVertex, kClipPolyMaxVertices> v{};
            size_t n = 0;

            size_t size() const { return n; }
            bool empty() const { return n == 0; }
            const RasterVertex& operator[](size_t i) const { return v[i]; }

            void push(const RasterVertex& vertex)
            {
                if (n < kClipPolyMaxVertices)
                {
                    v[n] = vertex;
                    ++n;
                }
            }

            void reset()
            {
                n = 0;
            }
        };

        template <typename PlaneDistFn>
        inline void clip_polygon_plane(const ClipPolygon& in_poly, ClipPolygon& out_poly, PlaneDistFn plane_dist_fn)
        {
            out_poly.reset();
            if (in_poly.empty()) return;

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
                    out_poly.push(nxt);
                }
                else if (cur_in && !nxt_in)
                {
                    const float denom = da - db;
                    if (std::abs(denom) > 1e-8f)
                    {
                        const float t = da / denom;
                        out_poly.push(lerp_rv(cur, nxt, t));
                    }
                }
                else if (!cur_in && nxt_in)
                {
                    const float denom = da - db;
                    if (std::abs(denom) > 1e-8f)
                    {
                        const float t = da / denom;
                        out_poly.push(lerp_rv(cur, nxt, t));
                    }
                    out_poly.push(nxt);
                }
            }
        }

        // Clips the input triangle against all 6 frustum half-spaces using two
        // stack-resident ping-pong buffers (6 passes => result in `poly`).
        inline ClipPolygon clip_polygon_frustum(const RasterVertex& rv0, const RasterVertex& rv1, const RasterVertex& rv2)
        {
            ClipPolygon poly{};
            poly.push(rv0);
            poly.push(rv1);
            poly.push(rv2);
            ClipPolygon scratch{};

            clip_polygon_plane(poly, scratch, plane_dist_left);
            clip_polygon_plane(scratch, poly, plane_dist_right);
            clip_polygon_plane(poly, scratch, plane_dist_bottom);
            clip_polygon_plane(scratch, poly, plane_dist_top);
            clip_polygon_plane(poly, scratch, plane_dist_near);
            clip_polygon_plane(scratch, poly, plane_dist_far);
            return poly;
        }

        // R2 (renderer-lib review 2026-09-18): a fully prepared screen-space
        // sub-triangle. All per-triangle work (perspective divisors,
        // edge-function setup, motion matrix, varying mask) happens ONCE;
        // the per-pixel loop only evaluates two edge functions at the pixel
        // center (FMA-friendly, span-independent) plus the shading work.
        struct PreparedTri
        {
            RasterVertex rv[3];
            glm::vec2 s[3]{};
            int minx = 0;
            int maxx = -1;
            int miny = 0;
            int maxy = -1;
            float invw[3]{1.0f, 1.0f, 1.0f};
            float inv_den = 0.0f; // 1 / signed_area2 (== barycentric denominator)
            // Edge functions for the v/w barycentric coordinates:
            //   E(x,y) = ex * x + ey * y + ec;  v = Ev * inv_den, w = Ew * inv_den,
            //   u = 1 - v - w  (the exact linear forms barycentric_2d evaluates).
            float ev_x = 0.0f, ev_y = 0.0f, ev_c = 0.0f;
            float ew_x = 0.0f, ew_y = 0.0f, ew_c = 0.0f;
            uint32_t varying_mask = 0;
            bool write_motion = false;
            glm::mat4 curr_to_prev_model{1.0f};
        };

        // The shared per-pixel inner loop. Both the streaming path and the
        // R3 tile path rasterize through this single implementation so the
        // two schedules cannot drift apart. (Body appended below.)
        template <typename ProgramT>
        inline void rasterize_prepared_span(
            const ProgramT& program,
            const ShaderUniforms& uniforms,
            RasterizerTarget target,
            const PreparedTri& t,
            int x0, int x1, int y0, int y1)
        {
            if (x0 > x1 || y0 > y1) return;
            const RasterVertex& rv0 = t.rv[0];
            const RasterVertex& rv1 = t.rv[1];
            const RasterVertex& rv2 = t.rv[2];
            const float invw0 = t.invw[0];
            const float invw1 = t.invw[1];
            const float invw2 = t.invw[2];
            const float inv_den = t.inv_den;
            const bool write_motion = t.write_motion;
            const uint32_t varying_mask = t.varying_mask;
            const int W = target.hdr->w;
            const int H = target.hdr->h;

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
            const float zc0 = rv0.clip.z * invw0;
            const float zc1 = rv1.clip.z * invw1;
            const float zc2 = rv2.clip.z * invw2;
            // R2-PER-PIXEL-LOOP
            for (int y = y0; y <= y1; ++y)
            {
                const float fy = (float)y + 0.5f; // pixel-center sample point
                for (int x = x0; x <= x1; ++x)
                {
                    // R2: edge functions evaluated DIRECTLY at the pixel
                    // center (FMA-friendly). Span-independent by construction,
                    // so any span of the same triangle produces bit-identical
                    // coverage — the property the R3 tile path relies on.
                    const float fx = (float)x + 0.5f;
                    const float v = (t.ev_x * fx + t.ev_y * fy + t.ev_c) * inv_den;
                    const float w = (t.ew_x * fx + t.ew_y * fy + t.ew_c) * inv_den;
                    const float u = 1.0f - v - w;
                    if (u < 0.0f || v < 0.0f || w < 0.0f) continue;

                    // 1/w interpolation: perspective-correct varying/position/uv тооцоо.
                    const float denom = u * invw0 + v * invw1 + w * invw2;
                    if (denom <= 1e-10f) continue;
                    const float inv_denom = 1.0f / denom;

                    const float z_clip = u * zc0 + v * zc1 + w * zc2;
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
                        fin.varyings[i] = (u * varw0[i] + v * varw1[i] + w * varw2[i]) * inv_denom;
                    }

                    fin.world_pos = (u * wpw0 + v * wpw1 + w * wpw2) * inv_denom;
                    fin.normal_ws = glm::normalize((u * npw0 + v * npw1 + w * npw2) * inv_denom);
                    fin.uv = (u * uvw0 + v * uvw1 + w * uvw2) * inv_denom;
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
                        const glm::vec4 prev_world = t.curr_to_prev_model * curr_world;
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

    // R1 (renderer-lib review 2026-09-18): rasterize_mesh is templated on the
    // shader program type. Any program-like type exposing vs()/fs() callables
    // and valid() is accepted; ShaderProgram (std::function) remains the
    // host-seam compatibility path, while concrete ShaderProgramFn instances
    // get fully inlined per-pixel fragment invocations.
    template <typename ProgramT>
    inline RasterizerStats rasterize_mesh(
        const MeshData& mesh,
        const ProgramT& program,
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

        // R3 (renderer-lib review 2026-09-18): tile binning on the
        // job-system path. Sub-triangles are prepared once, binned into
        // coarse screen tiles, and each tile is rasterized by ONE job into
        // disjoint memory — no per-triangle barriers. Bin order preserves
        // submission order, so per-pixel results (including depth ties) are
        // identical to the streaming path. The no-job-system path stays
        // allocation-free (frame zero-heap law).
        const bool use_tiled = (config.job_system != nullptr);
        std::vector<detail::PreparedTri> tiled_tris{};
        std::vector<std::vector<uint32_t>> tile_bins{};
        int tiles_x = 0;
        int tiles_y = 0;
        if (use_tiled)
        {
            const int tile = std::max(1, config.tile_size);
            tiles_x = (W + tile - 1) / tile;
            tiles_y = (H + tile - 1) / tile;
            tile_bins.resize((size_t)tiles_x * (size_t)tiles_y);
        }

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

            detail::ClipPolygon poly{};
            poly.push(rv0);
            poly.push(rv1);
            poly.push(rv2);
            // Ихэнх кадарт харагдаж буй трианглууд clip volume дотор байдаг тул clip-ийг алгасна.
            if (!(fully_inside_clip(rv0) && fully_inside_clip(rv1) && fully_inside_clip(rv2)))
            {
                poly = detail::clip_polygon_frustum(rv0, rv1, rv2);
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

                detail::PreparedTri tri{};
                tri.rv[0] = rv0;
                tri.rv[1] = rv1;
                tri.rv[2] = rv2;
                tri.s[0] = s0;
                tri.s[1] = s1;
                tri.s[2] = s2;
                tri.minx = minx;
                tri.maxx = maxx;
                tri.miny = miny;
                tri.maxy = maxy;
                tri.invw[0] = 1.0f / rv0.clip.w;
                tri.invw[1] = 1.0f / rv1.clip.w;
                tri.invw[2] = 1.0f / rv2.clip.w;
                tri.inv_den = 1.0f / signed_area2;
                // R2 (renderer-lib review 2026-09-18): edge-function setup -
                // the same linear forms barycentric_2d evaluated per pixel,
                // hoisted to the triangle; the per-pixel coverage test is now
                // two direct (FMA-friendly, span-independent) evaluations.
                tri.ev_x = e1.y;
                tri.ev_y = -e1.x;
                tri.ev_c = e1.x * s0.y - s0.x * e1.y;
                tri.ew_x = -e0.y;
                tri.ew_y = e0.x;
                tri.ew_c = s0.x * e0.y - e0.x * s0.y;
                tri.varying_mask = rv0.varying_mask | rv1.varying_mask | rv2.varying_mask;
                tri.write_motion = (target.depth_motion != nullptr) && uniforms.enable_motion_vectors;
                if (tri.write_motion)
                {
                    const float det_model = glm::determinant(uniforms.model);
                    if (std::abs(det_model) > 1e-10f)
                    {
                        tri.curr_to_prev_model = uniforms.prev_model * glm::inverse(uniforms.model);
                    }
                    else
                    {
                        tri.curr_to_prev_model = glm::mat4(1.0f);
                    }
                }

                auto raster_rows = [&](int yb, int ye)
                {
                    // barycentric_2d rejected EVERY pixel of a degenerate
                    // (|den| < 1e-8) triangle; reject once here instead.
                    // stats.tri_raster was already incremented above, matching
                    // the old count-then-skip behavior.
                    if (std::abs(signed_area2) < 1e-8f) return;
                    detail::rasterize_prepared_span(program, uniforms, target, tri, minx, maxx, yb, ye - 1);
                };

                if (use_tiled)
                {
                    // R3: defer to the tile pass after the triangle loop.
                    tiled_tris.push_back(tri);
                }
                else
                {
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
        }

        // R3: tile pass — bin every prepared sub-triangle into each overlapped
        // tile, then one wait-free job per non-empty tile. Jobs write disjoint
        // tile rectangles; the only synchronization is the single WaitGroup.
        if (use_tiled)
        {
            const int tile = std::max(1, config.tile_size);
            for (uint32_t sti = 0; sti < tiled_tris.size(); ++sti)
            {
                const detail::PreparedTri& t = tiled_tris[sti];
                const int tx0 = std::max(0, t.minx / tile);
                const int tx1 = std::min(tiles_x - 1, t.maxx / tile);
                const int ty0 = std::max(0, t.miny / tile);
                const int ty1 = std::min(tiles_y - 1, t.maxy / tile);
                for (int bty = ty0; bty <= ty1; ++bty)
                {
                    for (int btx = tx0; btx <= tx1; ++btx)
                    {
                        tile_bins[(size_t)bty * (size_t)tiles_x + (size_t)btx].push_back(sti);
                    }
                }
            }

            auto raster_tile = [&](int tx, int ty, const std::vector<uint32_t>& bin)
            {
                const int x0 = tx * tile;
                const int y0 = ty * tile;
                const int x1 = std::min(W - 1, x0 + tile - 1);
                const int y1 = std::min(H - 1, y0 + tile - 1);
                for (uint32_t sti : bin)
                {
                    const detail::PreparedTri& t = tiled_tris[sti];
                    const int cx0 = std::max(t.minx, x0);
                    const int cx1 = std::min(t.maxx, x1);
                    const int cy0 = std::max(t.miny, y0);
                    const int cy1 = std::min(t.maxy, y1);
                    if (cx0 > cx1 || cy0 > cy1) continue;
                    detail::rasterize_prepared_span(program, uniforms, target, t, cx0, cx1, cy0, cy1);
                }
            };

            // Degenerate job system (no workers): rasterize inline, same
            // order, identical output.
            const bool run_inline = (config.job_system->worker_count() == 0);
            task::WaitGroup wg{};
            for (int ty = 0; ty < tiles_y; ++ty)
            {
                for (int tx = 0; tx < tiles_x; ++tx)
                {
                    const std::vector<uint32_t>& bin = tile_bins[(size_t)ty * (size_t)tiles_x + (size_t)tx];
                    if (bin.empty()) continue;
                    if (run_inline)
                    {
                        raster_tile(tx, ty, bin);
                        continue;
                    }
                    wg.add(1);
                    config.job_system->enqueue([&raster_tile, &wg, tx, ty, &bin]()
                    {
                        raster_tile(tx, ty, bin);
                        wg.done();
                    });
                }
            }
            wg.wait();
        }
        return stats;
    }

    } // inline namespace render
}
