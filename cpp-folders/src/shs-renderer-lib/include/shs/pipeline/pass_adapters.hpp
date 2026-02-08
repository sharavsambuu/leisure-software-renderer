#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_adapters.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн pipeline модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <algorithm>
#include <cmath>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "shs/geometry/shape_cell_culling.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/lighting/light_set.hpp"
#include "shs/passes/pass_light_shafts.hpp"
#include "shs/passes/pass_motion_blur.hpp"
#include "shs/passes/pass_pbr_forward.hpp"
#include "shs/passes/pass_shadow_map.hpp"
#include "shs/passes/pass_tonemap.hpp"
#include "shs/pipeline/pass_registry.hpp"
#include "shs/pipeline/render_pass.hpp"
#include "shs/render/rasterizer.hpp"
#include "shs/resources/resource_registry.hpp"
#include "shs/shader/program.hpp"

namespace shs
{
    namespace detail
    {
        inline glm::mat4 make_item_model_matrix(const RenderItem& item)
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, item.tr.pos);
            model = glm::rotate(model, item.tr.rot_euler.x, glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::rotate(model, item.tr.rot_euler.y, glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::rotate(model, item.tr.rot_euler.z, glm::vec3(0.0f, 0.0f, 1.0f));
            model = glm::scale(model, item.tr.scl);
            return model;
        }

        inline Plane make_oriented_plane_from_points(
            const glm::vec3& a,
            const glm::vec3& b,
            const glm::vec3& c,
            const glm::vec3& inside_point)
        {
            Plane p{};
            glm::vec3 n = glm::cross(b - a, c - a);
            const float len2 = glm::dot(n, n);
            if (len2 <= 1e-12f)
            {
                p.normal = glm::vec3(0.0f, 1.0f, 0.0f);
                p.d = -glm::dot(p.normal, a);
                return p;
            }
            n *= 1.0f / std::sqrt(len2);
            p.normal = n;
            p.d = -glm::dot(n, a);
            if (p.signed_distance(inside_point) < 0.0f)
            {
                p.normal = -p.normal;
                p.d = -p.d;
            }
            return p;
        }

        inline glm::vec3 world_from_ndc(const glm::mat4& inv_view_proj, const glm::vec3& ndc)
        {
            const glm::vec4 hp = inv_view_proj * glm::vec4(ndc, 1.0f);
            if (std::abs(hp.w) <= 1e-8f) return glm::vec3(hp);
            return glm::vec3(hp) / hp.w;
        }

        inline ConvexCell make_screen_tile_convex_cell(
            const glm::mat4& view_proj,
            int viewport_w,
            int viewport_h,
            uint32_t tile_size,
            uint32_t tile_x,
            uint32_t tile_y)
        {
            ConvexCell cell{};
            cell.kind = ConvexCellKind::ScreenTileCell;
            cell.user_data = glm::uvec4(tile_x, tile_y, 0u, 0u);
            if (viewport_w <= 0 || viewport_h <= 0 || tile_size == 0u) return cell;

            const float inv_w = 1.0f / static_cast<float>(viewport_w);
            const float inv_h = 1.0f / static_cast<float>(viewport_h);
            const float px0 = static_cast<float>(tile_x * tile_size);
            const float px1 = static_cast<float>(std::min<uint32_t>((tile_x + 1u) * tile_size, static_cast<uint32_t>(viewport_w)));
            const float py0 = static_cast<float>(tile_y * tile_size);
            const float py1 = static_cast<float>(std::min<uint32_t>((tile_y + 1u) * tile_size, static_cast<uint32_t>(viewport_h)));

            const float nx0 = px0 * (2.0f * inv_w) - 1.0f;
            const float nx1 = px1 * (2.0f * inv_w) - 1.0f;
            const float ny_top = 1.0f - py0 * (2.0f * inv_h);
            const float ny_bottom = 1.0f - py1 * (2.0f * inv_h);

            const glm::mat4 inv_view_proj = glm::inverse(view_proj);
            const glm::vec3 nbl = world_from_ndc(inv_view_proj, glm::vec3(nx0, ny_bottom, -1.0f));
            const glm::vec3 nbr = world_from_ndc(inv_view_proj, glm::vec3(nx1, ny_bottom, -1.0f));
            const glm::vec3 ntl = world_from_ndc(inv_view_proj, glm::vec3(nx0, ny_top, -1.0f));
            const glm::vec3 ntr = world_from_ndc(inv_view_proj, glm::vec3(nx1, ny_top, -1.0f));
            const glm::vec3 fbl = world_from_ndc(inv_view_proj, glm::vec3(nx0, ny_bottom, 1.0f));
            const glm::vec3 fbr = world_from_ndc(inv_view_proj, glm::vec3(nx1, ny_bottom, 1.0f));
            const glm::vec3 ftl = world_from_ndc(inv_view_proj, glm::vec3(nx0, ny_top, 1.0f));
            const glm::vec3 ftr = world_from_ndc(inv_view_proj, glm::vec3(nx1, ny_top, 1.0f));
            const glm::vec3 inside = (nbl + nbr + ntl + ntr + fbl + fbr + ftl + ftr) * (1.0f / 8.0f);

            convex_cell_add_plane(cell, make_oriented_plane_from_points(nbl, nbr, ntr, inside)); // near
            convex_cell_add_plane(cell, make_oriented_plane_from_points(fbr, fbl, ftl, inside)); // far
            convex_cell_add_plane(cell, make_oriented_plane_from_points(nbl, ntl, ftl, inside)); // left
            convex_cell_add_plane(cell, make_oriented_plane_from_points(ntr, nbr, fbr, inside)); // right
            convex_cell_add_plane(cell, make_oriented_plane_from_points(nbr, nbl, fbl, inside)); // bottom
            convex_cell_add_plane(cell, make_oriented_plane_from_points(ntl, ntr, ftr, inside)); // top
            return cell;
        }

        inline void append_local_light_shapes_from_set(
            const LightSet& set,
            std::vector<ShapeVolume>& out_shapes)
        {
            out_shapes.reserve(out_shapes.size() + set.local_light_count());
            for (const PointLight& l : set.points)
            {
                ShapeVolume s{};
                s.value = point_light_culling_sphere(l);
                s.stable_id = static_cast<uint32_t>(out_shapes.size());
                out_shapes.push_back(s);
            }
            for (const SpotLight& l : set.spots)
            {
                const float range = std::max(l.common.range, 0.0f);
                const float outer = std::clamp(l.outer_angle_rad, 0.0f, glm::radians(89.0f));
                ConeFrustum cone{};
                cone.apex = l.common.position_ws;
                cone.axis = normalize_or(l.direction_ws, glm::vec3(0.0f, -1.0f, 0.0f));
                cone.near_distance = 0.0f;
                cone.far_distance = range;
                cone.near_radius = 0.0f;
                cone.far_radius = std::tan(outer) * range;

                ShapeVolume s{};
                s.value = cone;
                s.stable_id = static_cast<uint32_t>(out_shapes.size());
                out_shapes.push_back(s);
            }
            for (const RectAreaLight& l : set.rect_areas)
            {
                ShapeVolume s{};
                s.value = rect_area_light_culling_obb(l);
                s.stable_id = static_cast<uint32_t>(out_shapes.size());
                out_shapes.push_back(s);
            }
            for (const TubeAreaLight& l : set.tube_areas)
            {
                ShapeVolume s{};
                s.value = tube_area_light_culling_capsule(l);
                s.stable_id = static_cast<uint32_t>(out_shapes.size());
                out_shapes.push_back(s);
            }
        }

        inline bool technique_uses_light_culling(const FrameParams& fp)
        {
            return
                fp.technique.light_culling ||
                fp.technique.mode == TechniqueMode::ForwardPlus ||
                fp.technique.mode == TechniqueMode::TiledDeferred ||
                fp.technique.mode == TechniqueMode::ClusteredForward;
        }

        inline void execute_generic_light_culling(
            Context& ctx,
            const Scene& scene,
            const FrameParams& fp,
            RTRegistry& rtr,
            RT_Motion rt_motion,
            bool force_enable)
        {
            ctx.forward_plus.light_culling_valid = false;

            const bool light_culling_enabled = force_enable || technique_uses_light_culling(fp);
            if (!light_culling_enabled) return;
            if (fp.technique.depth_prepass && !ctx.forward_plus.depth_prepass_valid) return;

            int w = fp.w;
            int h = fp.h;
            if (rt_motion.valid())
            {
                auto* motion = static_cast<RT_ColorDepthMotion*>(rtr.get(rt_motion));
                if (motion && motion->w > 0 && motion->h > 0)
                {
                    w = motion->w;
                    h = motion->h;
                }
            }
            if (w <= 0 || h <= 0) return;

            const uint32_t tile_size = std::max<uint32_t>(1u, fp.technique.tile_size);
            const uint32_t tile_x = (uint32_t)((w + (int)tile_size - 1) / (int)tile_size);
            const uint32_t tile_y = (uint32_t)((h + (int)tile_size - 1) / (int)tile_size);
            const uint32_t total_tiles = tile_x * tile_y;
            const uint32_t max_per_tile = std::max<uint32_t>(1u, fp.technique.max_lights_per_tile);

            // Directional light нь tile бүрийг бүрэн хамарна (enabled үед).
            const uint32_t directional_light_count = (scene.sun.intensity > 0.0f) ? 1u : 0u;

            std::vector<ShapeVolume> local_light_shapes{};
            if (scene.local_lights)
            {
                append_local_light_shapes_from_set(*scene.local_lights, local_light_shapes);
            }

            if (!local_light_shapes.empty())
            {
                const ConvexCell camera_cell = extract_frustum_cell(
                    scene.cam.viewproj,
                    ConvexCellKind::CameraFrustumPerspective);
                CPUCullerConfig camera_cfg{};
                camera_cfg.use_broad_phase = true;
                camera_cfg.refine_intersections = true;
                camera_cfg.accept_broad_inside = true;
                camera_cfg.prefer_xsimd = true;
                camera_cfg.job_system = ctx.job_system;
                camera_cfg.parallel_min_items = 256;

                const CPUCullResult camera_cull = cull_shapes_cpu(camera_cell, local_light_shapes, camera_cfg);
                if (camera_cull.visible_indices.size() != local_light_shapes.size())
                {
                    std::vector<ShapeVolume> visible_shapes{};
                    visible_shapes.reserve(camera_cull.visible_indices.size());
                    for (size_t idx : camera_cull.visible_indices)
                    {
                        if (idx < local_light_shapes.size()) visible_shapes.push_back(local_light_shapes[idx]);
                    }
                    local_light_shapes.swap(visible_shapes);
                }
            }

            auto& fwdp = ctx.forward_plus;
            fwdp.tile_size = tile_size;
            fwdp.tile_count_x = tile_x;
            fwdp.tile_count_y = tile_y;
            fwdp.max_lights_per_tile = max_per_tile;
            fwdp.visible_light_count = directional_light_count + static_cast<uint32_t>(local_light_shapes.size());
            fwdp.tile_light_counts.assign((size_t)total_tiles, std::min(max_per_tile, directional_light_count));

            if (!local_light_shapes.empty())
            {
                CPUCullerConfig tile_cfg{};
                tile_cfg.use_broad_phase = true;
                tile_cfg.refine_intersections = true;
                tile_cfg.accept_broad_inside = true;
                tile_cfg.prefer_xsimd = true;
                tile_cfg.job_system = nullptr;

                for (uint32_t ty = 0; ty < tile_y; ++ty)
                {
                    for (uint32_t tx = 0; tx < tile_x; ++tx)
                    {
                        const uint32_t tile_index = ty * tile_x + tx;
                        const ConvexCell tile_cell = make_screen_tile_convex_cell(
                            scene.cam.viewproj,
                            w,
                            h,
                            tile_size,
                            tx,
                            ty);

                        uint32_t local_visible = 0;
                        for (const ShapeVolume& shape : local_light_shapes)
                        {
                            const CullClass c = classify_cpu(shape, tile_cell, tile_cfg);
                            if (!cull_class_visible(c, true)) continue;
                            ++local_visible;
                            if (directional_light_count + local_visible >= max_per_tile) break;
                        }

                        fwdp.tile_light_counts[(size_t)tile_index] = std::min(
                            max_per_tile,
                            directional_light_count + local_visible);
                    }
                }
            }
            fwdp.light_culling_valid = true;
        }

        inline ShaderProgram make_depth_prepass_program()
        {
            ShaderProgram p{};
            p.vs = [](const ShaderVertex& vin, const ShaderUniforms& u) -> VertexOut {
                VertexOut out{};
                const glm::vec4 wp4 = u.model * glm::vec4(vin.position, 1.0f);
                out.world_pos = glm::vec3(wp4);
                out.clip = u.viewproj * wp4;
                return out;
            };
            p.fs = [](const FragmentIn& fin, const ShaderUniforms& u) -> FragmentOut {
                (void)fin;
                (void)u;
                FragmentOut out{};
                out.color = ColorF{0.0f, 0.0f, 0.0f, 1.0f};
                return out;
            };
            return p;
        }
    }

    class PassShadowMapAdapter final : public IRenderPass
    {
    public:
        explicit PassShadowMapAdapter(RT_Shadow rt_shadow)
            : rt_shadow_(rt_shadow)
        {}

        const char* id() const override { return "shadow_map"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::Visibility;
            c.supported_modes_mask = technique_mode_mask_all();
            c.semantics = {
                write_semantic(PassSemantic::ShadowMap, ContractDomain::Software, "shadow")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.write(make_rt_resource_ref(static_cast<const RTHandle&>(rt_shadow_), PassResourceType::Shadow, "shadow", PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            PassShadowMap::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_shadow = rt_shadow_;
            pass_.execute(ctx, in);
        }

    private:
        RT_Shadow rt_shadow_{};
        PassShadowMap pass_{};
    };

    class PassDepthPrepassAdapter final : public IRenderPass
    {
    public:
        PassDepthPrepassAdapter(RT_Motion rt_motion, RTHandle rt_scratch_hdr = {})
            : rt_motion_(rt_motion), rt_scratch_hdr_(rt_scratch_hdr)
        {}

        const char* id() const override { return "depth_prepass"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::Visibility;
            c.supported_modes_mask =
                technique_mode_bit(TechniqueMode::ForwardPlus) |
                technique_mode_bit(TechniqueMode::TiledDeferred) |
                technique_mode_bit(TechniqueMode::ClusteredForward);
            c.semantics = {
                write_semantic(PassSemantic::Depth, ContractDomain::Software, "depth")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.write(make_named_resource_ref("technique.depth_prepass", PassResourceType::Temp, PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            ctx.forward_plus.depth_prepass_valid = false;
            if (!fp.technique.depth_prepass) return;
            if (!rt_motion_.valid()) return;

            auto* motion = static_cast<RT_ColorDepthMotion*>(rtr.get(rt_motion_));
            if (!motion || motion->w <= 0 || motion->h <= 0) return;

            motion->depth.clear(1.0f);
            motion->motion.clear(Motion2f{});

            RTHandle scratch_hdr = rt_scratch_hdr_;
            if (!scratch_hdr.valid())
            {
                scratch_hdr = rtr.ensure_transient_color_hdr("depth_prepass.auto_hdr", motion->w, motion->h);
            }
            auto* hdr = static_cast<RT_ColorHDR*>(rtr.get(scratch_hdr));
            if (!hdr || hdr->w <= 0 || hdr->h <= 0) return;

            hdr->clear(ColorF{0.0f, 0.0f, 0.0f, 1.0f});

            const ShaderProgram depth_prog = detail::make_depth_prepass_program();
            RasterizerTarget target{};
            target.hdr = hdr;
            target.depth_motion = motion;

            RasterizerConfig rast_cfg{};
            rast_cfg.front_face_ccw = fp.front_face_ccw;
            rast_cfg.job_system = ctx.job_system;
            switch (fp.cull_mode)
            {
                case CullMode::None: rast_cfg.cull_mode = RasterizerCullMode::None; break;
                case CullMode::Front: rast_cfg.cull_mode = RasterizerCullMode::Front; break;
                case CullMode::Back:
                default: rast_cfg.cull_mode = RasterizerCullMode::Back; break;
            }

            for (const auto& item : scene.items)
            {
                if (!item.visible) continue;
                if (!scene.resources) continue;
                const MeshData* mesh = scene.resources->get_mesh((MeshAssetHandle)item.mesh);
                if (!mesh || mesh->empty()) continue;

                ShaderUniforms uniforms{};
                uniforms.model = detail::make_item_model_matrix(item);
                uniforms.viewproj = scene.cam.viewproj;
                uniforms.enable_motion_vectors = false;
                (void)rasterize_mesh(*mesh, depth_prog, uniforms, target, rast_cfg);
            }

            ctx.forward_plus.depth_prepass_valid = true;
        }

    private:
        RT_Motion rt_motion_{};
        RTHandle rt_scratch_hdr_{};
    };

    class PassLightCullingAdapter final : public IRenderPass
    {
    public:
        explicit PassLightCullingAdapter(RT_Motion rt_motion)
            : rt_motion_(rt_motion)
        {}

        const char* id() const override { return "light_culling"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        RHIQueueClass preferred_queue() const override { return RHIQueueClass::Compute; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::LightCulling;
            c.supported_modes_mask =
                technique_mode_bit(TechniqueMode::ForwardPlus) |
                technique_mode_bit(TechniqueMode::TiledDeferred) |
                technique_mode_bit(TechniqueMode::ClusteredForward);
            c.requires_depth_prepass = true;
            c.prefer_async_compute = true;
            c.semantics = {
                read_semantic(PassSemantic::Depth, ContractDomain::Software, "depth"),
                write_semantic(PassSemantic::LightGrid, ContractDomain::Software, "light_grid"),
                write_semantic(PassSemantic::LightIndexList, ContractDomain::Software, "light_index_list")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_named_resource_ref("technique.depth_prepass", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_named_resource_ref("technique.light_grid", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_named_resource_ref("technique.light_index_list", PassResourceType::Temp, PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            detail::execute_generic_light_culling(ctx, scene, fp, rtr, rt_motion_, false);
        }

    private:
        RT_Motion rt_motion_{};
    };

    class PassClusterBuildAdapter final : public IRenderPass
    {
    public:
        explicit PassClusterBuildAdapter(RT_Motion rt_motion)
            : rt_motion_(rt_motion)
        {}

        const char* id() const override { return "cluster_build"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        RHIQueueClass preferred_queue() const override { return RHIQueueClass::Compute; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::LightCulling;
            c.supported_modes_mask = technique_mode_bit(TechniqueMode::ClusteredForward);
            c.requires_depth_prepass = true;
            c.prefer_async_compute = true;
            c.semantics = {
                read_semantic(PassSemantic::Depth, ContractDomain::Software, "depth"),
                write_semantic(PassSemantic::LightClusters, ContractDomain::Software, "clusters")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_named_resource_ref("technique.depth_prepass", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_named_resource_ref("technique.cluster_grid", PassResourceType::Temp, PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            (void)scene;
            if (fp.technique.depth_prepass && !ctx.forward_plus.depth_prepass_valid) return;

            int w = fp.w;
            int h = fp.h;
            if (rt_motion_.valid())
            {
                auto* motion = static_cast<RT_ColorDepthMotion*>(rtr.get(rt_motion_));
                if (motion && motion->w > 0 && motion->h > 0)
                {
                    w = motion->w;
                    h = motion->h;
                }
            }
            if (w <= 0 || h <= 0) return;

            auto& fwdp = ctx.forward_plus;
            fwdp.tile_size = std::max<uint32_t>(1u, fp.technique.tile_size);
            fwdp.tile_count_x = (uint32_t)((w + (int)fwdp.tile_size - 1) / (int)fwdp.tile_size);
            fwdp.tile_count_y = (uint32_t)((h + (int)fwdp.tile_size - 1) / (int)fwdp.tile_size);
            if (fwdp.tile_light_counts.size() != (size_t)fwdp.tile_count_x * (size_t)fwdp.tile_count_y)
            {
                fwdp.tile_light_counts.assign((size_t)fwdp.tile_count_x * (size_t)fwdp.tile_count_y, 0u);
            }
        }

    private:
        RT_Motion rt_motion_{};
    };

    class PassClusterLightAssignAdapter final : public IRenderPass
    {
    public:
        explicit PassClusterLightAssignAdapter(RT_Motion rt_motion)
            : rt_motion_(rt_motion)
        {}

        const char* id() const override { return "cluster_light_assign"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        RHIQueueClass preferred_queue() const override { return RHIQueueClass::Compute; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::LightCulling;
            c.supported_modes_mask = technique_mode_bit(TechniqueMode::ClusteredForward);
            c.requires_depth_prepass = true;
            c.prefer_async_compute = true;
            c.semantics = {
                read_semantic(PassSemantic::Depth, ContractDomain::Software, "depth"),
                read_semantic(PassSemantic::LightClusters, ContractDomain::Software, "clusters"),
                write_semantic(PassSemantic::LightGrid, ContractDomain::Software, "light_grid"),
                write_semantic(PassSemantic::LightIndexList, ContractDomain::Software, "light_index_list")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_named_resource_ref("technique.depth_prepass", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.cluster_grid", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_named_resource_ref("technique.light_grid", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_named_resource_ref("technique.light_index_list", PassResourceType::Temp, PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            detail::execute_generic_light_culling(ctx, scene, fp, rtr, rt_motion_, true);
        }

    private:
        RT_Motion rt_motion_{};
    };

    class PassGBufferAdapter final : public IRenderPass
    {
    public:
        const char* id() const override { return "gbuffer"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::GBuffer;
            c.supported_modes_mask =
                technique_mode_bit(TechniqueMode::Deferred) |
                technique_mode_bit(TechniqueMode::TiledDeferred);
            c.semantics = {
                write_semantic(PassSemantic::GBufferA, ContractDomain::Software, "gbuffer_a"),
                write_semantic(PassSemantic::GBufferB, ContractDomain::Software, "gbuffer_b"),
                write_semantic(PassSemantic::GBufferC, ContractDomain::Software, "gbuffer_c")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.write(make_named_resource_ref("technique.gbuffer_a", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_named_resource_ref("technique.gbuffer_b", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_named_resource_ref("technique.gbuffer_c", PassResourceType::Temp, PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            (void)ctx;
            (void)scene;
            (void)fp;
            (void)rtr;
        }
    };

    class PassDeferredLightingAdapter final : public IRenderPass
    {
    public:
        PassDeferredLightingAdapter(RTHandle rt_hdr, RT_Motion rt_motion, RTHandle rt_shadow)
            : rt_hdr_(rt_hdr), rt_motion_(rt_motion), rt_shadow_(rt_shadow)
        {}

        const char* id() const override { return "deferred_lighting"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::Lighting;
            c.supported_modes_mask = technique_mode_bit(TechniqueMode::Deferred);
            c.semantics = {
                read_semantic(PassSemantic::ShadowMap, ContractDomain::Software, "shadow"),
                read_semantic(PassSemantic::GBufferA, ContractDomain::Software, "gbuffer_a"),
                read_semantic(PassSemantic::GBufferB, ContractDomain::Software, "gbuffer_b"),
                read_semantic(PassSemantic::GBufferC, ContractDomain::Software, "gbuffer_c"),
                write_semantic(PassSemantic::ColorHDR, ContractDomain::Software, "hdr"),
                write_semantic(PassSemantic::MotionVectors, ContractDomain::Software, "motion")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_rt_resource_ref(rt_shadow_, PassResourceType::Shadow, "shadow", PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.gbuffer_a", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.gbuffer_b", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.gbuffer_c", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_rt_resource_ref(rt_hdr_, PassResourceType::ColorHDR, "hdr", PassResourceDomain::Software));
            io.write(make_rt_resource_ref(static_cast<const RTHandle&>(rt_motion_), PassResourceType::Motion, "motion", PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            PassPBRForward::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_hdr = rt_hdr_;
            in.rt_motion = rt_motion_;
            in.rt_shadow = rt_shadow_;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_hdr_{};
        RT_Motion rt_motion_{};
        RTHandle rt_shadow_{};
        PassPBRForward pass_{};
    };

    class PassDeferredLightingTiledAdapter final : public IRenderPass
    {
    public:
        PassDeferredLightingTiledAdapter(RTHandle rt_hdr, RT_Motion rt_motion, RTHandle rt_shadow)
            : rt_hdr_(rt_hdr), rt_motion_(rt_motion), rt_shadow_(rt_shadow)
        {}

        const char* id() const override { return "deferred_lighting_tiled"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::Lighting;
            c.supported_modes_mask = technique_mode_bit(TechniqueMode::TiledDeferred);
            c.requires_depth_prepass = true;
            c.requires_light_culling = true;
            c.semantics = {
                read_semantic(PassSemantic::ShadowMap, ContractDomain::Software, "shadow"),
                read_semantic(PassSemantic::GBufferA, ContractDomain::Software, "gbuffer_a"),
                read_semantic(PassSemantic::GBufferB, ContractDomain::Software, "gbuffer_b"),
                read_semantic(PassSemantic::GBufferC, ContractDomain::Software, "gbuffer_c"),
                read_semantic(PassSemantic::Depth, ContractDomain::Software, "depth"),
                read_semantic(PassSemantic::LightGrid, ContractDomain::Software, "light_grid"),
                read_semantic(PassSemantic::LightIndexList, ContractDomain::Software, "light_index_list"),
                write_semantic(PassSemantic::ColorHDR, ContractDomain::Software, "hdr"),
                write_semantic(PassSemantic::MotionVectors, ContractDomain::Software, "motion")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_rt_resource_ref(rt_shadow_, PassResourceType::Shadow, "shadow", PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.gbuffer_a", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.gbuffer_b", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.gbuffer_c", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.depth_prepass", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.light_grid", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.light_index_list", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_rt_resource_ref(rt_hdr_, PassResourceType::ColorHDR, "hdr", PassResourceDomain::Software));
            io.write(make_rt_resource_ref(static_cast<const RTHandle&>(rt_motion_), PassResourceType::Motion, "motion", PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            const bool depth_ready = (!fp.technique.depth_prepass) || ctx.forward_plus.depth_prepass_valid;
            const bool culling_ready = (!detail::technique_uses_light_culling(fp)) || ctx.forward_plus.light_culling_valid;

            PassPBRForward::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_hdr = rt_hdr_;
            in.rt_motion = rt_motion_;
            in.rt_shadow = rt_shadow_;
            in.preserve_existing_depth = depth_ready && culling_ready && fp.technique.depth_prepass;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_hdr_{};
        RT_Motion rt_motion_{};
        RTHandle rt_shadow_{};
        PassPBRForward pass_{};
    };

    class PassPBRForwardClusteredAdapter final : public IRenderPass
    {
    public:
        PassPBRForwardClusteredAdapter(RTHandle rt_hdr, RT_Motion rt_motion, RTHandle rt_shadow)
            : rt_hdr_(rt_hdr), rt_motion_(rt_motion), rt_shadow_(rt_shadow)
        {}

        const char* id() const override { return "pbr_forward_clustered"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::ForwardOpaque;
            c.supported_modes_mask = technique_mode_bit(TechniqueMode::ClusteredForward);
            c.requires_depth_prepass = true;
            c.requires_light_culling = true;
            c.semantics = {
                read_semantic(PassSemantic::ShadowMap, ContractDomain::Software, "shadow"),
                read_semantic(PassSemantic::Depth, ContractDomain::Software, "depth"),
                read_semantic(PassSemantic::LightGrid, ContractDomain::Software, "light_grid"),
                read_semantic(PassSemantic::LightIndexList, ContractDomain::Software, "light_index_list"),
                write_semantic(PassSemantic::ColorHDR, ContractDomain::Software, "hdr"),
                write_semantic(PassSemantic::MotionVectors, ContractDomain::Software, "motion")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_rt_resource_ref(rt_shadow_, PassResourceType::Shadow, "shadow", PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.depth_prepass", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.light_grid", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.light_index_list", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_rt_resource_ref(rt_hdr_, PassResourceType::ColorHDR, "hdr", PassResourceDomain::Software));
            io.write(make_rt_resource_ref(static_cast<const RTHandle&>(rt_motion_), PassResourceType::Motion, "motion", PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            const bool depth_ready = (!fp.technique.depth_prepass) || ctx.forward_plus.depth_prepass_valid;
            const bool culling_ready = (!detail::technique_uses_light_culling(fp)) || ctx.forward_plus.light_culling_valid;

            PassPBRForward::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_hdr = rt_hdr_;
            in.rt_motion = rt_motion_;
            in.rt_shadow = rt_shadow_;
            in.preserve_existing_depth = depth_ready && culling_ready && fp.technique.depth_prepass;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_hdr_{};
        RT_Motion rt_motion_{};
        RTHandle rt_shadow_{};
        PassPBRForward pass_{};
    };

    class PassPBRForwardAdapter final : public IRenderPass
    {
    public:
        PassPBRForwardAdapter(RTHandle rt_hdr, RT_Motion rt_motion, RTHandle rt_shadow)
            : rt_hdr_(rt_hdr), rt_motion_(rt_motion), rt_shadow_(rt_shadow)
        {}

        const char* id() const override { return "pbr_forward"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::ForwardOpaque;
            c.supported_modes_mask =
                technique_mode_bit(TechniqueMode::Forward) |
                technique_mode_bit(TechniqueMode::ForwardPlus) |
                technique_mode_bit(TechniqueMode::ClusteredForward);
            c.semantics = {
                read_semantic(PassSemantic::ShadowMap, ContractDomain::Software, "shadow"),
                write_semantic(PassSemantic::ColorHDR, ContractDomain::Software, "hdr"),
                write_semantic(PassSemantic::MotionVectors, ContractDomain::Software, "motion")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_rt_resource_ref(rt_shadow_, PassResourceType::Shadow, "shadow", PassResourceDomain::Software));
            io.write(make_rt_resource_ref(rt_hdr_, PassResourceType::ColorHDR, "hdr", PassResourceDomain::Software));
            io.write(make_rt_resource_ref(static_cast<const RTHandle&>(rt_motion_), PassResourceType::Motion, "motion", PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            PassPBRForward::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_hdr = rt_hdr_;
            in.rt_motion = rt_motion_;
            in.rt_shadow = rt_shadow_;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_hdr_{};
        RT_Motion rt_motion_{};
        RTHandle rt_shadow_{};
        PassPBRForward pass_{};
    };

    class PassPBRForwardPlusAdapter final : public IRenderPass
    {
    public:
        PassPBRForwardPlusAdapter(RTHandle rt_hdr, RT_Motion rt_motion, RTHandle rt_shadow)
            : rt_hdr_(rt_hdr), rt_motion_(rt_motion), rt_shadow_(rt_shadow)
        {}

        const char* id() const override { return "pbr_forward_plus"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::ForwardOpaque;
            c.supported_modes_mask = technique_mode_bit(TechniqueMode::ForwardPlus);
            c.requires_depth_prepass = true;
            c.requires_light_culling = true;
            c.semantics = {
                read_semantic(PassSemantic::ShadowMap, ContractDomain::Software, "shadow"),
                read_semantic(PassSemantic::Depth, ContractDomain::Software, "depth"),
                read_semantic(PassSemantic::LightGrid, ContractDomain::Software, "light_grid"),
                read_semantic(PassSemantic::LightIndexList, ContractDomain::Software, "light_index_list"),
                write_semantic(PassSemantic::ColorHDR, ContractDomain::Software, "hdr"),
                write_semantic(PassSemantic::MotionVectors, ContractDomain::Software, "motion")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_rt_resource_ref(rt_shadow_, PassResourceType::Shadow, "shadow", PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.depth_prepass", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.light_grid", PassResourceType::Temp, PassResourceDomain::Software));
            io.read(make_named_resource_ref("technique.light_index_list", PassResourceType::Temp, PassResourceDomain::Software));
            io.write(make_rt_resource_ref(rt_hdr_, PassResourceType::ColorHDR, "hdr", PassResourceDomain::Software));
            io.write(make_rt_resource_ref(static_cast<const RTHandle&>(rt_motion_), PassResourceType::Motion, "motion", PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            const bool light_culling_enabled = fp.technique.light_culling || fp.technique.mode == TechniqueMode::ForwardPlus;
            const bool depth_ready = (!fp.technique.depth_prepass) || ctx.forward_plus.depth_prepass_valid;
            const bool culling_ready = (!light_culling_enabled) || ctx.forward_plus.light_culling_valid;

            PassPBRForward::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_hdr = rt_hdr_;
            in.rt_motion = rt_motion_;
            in.rt_shadow = rt_shadow_;
            in.preserve_existing_depth = depth_ready && culling_ready && fp.technique.depth_prepass;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_hdr_{};
        RT_Motion rt_motion_{};
        RTHandle rt_shadow_{};
        PassPBRForward pass_{};
    };

    class PassTonemapAdapter final : public IRenderPass
    {
    public:
        PassTonemapAdapter(RTHandle rt_hdr, RTHandle rt_ldr)
            : rt_hdr_(rt_hdr), rt_ldr_(rt_ldr)
        {}

        const char* id() const override { return "tonemap"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::Composite;
            c.supported_modes_mask = technique_mode_mask_all();
            c.semantics = {
                read_semantic(PassSemantic::ColorHDR, ContractDomain::Software, "hdr"),
                write_semantic(PassSemantic::ColorLDR, ContractDomain::Software, "ldr")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read(make_rt_resource_ref(rt_hdr_, PassResourceType::ColorHDR, "hdr", PassResourceDomain::Software));
            io.write(make_rt_resource_ref(rt_ldr_, PassResourceType::ColorLDR, "ldr", PassResourceDomain::Software));
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            (void)scene;
            PassTonemap::Inputs in{};
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_hdr = rt_hdr_;
            in.rt_ldr = rt_ldr_;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_hdr_{};
        RTHandle rt_ldr_{};
        PassTonemap pass_{};
    };

    class PassLightShaftsAdapter final : public IRenderPass
    {
    public:
        PassLightShaftsAdapter(RTHandle rt_ldr_inout, RTHandle rt_depth_like, RTHandle rt_shafts_tmp)
            : rt_ldr_(rt_ldr_inout), rt_depth_like_(rt_depth_like), rt_shafts_tmp_(rt_shafts_tmp)
        {}

        const char* id() const override { return "light_shafts"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::PostProcess;
            c.supported_modes_mask = technique_mode_mask_all();
            // Light shafts can run without a dedicated depth-prepass; it consumes
            // the motion/depth-like buffer produced by the forward pass.
            c.semantics = {
                read_write_semantic(PassSemantic::ColorLDR, ContractDomain::Software, "ldr"),
                read_semantic(PassSemantic::MotionVectors, ContractDomain::Software, "depth_like")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read_write(make_rt_resource_ref(rt_ldr_, PassResourceType::ColorLDR, "ldr", PassResourceDomain::Software));
            io.read(make_rt_resource_ref(rt_depth_like_, PassResourceType::Motion, "motion", PassResourceDomain::Software));
            if (rt_shafts_tmp_.valid())
            {
                io.write(make_rt_resource_ref(rt_shafts_tmp_, PassResourceType::Temp, "shafts_tmp", PassResourceDomain::Software));
            }
            else
            {
                io.write(make_named_resource_ref("light_shafts.auto_tmp", PassResourceType::Temp, PassResourceDomain::Software));
            }
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            PassLightShafts::Inputs in{};
            in.scene = &scene;
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_input_ldr = rt_ldr_;
            in.rt_output_ldr = rt_ldr_;
            in.rt_depth_like = rt_depth_like_;
            RTHandle tmp = rt_shafts_tmp_;
            if (!tmp.valid())
            {
                auto* ldr = static_cast<RT_ColorLDR*>(rtr.get(rt_ldr_));
                if (ldr) tmp = rtr.ensure_transient_color_ldr("light_shafts.auto_tmp", ldr->w, ldr->h);
            }
            in.rt_shafts_tmp = tmp;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_ldr_{};
        RTHandle rt_depth_like_{};
        RTHandle rt_shafts_tmp_{};
        PassLightShafts pass_{};
    };

    class PassMotionBlurAdapter final : public IRenderPass
    {
    public:
        PassMotionBlurAdapter(RTHandle rt_ldr_inout, RTHandle rt_motion, RTHandle rt_tmp)
            : rt_ldr_(rt_ldr_inout), rt_motion_(rt_motion), rt_tmp_(rt_tmp)
        {}

        const char* id() const override { return "motion_blur"; }
        RenderBackendType preferred_backend() const override { return RenderBackendType::Software; }
        bool supports_backend(RenderBackendType backend) const override { return backend == RenderBackendType::Software; }
        TechniquePassContract describe_contract() const override
        {
            TechniquePassContract c{};
            c.role = TechniquePassRole::PostProcess;
            c.supported_modes_mask = technique_mode_mask_all();
            c.semantics = {
                read_write_semantic(PassSemantic::ColorLDR, ContractDomain::Software, "ldr"),
                read_semantic(PassSemantic::MotionVectors, ContractDomain::Software, "motion")
            };
            return c;
        }
        PassIODesc describe_io() const override
        {
            PassIODesc io{};
            io.read_write(make_rt_resource_ref(rt_ldr_, PassResourceType::ColorLDR, "ldr", PassResourceDomain::Software));
            io.read(make_rt_resource_ref(rt_motion_, PassResourceType::Motion, "motion", PassResourceDomain::Software));
            if (rt_tmp_.valid())
            {
                io.write(make_rt_resource_ref(rt_tmp_, PassResourceType::Temp, "motion_tmp", PassResourceDomain::Software));
            }
            else
            {
                io.write(make_named_resource_ref("motion_blur.auto_tmp", PassResourceType::Temp, PassResourceDomain::Software));
            }
            return io;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            (void)scene;
            PassMotionBlur::Inputs in{};
            in.fp = &fp;
            in.rtr = &rtr;
            in.rt_input_ldr = rt_ldr_;
            in.rt_output_ldr = rt_ldr_;
            in.rt_motion = rt_motion_;
            RTHandle tmp = rt_tmp_;
            if (!tmp.valid())
            {
                auto* ldr = static_cast<RT_ColorLDR*>(rtr.get(rt_ldr_));
                if (ldr) tmp = rtr.ensure_transient_color_ldr("motion_blur.auto_tmp", ldr->w, ldr->h);
            }
            in.rt_tmp = tmp;
            pass_.execute(ctx, in);
        }

    private:
        RTHandle rt_ldr_{};
        RTHandle rt_motion_{};
        RTHandle rt_tmp_{};
        PassMotionBlur pass_{};
    };

    inline PassFactoryRegistry make_standard_pass_factory_registry(
        RT_Shadow rt_shadow,
        RTHandle rt_hdr,
        RT_Motion rt_motion,
        RTHandle rt_ldr,
        RTHandle rt_shafts_tmp,
        RTHandle rt_motion_blur_tmp
    )
    {
        PassFactoryRegistry reg{};
        reg.register_factory("shadow_map", [=]() {
            return std::make_unique<PassShadowMapAdapter>(rt_shadow);
        });
        reg.register_factory("pbr_forward", [=]() {
            return std::make_unique<PassPBRForwardAdapter>(rt_hdr, rt_motion, RTHandle{rt_shadow.id});
        });
        reg.register_factory("depth_prepass", [=]() {
            return std::make_unique<PassDepthPrepassAdapter>(rt_motion);
        });
        reg.register_factory("light_culling", [=]() {
            return std::make_unique<PassLightCullingAdapter>(rt_motion);
        });
        reg.register_factory("cluster_build", [=]() {
            return std::make_unique<PassClusterBuildAdapter>(rt_motion);
        });
        reg.register_factory("cluster_light_assign", [=]() {
            return std::make_unique<PassClusterLightAssignAdapter>(rt_motion);
        });
        reg.register_factory("pbr_forward_plus", [=]() {
            return std::make_unique<PassPBRForwardPlusAdapter>(rt_hdr, rt_motion, RTHandle{rt_shadow.id});
        });
        reg.register_factory("pbr_forward_clustered", [=]() {
            return std::make_unique<PassPBRForwardClusteredAdapter>(rt_hdr, rt_motion, RTHandle{rt_shadow.id});
        });
        reg.register_factory("gbuffer", [=]() {
            return std::make_unique<PassGBufferAdapter>();
        });
        reg.register_factory("deferred_lighting", [=]() {
            return std::make_unique<PassDeferredLightingAdapter>(rt_hdr, rt_motion, RTHandle{rt_shadow.id});
        });
        reg.register_factory("deferred_lighting_tiled", [=]() {
            return std::make_unique<PassDeferredLightingTiledAdapter>(rt_hdr, rt_motion, RTHandle{rt_shadow.id});
        });
        reg.register_factory("tonemap", [=]() {
            return std::make_unique<PassTonemapAdapter>(rt_hdr, rt_ldr);
        });
        reg.register_factory("light_shafts", [=]() {
            return std::make_unique<PassLightShaftsAdapter>(rt_ldr, rt_motion, rt_shafts_tmp);
        });
        reg.register_factory("motion_blur", [=]() {
            return std::make_unique<PassMotionBlurAdapter>(rt_ldr, rt_motion, rt_motion_blur_tmp);
        });
        return reg;
    }
}
