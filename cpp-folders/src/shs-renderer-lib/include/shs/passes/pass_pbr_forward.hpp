#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_pbr_forward.hpp
    МОДУЛЬ: passes
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн passes модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/sw_render/rasterizer.hpp"
#include "shs/resources/resource_registry.hpp"
#include "shs/scene/scene_types.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"
#include "shs/gfx/rt_shadow.hpp"
#include "shs/job/parallel_for.hpp"
#include "shs/shader/builtin_shaders.hpp"
#include "shs/sky/skybox_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace shs
{
    struct Context;

    class PassPBRForward
    {
    public:
        struct Inputs
        {
            const Scene*       scene = nullptr;
            const FrameParams* fp    = nullptr;
            RTRegistry*        rtr   = nullptr;

            RTHandle rt_hdr{};
            RTHandle rt_motion{};
            RTHandle rt_shadow{};
            // Forward+ зэрэг техникүүд depth prepass-аар depth-ийг урьдчилж бөглөсөн үед ашиглана.
            bool preserve_existing_depth = false;
        };

        void execute(Context& ctx, const Inputs& in)
        {
            if (!in.scene || !in.fp || !in.rtr) return;
            if (!in.rt_hdr.valid()) return;
            ctx.debug.tri_input = 0;
            ctx.debug.tri_after_clip = 0;
            ctx.debug.tri_raster = 0;

            auto* hdr = static_cast<RT_ColorHDR*>(in.rtr->get(in.rt_hdr));
            if (!hdr || hdr->w <= 0 || hdr->h <= 0) return;

            // Motion/shadow RT нь тухайн pipeline дээр тохируулагдсан үед л ашиглагдана.
            auto* motion = in.rt_motion.valid() ? static_cast<RT_ColorDepthMotion*>(in.rtr->get(in.rt_motion)) : nullptr;
            auto* shadow = in.rt_shadow.valid() ? static_cast<RT_ShadowDepth*>(in.rtr->get(in.rt_shadow)) : nullptr;

            if (in.scene->sky)
            {
                render_skybox_to_hdr(*hdr, *in.scene, *in.scene->sky, ctx.job_system);
            }
            else
            {
                // Sky model байхгүй үед HDR background градиент зурна.
                parallel_for_1d(ctx.job_system, 0, hdr->h, 8, [&](int yb, int ye)
                {
                    for (int y = yb; y < ye; ++y)
                    {
                        const float t = (float)y / (float)std::max(1, hdr->h - 1);
                        const ColorF clear = {
                            0.06f + 0.08f * t,
                            0.08f + 0.10f * t,
                            0.12f + 0.12f * t,
                            1.0f
                        };
                        for (int x = 0; x < hdr->w; ++x) hdr->color.at(x, y) = clear;
                    }
                });
            }

            if (motion && motion->w == hdr->w && motion->h == hdr->h)
            {
                if (in.preserve_existing_depth)
                {
                    motion->color.clear(motion->clear);
                    motion->motion.clear(Motion2f{});
                }
                else
                {
                    motion->clear_all();
                }
            }

            ShaderProgram prog = make_pbr_mr_program();
            if (in.fp->shading_model == ShadingModel::BlinnPhong)
            {
                prog = make_blinn_phong_program();
            }
            if (in.fp->debug_view != DebugViewMode::Final)
            {
                prog = make_debug_view_shader_program(in.fp->debug_view);
            }
            RasterizerTarget tgt{};
            tgt.hdr = hdr;
            tgt.depth_motion = (motion && motion->w == hdr->w && motion->h == hdr->h) ? motion : nullptr;
            RasterizerConfig rast_cfg{};
            rast_cfg.front_face_ccw = in.fp->front_face_ccw;
            rast_cfg.job_system = ctx.job_system;
            switch (in.fp->cull_mode)
            {
            case CullMode::None: rast_cfg.cull_mode = RasterizerCullMode::None; break;
            case CullMode::Front: rast_cfg.cull_mode = RasterizerCullMode::Front; break;
            case CullMode::Back:
            default: rast_cfg.cull_mode = RasterizerCullMode::Back; break;
            }

            std::unordered_map<uint64_t, glm::mat4> next_prev_model_by_object{};
            next_prev_model_by_object.reserve(in.scene->items.size() * 2 + 1);

            for (size_t item_index = 0; item_index < in.scene->items.size(); ++item_index)
            {
                const auto& item = in.scene->items[item_index];
                if (!item.visible) continue;
                if (!in.scene->resources) continue;

                const MeshData* mesh = in.scene->resources->get_mesh((MeshAssetHandle)item.mesh);
                if (!mesh || mesh->empty()) continue;
                const MaterialData* mat = in.scene->resources->get_material((MaterialAssetHandle)item.mat);

                glm::mat4 model(1.0f);
                model = glm::translate(model, item.tr.pos);
                model = glm::rotate(model, item.tr.rot_euler.x, glm::vec3(1.0f, 0.0f, 0.0f));
                model = glm::rotate(model, item.tr.rot_euler.y, glm::vec3(0.0f, 1.0f, 0.0f));
                model = glm::rotate(model, item.tr.rot_euler.z, glm::vec3(0.0f, 0.0f, 1.0f));
                model = glm::scale(model, item.tr.scl);

                uint64_t motion_key = item.object_id;
                if (motion_key == 0)
                {
                    motion_key = ((uint64_t)item.mesh << 32) ^ (uint64_t)item.mat ^ ((uint64_t)item_index + 1u);
                    if (motion_key == 0) motion_key = 1;
                }
                glm::mat4 prev_model = model;
                const auto it_prev = ctx.history.prev_model_by_object.find(motion_key);
                if (ctx.history.has_prev_frame && it_prev != ctx.history.prev_model_by_object.end())
                {
                    prev_model = it_prev->second;
                }
                next_prev_model_by_object[motion_key] = model;

                ShaderUniforms u{};
                u.model = model;
                u.viewproj = in.scene->cam.viewproj;
                u.prev_model = prev_model;
                u.prev_viewproj = ctx.history.has_prev_frame ? in.scene->cam.prev_viewproj : in.scene->cam.viewproj;
                u.light_dir_ws = in.scene->sun.dir_ws;
                u.light_color = in.scene->sun.color;
                u.light_intensity = in.scene->sun.intensity;
                u.camera_pos = in.scene->cam.pos;
                u.enable_motion_vectors = in.fp->pass.motion_vectors.enable;
                if (mat)
                {
                    u.base_color = mat->base_color;
                    u.metallic = mat->metallic;
                    u.roughness = mat->roughness;
                    u.ao = mat->ao;
                    if (mat->base_color_tex != 0 && in.scene->resources)
                    {
                        u.base_color_tex = in.scene->resources->get_texture(mat->base_color_tex);
                    }
                }
                else
                {
                    u.base_color = glm::vec3(0.8f, 0.5f, 0.2f);
                    u.metallic = 0.1f;
                    u.roughness = 0.5f;
                    u.ao = 1.0f;
                }

                if (in.fp->pass.shadow.enable && shadow && ctx.shadow.valid)
                {
                    u.shadow_map = shadow;
                    u.light_viewproj = ctx.shadow.light_viewproj;
                    u.shadow_bias_const = in.fp->pass.shadow.bias_const;
                    u.shadow_bias_slope = in.fp->pass.shadow.bias_slope;
                    u.shadow_pcf_radius = in.fp->pass.shadow.pcf_radius;
                    u.shadow_pcf_step = in.fp->pass.shadow.pcf_step;
                    u.shadow_strength = in.fp->pass.shadow.strength;
                }

                // Generic uniform slots for future shader permutations.
                set_uniform_mat4(u, 0, u.model);
                set_uniform_mat4(u, 1, u.viewproj);
                set_uniform_vec4(u, 0, glm::vec4(u.base_color, 1.0f));
                set_uniform_vec4(u, 1, glm::vec4(u.light_dir_ws, 0.0f));
                set_uniform_vec4(u, 2, glm::vec4(u.light_color, u.light_intensity));
                set_uniform_vec4(u, 3, glm::vec4(u.camera_pos, 1.0f));
                set_uniform_vec4(u, 4, glm::vec4(u.metallic, u.roughness, u.ao, 0.0f));

                const RasterizerStats rs = rasterize_mesh(*mesh, prog, u, tgt, rast_cfg);
                ctx.debug.tri_input += rs.tri_input;
                ctx.debug.tri_after_clip += rs.tri_after_clip;
                ctx.debug.tri_raster += rs.tri_raster;
            }

            ctx.history.prev_model_by_object.swap(next_prev_model_by_object);
            ctx.history.has_prev_frame = true;
        }
    };
}
