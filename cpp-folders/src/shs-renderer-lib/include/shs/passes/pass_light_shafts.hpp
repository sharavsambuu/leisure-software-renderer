#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_light_shafts.hpp
    МОДУЛЬ: passes
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн passes модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/passes/pass_context.hpp"
#include "shs/scene/scene_types.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"
#include "shs/job/parallel_for.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <glm/glm.hpp>

namespace shs
{
    struct Context;

    class PassLightShafts
    {
    public:
        struct Inputs
        {
            const Scene* scene = nullptr;
            const FrameParams* fp = nullptr;
            RTRegistry* rtr = nullptr;

            RTHandle rt_input_ldr{};
            RTHandle rt_output_ldr{};

            RTHandle rt_depth_like{};
            RTHandle rt_shafts_tmp{};
        };

        void execute(Context& ctx, const Inputs& in)
        {
            if (!in.scene || !in.fp || !in.rtr) return;
            if (!in.rt_input_ldr.valid() || !in.rt_output_ldr.valid()) return;

            auto* inldr = static_cast<RT_ColorLDR*>(in.rtr->get(in.rt_input_ldr));
            auto* outldr = static_cast<RT_ColorLDR*>(in.rtr->get(in.rt_output_ldr));
            if (!inldr || !outldr || inldr->w <= 0 || inldr->h <= 0 || outldr->w <= 0 || outldr->h <= 0) return;

            // Light shafts унтраалттай үед input-ийг output руу шууд дамжуулна.
            if (!in.fp->pass.light_shafts.enable)
            {
                if (inldr == outldr) return;
                const int w = std::min(inldr->w, outldr->w);
                const int h = std::min(inldr->h, outldr->h);
                parallel_for_1d(ctx.job_system, 0, h, 8, [&](int yb, int ye)
                {
                    for (int y = yb; y < ye; ++y)
                    {
                        for (int x = 0; x < w; ++x) outldr->color.at(x, y) = inldr->color.at(x, y);
                    }
                });
                return;
            }

            auto* depth_like = in.rt_depth_like.valid() ? static_cast<RT_ColorDepthMotion*>(in.rtr->get(in.rt_depth_like)) : nullptr;
            auto* tmp = in.rt_shafts_tmp.valid() ? static_cast<RT_ColorLDR*>(in.rtr->get(in.rt_shafts_tmp)) : nullptr;

            const int w = std::min(inldr->w, outldr->w);
            const int h = std::min(inldr->h, outldr->h);
            if (tmp && (tmp->w != w || tmp->h != h)) tmp = nullptr;
            const bool in_place_no_tmp = (tmp == nullptr && inldr == outldr);

            // Estimate sun position on screen from scene camera + sun direction.
            glm::vec2 sun_uv(0.5f, 0.2f);
            bool sun_valid = false;
            {
                const glm::vec3 sun_pos_ws = in.scene->cam.pos + (-in.scene->sun.dir_ws) * 100.0f;
                const glm::vec4 clip = in.scene->cam.viewproj * glm::vec4(sun_pos_ws, 1.0f);
                if (std::abs(clip.w) > 1e-6f)
                {
                    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                    sun_uv = glm::vec2(ndc.x * 0.5f + 0.5f, ndc.y * 0.5f + 0.5f);
                    sun_valid = (clip.w > 0.0f) &&
                                (ndc.z >= -1.0f && ndc.z <= 1.0f) &&
                                (sun_uv.x >= 0.0f && sun_uv.x <= 1.0f) &&
                                (sun_uv.y >= 0.0f && sun_uv.y <= 1.0f);
                }
            }

            // Нар дэлгэц дээр хүчинтэй проекцлогдоогүй үед эффектийг алгасна.
            if (!sun_valid)
            {
                if (inldr == outldr) return;
                parallel_for_1d(ctx.job_system, 0, h, 8, [&](int yb, int ye)
                {
                    for (int y = yb; y < ye; ++y)
                    {
                        for (int x = 0; x < w; ++x) outldr->color.at(x, y) = inldr->color.at(x, y);
                    }
                });
                return;
            }

            // Luma-г нэг удаа урьдчилан тооцоолж, ray marching доторх sample хөрвүүлэлтийн зардлыг бууруулна.
            std::vector<float> luma{};
            luma.resize((size_t)w * (size_t)h, 0.0f);
            parallel_for_1d(ctx.job_system, 0, h, 8, [&](int yb, int ye)
            {
                for (int y = yb; y < ye; ++y)
                {
                    for (int x = 0; x < w; ++x)
                    {
                        const Color c = inldr->color.at(x, y);
                        const float r = (float)c.r / 255.0f;
                        const float g = (float)c.g / 255.0f;
                        const float b = (float)c.b / 255.0f;
                        luma[(size_t)y * (size_t)w + (size_t)x] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                    }
                }
            });

            auto sample_luma = [&](int sx, int sy) -> float
            {
                sx = std::clamp(sx, 0, w - 1);
                sy = std::clamp(sy, 0, h - 1);
                return luma[(size_t)sy * (size_t)w + (size_t)sx];
            };

            const int steps = std::max(8, in.fp->pass.light_shafts.steps);
            const float density = std::max(0.0f, in.fp->pass.light_shafts.density);
            const float weight = std::max(0.0f, in.fp->pass.light_shafts.weight);
            const float decay = std::clamp(in.fp->pass.light_shafts.decay, 0.0f, 1.0f);

            std::vector<Color> scratch{};
            if (in_place_no_tmp) scratch.resize((size_t)w * (size_t)h);

            parallel_for_1d(ctx.job_system, 0, h, 4, [&](int yb, int ye)
            {
                for (int y = yb; y < ye; ++y)
                {
                    for (int x = 0; x < w; ++x)
                    {
                        const float u = (float)x / (float)std::max(1, w - 1);
                        const float v = (float)y / (float)std::max(1, h - 1);

                        float illum_decay = 1.0f;
                        float accum = 0.0f;
                        for (int i = 0; i < steps; ++i)
                        {
                            const float t = (float)i / (float)steps;
                            const float su = u + (sun_uv.x - u) * t * density;
                            const float sv = v + (sun_uv.y - v) * t * density;
                            const int sx = (int)std::lround(su * (float)(w - 1));
                            const int sy = (int)std::lround(sv * (float)(h - 1));

                            float s = sample_luma(sx, sy);
                            if (depth_like && depth_like->w == w && depth_like->h == h)
                            {
                                // Depth нь [near=0 .. far=1] тул sky/far пикселүүд дээр shafts үлдээнэ.
                                const int dx = std::clamp(sx, 0, w - 1);
                                const int dy = std::clamp(sy, 0, h - 1);
                                const float d = depth_like->depth.at(dx, dy);
                                s *= std::clamp(d, 0.0f, 1.0f);
                            }

                            accum += s * illum_decay * weight;
                            illum_decay *= decay;
                        }

                        const Color base = inldr->color.at(x, y);
                        const int boost = std::clamp((int)std::lround(accum * 80.0f), 0, 120);
                        Color out{
                            (uint8_t)std::clamp((int)base.r + boost, 0, 255),
                            (uint8_t)std::clamp((int)base.g + boost, 0, 255),
                            (uint8_t)std::clamp((int)base.b + boost / 2, 0, 255),
                            255
                        };
                        if (tmp) tmp->color.at(x, y) = out;
                        else if (in_place_no_tmp) scratch[(size_t)y * (size_t)w + (size_t)x] = out;
                        else outldr->color.at(x, y) = out;
                    }
                }
            });

            if (tmp)
            {
                parallel_for_1d(ctx.job_system, 0, h, 8, [&](int yb, int ye)
                {
                    for (int y = yb; y < ye; ++y)
                    {
                        for (int x = 0; x < w; ++x)
                        {
                            outldr->color.at(x, y) = tmp->color.at(x, y);
                        }
                    }
                });
            }
            else if (in_place_no_tmp)
            {
                parallel_for_1d(ctx.job_system, 0, h, 8, [&](int yb, int ye)
                {
                    for (int y = yb; y < ye; ++y)
                    {
                        for (int x = 0; x < w; ++x)
                        {
                            outldr->color.at(x, y) = scratch[(size_t)y * (size_t)w + (size_t)x];
                        }
                    }
                });
            }
        }
    };
}
