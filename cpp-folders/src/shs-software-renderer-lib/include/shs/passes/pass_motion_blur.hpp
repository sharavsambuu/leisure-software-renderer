#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_motion_blur.hpp
    МОДУЛЬ: passes
    ЗОРИЛГО: Camera + per-object хөдөлгөөний вектор дээр тулгуурласан пост-процесс
            motion blur хэрэгжүүлнэ.
*/


#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"
#include "shs/job/parallel_for.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace shs
{
    struct Context;

    class PassMotionBlur
    {
    public:
        struct Inputs
        {
            const FrameParams* fp = nullptr;
            RTRegistry* rtr = nullptr;

            RTHandle rt_input_ldr{};
            RTHandle rt_output_ldr{};
            RTHandle rt_motion{};
            RTHandle rt_tmp{};
        };

        void execute(Context& ctx, const Inputs& in)
        {
            if (!in.fp || !in.rtr) return;
            if (!in.rt_input_ldr.valid() || !in.rt_output_ldr.valid()) return;

            auto* src = static_cast<RT_ColorLDR*>(in.rtr->get(in.rt_input_ldr));
            auto* dst = static_cast<RT_ColorLDR*>(in.rtr->get(in.rt_output_ldr));
            auto* motion = in.rt_motion.valid() ? static_cast<RT_ColorDepthMotion*>(in.rtr->get(in.rt_motion)) : nullptr;
            auto* tmp = in.rt_tmp.valid() ? static_cast<RT_ColorLDR*>(in.rtr->get(in.rt_tmp)) : nullptr;
            if (!src || !dst || !motion) return;

            const int w = std::min({src->w, dst->w, motion->w});
            const int h = std::min({src->h, dst->h, motion->h});
            if (w <= 0 || h <= 0) return;

            if (!in.fp->pass.motion_blur.enable)
            {
                copy_ldr(ctx, *src, *dst, w, h);
                return;
            }

            if (tmp && (tmp->w != w || tmp->h != h))
            {
                tmp = nullptr;
            }
            const bool in_place = (src == dst);

            std::vector<Color> scratch{};
            if (!tmp && in_place)
            {
                scratch.resize((size_t)w * (size_t)h);
            }

            const int samples = std::clamp(in.fp->pass.motion_blur.samples, 4, 32);
            const float strength = std::max(0.0f, in.fp->pass.motion_blur.strength);
            const float max_vel = std::max(1.0f, in.fp->pass.motion_blur.max_velocity_px);
            const float min_vel = std::max(0.0f, in.fp->pass.motion_blur.min_velocity_px);
            const float depth_eps = std::max(0.0f, in.fp->pass.motion_blur.depth_reject);
            const float dt_scale = std::clamp(std::max(in.fp->dt, 1e-4f) * 60.0f, 0.5f, 2.5f);

            auto sample_color = [&](int sx, int sy) -> Color
            {
                sx = std::clamp(sx, 0, w - 1);
                sy = std::clamp(sy, 0, h - 1);
                return src->color.at(sx, sy);
            };

            auto sample_depth = [&](int sx, int sy) -> float
            {
                sx = std::clamp(sx, 0, w - 1);
                sy = std::clamp(sy, 0, h - 1);
                return motion->depth.at(sx, sy);
            };

            auto write_pixel = [&](int x, int y, const Color& c)
            {
                if (tmp)
                {
                    tmp->color.at(x, y) = c;
                }
                else if (!scratch.empty())
                {
                    scratch[(size_t)y * (size_t)w + (size_t)x] = c;
                }
                else
                {
                    dst->color.at(x, y) = c;
                }
            };

            parallel_for_1d(ctx.job_system, 0, h, 4, [&](int yb, int ye)
            {
                for (int y = yb; y < ye; ++y)
                {
                    for (int x = 0; x < w; ++x)
                    {
                        Motion2f mv = motion->motion.at(x, y);
                        float vx = mv.x * strength * dt_scale;
                        float vy = mv.y * strength * dt_scale;
                        float len = std::sqrt(vx * vx + vy * vy);
                        if (len < min_vel)
                        {
                            write_pixel(x, y, src->color.at(x, y));
                            continue;
                        }
                        if (len > max_vel && len > 1e-6f)
                        {
                            const float s = max_vel / len;
                            vx *= s;
                            vy *= s;
                        }

                        const float center_depth = motion->depth.at(x, y);
                        float ar = 0.0f;
                        float ag = 0.0f;
                        float ab = 0.0f;
                        float aw = 0.0f;
                        for (int i = 0; i < samples; ++i)
                        {
                            const float t = ((float)i / (float)(samples - 1) - 0.5f);
                            const int sx = (int)std::lround((float)x + vx * t);
                            const int sy = (int)std::lround((float)y + vy * t);
                            const float sd = sample_depth(sx, sy);
                            if (std::abs(sd - center_depth) > depth_eps) continue;
                            const Color sc = sample_color(sx, sy);
                            ar += (float)sc.r;
                            ag += (float)sc.g;
                            ab += (float)sc.b;
                            aw += 1.0f;
                        }

                        if (aw < 1.0f)
                        {
                            write_pixel(x, y, src->color.at(x, y));
                            continue;
                        }

                        const Color out{
                            (uint8_t)std::clamp((int)std::lround(ar / aw), 0, 255),
                            (uint8_t)std::clamp((int)std::lround(ag / aw), 0, 255),
                            (uint8_t)std::clamp((int)std::lround(ab / aw), 0, 255),
                            255
                        };
                        write_pixel(x, y, out);
                    }
                }
            });

            if (tmp)
            {
                copy_ldr(ctx, *tmp, *dst, w, h);
            }
            else if (!scratch.empty())
            {
                parallel_for_1d(ctx.job_system, 0, h, 8, [&](int yb, int ye)
                {
                    for (int y = yb; y < ye; ++y)
                    {
                        for (int x = 0; x < w; ++x)
                        {
                            dst->color.at(x, y) = scratch[(size_t)y * (size_t)w + (size_t)x];
                        }
                    }
                });
            }
        }

    private:
        static void copy_ldr(Context& ctx, const RT_ColorLDR& src, RT_ColorLDR& dst, int w, int h)
        {
            parallel_for_1d(ctx.job_system, 0, h, 8, [&](int yb, int ye)
            {
                for (int y = yb; y < ye; ++y)
                {
                    for (int x = 0; x < w; ++x)
                    {
                        dst.color.at(x, y) = src.color.at(x, y);
                    }
                }
            });
        }
    };
}
