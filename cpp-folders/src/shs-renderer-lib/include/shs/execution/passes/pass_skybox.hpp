#pragma once

/*
    SHS RENDERER SAN

    FILE: pass_skybox.hpp
    MODULE: execution/passes (edge dispatch; R2 P2.4)
    PURPOSE: Parallel dispatch wrapper over the sky domain's pure
             shade_skybox_rows(). Same signature the forward pass used;
             the math stays headlessly testable in domains/sky.
*/

#include "shs/domains/sky/skybox_renderer.hpp"
#include "shs/execution/job/job_system.hpp"
#include "shs/execution/job/parallel_for.hpp"

namespace shs
{
    inline void render_skybox_to_hdr(RT_ColorHDR& out_hdr, const Scene& scene, const ISkyModel& sky, IJobSystem* jobs = nullptr)
    {
        if (out_hdr.w <= 0 || out_hdr.h <= 0) return;
        parallel_for_1d(jobs, 0, out_hdr.h, 8, [&](int yb, int ye)
        {
            shade_skybox_rows(out_hdr, scene, sky, yb, ye);
        });
    }
} // namespace shs
