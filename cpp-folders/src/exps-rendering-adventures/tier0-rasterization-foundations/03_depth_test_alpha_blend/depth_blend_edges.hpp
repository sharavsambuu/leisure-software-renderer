#pragma once

/*
    tier0 demo 03 (AD3 pilot) — EDGE stages: software execution and PNG output.

    Split from depth_blend_plan.hpp on purpose. The pure zone must not link
    against the framebuffer, the rasterizer's storage or the PNG encoder, so
    "no file or driver I/O in preparation" is a property of the file itself and
    not of a convention someone has to remember. The Vulkan twin keeps its own
    executor (it owns the device, the SPIR-V and the uploads) but consumes the
    SAME plan and reports failures in the SAME vocabulary.

    Both stages return std::expected, so they compose with prepare_depth_blend()
    into one typed chain at the host edge.
*/

#include <expected>
#include <string>
#include <vector>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_sw_raster.hpp"
#include "depth_blend_plan.hpp"

namespace adventures::t0_03
{
    // Software realization of the plan, exposed as the executor's whole target:
    // the image PLUS the z-buffer and stencil plane it owns. Presentation needs
    // only the image (execute_software below), but a caller that must inspect
    // depth or stencil — the AD4 known-answer checks do — gets the real storage
    // from the real plan instead of re-implementing the draw sequence.
    struct SoftwareTarget
    {
        Frame                frame;
        std::vector<float>   depth;
        std::vector<uint8_t> stencil;
    };

    // Trusts nothing: the plan is public aggregate data, so a hand-built plan
    // with an out-of-range or partial draw is rejected instead of reading past
    // its vertices or silently dropping a stray vertex.
    inline std::expected<SoftwareTarget, DepthBlendError> execute_software_target(const DepthBlendPlan& plan)
    {
        if (plan.width <= 0 || plan.height <= 0)
        {
            return std::unexpected(DepthBlendError::InvalidFrameExtent);
        }
        for (const DepthBlendPass& pass : plan.passes)
        {
            if (size_t(pass.first_vertex) + size_t(pass.vertex_count) > plan.vertices.size())
            {
                return std::unexpected(DepthBlendError::DrawRangeOutOfBounds);
            }
            if (pass.vertex_count == 0 || pass.vertex_count % 3u != 0)
            {
                return std::unexpected(DepthBlendError::DrawNotTriangleList);
            }
        }

        SoftwareTarget target{};
        target.frame = Frame(plan.width, plan.height);
        target.frame.clear(12, 12, 16); // the clear every tier0 demo uses
        SwRaster raster(target.frame);
        for (const DepthBlendPass& pass : plan.passes)
        {
            const std::vector<T0Vertex> draw(plan.vertices.begin() + pass.first_vertex,
                                             plan.vertices.begin() + pass.first_vertex + pass.vertex_count);
            draw_triangles(raster, pass.policy, draw); // the plan's policy, verbatim
        }
        target.depth   = std::move(raster.depth);
        target.stencil = std::move(raster.stencil);
        return target;
    }

    // The presentation-shaped stage: same execution, image only.
    inline std::expected<Frame, DepthBlendError> execute_software(const DepthBlendPlan& plan)
    {
        auto target = execute_software_target(plan);
        if (!target) return std::unexpected(target.error());
        return std::move(target->frame);
    }

    // Output edge: the only place this pilot touches the filesystem. Returns the
    // path on success so the host edge can log it without a second source of
    // truth.
    inline std::expected<std::string, DepthBlendError> write_png(const Frame& frame, std::string path)
    {
        if (!frame.save_png(path)) return std::unexpected(DepthBlendError::OutputUnwritable);
        return path;
    }
}
