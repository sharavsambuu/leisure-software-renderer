#pragma once

/*
    tier0 demo 03 (AD3 pilot) — PURE preparation and validation.

    Demo 03's lesson is that draw ORDER and per-pass fixed-function policy
    decide the image (near opaque wins the overlap; the translucent quad is
    drawn last, blends, and must not write depth). This header turns that lesson
    into a value: a request goes in, a backend-neutral plan comes out, and every
    rejection is a named member of a closed error vocabulary.

    What this zone may touch: plain values, the shared PassPolicy, and the
    scene's vertex type. What it must NOT touch (AD3 acceptance): drivers,
    files, logging, clocks, or any backend header. Both twins — the software
    rasterizer and the Vulkan executor — consume the SAME plan; neither can
    restate the order or the policy differently, because the plan is where they
    are decided.

    The plan OWNS its vertices (moved out of the request), so no borrowed
    payload outlives its owner and the plan stays valid after the caller's scene
    vector is gone.
*/

#include <cstdint>
#include <expected>
#include <utility>
#include <vector>

#include "../common/adventures_pass_policy.hpp"
#include "../common/t0_scenes.hpp"

namespace adventures::t0_03
{
    // Which draw of the lesson a pass is. Closed: the demo has exactly two.
    enum class DepthBlendLayer : uint8_t
    {
        Opaque,      // depth test + write, no blend — resolves the overlap
        Translucent, // depth test on, depth write OFF, blend on — drawn last
    };

    // Closed failure vocabulary. Every value is a failure this pipeline can
    // actually hit, at the stage that can hit it; no value stands in for
    // infallible math. depth_blend_stage() is the single error->stage mapping,
    // so diagnostics and exit codes cannot disagree.
    enum class DepthBlendError : uint8_t
    {
        // --- preparation (pure; nothing allocated or opened yet) ---
        InvalidFrameExtent,          // width or height <= 0: no target to render
        EmptyScene,                  // no vertices at all
        OpaqueNotTriangleList,       // opaque count 0 or not a multiple of 3
        TranslucentNotTriangleList,  // translucent count 0 or not a multiple of 3
        DrawCountsMismatch,          // the two ranges do not add up to the scene
        TranslucentBeforeOpaque,     // the lesson's ordering requirement is broken
        // --- execution: software realization ---
        DrawRangeOutOfBounds,        // a plan (possibly hand-built) points past its vertices
        DrawNotTriangleList,         // a plan (possibly hand-built) carries a partial triangle
        // --- execution: Vulkan realization (never produced by the software path) ---
        DeviceUnavailable,           // no usable Vulkan device for this target
        ShaderModuleUnreadable,      // SPIR-V file missing/unreadable (adapted bool)
        PipelineUnavailable,         // vkCreateGraphicsPipelines refused (adapted index)
        VertexUploadFailed,          // staging/upload refused (adapted bool)
        FrameSubmitFailed,           // record/submit/readback refused (adapted bool)
        // --- output edge ---
        OutputUnwritable,            // the PNG encoder could not write the file
    };

    enum class DepthBlendStage : uint8_t
    {
        Preparation,
        Execution,
        Output,
    };

    inline DepthBlendStage depth_blend_stage(DepthBlendError e)
    {
        switch (e)
        {
            case DepthBlendError::InvalidFrameExtent:
            case DepthBlendError::EmptyScene:
            case DepthBlendError::OpaqueNotTriangleList:
            case DepthBlendError::TranslucentNotTriangleList:
            case DepthBlendError::DrawCountsMismatch:
            case DepthBlendError::TranslucentBeforeOpaque:
                return DepthBlendStage::Preparation;
            case DepthBlendError::DrawRangeOutOfBounds:
            case DepthBlendError::DrawNotTriangleList:
            case DepthBlendError::DeviceUnavailable:
            case DepthBlendError::ShaderModuleUnreadable:
            case DepthBlendError::PipelineUnavailable:
            case DepthBlendError::VertexUploadFailed:
            case DepthBlendError::FrameSubmitFailed:
                return DepthBlendStage::Execution;
            case DepthBlendError::OutputUnwritable:
                return DepthBlendStage::Output;
        }
        return DepthBlendStage::Preparation; // unreachable for a valid enumerator
    }

    inline const char* depth_blend_stage_name(DepthBlendStage s)
    {
        switch (s)
        {
            case DepthBlendStage::Preparation: return "preparation";
            case DepthBlendStage::Execution:   return "execution";
            case DepthBlendStage::Output:      return "output";
        }
        return "unknown";
    }

    inline const char* depth_blend_message(DepthBlendError e)
    {
        switch (e)
        {
            case DepthBlendError::InvalidFrameExtent:         return "frame extent must be positive";
            case DepthBlendError::EmptyScene:                 return "scene has no vertices";
            case DepthBlendError::OpaqueNotTriangleList:      return "opaque draw is not a triangle list";
            case DepthBlendError::TranslucentNotTriangleList: return "translucent draw is not a triangle list";
            case DepthBlendError::DrawCountsMismatch:         return "draw counts do not sum to the scene size";
            case DepthBlendError::TranslucentBeforeOpaque:    return "translucent draw must follow the opaque draw";
            case DepthBlendError::DrawRangeOutOfBounds:       return "draw range exceeds the plan's vertices";
            case DepthBlendError::DrawNotTriangleList:        return "draw range is not a whole number of triangles";
            case DepthBlendError::DeviceUnavailable:          return "no usable Vulkan device";
            case DepthBlendError::ShaderModuleUnreadable:     return "SPIR-V module could not be read";
            case DepthBlendError::PipelineUnavailable:        return "graphics pipeline could not be created";
            case DepthBlendError::VertexUploadFailed:         return "vertex upload failed";
            case DepthBlendError::FrameSubmitFailed:          return "frame record/submit/readback failed";
            case DepthBlendError::OutputUnwritable:           return "PNG could not be written";
        }
        return "unknown failure";
    }

    // Exit codes live beside the stage mapping so a new error cannot silently
    // inherit the wrong code. Tooling only distinguishes zero from non-zero.
    inline int depth_blend_exit_code(DepthBlendError e)
    {
        switch (depth_blend_stage(e))
        {
            case DepthBlendStage::Preparation: return 1; // bad inputs / broken lesson invariant
            case DepthBlendStage::Execution:   return 2; // renderer could not run
            case DepthBlendStage::Output:      return 3; // image could not be written
        }
        return 1;
    }

    struct DepthBlendPass
    {
        DepthBlendLayer layer        = DepthBlendLayer::Opaque;
        uint32_t        first_vertex = 0;
        uint32_t        vertex_count = 0;
        PassPolicy      policy{}; // decided HERE, consumed by both backends
    };

    struct DepthBlendPlan
    {
        int                         width  = 0;
        int                         height = 0;
        std::vector<T0Vertex>       vertices{}; // owned (no borrowed payload)
        std::vector<DepthBlendPass> passes{};   // opaque first, translucent last

        const DepthBlendPass* find(DepthBlendLayer layer) const
        {
            for (const DepthBlendPass& p : passes)
            {
                if (p.layer == layer) return &p;
            }
            return nullptr;
        }

        uint32_t total_vertices() const
        {
            uint32_t n = 0;
            for (const DepthBlendPass& p : passes) n += p.vertex_count;
            return n;
        }
    };

    // The explicit draw inputs of demo 03: nothing is inferred from a global or
    // from a previous call.
    struct DepthBlendRequest
    {
        int                   width  = 0;
        int                   height = 0;
        std::vector<T0Vertex> vertices{};
        uint32_t              opaque_vertex_count      = 0;
        uint32_t              translucent_vertex_count = 0;
        bool                  translucent_drawn_last   = true;
    };

    // PURE: validates the request and produces the plan. No driver, no file
    // I/O, no logging — every rejection is returned, never printed.
    inline std::expected<DepthBlendPlan, DepthBlendError> prepare_depth_blend(DepthBlendRequest request)
    {
        if (request.width <= 0 || request.height <= 0)
        {
            return std::unexpected(DepthBlendError::InvalidFrameExtent);
        }
        if (request.vertices.empty()) return std::unexpected(DepthBlendError::EmptyScene);
        if (request.opaque_vertex_count == 0 || request.opaque_vertex_count % 3u != 0)
        {
            return std::unexpected(DepthBlendError::OpaqueNotTriangleList);
        }
        if (request.translucent_vertex_count == 0 || request.translucent_vertex_count % 3u != 0)
        {
            return std::unexpected(DepthBlendError::TranslucentNotTriangleList);
        }
        if (static_cast<size_t>(request.opaque_vertex_count) + size_t(request.translucent_vertex_count) !=
            request.vertices.size())
        {
            return std::unexpected(DepthBlendError::DrawCountsMismatch);
        }
        if (!request.translucent_drawn_last) return std::unexpected(DepthBlendError::TranslucentBeforeOpaque);

        DepthBlendPlan plan{};
        plan.width    = request.width;
        plan.height   = request.height;
        plan.vertices = std::move(request.vertices); // ownership transfer

        DepthBlendPass opaque{};
        opaque.layer        = DepthBlendLayer::Opaque;
        opaque.first_vertex = 0;
        opaque.vertex_count = request.opaque_vertex_count;
        // opaque uses the PassPolicy defaults: depth test + write, no blend

        DepthBlendPass translucent{};
        translucent.layer                = DepthBlendLayer::Translucent;
        translucent.first_vertex         = request.opaque_vertex_count;
        translucent.vertex_count         = request.translucent_vertex_count;
        translucent.policy.depth_write   = false; // blended fragments must not poison the z-buffer
        translucent.policy.blend         = true;  // SRC_ALPHA / ONE_MINUS_SRC_ALPHA

        plan.passes.push_back(opaque);      // order is the lesson: opaque first
        plan.passes.push_back(translucent); // translucent last
        return plan;
    }
}
