// tier0 AD3 — typed composition pilot (demo 03), GPU-free and always active.
// Location: .../tier0-rasterization-foundations/tools/t0_composition_tests.cpp
//
// AD3 replaces demo 03's straight-line main() with a value pipeline:
//
//     prepare_depth_blend(request)   PURE   -> expected<DepthBlendPlan, Error>
//       .and_then(execute_software)  EDGE   -> expected<Frame, Error>
//       .and_then(write_png)         EDGE   -> expected<string, Error>
//
// This gate checks the properties that make that more than renamed calls:
//   1. the plan's shape, order and per-pass policy (the lesson as a value)
//   2. every preparation rejection in the closed vocabulary, one at a time
//   3. short-circuiting: after a failure no later stage runs, and the ORIGINAL
//      error value survives the diagnostic mapping
//   4. execution refuses hand-built plans that were never validated
//   5. the output edge reports failure and releases its resources
//   6. the plan owns its vertices: it stays valid after its request is gone
//   7. the error -> stage -> exit code mapping is total and consistent
//
// Run: t0_composition_tests

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h> // getpid, for the scratch path

#include "../03_depth_test_alpha_blend/depth_blend_edges.hpp"
#include "../03_depth_test_alpha_blend/depth_blend_plan.hpp"

using namespace adventures;
using namespace adventures::t0_03;

namespace
{
    int g_checks   = 0;
    int g_failures = 0;

    void check(bool ok, const char* what)
    {
        ++g_checks;
        if (!ok)
        {
            ++g_failures;
            std::fprintf(stderr, "FAIL: %s\n", what);
        }
    }

    constexpr uint32_t kOpaqueVertices      = 6;
    constexpr uint32_t kTranslucentVertices = 6;

    // Explicit draw inputs for the real demo scene.
    DepthBlendRequest request_from_scene(int width = 640, int height = 480)
    {
        DepthBlendRequest r{};
        r.width                    = width;
        r.height                   = height;
        r.vertices                 = scene_depth_blend(); // 6 opaque + 6 quad verts
        r.opaque_vertex_count      = kOpaqueVertices;
        r.translucent_vertex_count = kTranslucentVertices;
        r.translucent_drawn_last   = true;
        return r;
    }

    std::string scratch_path(const char* name)
    {
        return (std::filesystem::temp_directory_path() /
                ("t0_composition_" + std::to_string(::getpid()) + "_" + name))
            .string();
    }

    // Open file descriptors of this process: used to show that a failed output
    // edge does not leave a handle behind.
    int open_fd_count()
    {
        int n = 0;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd", ec))
        {
            (void)entry;
            ++n;
        }
        return ec ? -1 : n;
    }

    bool all_clear(const Frame& f)
    {
        for (size_t i = 0; i < f.rgba.size(); i += 4u)
        {
            if (f.rgba[i + 0] != 12u || f.rgba[i + 1] != 12u || f.rgba[i + 2] != 16u) return false;
        }
        return true;
    }

    // ----------------------------------------------------------------------
    // 1. The plan is the lesson as a value: order, ranges and per-pass policy.
    // ----------------------------------------------------------------------
    void test_plan_shape()
    {
        const auto prepared = prepare_depth_blend(request_from_scene());
        check(prepared.has_value(), "plan: the demo request prepares");
        if (!prepared) return;
        const DepthBlendPlan& plan = *prepared;

        check(plan.width == 640 && plan.height == 480, "plan: carries the requested extent");
        check(plan.vertices.size() == kOpaqueVertices + kTranslucentVertices, "plan: owns every vertex");
        check(plan.total_vertices() == kOpaqueVertices + kTranslucentVertices,
              "plan: pass ranges cover the scene");
        check(plan.passes.size() == 2, "plan: exactly two passes");
        if (plan.passes.size() != 2) return;

        const DepthBlendPass& first  = plan.passes[0];
        const DepthBlendPass& second = plan.passes[1];
        check(first.layer == DepthBlendLayer::Opaque && second.layer == DepthBlendLayer::Translucent,
              "plan: opaque first, translucent last");
        check(first.first_vertex == 0 && first.vertex_count == kOpaqueVertices,
              "plan: opaque range is the first span");
        check(second.first_vertex == kOpaqueVertices && second.vertex_count == kTranslucentVertices,
              "plan: translucent range follows the opaque one");
        check(plan.find(DepthBlendLayer::Opaque) == &first && plan.find(DepthBlendLayer::Translucent) == &second,
              "plan: passes are findable by layer");

        // opaque: the documented PassPolicy defaults (depth test + write, no blend)
        check(first.policy.depth_test && first.policy.depth_write && !first.policy.blend,
              "plan: opaque pass uses the documented defaults");
        // translucent: depth tested, not written, blended
        check(second.policy.depth_test && !second.policy.depth_write && second.policy.blend,
              "plan: translucent pass tests depth, writes none, blends");
        check(first.policy.stencil == StencilMode::Disabled &&
                  second.policy.stencil == StencilMode::Disabled,
              "plan: neither pass touches the stencil plane");
    }

    // ----------------------------------------------------------------------
    // 2. Preparation rejections, one at a time, each with its own name.
    // ----------------------------------------------------------------------
    void test_preparation_failures()
    {
        DepthBlendRequest no_width = request_from_scene();
        no_width.width = 0;
        DepthBlendRequest no_height = request_from_scene();
        no_height.height = -1;
        DepthBlendRequest empty = request_from_scene();
        empty.vertices.clear();
        DepthBlendRequest opaque_partial = request_from_scene();
        opaque_partial.opaque_vertex_count = 4;
        DepthBlendRequest opaque_zero = request_from_scene();
        opaque_zero.opaque_vertex_count      = 0;
        opaque_zero.translucent_vertex_count = 12;
        DepthBlendRequest translucent_partial = request_from_scene();
        translucent_partial.translucent_vertex_count = 4;
        DepthBlendRequest mismatch = request_from_scene();
        mismatch.translucent_vertex_count = 3;
        DepthBlendRequest reversed = request_from_scene();
        reversed.translucent_drawn_last = false;

        struct Case
        {
            const char*       what;
            DepthBlendRequest request;
            DepthBlendError   want;
        };
        std::vector<Case> cases{};
        cases.push_back({ "plan rejection: zero width", std::move(no_width),
                          DepthBlendError::InvalidFrameExtent });
        cases.push_back({ "plan rejection: negative height", std::move(no_height),
                          DepthBlendError::InvalidFrameExtent });
        cases.push_back({ "plan rejection: empty scene", std::move(empty), DepthBlendError::EmptyScene });
        cases.push_back({ "plan rejection: opaque not a triangle list", std::move(opaque_partial),
                          DepthBlendError::OpaqueNotTriangleList });
        cases.push_back({ "plan rejection: no opaque draw", std::move(opaque_zero),
                          DepthBlendError::OpaqueNotTriangleList });
        cases.push_back({ "plan rejection: translucent not a triangle list", std::move(translucent_partial),
                          DepthBlendError::TranslucentNotTriangleList });
        cases.push_back({ "plan rejection: counts do not sum to the scene", std::move(mismatch),
                          DepthBlendError::DrawCountsMismatch });
        cases.push_back({ "plan rejection: translucent before opaque", std::move(reversed),
                          DepthBlendError::TranslucentBeforeOpaque });

        for (Case& c : cases)
        {
            const auto prepared = prepare_depth_blend(std::move(c.request));
            check(!prepared.has_value() && prepared.error() == c.want, c.what);
            if (prepared) std::fprintf(stderr, "  (unexpectedly prepared: %s)\n", c.what);
        }

        // the untouched request still prepares, so each case isolates one rule
        check(prepare_depth_blend(request_from_scene()).has_value(),
              "plan: the untouched request still prepares");
    }

    // ----------------------------------------------------------------------
    // 3. Short-circuiting and error preservation along the chain.
    // ----------------------------------------------------------------------
    void test_short_circuit()
    {
        int  executed = 0;
        int  output   = 0;
        auto chain    = prepare_depth_blend(DepthBlendRequest{}) // invalid extent
                         .and_then([&](DepthBlendPlan plan) {
                             ++executed;
                             return execute_software(plan);
                         })
                         .and_then([&](Frame frame) {
                             ++output;
                             return write_png(frame, scratch_path("never.png"));
                         });

        check(!chain.has_value(), "chain: a preparation failure fails the chain");
        check(chain.error() == DepthBlendError::InvalidFrameExtent, "chain: the original error value survives");
        check(executed == 0 && output == 0, "chain: no later stage ran after the failure");

        // the diagnostic mapping observes the failure without rewriting it
        int  mapped    = 0;
        auto diagnosed = chain.or_else([&](DepthBlendError e) -> std::expected<std::string, DepthBlendError> {
            ++mapped;
            return std::unexpected(e);
        });
        check(mapped == 1, "chain: or_else() observes the failure exactly once");
        check(diagnosed.error() == DepthBlendError::InvalidFrameExtent,
              "chain: or_else() keeps the original error");
    }

    // ----------------------------------------------------------------------
    // 4. Execution refuses plans that were never validated: a plan is public
    //    aggregate data, so the executor cannot assume preparation ran.
    // ----------------------------------------------------------------------
    void test_execution_refuses_bad_plans()
    {
        const auto prepared = prepare_depth_blend(request_from_scene());
        check(prepared.has_value(), "execution: the demo plan prepares");
        if (!prepared) return;

        // a range pointing past the owned vertices
        DepthBlendPlan out_of_range = *prepared;
        out_of_range.passes[0].vertex_count = uint32_t(out_of_range.vertices.size()) + 3u;
        const auto rejected = execute_software(out_of_range);
        check(!rejected.has_value() && rejected.error() == DepthBlendError::DrawRangeOutOfBounds,
              "execution: out-of-range draw range is rejected");

        // a range that is not a whole number of triangles
        DepthBlendPlan partial = *prepared;
        partial.passes[0].vertex_count = 4;
        const auto rejected_partial = execute_software(partial);
        check(!rejected_partial.has_value() && rejected_partial.error() == DepthBlendError::DrawNotTriangleList,
              "execution: partial triangle is rejected, not silently dropped");

        // no passes at all renders an untouched frame rather than reading anything
        DepthBlendPlan no_passes{};
        no_passes.width  = 8;
        no_passes.height = 8;
        const auto empty_frame = execute_software(no_passes);
        check(empty_frame.has_value() && all_clear(*empty_frame),
              "execution: a pass-less plan renders an untouched frame");

        // a non-positive extent in a hand-built plan is still a real rejection
        DepthBlendPlan bad_extent{};
        bad_extent.width  = 0;
        bad_extent.height = 8;
        const auto no_target = execute_software(bad_extent);
        check(!no_target.has_value() && no_target.error() == DepthBlendError::InvalidFrameExtent,
              "execution: non-positive extent is rejected");
    }

    // ----------------------------------------------------------------------
    // 5. The output edge: success writes the file; failure reports its own error
    //    and leaves no descriptor behind.
    // ----------------------------------------------------------------------
    void test_output_edge()
    {
        const auto rendered = prepare_depth_blend(request_from_scene(64, 64)).and_then(execute_software);
        check(rendered.has_value(), "output: the software stage renders the plan");
        if (!rendered) return;
        check(!all_clear(*rendered), "output: the rendered frame is not just the clear color");

        const std::string good = scratch_path("written.png");
        const auto        path = write_png(*rendered, good);
        check(path.has_value() && *path == good, "output: write_png returns the written path");
        std::error_code ec;
        const auto      size = std::filesystem::file_size(good, ec);
        check(!ec && size > 0, "output: the PNG exists and is not empty");
        std::filesystem::remove(good, ec);

        // an unwritable destination: a directory that does not exist
        const int  before = open_fd_count();
        const auto failed = write_png(*rendered, scratch_path("missing_dir/never.png"));
        const int  after  = open_fd_count();
        check(!failed.has_value() && failed.error() == DepthBlendError::OutputUnwritable,
              "output: an unwritable path is reported as OutputUnwritable");
        check(before > 0 && before == after, "output: a failed write leaves no file descriptor behind");
    }

    // ----------------------------------------------------------------------
    // 6. The plan owns its payload: it stays usable after the request and the
    //    scene vector that produced it are gone (no borrowed plan payload).
    // ----------------------------------------------------------------------
    void test_plan_owns_its_vertices()
    {
        std::optional<DepthBlendPlan> kept{};
        {
            const auto prepared = prepare_depth_blend(request_from_scene(32, 32));
            check(prepared.has_value(), "ownership: the request prepares");
            if (!prepared) return;
            kept = *prepared; // the request (and its vertex vector) dies here
        }

        const auto frame = execute_software(*kept);
        check(frame.has_value(), "ownership: the plan still executes after its request is gone");
        if (!frame) return;
        check(frame->width == 32 && frame->height == 32, "ownership: the plan kept its extent");
        check(!all_clear(*frame), "ownership: the plan kept its geometry");
    }

    // ----------------------------------------------------------------------
    // 8. Policy propagation: the executor applies each pass's prepared policy.
    //    Two variants of the same plan differ ONLY in one policy bit, so any
    //    difference in the result is that bit reaching the rasterizer — and the
    //    depth variant proves depth_write=false really kept the opaque depth.
    // ----------------------------------------------------------------------
    void test_policy_propagation()
    {
        const auto prepared = prepare_depth_blend(request_from_scene());
        check(prepared.has_value(), "propagation: the demo plan prepares");
        if (!prepared) return;

        const auto target = execute_software_target(*prepared);
        check(target.has_value(), "propagation: the plan executes with its storage");
        if (!target) return;

        // blended vs unblended translucent pass -> the image must change
        DepthBlendPlan unblended_plan = *prepared;
        unblended_plan.passes[1].policy.blend = false;
        const auto unblended = execute_software_target(unblended_plan);
        check(unblended.has_value(), "propagation: the unblended variant executes");
        if (!unblended) return;

        size_t colour_differences = 0;
        for (size_t i = 0; i + 3 < target->frame.rgba.size(); i += 4u)
        {
            if (target->frame.rgba[i + 0] != unblended->frame.rgba[i + 0] ||
                target->frame.rgba[i + 1] != unblended->frame.rgba[i + 1] ||
                target->frame.rgba[i + 2] != unblended->frame.rgba[i + 2])
            {
                ++colour_differences;
            }
        }
        check(colour_differences > 0, "propagation: the blended pass's policy reaches the rasterizer");

        // depth_write=false vs true -> wherever the blended quad is nearer, the
        // prepared plan must still hold the OPAQUE depth
        DepthBlendPlan depth_writing_plan = *prepared;
        depth_writing_plan.passes[1].policy.depth_write = true;
        const auto depth_writing = execute_software_target(depth_writing_plan);
        check(depth_writing.has_value(), "propagation: the depth-writing variant executes");
        if (!depth_writing) return;

        size_t depth_differences = 0;
        size_t shallower        = 0;
        for (size_t i = 0; i < target->depth.size(); ++i)
        {
            if (depth_writing->depth[i] != target->depth[i])
            {
                ++depth_differences;
                if (depth_writing->depth[i] < target->depth[i]) ++shallower;
            }
        }
        check(depth_differences > 0 && depth_differences == shallower,
              "propagation: depth_write=false kept the opaque depth where the blended quad is nearer");

        // neither variant touched the stencil plane (both passes are Disabled)
        bool stencil_untouched = true;
        for (uint8_t s : target->stencil)
        {
            if (s != 0) stencil_untouched = false;
        }
        check(stencil_untouched, "propagation: the prepared plan leaves the stencil plane alone");
    }

    // ----------------------------------------------------------------------
    // 7. The vocabulary is total and consistent: every error has a stage, a
    //    message, and the exit code that stage implies.
    // ----------------------------------------------------------------------
    void test_vocabulary()
    {
        struct Row
        {
            DepthBlendError code;
            DepthBlendStage stage;
            int             exit_code;
        };
        const Row rows[] = {
            { DepthBlendError::InvalidFrameExtent, DepthBlendStage::Preparation, 1 },
            { DepthBlendError::EmptyScene, DepthBlendStage::Preparation, 1 },
            { DepthBlendError::OpaqueNotTriangleList, DepthBlendStage::Preparation, 1 },
            { DepthBlendError::TranslucentNotTriangleList, DepthBlendStage::Preparation, 1 },
            { DepthBlendError::DrawCountsMismatch, DepthBlendStage::Preparation, 1 },
            { DepthBlendError::TranslucentBeforeOpaque, DepthBlendStage::Preparation, 1 },
            { DepthBlendError::DrawRangeOutOfBounds, DepthBlendStage::Execution, 2 },
            { DepthBlendError::DrawNotTriangleList, DepthBlendStage::Execution, 2 },
            { DepthBlendError::DeviceUnavailable, DepthBlendStage::Execution, 2 },
            { DepthBlendError::ShaderModuleUnreadable, DepthBlendStage::Execution, 2 },
            { DepthBlendError::PipelineUnavailable, DepthBlendStage::Execution, 2 },
            { DepthBlendError::VertexUploadFailed, DepthBlendStage::Execution, 2 },
            { DepthBlendError::FrameSubmitFailed, DepthBlendStage::Execution, 2 },
            { DepthBlendError::OutputUnwritable, DepthBlendStage::Output, 3 },
        };

        for (const Row& r : rows)
        {
            check(depth_blend_stage(r.code) == r.stage, "vocabulary: stage mapping is as documented");
            check(depth_blend_exit_code(r.code) == r.exit_code, "vocabulary: exit code follows the stage");
            const char* message = depth_blend_message(r.code);
            check(message != nullptr && message[0] != '\0', "vocabulary: every error has a message");
        }

        check(std::string(depth_blend_stage_name(DepthBlendStage::Preparation)) == "preparation" &&
                  std::string(depth_blend_stage_name(DepthBlendStage::Execution)) == "execution" &&
                  std::string(depth_blend_stage_name(DepthBlendStage::Output)) == "output",
              "vocabulary: every stage has a name");
    }
}

int main()
{
    test_plan_shape();
    test_preparation_failures();
    test_short_circuit();
    test_execution_refuses_bad_plans();
    test_output_edge();
    test_plan_owns_its_vertices();
    test_policy_propagation();
    test_vocabulary();

    if (g_failures != 0)
    {
        std::fprintf(stderr, "t0_composition_tests: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    std::printf("t0_composition_tests: %d checks passed (AD3 typed composition pilot)\n", g_checks);
    return 0;
}
