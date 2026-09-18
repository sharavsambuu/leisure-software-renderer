// tier0 demo 03 — depth test + alpha blending + draw-order sensitivity.
//
// AD3 pilot. The lesson is no longer a sequence of statements in main():
//   * preparation is PURE (depth_blend_plan.hpp): the explicit draw inputs go
//     in, a backend-neutral plan comes out, or a named error does;
//   * execution is an EDGE (depth_blend_edges.hpp here, the Vulkan executor in
//     the twin) that consumes that plan and returns std::expected;
//   * output is another EDGE (PNG + the optional window front-end).
// The three compose into one typed chain, and failures are reported once, at
// the host boundary, with a stage-specific diagnostic and exit code.
//
// Lesson content is unchanged: the two opaque triangles overlap and the
// z-buffer resolves them by depth (near red beats far blue); the translucent
// green quad is drawn LAST, with blending and depth writes disabled, because
// transparent geometry must follow opaque geometry regardless of z.
//
// Run: t0_depth_blend_sw [out.png]

#include <cstdio>
#include <expected>
#include <string>
#include <utility>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_window.hpp"
#include "../common/t0_scenes.hpp"
#include "depth_blend_edges.hpp"
#include "depth_blend_plan.hpp"

using namespace adventures;
using namespace adventures::t0_03;

namespace
{
    // Host-edge value: the rendered frame plus the path it was written to, so the
    // optional window front-end can present the same image the chain wrote.
    struct DemoOutput
    {
        Frame       frame;
        std::string path;
    };

    // The single diagnostic mapping, at the host boundary. or_else() observes the
    // failure without rewriting it: the original error value survives, so the
    // exit code stays accurate to the stage that produced it.
    template <typename T>
    int report_and_code(const std::expected<T, DepthBlendError>& failed)
    {
        const auto diagnosed = failed.or_else([](DepthBlendError e) -> std::expected<T, DepthBlendError> {
            std::fprintf(stderr, "t0 03 (software): %s stage failed: %s\n",
                         depth_blend_stage_name(depth_blend_stage(e)), depth_blend_message(e));
            return std::unexpected(e);
        });
        (void)diagnosed;
        return depth_blend_exit_code(failed.error());
    }
}

int main(int argc, char* argv[])
{
    const DemoArgs args = parse_demo_args(argc, argv, "t0_03_depth_blend_sw.png");

    // Explicit draw inputs. The geometry still comes from the tier0 shared scene
    // (single source of vertices); the counts and the ordering requirement are
    // stated here rather than inferred.
    DepthBlendRequest request{};
    request.width                    = 640;
    request.height                   = 480;
    request.vertices                 = scene_depth_blend(); // 6 opaque + 6 quad verts
    request.opaque_vertex_count      = 6;
    request.translucent_vertex_count = 6;
    request.translucent_drawn_last   = true;

    // One typed chain: pure preparation -> software execution -> PNG output.
    // A failure at any stage means the later stages never run.
    const std::expected<DemoOutput, DepthBlendError> run =
        prepare_depth_blend(std::move(request))
            .and_then(execute_software)
            .and_then([&args](Frame frame) -> std::expected<DemoOutput, DepthBlendError> {
                auto path = write_png(frame, args.out_path);
                if (!path) return std::unexpected(path.error());
                return DemoOutput{ std::move(frame), std::move(*path) };
            });

    if (!run) return report_and_code(run);

    std::printf("wrote %s (%dx%d) — red triangle (near) wins overlap; quad blended last\n",
                run->path.c_str(), run->frame.width, run->frame.height);
    if (args.windowed)
    {
        present_frame_windowed(run->frame, "t0 03 — depth test + alpha blend (software)", run->path,
                               args.backend);
    }
    return 0;
}
