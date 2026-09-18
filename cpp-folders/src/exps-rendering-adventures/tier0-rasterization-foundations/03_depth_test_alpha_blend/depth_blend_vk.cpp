// tier0 demo 03 — Vulkan/Slang twin: depth test + alpha blending.
//
// AD3 pilot, Vulkan side. The SAME pure plan as the software twin
// (depth_blend_plan.hpp) drives this executor: one graphics pipeline per
// prepared pass, built from that pass's PassPolicy, and one upload of the
// plan's owned vertices. Vulkan's own failures (device, SPIR-V, pipeline,
// upload, submit) are adapted into the shared closed vocabulary instead of
// ad-hoc ints, and the PNG output goes through the same edge function the
// software twin uses.
//
// Run: t0_depth_blend_vk [out.png]

#include <cstdio>
#include <cstring>
#include <expected>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_vk.hpp"
#include "../common/adventures_window.hpp"
#include "../common/t0_scenes.hpp"
#include "depth_blend_edges.hpp" // write_png: the shared output edge
#include "depth_blend_plan.hpp"

using namespace adventures;
using namespace adventures::t0_03;

#ifndef SHS_ADVENTURES_SHADER_DIR
#define SHS_ADVENTURES_SHADER_DIR "."
#endif

namespace
{
    // The harness collapses "SPIR-V unreadable" and "pipeline refused" into one
    // -1, so the edge probes readability first to keep the diagnostic specific.
    // This reads no content and takes no decision the plan should own.
    bool readable(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        return file.good();
    }
}

int main(int argc, char* argv[])
{
    const DemoArgs args = parse_demo_args(argc, argv, "t0_03_depth_blend_vk.png");
    const std::string spv_dir = SHS_ADVENTURES_SHADER_DIR;

    // The single diagnostic mapping of this edge: one place prints, one place
    // decides the exit code, and both read the shared vocabulary.
    const auto fail = [](DepthBlendError e) {
        std::fprintf(stderr, "t0 03 (vulkan): %s stage failed: %s\n",
                     depth_blend_stage_name(depth_blend_stage(e)), depth_blend_message(e));
        return depth_blend_exit_code(e);
    };

    // Same explicit draw inputs and the same pure preparation as the software
    // twin — this is what makes the plan backend-neutral rather than a story.
    DepthBlendRequest request{};
    request.width                    = 640;
    request.height                   = 480;
    request.vertices                 = scene_depth_blend(); // 6 opaque + 6 quad verts
    request.opaque_vertex_count      = 6;
    request.translucent_vertex_count = 6;
    request.translucent_drawn_last   = true;

    const std::expected<DepthBlendPlan, DepthBlendError> prepared = prepare_depth_blend(std::move(request));
    if (!prepared) return fail(prepared.error());
    const DepthBlendPlan& plan = *prepared;

    OffscreenVulkan vk;
    if (!vk.init(uint32_t(plan.width), uint32_t(plan.height)))
    {
        return fail(DepthBlendError::DeviceUnavailable);
    }

    const std::string vs_path = spv_dir + "/depth_blend_vs.spv";
    const std::string fs_path = spv_dir + "/depth_blend_fs.spv";
    if (!readable(vs_path) || !readable(fs_path))
    {
        return fail(DepthBlendError::ShaderModuleUnreadable);
    }

    // One pipeline per prepared pass, configured from that pass's policy: the
    // opaque pass uses the PassPolicy defaults (depth test + write, no blend),
    // the translucent pass carries depth_write=false + blend=true. The twin
    // restates neither.
    std::vector<uint32_t> pipelines{};
    for (const DepthBlendPass& pass : plan.passes)
    {
        VkPipelineSetup setup{};
        setup.vs_spv_path = vs_path.c_str();
        setup.fs_spv_path = fs_path.c_str();
        setup.policy      = pass.policy;
        const int pipe    = vk.add_pipeline(setup);
        if (pipe < 0) return fail(DepthBlendError::PipelineUnavailable);
        pipelines.push_back(uint32_t(pipe));
    }

    if (!vk.upload_vertices(plan.vertices.data(), plan.vertices.size() * sizeof(T0Vertex)))
    {
        return fail(DepthBlendError::VertexUploadFailed);
    }

    const T0Push        push = make_push(glm::mat4(1.0f)); // scene is clip-space already
    std::vector<VkDraw> draws{};
    for (size_t i = 0; i < plan.passes.size(); ++i)
    {
        const DepthBlendPass& pass = plan.passes[i];
        draws.push_back(VkDraw{ pipelines[i], pass.first_vertex, pass.vertex_count, &push, sizeof(push), -1 });
    }
    if (!vk.render(draws)) return fail(DepthBlendError::FrameSubmitFailed);

    const void* readback = vk.color_readback_data();
    if (readback == nullptr) return fail(DepthBlendError::FrameSubmitFailed);
    Frame frame(int(vk.width()), int(vk.height()));
    std::memcpy(frame.rgba.data(), readback, frame.rgba.size());

    auto written = write_png(frame, args.out_path);
    if (!written) return fail(written.error());

    std::printf("wrote %s (%ux%u) — z-buffered tris + blended quad (Slang)\n",
                written->c_str(), vk.width(), vk.height());
    if (args.windowed)
    {
        present_frame_windowed(frame, "t0 03 — depth test + alpha blend (Vulkan/Slang)", *written,
                               args.backend);
    }
    return 0;
}
