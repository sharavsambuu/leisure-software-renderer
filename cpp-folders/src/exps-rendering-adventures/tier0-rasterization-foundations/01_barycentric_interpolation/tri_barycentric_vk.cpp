// tier0 demo 01 — Vulkan/Slang twin: triangle with barycentric color
// interpolation. Coverage, weights and varying interpolation run on the GPU
// rasterizer; the shader logic lives in tri_barycentric.slang (compiled to
// SPIR-V at build time by slangc). Compare the output PNG with the _sw twin.
//
// Run: t0_tri_barycentric_vk [out.png]

#include <cstdio>
#include <string>

#include "../common/adventures_vk.hpp"
#include "../common/t0_scenes.hpp"

using namespace adventures;

#ifndef SHS_ADVENTURES_SHADER_DIR
#define SHS_ADVENTURES_SHADER_DIR "."
#endif

int main(int argc, char* argv[])
{
    const std::string out_path = argc > 1 ? argv[1] : "t0_01_barycentric_vk.png";
    const std::string spv_dir  = SHS_ADVENTURES_SHADER_DIR;

    OffscreenVulkan vk;
    if (!vk.init(640, 480))
    {
        std::fprintf(stderr, "vulkan init failed (no device?)\n");
        return 2;
    }

    const std::string vs_path = spv_dir + "/tri_barycentric_vs.spv";
    const std::string fs_path = spv_dir + "/tri_barycentric_fs.spv";
    VkPipelineSetup setup{};
    setup.vs_spv_path = vs_path.c_str();
    setup.fs_spv_path = fs_path.c_str();
    setup.depth_test   = false; // single triangle in a constant-z plane
    const int pipeline = vk.add_pipeline(setup);
    if (pipeline < 0) return 2;

    const std::vector<T0Vertex> tri = scene_tri_barycentric();
    if (!vk.upload_vertices(tri.data(), tri.size() * sizeof(T0Vertex))) return 2;

    // identity: scene vertices are already in clip space (constant w)
    const T0Push push = make_push(glm::mat4(1.0f));
    const VkDraw draw{ uint32_t(pipeline), 0, uint32_t(tri.size()), &push, sizeof(push), -1 };
    if (!vk.render({ draw })) return 2;
    if (!vk.save_png(out_path.c_str()))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%ux%u) — GPU barycentric interpolation (Slang)\n",
                out_path.c_str(), vk.width(), vk.height());
    return 0;
}
