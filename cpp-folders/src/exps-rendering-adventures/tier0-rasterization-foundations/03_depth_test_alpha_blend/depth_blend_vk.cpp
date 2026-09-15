// tier0 demo 03 — Vulkan/Slang twin: depth test + alpha blending.
// Pipeline A: depth test + write, no blend (the two opaque triangles).
// Pipeline B: depth test on, depth WRITE off, alpha blend on — the
// translucent quad, drawn last (transparent-after-opaque draw ordering).
// States mirror depth_blend_sw.cpp exactly.
//
// Run: t0_depth_blend_vk [out.png]

#include <cstdio>
#include <string>

#include <glm/glm.hpp>

#include "../common/adventures_vk.hpp"
#include "../common/t0_scenes.hpp"

using namespace adventures;

#ifndef SHS_ADVENTURES_SHADER_DIR
#define SHS_ADVENTURES_SHADER_DIR "."
#endif

int main(int argc, char* argv[])
{
    const std::string out_path = argc > 1 ? argv[1] : "t0_03_depth_blend_vk.png";
    const std::string spv_dir  = SHS_ADVENTURES_SHADER_DIR;

    OffscreenVulkan vk;
    if (!vk.init(640, 480))
    {
        std::fprintf(stderr, "vulkan init failed (no device?)\n");
        return 2;
    }

    const std::string vs_path = spv_dir + "/depth_blend_vs.spv";
    const std::string fs_path = spv_dir + "/depth_blend_fs.spv";
    VkPipelineSetup opaque_setup{};
    opaque_setup.vs_spv_path  = vs_path.c_str();
    opaque_setup.fs_spv_path  = fs_path.c_str();
    opaque_setup.depth_test   = true;
    opaque_setup.depth_write  = true;
    opaque_setup.blend        = false;
    const int opaque_pipe = vk.add_pipeline(opaque_setup);

    VkPipelineSetup blend_setup = opaque_setup;
    blend_setup.depth_write = false; // blended fragments must not poison the z-buffer
    blend_setup.blend       = true;  // SRC_ALPHA / ONE_MINUS_SRC_ALPHA
    const int blend_pipe = vk.add_pipeline(blend_setup);
    if (opaque_pipe < 0 || blend_pipe < 0) return 2;

    const std::vector<T0Vertex> scene = scene_depth_blend(); // 6 opaque + 6 quad verts
    if (!vk.upload_vertices(scene.data(), scene.size() * sizeof(T0Vertex))) return 2;

    const T0Push push = make_push(glm::mat4(1.0f)); // scene is clip-space already
    const VkDraw draws[2] = {
        { uint32_t(opaque_pipe), 0, 6, &push, sizeof(push), -1 },
        { uint32_t(blend_pipe),  6, 6, &push, sizeof(push), -1 },
    };
    if (!vk.render({ draws[0], draws[1] })) return 2;
    if (!vk.save_png(out_path.c_str()))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%ux%u) — z-buffered tris + blended quad (Slang)\n",
                out_path.c_str(), vk.width(), vk.height());
    return 0;
}
