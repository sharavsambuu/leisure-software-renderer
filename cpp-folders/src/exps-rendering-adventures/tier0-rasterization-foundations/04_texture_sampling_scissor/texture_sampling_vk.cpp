// tier0 demo 04 — Vulkan/Slang twin: texture sampling + scissor.
// One 16x16 RGBA8 checkerboard uploaded as a sampled image; two combined
// image samplers on the same image (NEAREST + LINEAR, both REPEAT wrap).
// Left quad draw binds the NEAREST set, right quad the LINEAR set. A
// scissor band (rows 300+) matches the _sw twin.
// The Slang fragment shader does tex0.SampleLevel(uv, 0).
//
// Run: t0_texture_sampling_vk [out.png]

#include <cstdio>
#include <string>

#include <glm/glm.hpp>

#include "../common/adventures_vk.hpp"
#include "../common/adventures_frame.hpp"
#include "../common/adventures_window.hpp"
#include "../common/t0_scenes.hpp"

using namespace adventures;

#ifndef SHS_ADVENTURES_SHADER_DIR
#define SHS_ADVENTURES_SHADER_DIR "."
#endif

namespace
{
    // same procedural texture as the _sw twin
    std::vector<uint8_t> make_checker_rgba(int size = 16)
    {
        std::vector<uint8_t> rgba(size_t(size) * size * 4, 255);
        for (int y = 0; y < size; ++y)
        {
            for (int x = 0; x < size; ++x)
            {
                const bool even = ((x / (size / 2)) + (y / (size / 2))) % 2 == 0;
                uint8_t    r    = even ? 230 : 40;
                uint8_t    g    = even ? 230 : 40;
                uint8_t    b    = even ? 230 : 220;
                if (x == size / 2 || y == size / 2) { r = 255; g = 60; b = 60; }
                const size_t i = (size_t(y) * size + x) * 4;
                rgba[i]     = r;
                rgba[i + 1] = g;
                rgba[i + 2] = b;
            }
        }
        return rgba;
    }
}

int main(int argc, char* argv[])
{
    const DemoArgs args = parse_demo_args(argc, argv, "t0_04_texture_sampling_vk.png");
    const std::string out_path = args.out_path;
    const std::string spv_dir  = SHS_ADVENTURES_SHADER_DIR;

    OffscreenVulkan vk;
    if (!vk.init(640, 480))
    {
        std::fprintf(stderr, "vulkan init failed (no device?)\n");
        return 2;
    }

    const std::string vs_path = spv_dir + "/texture_sampling_vs.spv";
    const std::string fs_path = spv_dir + "/texture_sampling_fs.spv";
    VkPipelineSetup setup{};
    setup.vs_spv_path  = vs_path.c_str();
    setup.fs_spv_path  = fs_path.c_str();
    setup.depth_test   = false;
    setup.textured     = true;
    const int pipeline = vk.add_pipeline(setup);
    if (pipeline < 0) return 2;

    const std::vector<uint8_t> tex = make_checker_rgba();
    const int set_nearest = vk.upload_texture_rgba(tex.data(), 16, 16, /*bilinear*/ false);
    const int set_linear  = vk.upload_texture_rgba(tex.data(), 16, 16, /*bilinear*/ true);
    if (set_nearest < 0 || set_linear < 0) return 2;

    const std::vector<T0Vertex> quads = scene_texture_quads(); // 6 left + 6 right
    if (!vk.upload_vertices(quads.data(), quads.size() * sizeof(T0Vertex))) return 2;

    const T0Push push = make_push(glm::mat4(1.0f));
    const VkDraw draws[2] = {
        { uint32_t(pipeline), 0, 6, &push, sizeof(push), set_nearest },
        { uint32_t(pipeline), 6, 6, &push, sizeof(push), set_linear  },
    };
    const VkRect2D scissor{ { 0, 300 }, { 640, 180 } }; // band rows 300..479, like _sw
    if (!vk.render({ draws[0], draws[1] }, &scissor)) return 2;
    if (!vk.save_png(out_path.c_str()))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%ux%u) — left: NEAREST, right: LINEAR, scissor rows 300+ (Slang)\n",
                out_path.c_str(), vk.width(), vk.height());
    if (args.windowed && vk.color_readback_data() != nullptr)
    {
        Frame frame(int(vk.width()), int(vk.height()));
        std::memcpy(frame.rgba.data(), vk.color_readback_data(), frame.rgba.size());
        present_frame_windowed(frame, "t0 04 — texture sampling + scissor (Vulkan/Slang)", out_path, args.backend);
    }
    return 0;
}
