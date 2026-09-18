// tier1 demo 08 — Vulkan/Slang twin: normal mapping.
// Procedural bump RGBA8 uploaded once; two draws over the same quad pair
// with push params.x selecting flat (0) vs mapped (1). Light, albedo and
// ambient ride as literals mirrored from the _sw twin; params.yzw carries
// the pre-normalized light dir so both sides shade identical floats.
//
// Run: t1_normal_mapping_vk [out.png]

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "../common/adventures_vk.hpp"
#include "../common/adventures_frame.hpp"
#include "../common/adventures_window.hpp"
#include "../common/t1_scenes.hpp"

using namespace adventures;

#ifndef SHS_ADVENTURES_SHADER_DIR
#define SHS_ADVENTURES_SHADER_DIR "."
#endif

namespace
{
    constexpr float k_bump_strength = 0.08f;

    // Bit-identical texels to the _sw twin (same formula, same order).
    std::vector<uint8_t> make_bump_rgba(int size = 16)
    {
        std::vector<uint8_t> rgba(size_t(size) * size * 4, 255);
        constexpr float k_two_pi = 6.283185307179586f;
        for (int y = 0; y < size; ++y)
        {
            for (int x = 0; x < size; ++x)
            {
                const float u = (float(x) + 0.5f) / float(size);
                const float v = (float(y) + 0.5f) / float(size);
                const float dhdu = 4.0f * k_two_pi * 0.5f * std::cos(2.0f * k_two_pi * u) * std::cos(2.0f * k_two_pi * v);
                const float dhdv = -4.0f * k_two_pi * 0.5f * std::sin(2.0f * k_two_pi * u) * std::sin(2.0f * k_two_pi * v);
                const glm::vec3 n = glm::normalize(glm::vec3(-k_bump_strength * dhdu, -k_bump_strength * dhdv, 1.0f));
                const size_t i = (size_t(y) * size + x) * 4;
                rgba[i]     = uint8_t(n.x * 127.5f + 127.5f);
                rgba[i + 1] = uint8_t(n.y * 127.5f + 127.5f);
                rgba[i + 2] = uint8_t(n.z * 127.5f + 127.5f);
            }
        }
        return rgba;
    }
} // namespace

int main(int argc, char* argv[])
{
    const DemoArgs args = parse_demo_args(argc, argv, "t1_08_normal_mapping_vk.png");
    const std::string out_path = args.out_path;
    const std::string spv_dir  = SHS_ADVENTURES_SHADER_DIR;

    OffscreenVulkan vk;
    if (!vk.init(640, 480))
    {
        std::fprintf(stderr, "vulkan init failed (no device?)\n");
        return 2;
    }

    const std::string vs_path = spv_dir + "/normal_mapping_vs.spv";
    const std::string fs_path = spv_dir + "/normal_mapping_fs.spv";
    VkPipelineSetup setup{};
    setup.vs_spv_path = vs_path.c_str();
    setup.fs_spv_path = fs_path.c_str();
    setup.depth_test  = false;
    setup.textured    = true;
    const int pipeline = vk.add_pipeline(setup);
    if (pipeline < 0) return 2;

    const std::vector<uint8_t> bump = make_bump_rgba();
    const int set_linear = vk.upload_texture_rgba(bump.data(), 16, 16, /*bilinear*/ true);
    if (set_linear < 0) return 2;

    // T0Vertex reuse: normal rides COLOR0.xyz (see t1_scenes.hpp).
    const std::vector<T0Vertex> quads = scene_nmap_quads_t0();
    if (!vk.upload_vertices(quads.data(), quads.size() * sizeof(T0Vertex))) return 2;

    const glm::vec3 light = glm::normalize(glm::vec3(-0.35f, 0.5f, 0.8f));
    const T0Push push_flat   = make_push(glm::mat4(1.0f), glm::vec4(0.0f, light.x, light.y, light.z));
    const T0Push push_mapped = make_push(glm::mat4(1.0f), glm::vec4(1.0f, light.x, light.y, light.z));
    const VkDraw draws[2] = {
        { uint32_t(pipeline), 0, 6, &push_flat,   sizeof(push_flat),   set_linear },
        { uint32_t(pipeline), 6, 6, &push_mapped, sizeof(push_mapped), set_linear },
    };
    if (!vk.render({ draws[0], draws[1] })) return 2;
    if (!vk.save_png(out_path.c_str()))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%ux%u) — left: flat, right: normal-mapped (Slang)\n",
                out_path.c_str(), vk.width(), vk.height());
    if (args.windowed && vk.color_readback_data() != nullptr)
    {
        Frame frame(int(vk.width()), int(vk.height()));
        std::memcpy(frame.rgba.data(), vk.color_readback_data(), frame.rgba.size());
        present_frame_windowed(frame, "t1 08 — normal mapping (Vulkan/Slang)", out_path, args.backend);
    }
    return 0;
}
