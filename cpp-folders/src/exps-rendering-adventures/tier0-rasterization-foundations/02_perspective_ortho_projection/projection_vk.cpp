// tier0 demo 02 — Vulkan/Slang twin: perspective vs orthographic projection.
// One pipeline, two draws; each draw pushes its own MVP (perspective for the
// left half, ortho for the right). The matrices match projection_sw.cpp
// exactly — GLM computes them on the CPU in both realizations.
//
// Run: t0_projection_vk [out.png]

#include <cstdio>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../common/adventures_vk.hpp"
#include "../common/adventures_frame.hpp"
#include "../common/adventures_window.hpp"
#include "../common/t0_scenes.hpp"

using namespace adventures;

#ifndef SHS_ADVENTURES_SHADER_DIR
#define SHS_ADVENTURES_SHADER_DIR "."
#endif

int main(int argc, char* argv[])
{
    const DemoArgs args = parse_demo_args(argc, argv, "t0_02_projection_vk.png");
    const std::string out_path = args.out_path;
    const std::string spv_dir  = SHS_ADVENTURES_SHADER_DIR;

    OffscreenVulkan vk;
    if (!vk.init(640, 480))
    {
        std::fprintf(stderr, "vulkan init failed (no device?)\n");
        return 2;
    }

    const std::string vs_path = spv_dir + "/projection_vs.spv";
    const std::string fs_path = spv_dir + "/projection_fs.spv";
    VkPipelineSetup setup{};
    setup.vs_spv_path  = vs_path.c_str();
    setup.fs_spv_path  = fs_path.c_str();
    setup.depth_test   = true;
    const int pipeline = vk.add_pipeline(setup);
    if (pipeline < 0) return 2;

    const std::vector<T0Vertex> cube = scene_projection_cube();
    if (!vk.upload_vertices(cube.data(), cube.size() * sizeof(T0Vertex))) return 2;

    // same transforms as the _sw twin
    const glm::mat4 model = glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(35.0f), glm::vec3(0, 1, 0)),
                                        glm::radians(20.0f), glm::vec3(1, 0, 0));
    const glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -3.0f));
    glm::mat4       proj = glm::perspective(glm::radians(60.0f), 0.5f, 0.1f, 10.0f);
    // half-viewport maps: x' = 0.5*x -/+ 0.5 (translate applied AFTER scale)
    const glm::mat4 half_left = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)),
                                               glm::vec3(-1.0f, 0.0f, 0.0f));
    const glm::mat4 half_right = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)),
                                                glm::vec3(1.0f, 0.0f, 0.0f));

    const T0Push push_left = make_push_row_major(half_left * (proj * view * model));
    proj = glm::ortho(-1.6f, 1.6f, -1.0f, 1.0f, 0.1f, 10.0f);
    const T0Push push_right = make_push_row_major(half_right * (proj * view * model));

    const VkDraw draws[2] = {
        { uint32_t(pipeline), 0, uint32_t(cube.size()), &push_left,  sizeof(push_left),  -1 },
        { uint32_t(pipeline), 0, uint32_t(cube.size()), &push_right, sizeof(push_right), -1 },
    };
    if (!vk.render({ draws[0], draws[1] })) return 2;
    if (!vk.save_png(out_path.c_str()))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%ux%u) — left: perspective, right: ortho (Slang vertex shader)\n",
                out_path.c_str(), vk.width(), vk.height());
    if (args.windowed && vk.color_readback_data() != nullptr)
    {
        Frame frame(int(vk.width()), int(vk.height()));
        std::memcpy(frame.rgba.data(), vk.color_readback_data(), frame.rgba.size());
        present_frame_windowed(frame, "t0 02 — perspective vs ortho (Vulkan/Slang)", out_path, args.backend);
    }
    return 0;
}
