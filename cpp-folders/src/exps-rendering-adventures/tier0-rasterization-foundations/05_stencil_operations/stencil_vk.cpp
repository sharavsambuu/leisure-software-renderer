// tier0 demo 05 — Vulkan/Slang twin: stencil buffer operations.
// Pipeline A: stencil WRITE (ALWAYS + REPLACE ref 1) for the triangle.
// Pipeline B: stencil TEST EQUAL, quad squeezed into the left half.
// Pipeline C: stencil TEST NOT_EQUAL, quad squeezed into the right half.
// Op states mirror stencil_sw.cpp.
//
// Run: t0_stencil_vk [out.png]

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
    const DemoArgs args = parse_demo_args(argc, argv, "t0_05_stencil_vk.png");
    const std::string out_path = args.out_path;
    const std::string spv_dir  = SHS_ADVENTURES_SHADER_DIR;

    OffscreenVulkan vk;
    if (!vk.init(640, 480))
    {
        std::fprintf(stderr, "vulkan init failed (no device?)\n");
        return 2;
    }

    const std::string vs_path = spv_dir + "/stencil_vs.spv";
    const std::string fs_path = spv_dir + "/stencil_fs.spv";
    VkPipelineSetup write_setup{};
    write_setup.vs_spv_path = vs_path.c_str();
    write_setup.fs_spv_path = fs_path.c_str();
    write_setup.depth_test  = false;
    write_setup.stencil_write = true; // ALWAYS + REPLACE ref 1
    const int write_pipe = vk.add_pipeline(write_setup);

    VkPipelineSetup test_setup = write_setup;
    test_setup.stencil_write = false;
    test_setup.stencil_test  = true; // EQUAL ref 1
    const int equal_pipe = vk.add_pipeline(test_setup);

    VkPipelineSetup invert_setup = test_setup;
    invert_setup.stencil_invert = true; // NOT_EQUAL ref 1
    const int invert_pipe = vk.add_pipeline(invert_setup);
    if (write_pipe < 0 || equal_pipe < 0 || invert_pipe < 0) return 2;

    const std::vector<T0Vertex> tri = scene_stencil_triangle();
    const std::vector<T0Vertex> quad = scene_stencil_quad();
    std::vector<T0Vertex> all(tri);
    all.insert(all.end(), quad.begin(), quad.end());
    if (!vk.upload_vertices(all.data(), all.size() * sizeof(T0Vertex))) return 2;

    const T0Push push_tri = make_push(glm::mat4(1.0f));
    // quad squeezed into left half / right half via push-constant MVP
    // half-viewport maps: x' = 0.5*x -/+ 0.5 (translate applied AFTER scale)
    const glm::mat4 half_left = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)),
                                               glm::vec3(-1.0f, 0.0f, 0.0f));
    const glm::mat4 half_right = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)),
                                                glm::vec3(1.0f, 0.0f, 0.0f));
    const T0Push push_left  = make_push_row_major(half_left);
    const T0Push push_right = make_push_row_major(half_right);

    const VkDraw draws[3] = {
        { uint32_t(write_pipe),  0,                      uint32_t(tri.size()),  &push_tri,   sizeof(push_tri),   -1 },
        { uint32_t(equal_pipe),  uint32_t(tri.size()),   uint32_t(quad.size()), &push_left,  sizeof(push_left),  -1 },
        { uint32_t(invert_pipe), uint32_t(tri.size()),   uint32_t(quad.size()), &push_right, sizeof(push_right), -1 },
    };
    if (!vk.render({ draws[0], draws[1], draws[2] })) return 2;
    if (!vk.save_png(out_path.c_str()))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%ux%u) — left: quad masked by stencil, right: inverted (Slang)\n",
                out_path.c_str(), vk.width(), vk.height());
    if (args.windowed && vk.color_readback_data() != nullptr)
    {
        Frame frame(int(vk.width()), int(vk.height()));
        std::memcpy(frame.rgba.data(), vk.color_readback_data(), frame.rgba.size());
        present_frame_windowed(frame, "t0 05 — stencil operations (Vulkan/Slang)", out_path, args.backend);
    }
    return 0;
}
