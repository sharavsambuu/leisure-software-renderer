// tier0 demo 02 — perspective vs orthographic projection.
// Same cube, same model transform, two projections: glm::perspective (left
// half — farther edges shrink) vs glm::ortho (right half — parallel edges
// stay parallel). The *_vk twin pushes the identical MVPs to the Slang
// vertex shader.
//
// Run: t0_projection_sw [out.png]

#include <cstdio>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_window.hpp"
#include "../common/adventures_sw_raster.hpp"
#include "../common/t0_scenes.hpp"

using namespace adventures;

int main(int argc, char* argv[])
{
    const DemoArgs args = parse_demo_args(argc, argv, "t0_02_projection_sw.png");
    const std::string out_path = args.out_path;

    Frame frame(640, 480);
    frame.clear(12, 12, 16);
    SwRaster raster(frame);

    const std::vector<T0Vertex> cube = scene_projection_cube();

    // shared model transform: slight rotation so both projections show depth
    const glm::mat4 model = glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(35.0f), glm::vec3(0, 1, 0)),
                                        glm::radians(20.0f), glm::vec3(1, 0, 0));
    const glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -3.0f));

    // left half: perspective — FOV 60°, near 0.1, far 10
    // half-viewport map = translate AFTER scale: ndc.x*0.5 - 0.5
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 0.5f, 0.1f, 10.0f);
    const glm::mat4 vp_left = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)),
                                             glm::vec3(-1.0f, 0.0f, 0.0f)) * proj * view * model;

    // right half: orthographic — the same box, no foreshortening
    proj = glm::ortho(-1.6f, 1.6f, -1.0f, 1.0f, 0.1f, 10.0f);
    const glm::mat4 vp_right = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)),
                                              glm::vec3(1.0f, 0.0f, 0.0f)) * proj * view * model;

    // mvp is baked into the vertices here; the draw call sees clip space
    // (the *_vk twin sends these exact matrices as push constants instead)
    // mvp baked into the vertices as POST-w-divide NDC coords, so the
    // constant-w clip->screen transform in the rasterizer is exact (the *_vk
    // twin sends these matrices as push constants instead)
    std::vector<T0Vertex> left(cube);
    for (auto& v : left)
    {
        const glm::vec4 clip = vp_left * glm::vec4(glm::vec3(v.pos[0], v.pos[1], v.pos[2]), 1.0f);
        v.pos[0] = clip.x / clip.w;
        v.pos[1] = clip.y / clip.w;
        v.pos[2] = clip.z / clip.w;
    }
    std::vector<T0Vertex> right(cube);
    for (auto& v : right)
    {
        const glm::vec4 clip = vp_right * glm::vec4(glm::vec3(v.pos[0], v.pos[1], v.pos[2]), 1.0f);
        v.pos[0] = clip.x / clip.w;
        v.pos[1] = clip.y / clip.w;
        v.pos[2] = clip.z / clip.w;
    }

    // AD2: the pass policy is an explicit argument, and the old local SwState
    // here was never applied — the demo silently relied on the rasterizer's
    // default state. Stating the default makes that choice visible: depth test
    // on, so back faces lose the depth test (no culling — on purpose).
    const PassPolicy policy{};
    draw_triangles(raster, policy, left);
    draw_triangles(raster, policy, right);

    if (!frame.save_png(out_path))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d) — left: perspective, right: orthographic\n",
                out_path.c_str(), frame.width, frame.height);
    if (args.windowed)
    {
        present_frame_windowed(frame, "t0 02 — perspective vs orthographic projection (software)", out_path, args.backend);
    }
    return 0;
}
