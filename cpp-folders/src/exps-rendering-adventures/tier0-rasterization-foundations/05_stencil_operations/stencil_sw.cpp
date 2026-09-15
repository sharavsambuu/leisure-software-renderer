// tier0 demo 05 — stencil buffer operations.
// Pipeline A draws the triangle with stencil WRITE (stencil := 1 where it
// lands). Pipeline B draws the full-viewport quad with a stencil TEST:
// left half keeps only stencil==1 pixels (quad appears only inside the
// triangle silhouette), right half keeps only stencil!=1 (inverted mask).
// The *_vk twin runs identical VkStencilOpState configurations.
//
// Run: t0_stencil_sw [out.png]

#include <cstdio>
#include <string>

#include <glm/glm.hpp>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_sw_raster.hpp"
#include "../common/t0_scenes.hpp"

using namespace adventures;

int main(int argc, char* argv[])
{
    const std::string out_path = argc > 1 ? argv[1] : "t0_05_stencil_sw.png";

    Frame frame(640, 480);
    frame.clear(12, 12, 16);
    SwRaster raster(frame);

    // pass A: triangle writes stencil ref 1 (and its own color)
    // (depth disabled to mirror the *_vk twin's pipeline state — the quad in
    // pass B sits at the same z and must not be depth-rejected)
    SwState write_state{};
    write_state.depth_test    = false;
    write_state.depth_write   = false;
    write_state.stencil_write = true;
    write_state.stencil_ref   = 1;
    raster.state = write_state;
    draw_triangles(raster, scene_stencil_triangle());

    // pass B left: quad masked to the triangle silhouette (EQUAL)
    SwState test_state{};
    test_state.depth_test   = false;
    test_state.depth_write  = false;
    test_state.stencil_test = true;
    test_state.stencil_ref  = 1;
    raster.state = test_state;
    std::vector<T0Vertex> quad = scene_stencil_quad();
    for (auto& v : quad) { v.pos[0] = v.pos[0] * 0.5f - 0.5f; } // squeeze into left half
    draw_triangles(raster, quad);

    // pass B right: inverted mask (NOT_EQUAL)
    test_state.stencil_invert = true;
    raster.state = test_state;
    quad = scene_stencil_quad();
    for (auto& v : quad) { v.pos[0] = v.pos[0] * 0.5f + 0.5f; } // squeeze into right half
    draw_triangles(raster, quad);

    if (!frame.save_png(out_path))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d) — left: quad inside stencil mask, right: inverted mask\n",
                out_path.c_str(), frame.width, frame.height);
    return 0;
}
