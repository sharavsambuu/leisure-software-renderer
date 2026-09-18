// tier0 demo 03 — depth test + alpha blending + draw-order sensitivity.
// The two opaque triangles overlap; the z-buffer resolves them by depth
// (near red beats far blue). The translucent green quad is drawn LAST —
// with alpha blending, transparent geometry must be drawn after opaque
// geometry (back-to-front), regardless of z. The *_vk twin configures the
// same pipeline states: depth test LESS, blend SRC_ALPHA/ONE_MINUS_SRC_ALPHA.
//
// Run: t0_depth_blend_sw [out.png]

#include <cstdio>
#include <string>

#include <glm/glm.hpp>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_window.hpp"
#include "../common/adventures_sw_raster.hpp"
#include "../common/t0_scenes.hpp"

using namespace adventures;

int main(int argc, char* argv[])
{
    const DemoArgs args = parse_demo_args(argc, argv, "t0_03_depth_blend_sw.png");
    const std::string out_path = args.out_path;

    Frame frame(640, 480);
    frame.clear(12, 12, 16);
    SwRaster raster(frame);

    const std::vector<T0Vertex> scene = scene_depth_blend();
    // scene layout: 6 opaque verts (near red tri + far blue tri), 6 quad verts

    SwState opaque{};
    opaque.depth_test  = true;
    opaque.depth_write = true;
    opaque.blend       = false;
    raster.state = opaque; // pipeline state must be set before drawing
    draw_triangles(raster, { scene.begin(), scene.begin() + 6 });

    SwState translucent{};
    translucent.depth_test  = true;  // still occluded by nearer opaque pixels
    translucent.depth_write = false; // blended fragments must not poison the z-buffer
    translucent.blend       = true;
    raster.state = translucent; // pipeline state must be set before drawing
    draw_triangles(raster, { scene.begin() + 6, scene.end() });

    if (!frame.save_png(out_path))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d) — red triangle (near) wins overlap; quad blended last\n",
                out_path.c_str(), frame.width, frame.height);
    if (args.windowed)
    {
        present_frame_windowed(frame, "t0 03 — depth test + alpha blend (software)", out_path, args.backend);
    }
    return 0;
}
