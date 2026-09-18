// tier0 AD2 — pass-policy semantics through the shared software kernel.
// Location: .../tier0-rasterization-foundations/tools/t0_policy_tests.cpp
//
// AD2 replaced the pair of parallel state objects (SwState and the
// depth/blend/stencil fields of VkPipelineSetup) with one execution-neutral
// PassPolicy that every draw states for itself. This gate exercises that
// vocabulary against the real shared rasterizer (adventures_sw_raster.hpp) —
// the same kernel the *_sw demos and the AD4 known-answer tool use — with
// checks derived from the policy semantics, not from twin image agreement:
//
//   1. documented defaults
//   2. depth test / depth write, inspected in the z-buffer
//   3. opaque-then-translucent vs translucent-then-opaque ordering, with an
//      analytic source-over value and a z-buffer-poisoning check
//   4. stencil WriteRef / TestEqual / TestNotEqual / Disabled, inspected in the
//      stencil plane
//   5. scissor bounds (outside untouched, inside drawn, empty band draws
//      nothing) plus the Vulkan-side clamp adapter
//   6. policy isolation: a hostile policy in one draw must not leak into the
//      next draw that states the defaults
//
// GPU-free and always active: the software rasterizer needs no device.
//
// Run: t0_policy_tests

#include <cstdint>
#include <cstdio>
#include <vector>

#include <glm/glm.hpp>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_pass_policy.hpp"
#include "../common/adventures_sw_raster.hpp"
#include "../common/t0_scenes.hpp"

using namespace adventures;

namespace
{
    constexpr int kWidth  = 64;
    constexpr int kHeight = 64;
    constexpr int kMid    = kWidth / 2;

    int g_checks   = 0;
    int g_failures = 0;

    void check(bool ok, const char* what)
    {
        ++g_checks;
        if (!ok)
        {
            ++g_failures;
            std::fprintf(stderr, "FAIL: %s\n", what);
        }
    }

    Frame make_frame()
    {
        Frame f(kWidth, kHeight);
        f.clear(12, 12, 16); // the clear every tier0 demo uses
        return f;
    }

    // Two triangles covering the NDC rectangle [x0, x1] x [-1, 1] at depth z,
    // with a constant vertex color. NDC +Y is the top of the target (to_screen
    // flips y), and pixel centers never sit on the NDC boundary, so the full
    // rectangle [-1, 1] covers every pixel exactly once.
    std::vector<T0Vertex> ndc_quad(float x0, float x1, float z, const glm::vec4& c)
    {
        return {
            t0_vertex({ x0, -1.0f, z }, c), t0_vertex({ x1, -1.0f, z }, c), t0_vertex({ x1, 1.0f, z }, c),
            t0_vertex({ x0, -1.0f, z }, c), t0_vertex({ x1, 1.0f, z }, c), t0_vertex({ x0, 1.0f, z }, c),
        };
    }

    struct Rgba
    {
        uint8_t r, g, b, a;
        bool operator==(const Rgba& o) const { return r == o.r && g == o.g && b == o.b && a == o.a; }
    };
    constexpr Rgba kClear{ 12, 12, 16, 255 };

    Rgba pixel(const Frame& f, int x, int y)
    {
        const size_t i = (size_t(y) * size_t(f.width) + size_t(x)) * 4u;
        return Rgba{ f.rgba[i + 0], f.rgba[i + 1], f.rgba[i + 2], f.rgba[i + 3] };
    }

    int count_pixels(const Frame& f, const Rgba& want)
    {
        int n = 0;
        for (int y = 0; y < f.height; ++y)
        {
            for (int x = 0; x < f.width; ++x)
            {
                if (pixel(f, x, y) == want) ++n;
            }
        }
        return n;
    }

    bool row_untouched(const Frame& f, int y)
    {
        for (int x = 0; x < f.width; ++x)
        {
            if (!(pixel(f, x, y) == kClear)) return false;
        }
        return true;
    }

    int count_stencil(const SwRaster& r, uint8_t want, int x0, int x1)
    {
        int n = 0;
        for (int y = 0; y < kHeight; ++y)
        {
            for (int x = x0; x < x1; ++x)
            {
                if (r.stencil[size_t(y) * size_t(kWidth) + size_t(x)] == want) ++n;
            }
        }
        return n;
    }

    bool depth_all(const SwRaster& r, float want)
    {
        for (float d : r.depth)
        {
            if (d != want) return false;
        }
        return true;
    }

    // ----------------------------------------------------------------------
    // 1. Documentation is part of the contract: the defaults are asserted, so a
    //    silent default change cannot pass review unnoticed.
    // ----------------------------------------------------------------------
    void test_defaults()
    {
        const PassPolicy d{};
        check(d.depth_test && d.depth_write && !d.blend,
              "default policy: depth test on, depth write on, blend off");
        check(d.stencil == StencilMode::Disabled && d.stencil_ref == 1,
              "default policy: stencil disabled with ref 1");
        check(!d.scissor_enabled, "default policy: no scissor");
        check(scissor_allows(d, 0, 0) && scissor_allows(d, kWidth - 1, kHeight - 1),
              "disabled scissor allows every pixel");

        check(!stencil_enabled(StencilMode::Disabled) && stencil_enabled(StencilMode::WriteRef),
              "stencil_enabled: Disabled is the only inactive mode");
        check(stencil_writes(StencilMode::WriteRef) && !stencil_writes(StencilMode::TestEqual),
              "stencil_writes only for WriteRef");
        check(stencil_tests(StencilMode::TestEqual) && stencil_tests(StencilMode::TestNotEqual) &&
                  !stencil_tests(StencilMode::WriteRef),
              "stencil_tests covers exactly the two test modes");
        check(stencil_inverts(StencilMode::TestNotEqual) && !stencil_inverts(StencilMode::TestEqual),
              "stencil_inverts only for TestNotEqual");
    }

    // ----------------------------------------------------------------------
    // 2. Depth: rejected fragments must not write color or depth, and
    //    depth_write off must leave the z-buffer untouched.
    // ----------------------------------------------------------------------
    void test_depth()
    {
        const Rgba red{ 255, 0, 0, 255 };
        const Rgba green{ 0, 255, 0, 255 };

        Frame    f = make_frame();
        SwRaster r(f);
        const PassPolicy depth_on{}; // defaults: test + write
        draw_triangles(r, depth_on, ndc_quad(-1.0f, 1.0f, 0.25f, glm::vec4(1, 0, 0, 1)));
        check(count_pixels(f, red) == kWidth * kHeight, "depth: near opaque quad covers the target");
        check(depth_all(r, 0.25f), "depth: depth_write stored the fragment depth");

        // farther fragment under the same policy -> rejected everywhere (LESS)
        draw_triangles(r, depth_on, ndc_quad(-1.0f, 1.0f, 0.75f, glm::vec4(0, 1, 0, 1)));
        check(count_pixels(f, green) == 0, "depth: farther fragment is rejected everywhere");
        check(count_pixels(f, red) == kWidth * kHeight, "depth: rejected fragment left color untouched");
        check(depth_all(r, 0.25f), "depth: rejected fragment left the z-buffer untouched");

        // depth_write off -> color draws, z-buffer unchanged
        Frame    f2 = make_frame();
        SwRaster r2(f2);
        PassPolicy no_write{};
        no_write.depth_write = false;
        draw_triangles(r2, no_write, ndc_quad(-1.0f, 1.0f, 0.25f, glm::vec4(0, 1, 0, 1)));
        check(count_pixels(f2, green) == kWidth * kHeight, "depth_write off: color still drawn");
        check(depth_all(r2, 1.0f), "depth_write off: z-buffer stays at the clear value");

        // depth_test off -> a farther fragment wins
        Frame    f3 = make_frame();
        SwRaster r3(f3);
        draw_triangles(r3, depth_on, ndc_quad(-1.0f, 1.0f, 0.25f, glm::vec4(1, 0, 0, 1)));
        PassPolicy no_test{};
        no_test.depth_test = false;
        draw_triangles(r3, no_test, ndc_quad(-1.0f, 1.0f, 0.75f, glm::vec4(0, 1, 0, 1)));
        check(count_pixels(f3, green) == kWidth * kHeight, "depth_test off: farther fragment draws");
    }

    // ----------------------------------------------------------------------
    // 3. Alpha blending and draw-order sensitivity (demo 03's lesson), with an
    //    independently derived source-over value and a z-buffer check.
    // ----------------------------------------------------------------------
    void test_blend_and_order()
    {
        const Rgba red{ 255, 0, 0, 255 };

        Frame    f = make_frame();
        SwRaster r(f);
        draw_triangles(r, PassPolicy{}, ndc_quad(-1.0f, 1.0f, 0.25f, glm::vec4(1, 0, 0, 1)));
        PassPolicy translucent{};
        translucent.depth_write = false; // must not poison the z-buffer
        translucent.blend       = true;
        draw_triangles(r, translucent, ndc_quad(-1.0f, 1.0f, 0.10f, glm::vec4(0, 0, 1, 0.5)));
        // analytic: a=0.5 over an opaque (1,0,0,1) target -> (0.5, 0, 0.5), round-half-up
        check(pixel(f, kMid, kMid) == Rgba{ 128, 0, 128, 255 },
              "blend: translucent-over-opaque matches source-over (128,0,128,255)");
        check(depth_all(r, 0.25f), "blend: depth_write off kept the opaque depth");

        // reversed order: the opaque draw overwrites the blended result
        Frame    f2 = make_frame();
        SwRaster r2(f2);
        draw_triangles(r2, translucent, ndc_quad(-1.0f, 1.0f, 0.10f, glm::vec4(0, 0, 1, 0.5)));
        draw_triangles(r2, PassPolicy{}, ndc_quad(-1.0f, 1.0f, 0.25f, glm::vec4(1, 0, 0, 1)));
        check(pixel(f2, kMid, kMid) == red, "order: translucent-then-opaque ends opaque red");
        check(!(pixel(f, kMid, kMid) == pixel(f2, kMid, kMid)),
              "order: draw order changes the composed pixel");

        // blend off stores the source color and alpha unchanged
        Frame    f3 = make_frame();
        SwRaster r3(f3);
        draw_triangles(r3, PassPolicy{}, ndc_quad(-1.0f, 1.0f, 0.5f, glm::vec4(0, 0, 1, 0.5)));
        check(pixel(f3, 1, 1) == Rgba{ 0, 0, 255, 128 },
              "blend off: fragment color and alpha stored unchanged");
    }

    // ----------------------------------------------------------------------
    // 4. Stencil: write/equal/invert/disabled, inspected in the stencil plane.
    // ----------------------------------------------------------------------
    void test_stencil()
    {
        const Rgba red{ 255, 0, 0, 255 };
        const Rgba blue{ 0, 0, 255, 255 };

        // WriteRef, scissored to the left half
        Frame    f = make_frame();
        SwRaster r(f);
        PassPolicy write{};
        write.depth_write     = false;
        write.stencil         = StencilMode::WriteRef;
        write.stencil_ref     = 7;
        write.scissor_enabled = true;
        write.scissor         = ScissorRect{ 0, 0, kMid, kHeight };
        draw_triangles(r, write, ndc_quad(-1.0f, 1.0f, 0.5f, glm::vec4(1, 0, 0, 1)));
        check(count_stencil(r, 7, 0, kMid) == kMid * kHeight,
              "stencil WriteRef: covered pixels store ref 7");
        check(count_stencil(r, 0, kMid, kWidth) == kMid * kHeight,
              "stencil WriteRef: unscissored half stays 0");
        check(count_pixels(f, red) == kMid * kHeight, "stencil WriteRef: color written in the band only");

        // TestEqual keeps only ref pixels — the mask, not a scissor, does the cutting
        Frame    f2 = make_frame();
        SwRaster r2(f2);
        draw_triangles(r2, write, ndc_quad(-1.0f, 1.0f, 0.5f, glm::vec4(1, 0, 0, 1)));
        PassPolicy equal = write;
        equal.scissor_enabled = false;
        equal.stencil         = StencilMode::TestEqual;
        draw_triangles(r2, equal, ndc_quad(-1.0f, 1.0f, 0.5f, glm::vec4(0, 0, 1, 1)));
        check(count_pixels(f2, blue) == kMid * kHeight, "stencil TestEqual: only ref pixels kept");
        check(count_stencil(r2, 7, kMid, kWidth) == 0,
              "stencil TestEqual: no ref outside the mask -> nothing drawn there");
        check(count_stencil(r2, 7, 0, kMid) == kMid * kHeight,
              "stencil TestEqual: the reference is not overwritten");
        // x = kMid-1 is inside the ref mask (drawn blue); x = kMid is outside it
        // and was never drawn by the scissored write pass (still the clear color).
        check(pixel(f2, kMid - 1, kMid) == blue && pixel(f2, kMid, kMid) == kClear,
              "stencil TestEqual: mask boundary is the ref boundary");

        // TestNotEqual is the inverted mask
        Frame    f3 = make_frame();
        SwRaster r3(f3);
        draw_triangles(r3, write, ndc_quad(-1.0f, 1.0f, 0.5f, glm::vec4(1, 0, 0, 1)));
        PassPolicy invert_state = write;
        invert_state.scissor_enabled = false;
        invert_state.stencil         = StencilMode::TestNotEqual;
        draw_triangles(r3, invert_state, ndc_quad(-1.0f, 1.0f, 0.5f, glm::vec4(0, 0, 1, 1)));
        check(count_pixels(f3, blue) == kMid * kHeight, "stencil TestNotEqual: only non-ref pixels kept");
        check(pixel(f3, kMid - 1, kMid) == red && pixel(f3, kMid, kMid) == blue,
              "stencil TestNotEqual: inverted mask boundary");

        // Disabled leaves the plane alone and masks nothing
        Frame    f4 = make_frame();
        SwRaster r4(f4);
        draw_triangles(r4, PassPolicy{}, ndc_quad(-1.0f, 1.0f, 0.5f, glm::vec4(0, 0, 1, 1)));
        check(count_stencil(r4, 0, 0, kWidth) == kWidth * kHeight, "stencil Disabled: plane untouched");
        check(count_pixels(f4, blue) == kWidth * kHeight, "stencil Disabled: nothing is masked");
    }

    // ----------------------------------------------------------------------
    // 5. Scissor bounds, plus the clamp adapter the Vulkan twin uses.
    // ----------------------------------------------------------------------
    void test_scissor()
    {
        const Rgba    green{ 0, 255, 0, 255 };
        constexpr int kY0 = 16;
        constexpr int kY1 = 48;

        Frame    f = make_frame();
        SwRaster r(f);
        PassPolicy band{};
        band.scissor_enabled = true;
        band.scissor         = ScissorRect{ 0, kY0, kWidth, kY1 };
        draw_triangles(r, band, ndc_quad(-1.0f, 1.0f, 0.5f, glm::vec4(0, 1, 0, 1)));
        check(count_pixels(f, green) == kWidth * (kY1 - kY0), "scissor: only the band rows are drawn");
        check(row_untouched(f, 0) && row_untouched(f, kY0 - 1), "scissor: rows above the band untouched");
        check(row_untouched(f, kY1) && row_untouched(f, kHeight - 1), "scissor: rows below the band untouched");

        // an empty band draws nothing at all
        Frame    f2 = make_frame();
        SwRaster r2(f2);
        PassPolicy empty_band{};
        empty_band.scissor_enabled = true;
        empty_band.scissor         = ScissorRect{ 10, 10, 10, 40 }; // x1 == x0
        draw_triangles(r2, empty_band, ndc_quad(-1.0f, 1.0f, 0.5f, glm::vec4(0, 1, 0, 1)));
        check(count_pixels(f2, kClear) == kWidth * kHeight, "scissor: empty band draws nothing");

        // the Vulkan adapter clamps into the target (Vulkan requires it inside)
        const ScissorRect clamped = clamp_scissor(ScissorRect{ -5, -5, 100, 100 }, kWidth, kHeight);
        check(clamped.x0 == 0 && clamped.y0 == 0 && clamped.x1 == kWidth && clamped.y1 == kHeight,
              "clamp_scissor: oversized rectangle clamps to the target");
        const ScissorRect clipped = clamp_scissor(ScissorRect{ 30, 0, 200, 20 }, kWidth, kHeight);
        check(clipped.x0 == 30 && clipped.x1 == kWidth, "clamp_scissor: right edge clamps to width");
        check(clamp_scissor(ScissorRect{ 90, 90, 10, 10 }, kWidth, kHeight).empty(),
              "clamp_scissor: inverted rectangle stays empty");
    }

    // ----------------------------------------------------------------------
    // 6. The acceptance property: a deliberately hostile policy in one draw must
    //    not be inherited by the next draw, which states the defaults. Before
    //    AD2 this was exactly the failure mode of the `raster.state = ...`
    //    prerequisite sequencing.
    // ----------------------------------------------------------------------
    void test_policy_isolation()
    {
        const Rgba red{ 255, 0, 0, 255 };
        const Rgba green{ 0, 255, 0, 255 };

        Frame    f = make_frame();
        SwRaster r(f);

        // draw 1: no depth write, stencil write ref 3, scissored to the left half
        PassPolicy hostile{};
        hostile.depth_write     = false;
        hostile.stencil         = StencilMode::WriteRef;
        hostile.stencil_ref     = 3;
        hostile.scissor_enabled = true;
        hostile.scissor         = ScissorRect{ 0, 0, kMid, kHeight };
        draw_triangles(r, hostile, ndc_quad(-1.0f, 1.0f, 0.9f, glm::vec4(1, 0, 0, 1)));
        check(count_pixels(f, red) == kMid * kHeight, "isolation: first draw respected its own scissor");
        check(depth_all(r, 1.0f), "isolation: first draw respected its own depth_write");

        // draw 2: the defaults, stated as a fresh policy
        draw_triangles(r, PassPolicy{}, ndc_quad(-1.0f, 1.0f, 0.5f, glm::vec4(0, 1, 0, 1)));
        check(count_pixels(f, green) == kWidth * kHeight,
              "isolation: second draw covered the whole target (no scissor leaked)");
        check(count_stencil(r, 3, 0, kMid) == kMid * kHeight,
              "isolation: first draw's stencil reference survived");
        check(count_stencil(r, 3, kMid, kWidth) == 0,
              "isolation: second draw did not inherit stencil WriteRef");
        check(depth_all(r, 0.5f), "isolation: second draw wrote depth (no depth_write leak)");
    }
}

int main()
{
    test_defaults();
    test_depth();
    test_blend_and_order();
    test_stencil();
    test_scissor();
    test_policy_isolation();

    if (g_failures != 0)
    {
        std::fprintf(stderr, "t0_policy_tests: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    std::printf("t0_policy_tests: %d checks passed (AD2 pass-policy semantics, GPU-free)\n", g_checks);
    return 0;
}
