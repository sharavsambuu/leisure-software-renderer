// tier0 AD4 — independent known-answer checks (GPU-free, always active).
// Location: .../tier0-rasterization-foundations/tools/t0_known_answer_checks.cpp
//
// Every lesson gets numerical assertions derived independently of the
// *_vk twins (no parity allowed as evidence) and of the demo's own printed
// diagnostics. Coverage:
//   * 02/03/05 — driven IN-PROCESS through the shared SwRaster kernels so
//     depth/stencil storage can be inspected directly (DoD requirement).
//   * 01/04 — the real demo binaries are run and their PNGs checked against
//     analytic oracles derived here from first principles (independent
//     barycentric solve / checkerboard + sampler model).
// Checks fail the process (nonzero exit), never rely on assert(); the
// `--prove-wrong-expected` mode corrupts one expectation and requires the
// comparator to catch it, proving a correlated SW/VK mistake cannot pass
// through this tool silently.
//
// Run: t0_known_answer_checks [--prove-wrong-expected]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h> // getpid

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_sw_raster.hpp"
#include "../common/t0_scenes.hpp"

#ifndef SHS_KA_BIN_DIR
#define SHS_KA_BIN_DIR "."
#endif

// Defined in common/adventures_stb_load.cpp (single stb_image TU per tree).
namespace adventures
{
    bool ka_load_png_rgba(const std::string& path, std::vector<uint8_t>& rgba, int& w, int& h);
}

using namespace adventures;

namespace
{
    // ---- tiny diagnostic harness: failures collect here, summary prints at
    // ---- the host edge (never inside the kernels under test).
    struct Ka
    {
        static constexpr std::size_t k_max_reported = 64; // diagnostics stay bounded
        int                      failed = 0;
        std::string              first;
        std::vector<std::string> reported; // printed once, at the host edge

        void record(std::string msg)
        {
            ++failed;
            if (first.empty()) first = msg;
            if (reported.size() < k_max_reported) reported.push_back(std::move(msg));
        }
    };

    void near(Ka& k, const char* what, double got, double want, double tol)
    {
        if (!(std::abs(got - want) <= tol))
        {
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s: got %.6f want %.6f (tol %.4f)",
                          what, got, want, tol);
            k.record(buf);
        }
    }

    void require(Ka& k, const char* what, bool ok)
    {
        if (!ok) k.record(std::string(what));
    }

    // Same comparator, but the diagnostic carries the probe location and
    // channel — texture probes differ per channel and need the coordinate to
    // be actionable.
    void near_at(Ka& k, const char* what, int x, int y, int ch, double got, double want, double tol)
    {
        if (!(std::abs(got - want) <= tol))
        {
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s @(%d,%d) ch%d: got %.6f want %.6f (tol %.4f)",
                          what, x, y, ch, got, want, tol);
            k.record(buf);
        }
    }

    uint8_t pix(const std::vector<uint8_t>& img, int w, int x, int y, int c)
    {
        return img[(size_t(y) * size_t(w) + size_t(x)) * 4u + size_t(c)];
    }

    bool run_demo_binary(const std::string& bin_dir, const std::string& exe, const std::string& out_png)
    {
        const std::string cmd = "\"" + bin_dir + "/" + exe + "\" \"" + out_png + "\" >/dev/null 2>&1";
        return std::system(cmd.c_str()) == 0;
    }

    // Independent barycentric solve (2x2 linear system, no edge functions —
    // deliberately not the rasterizer's area-ratio formulation).
    void bary2d(const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c,
                const glm::dvec2& p, double w[3])
    {
        const double den = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
        w[0] = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / den;
        w[1] = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / den;
        w[2] = 1.0 - w[0] - w[1];
    }

    glm::dvec2 to_px(double clip_x, double clip_y, int w, int h)
    {
        return {(clip_x * 0.5 + 0.5) * double(w), (1.0 - (clip_y * 0.5 + 0.5)) * double(h)};
    }

    // ---------------------------------------------------------------------------
    // Demo 01 — interior barycentric color vs analytic weights at the sampled
    // pixel center, with an explicit quantization tolerance (±1.5 LSB: one
    // 8-bit rounding per channel plus a half-step for float accumulation).
    // PNG-based: runs the real t0_tri_barycentric_sw binary.
    // ---------------------------------------------------------------------------
    void check_01(Ka& k, const std::vector<uint8_t>& img, int w, int h, double bump)
    {
        const auto tri = scene_tri_barycentric();
        const glm::dvec2 a = to_px(tri[0].pos[0], tri[0].pos[1], w, h);
        const glm::dvec2 b = to_px(tri[1].pos[0], tri[1].pos[1], w, h);
        const glm::dvec2 c = to_px(tri[2].pos[0], tri[2].pos[1], w, h);

        constexpr double k_tol = 1.5 / 255.0; // explicit quantization tolerance
        const int probes[3][2] = {{320, 284}, {200, 300}, {400, 200}};
        for (const auto& pxy : probes)
        {
            double wgt[3];
            bary2d(a, b, c, {pxy[0] + 0.5, pxy[1] + 0.5}, wgt);
            require(k, "ka01 sample point unexpectedly outside triangle",
                    wgt[0] > 0.01 && wgt[1] > 0.01 && wgt[2] > 0.01);
            for (int ch = 0; ch < 3; ++ch)
            {
                const double want = wgt[0] * tri[0].col[ch] + wgt[1] * tri[1].col[ch] +
                                    wgt[2] * tri[2].col[ch] + double(ch == 0) * bump;
                const double got = double(pix(img, w, pxy[0], pxy[1], ch)) / 255.0;
                near(k, "ka01 interior barycentric color", got, want, k_tol);
            }
        }
        // exterior: pixel outside the triangle must remain the clear color
        for (int ch = 0; ch < 3; ++ch)
        {
            const double clear_ch = double(ch == 2 ? 16 : 12) / 255.0;
            near(k, "ka01 exterior stays clear", double(pix(img, w, 600, 100, ch)) / 255.0,
                 clear_ch, k_tol);
        }
        // centroid: equal-weight average of the vertex colors (independent check
        // that the weights are affine and normalized)
        double wgt[3];
        const glm::dvec2 ctr{(a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0};
        bary2d(a, b, c, ctr, wgt);
        near(k, "ka01 centroid weight w0", wgt[0], 1.0 / 3.0, 1e-3);
        near(k, "ka01 centroid weight w1", wgt[1], 1.0 / 3.0, 1e-3);
        near(k, "ka01 centroid weight w2", wgt[2], 1.0 / 3.0, 1e-3);
    }

    // ---------------------------------------------------------------------------
    // Demo 02 — transformed coordinates and depth outcomes, independent of the
    // twin image. The test derives NDC by hand (textbook trig + projection
    // formulas, double precision) and compares against the GLM chain the demo
    // pins; depth outcomes are read from the rasterizer's z-buffer after an
    // in-process render through the shared kernels.
    // ---------------------------------------------------------------------------
    struct ManualMath
    {
        static constexpr double k_c1 = 0.8191520442889918;    // cos 35deg
        static constexpr double k_s1 = 0.5735764363510460;    // sin 35deg
        static constexpr double k_c2 = 0.9396926207859084;    // cos 20deg
        static constexpr double k_s2 = 0.3420201433256687;    // sin 20deg
        static constexpr double k_focal = 1.7320508075688774; // cot 30deg (fovy 60)

        // model = rotate(rotate(I, 35, Y), 20, X) = Ry(35) * Rx(20) (GLM
        // post-multiplies), then view z-3 (hand-derived).
        static glm::dvec3 eye(const glm::dvec3& p)
        {
            const double x1 = p.x;                     // Rx first
            const double y1 = k_c2 * p.y - k_s2 * p.z;
            const double z1 = k_s2 * p.y + k_c2 * p.z;
            const double x2 = k_c1 * x1 + k_s1 * z1;   // then Ry
            const double z2 = -k_s1 * x1 + k_c1 * z1;
            return {x2, y1, z2 - 3.0};
        }
        // perspective(60, 0.5, 0.1, 10) RH + zero-to-one, left-half remap.
        static glm::dvec3 persp_left(const glm::dvec3& p)
        {
            const glm::dvec3 e = eye(p);
            const double w = -e.z;
            return {e.x * (k_focal / 0.5) / w * 0.5 - 0.5, e.y * k_focal / w,
                    (e.z * (-10.0 / 9.9) - 1.0 / 9.9) / w};
        }
        // ortho(-1.6, 1.6, -1, 1, 0.1, 10) zero-to-one, right-half remap.
        // GLM ZO ortho (verified against the pinned headers): z_ndc =
        // -z/(f-n) - n/(f-n), mapping eye -n -> 0 and eye -f -> 1.
        static glm::dvec3 ortho_right(const glm::dvec3& p)
        {
            const glm::dvec3 e = eye(p);
            return {e.x / 1.6 * 0.5 + 0.5, e.y, -e.z / 9.9 - 0.1 / 9.9};
        }
    };

    double ndc_ch(const glm::vec4& clip, int ch)
    {
        const double c[4] = {clip.x, clip.y, clip.z, clip.w};
        return c[ch] / c[3];
    }

    void render_02(Frame& frame, SwRaster& raster) // mirrors projection_sw.cpp
    {
        const std::vector<T0Vertex> cube = scene_projection_cube();
        const glm::mat4 model = glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(35.0f), glm::vec3(0, 1, 0)),
                                            glm::radians(20.0f), glm::vec3(1, 0, 0));
        const glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -3.0f));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f), 0.5f, 0.1f, 10.0f);
        const glm::mat4 vp_left = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)),
                                                 glm::vec3(-1.0f, 0.0f, 0.0f)) * proj * view * model;
        proj = glm::ortho(-1.6f, 1.6f, -1.0f, 1.0f, 0.1f, 10.0f);
        const glm::mat4 vp_right = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)),
                                                  glm::vec3(1.0f, 0.0f, 0.0f)) * proj * view * model;
        const glm::mat4 vps[2] = {vp_left, vp_right};
        for (int half = 0; half < 2; ++half)
        {
            std::vector<T0Vertex> baked = cube; // bake post-w NDC, as the demo does
            for (auto& v : baked)
            {
                const glm::vec4 clip = vps[half] * glm::vec4(glm::vec3(v.pos[0], v.pos[1], v.pos[2]), 1.0f);
                v.pos[0] = clip.x / clip.w; v.pos[1] = clip.y / clip.w; v.pos[2] = clip.z / clip.w;
            }
            SwState st{};
            st.depth_test = true; // back faces lose the depth test (no culling, as in the demo)
            raster.state = st;
            draw_triangles(raster, baked);
        }
    }

    void check_02(Ka& k)
    {
        // (a) transformed coordinates: hand-derived NDC vs the demo's GLM chain
        const glm::dvec3 corners[8] = {
            {-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5}, {0.5, 0.5, -0.5}, {-0.5, 0.5, -0.5},
            {-0.5, -0.5, 0.5},  {0.5, -0.5, 0.5},  {0.5, 0.5, 0.5},  {-0.5, 0.5, 0.5},
        };
        const glm::mat4 model = glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(35.0f), glm::vec3(0, 1, 0)),
                                            glm::radians(20.0f), glm::vec3(1, 0, 0));
        const glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -3.0f));
        const glm::mat4 p_left = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)),
                                                glm::vec3(-1.0f, 0.0f, 0.0f)) *
                                 glm::perspective(glm::radians(60.0f), 0.5f, 0.1f, 10.0f) * view * model;
        const glm::mat4 p_right = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)),
                                                 glm::vec3(1.0f, 0.0f, 0.0f)) *
                                  glm::ortho(-1.6f, 1.6f, -1.0f, 1.0f, 0.1f, 10.0f) * view * model;
        double worst = 0.0;
        for (const auto& c : corners)
        {
            const glm::vec4 cl = p_left * glm::vec4(glm::vec3(c), 1.0f);
            const glm::dvec3 mw = ManualMath::persp_left(c);
            worst = std::max({worst, std::abs(ndc_ch(cl, 0) - mw.x), std::abs(ndc_ch(cl, 1) - mw.y),
                              std::abs(ndc_ch(cl, 2) - mw.z)});
            const glm::vec4 cr = p_right * glm::vec4(glm::vec3(c), 1.0f);
            const glm::dvec3 mo = ManualMath::ortho_right(c);
            worst = std::max({worst, std::abs(ndc_ch(cr, 0) - mo.x), std::abs(ndc_ch(cr, 1) - mo.y),
                              std::abs(ndc_ch(cr, 2) - mo.z)});
        }
        near(k, "ka02 hand-derived NDC vs GLM chain (8 corners, both projections)", worst, 0.0, 1e-5);

        // (b) depth outcomes + nearest-wins, read from rasterizer storage
        Frame frame(640, 480);
        frame.clear(12, 12, 16);
        SwRaster raster(frame);
        render_02(frame, raster);

        const glm::dvec3 quad_z[4] = {{-0.5, -0.5, 0.5}, {0.5, -0.5, 0.5}, {0.5, 0.5, 0.5}, {-0.5, 0.5, 0.5}};
        const glm::dvec3 back_z[4] = {{-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5}, {0.5, 0.5, -0.5}, {-0.5, 0.5, -0.5}};
        for (int half = 0; half < 2; ++half)
        {
            // hand-project the first drawn triangle of the +Z face (verts 4,5,6)
            glm::dvec3 P[3];
            glm::dvec2 s[3];
            for (int i = 0; i < 3; ++i)
            {
                P[i] = half == 0 ? ManualMath::persp_left(quad_z[i]) : ManualMath::ortho_right(quad_z[i]);
                s[i] = to_px(P[i].x, P[i].y, 640, 480);
            }
            const glm::dvec2 ctr{(s[0].x + s[1].x + s[2].x) / 3.0, (s[0].y + s[1].y + s[2].y) / 3.0};
            double wgt[3];
            bary2d(s[0], s[1], s[2], ctr, wgt);
            require(k, "ka02 sample point unexpectedly outside +Z triangle",
                    wgt[0] > 0.01 && wgt[1] > 0.01 && wgt[2] > 0.01);
            const double z_expect = wgt[0] * P[0].z + wgt[1] * P[1].z + wgt[2] * P[2].z;
            const int px = int(std::floor(ctr.x));
            const int py = int(std::floor(ctr.y));
            const double depth = double(raster.depth[size_t(py) * 640u + size_t(px)]);
            near(k, half == 0 ? "ka02 perspective depth at +Z face interior"
                              : "ka02 ortho depth at +Z face interior",
                 depth, z_expect, 1e-3);
            // nearest-wins: compare against the -Z face's OWN depth field at the
            // same pixel (interpolated over the face polygon, not a corner
            // average). The stored depth was already proven equal to the front
            // face's value above, so equality + a strictly-nearer back face is
            // the nearest-wins proof.
            // Depth the -Z face's PLANE would have at the probe pixel. The
            // barycentric weights are deliberately allowed to go negative
            // (plane extrapolation): the -Z face need not project over the
            // probe, but for a convex cube its plane is uniformly farther than
            // the +Z plane along the view axis. The stored depth was already
            // proven equal to the +Z face's plane depth above, so a strictly
            // nearer front value is the nearest-wins proof.
            glm::dvec3 B[4];
            glm::dvec2 bs[4];
            for (int i = 0; i < 4; ++i)
            {
                B[i]  = half == 0 ? ManualMath::persp_left(back_z[i]) : ManualMath::ortho_right(back_z[i]);
                bs[i] = to_px(B[i].x, B[i].y, 640, 480);
            }
            double bw[3];
            bary2d(bs[0], bs[1], bs[2], ctr, bw);
            const double z_back = bw[0] * B[0].z + bw[1] * B[1].z + bw[2] * B[2].z;
            // Perspective depth is nonlinear: at this cube scale the front/back
            // separation is only ~9e-3 (ortho ~8e-2). Require a measurable gap
            // before trusting a margin, which stays well inside it.
            require(k, "ka02 front/back depth separation is measurable", z_back - z_expect > 5e-3);
            require(k, "ka02 nearest surface wins the depth test", depth < z_back - 1e-3);
            // the +Z face is uniformly red (0.9, 0.2, 0.2)
            const uint8_t exp[3] = {uint8_t(0.9f * 255.0f + 0.5f), uint8_t(0.2f * 255.0f + 0.5f),
                                    uint8_t(0.2f * 255.0f + 0.5f)};
            for (int ch = 0; ch < 3; ++ch)
            {
                near(k, "ka02 front-face color", double(pix(frame.rgba, 640, px, py, ch)),
                     double(exp[ch]), 1.0);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Demo 04 — known NEAREST/BILINEAR samples, REPEAT wrap, scissor. PNG-based
    // on the real t0_texture_sampling_sw binary; the checkerboard, wrap rule,
    // and both filter models are re-derived here from the documented pattern.
    // ---------------------------------------------------------------------------
    struct CheckerOracle
    {
        // 16x16 checkerboard, 2x2 blocks of 8px, red cross marker at x==8|y==8.
        // Palette (the lesson's texture definition, shared by both twins):
        //   even block (230,230,230) | odd block (40,40,220) | marker (255,60,60)
        static void texel(int x, int y, double out[3])
        {
            x = ((x % 16) + 16) % 16; // REPEAT wrap
            y = ((y % 16) + 16) % 16;
            if (x == 8 || y == 8) { out[0] = 255.0; out[1] = 60.0; out[2] = 60.0; return; }
            if (((x / 8) + (y / 8)) % 2 == 0) { out[0] = 230.0; out[1] = 230.0; out[2] = 230.0; }
            else { out[0] = 40.0; out[1] = 40.0; out[2] = 220.0; }
        }
        // uv at pixel center, from the documented scene mapping:
        // left quad x -0.95..-0.05 -> u 0..8; y -0.6..0.6 -> v 0..8
        static glm::dvec2 uv_at(int px, int py, bool right_half)
        {
            const double ndc_x = (double(px) + 0.5) / 320.0 - 1.0;
            const double ndc_y = 1.0 - (double(py) + 0.5) / 240.0;
            if (!right_half) return {(ndc_x + 0.95) / 0.90 * 8.0, (ndc_y + 0.6) / 1.2 * 8.0};
            return {(ndc_x - 0.05) / 0.90 * 8.0, (ndc_y + 0.6) / 1.2 * 8.0};
        }
        // NEAREST with REPEAT: floor(u*16) (since floor(u*16-0.5+0.5)).
        static void nearest(const glm::dvec2& uv, double out[3])
        {
            texel(int(std::floor(uv.x * 16.0)), int(std::floor(uv.y * 16.0)), out);
        }
        // BILINEAR with REPEAT: 4-texel mix (same math as the documented model).
        static void bilinear(const glm::dvec2& uv, double out[3])
        {
            const double fx = uv.x * 16.0 - 0.5;
            const double fy = uv.y * 16.0 - 0.5;
            const int x0 = int(std::floor(fx));
            const int y0 = int(std::floor(fy));
            const double tx = fx - double(x0);
            const double ty = fy - double(y0);
            double c00[3], c10[3], c01[3], c11[3];
            texel(x0, y0, c00);
            texel(x0 + 1, y0, c10);
            texel(x0, y0 + 1, c01);
            texel(x0 + 1, y0 + 1, c11);
            for (int ch = 0; ch < 3; ++ch)
            {
                const double top = c00[ch] + (c10[ch] - c00[ch]) * tx;
                const double bottom = c01[ch] + (c11[ch] - c01[ch]) * tx;
                out[ch] = top + (bottom - top) * ty;
            }
        }
    };

    void check_04(Ka& k, const std::vector<uint8_t>& img, int w, int h, double bump)
    {
        // scissor: rows above the band (300) must be untouched clear color
        for (int row : {250, 299})
        {
            for (int x : {50, 160, 320, 480, 620})
            {
                near(k, "ka04 unchanged pixel outside scissor", double(pix(img, w, x, row, 0)), 12.0, 0.5);
                near(k, "ka04 unchanged pixel outside scissor", double(pix(img, w, x, row, 2)), 16.0, 0.5);
            }
        }
        // inside the band, both halves actually textured
        // (the quad spans rows 96..383; the scissor band starts at 300, so rows
        //  300..383 are the ONLY textured rows — probes must live there)
        require(k, "ka04 nearest half textured", pix(img, w, 100, 340, 0) != 12 || pix(img, w, 100, 340, 2) != 16);
        require(k, "ka04 bilinear half textured", pix(img, w, 400, 340, 0) != 12 || pix(img, w, 400, 340, 2) != 16);

        constexpr double k_tol = 2.0 / 255.0; // 8-bit truncation + float mix chain
        const int nearest_px[4][2] = {{100, 340}, {60, 310}, {250, 330}, {150, 360}};
        for (const auto& pxy : nearest_px)
        {
            const glm::dvec2 uv = CheckerOracle::uv_at(pxy[0], pxy[1], false);
            double want[3];
            CheckerOracle::texel(int(std::floor(uv.x * 16.0)), int(std::floor(uv.y * 16.0)), want);
            for (int ch = 0; ch < 3; ++ch)
            {
                near_at(k, "ka04 NEAREST + REPEAT sample", pxy[0], pxy[1], ch,
                        double(pix(img, w, pxy[0], pxy[1], ch)) / 255.0,
                        want[ch] / 255.0 + double(ch == 0) * bump, k_tol);
            }
        }
        const int bilinear_px[4][2] = {{400, 340}, {500, 350}, {600, 330}, {360, 320}};
        for (const auto& pxy : bilinear_px)
        {
            const glm::dvec2 uv = CheckerOracle::uv_at(pxy[0], pxy[1], true);
            const double fx = uv.x * 16.0 - 0.5;
            const double fy = uv.y * 16.0 - 0.5;
            const int x0 = int(std::floor(fx));
            const int y0 = int(std::floor(fy));
            const double tx = fx - double(x0);
            const double ty = fy - double(y0);
            double c00[3], c10[3], c01[3], c11[3];
            CheckerOracle::texel(x0, y0, c00);
            CheckerOracle::texel(x0 + 1, y0, c10);
            CheckerOracle::texel(x0, y0 + 1, c01);
            CheckerOracle::texel(x0 + 1, y0 + 1, c11);
            for (int ch = 0; ch < 3; ++ch)
            {
                const double want = ((c00[ch] + (c10[ch] - c00[ch]) * tx) * (1.0 - ty) +
                                     (c01[ch] + (c11[ch] - c01[ch]) * tx) * ty);
                near_at(k, "ka04 BILINEAR + REPEAT sample", pxy[0], pxy[1], ch,
                        double(pix(img, w, pxy[0], pxy[1], ch)) / 255.0,
                        want / 255.0 + double(ch == 0) * bump, k_tol);
            }
        }
        // wrap probe: (330,355) lands at u*16 ≈ 119.x — inside the 8th tile;
        // REPEAT must map it back into texels 7..8, not clamp at the edge.
        // (covered by the bilinear oracle above, which wraps internally.)
    }

    // ---------------------------------------------------------------------------
    // Demo 03 — opaque overlap resolution and alpha composition, with direct
    // depth-storage inspection proving transparent draws never write depth.
    // In-process through the shared kernels (same draw sequence as the demo).
    // ---------------------------------------------------------------------------
    void render_03(Frame& frame, SwRaster& raster) // mirrors depth_blend_sw.cpp
    {
        const std::vector<T0Vertex> scene = scene_depth_blend();
        SwState opaque{};
        opaque.depth_test  = true;
        opaque.depth_write = true;
        opaque.blend       = false;
        raster.state = opaque;
        draw_triangles(raster, {scene.begin(), scene.begin() + 6});
        SwState translucent{};
        translucent.depth_test  = true;
        translucent.depth_write = false;
        translucent.blend       = true;
        raster.state = translucent;
        draw_triangles(raster, {scene.begin() + 6, scene.end()});
    }

    void check_03(Ka& k)
    {
        Frame frame(640, 480);
        frame.clear(12, 12, 16);
        SwRaster raster(frame);
        render_03(frame, raster);

        const auto depth_at = [&raster](int x, int y) { return raster.depth[size_t(y) * 640u + size_t(x)]; };
        const auto rgb_at = [&](int x, int y, int ch) { return double(pix(frame.rgba, 640, x, y, ch)); };
        const float* src_col = scene_depth_blend()[6].col; // translucent green quad, a = 0.55
        // independent straight-alpha source-over oracle
        const auto over = [&](const double dst[3], double out[3]) {
            for (int ch = 0; ch < 3; ++ch) out[ch] = src_col[ch] * src_col[3] + dst[ch] * (1.0 - src_col[3]);
        };

        // (1) near-red / far-blue overlap: red (z 0.20) beats blue (z 0.70) —
        // depth must hold the NEAR value. This pixel is covered by both
        // triangles and NOT by the quad (the quad occupies x 176..464,
        // y 204..444 in this 640x480 frame, so the probe sits above it).
        near(k, "ka03 overlap depth stores near triangle", depth_at(320, 180), 0.20, 1e-6);
        const double red[3] = {0.95, 0.15, 0.15};
        for (int ch = 0; ch < 3; ++ch)
        {
            near(k, "ka03 overlap color = near red", rgb_at(320, 180, ch), red[ch] * 255.0 + 0.5, 1.0);
        }

        // (2) quad over far blue: blended color, depth stays 0.70 (quad's
        // depth_write = false must not poison the z-buffer).
        const double blue[3] = {0.15, 0.30, 0.95};
        double want[3];
        over(blue, want);
        for (int ch = 0; ch < 3; ++ch)
        {
            near(k, "ka03 alpha blend over far blue", rgb_at(200, 250, ch), want[ch] * 255.0 + 0.5, 1.0);
        }
        near(k, "ka03 transparent draw left far-blue depth intact", depth_at(200, 250), 0.70, 1e-6);

        // (3) quad over clear background only: color = src over clear, and the
        // depth buffer must still read 1.0 — the explicit storage proof that
        // transparent draws do not write depth.
        const double clear_bg[3] = {12.0 / 255.0, 12.0 / 255.0, 16.0 / 255.0};
        over(clear_bg, want);
        for (int ch = 0; ch < 3; ++ch)
        {
            near(k, "ka03 alpha blend over clear", rgb_at(200, 430, ch), want[ch] * 255.0 + 0.5, 1.0);
        }
        require(k, "ka03 transparent draw wrote NO depth (still 1.0)", depth_at(200, 430) == 1.0f);
        // alpha compositing accumulates to fully opaque over the opaque base
        near(k, "ka03 composited alpha", rgb_at(200, 250, 3), 255.0, 0.5);
    }

    // ---------------------------------------------------------------------------
    // Demo 05 — stencil storage + EQUAL/NOT_EQUAL masking at interior/exterior
    // sample points, with depth-storage inspection (depth_test off must leave
    // the z-buffer untouched). In-process through the shared kernels.
    // ---------------------------------------------------------------------------
    void render_05(Frame& frame, SwRaster& raster) // mirrors stencil_sw.cpp
    {
        SwState write_state{};
        write_state.depth_test    = false;
        write_state.depth_write   = false;
        write_state.stencil_write = true;
        write_state.stencil_ref   = 1;
        raster.state = write_state;
        draw_triangles(raster, scene_stencil_triangle());

        SwState test_state{};
        test_state.depth_test   = false;
        test_state.depth_write  = false;
        test_state.stencil_test = true;
        test_state.stencil_ref  = 1;
        raster.state = test_state;
        std::vector<T0Vertex> quad = scene_stencil_quad();
        for (auto& v : quad) { v.pos[0] = v.pos[0] * 0.5f - 0.5f; } // left half
        draw_triangles(raster, quad);

        test_state.stencil_invert = true;
        raster.state = test_state;
        quad = scene_stencil_quad();
        for (auto& v : quad) { v.pos[0] = v.pos[0] * 0.5f + 0.5f; } // right half
        draw_triangles(raster, quad);
    }

    void check_05(Ka& k)
    {
        Frame frame(640, 480);
        frame.clear(12, 12, 16);
        SwRaster raster(frame);
        render_05(frame, raster);

        const auto stencil_at = [&raster](int x, int y) { return raster.stencil[size_t(y) * 640u + size_t(x)]; };
        const auto rgb_at = [&](int x, int y, int ch) { return double(pix(frame.rgba, 640, x, y, ch)); };

        // stencil storage: 1 inside the triangle silhouette, 0 outside
        require(k, "ka05 stencil == 1 at interior point", stencil_at(320, 240) == 1u);
        require(k, "ka05 stencil == 0 at exterior point", stencil_at(100, 100) == 0u);
        // triangle color written during pass A (depth stayed untouched)
        near(k, "ka05 triangle color", rgb_at(320, 240, 0), 0.95 * 255.0 + 0.5, 1.0);
        require(k, "ka05 depth untouched (depth_test off)", raster.depth[size_t(240) * 640u + 320u] == 1.0f);

        // pass B left — EQUAL mask: quad only inside the triangle silhouette
        const uint8_t purple[3] = {uint8_t(0.55f * 255.0f + 0.5f), uint8_t(0.15f * 255.0f + 0.5f),
                                   uint8_t(0.85f * 255.0f + 0.5f)};
        near(k, "ka05 equal-mask keeps quad inside silhouette", rgb_at(300, 300, 0),
             double(purple[0]), 0.5);
        near(k, "ka05 equal-mask keeps quad inside silhouette (b)", rgb_at(300, 300, 2),
             double(purple[2]), 0.5);
        // left-half exterior point: quad rejected by EQUAL, stays clear. At row
        // 300 the triangle silhouette spans x 146.6..493.4, so x=80 (inside the
        // left-half quad, x 8..312) is safely outside the mask.
        for (int ch = 0; ch < 3; ++ch)
        {
            near(k, "ka05 equal-mask rejects exterior", rgb_at(80, 300, ch),
                 double(ch == 2 ? 16 : 12), 0.5);
        }
        // pass B right — NOT_EQUAL mask: exterior passes, interior rejected
        near(k, "ka05 inverted mask keeps quad outside silhouette", rgb_at(500, 300, 0),
             double(purple[0]), 0.5);
        const uint8_t yellow[3] = {uint8_t(0.95f * 255.0f + 0.5f), uint8_t(0.75f * 255.0f + 0.5f),
                                   uint8_t(0.15f * 255.0f + 0.5f)};
        near(k, "ka05 inverted mask rejects stencil==1 (interior)", rgb_at(350, 300, 0),
             double(yellow[0]), 0.5);
        near(k, "ka05 inverted mask keeps triangle color", rgb_at(350, 300, 1),
             double(yellow[1]), 0.5);
    }
} // namespace

int main(int argc, char* argv[])
{
    const bool prove_wrong = argc > 1 && std::string(argv[1]) == "--prove-wrong-expected";
    const std::string bin_dir = SHS_KA_BIN_DIR;

    // Temp output paths for the PNG-based probes (unique per process so
    // parallel ctest runs never collide).
    const std::string tmp_base =
        (std::filesystem::temp_directory_path() /
         ("shs_adventures_ka_" + std::to_string(::getpid())))
            .string();

    Ka k;

    if (prove_wrong)
    {
        // Deliberately corrupt one expectation per probe style by ~12 LSB: the
        // comparator MUST flag each, and BOTH must be caught for the prove to
        // pass. Two independent oracle styles are proven non-vacuous here:
        //   * 01 — the analytic barycentric-weight PNG probe;
        //   * 04 — the sampler/wrap/filter PNG probe.
        // The in-process depth/stencil STORAGE checks (02/03/05) run through the
        // same `near`/`require` comparators, so a comparator that catches these
        // cannot be silently tolerant there either. This is the guard against a
        // correlated SW/VK mistake passing through image parity alone.
        if (!run_demo_binary(bin_dir, "t0_tri_barycentric_sw", tmp_base + "_01.png") ||
            !run_demo_binary(bin_dir, "t0_texture_sampling_sw", tmp_base + "_04.png"))
        {
            std::fprintf(stderr, "prove mode: failed to run demo binaries\n");
            return 1;
        }
        std::vector<uint8_t> img01, img04;
        int w01 = 0, h01 = 0, w04 = 0, h04 = 0;
        if (!ka_load_png_rgba(tmp_base + "_01.png", img01, w01, h01) ||
            !ka_load_png_rgba(tmp_base + "_04.png", img04, w04, h04))
        {
            std::fprintf(stderr, "prove mode: failed to load demo PNGs\n");
            return 1;
        }
        Ka k01, k04;
        check_01(k01, img01, w01, h01, /*bump=*/12.0 / 255.0);
        check_04(k04, img04, w04, h04, /*bump=*/12.0 / 255.0);
        std::filesystem::remove(tmp_base + "_01.png");
        std::filesystem::remove(tmp_base + "_04.png");
        if (k01.failed > 0 && k04.failed > 0)
        {
            std::printf("PROVE OK: corrupted expectations caught (barycentric %d, sampler %d)\n",
                        k01.failed, k04.failed);
            return 0;
        }
        std::fprintf(stderr, "PROVE FAILED: barycentric=%d sampler=%d (both must be > 0)\n",
                     k01.failed, k04.failed);
        return 1;
    }

    // ---- demo 01 (PNG of the real binary vs independent barycentric oracle)
    if (!run_demo_binary(bin_dir, "t0_tri_barycentric_sw", tmp_base + "_01.png") ||
        !run_demo_binary(bin_dir, "t0_texture_sampling_sw", tmp_base + "_04.png"))
    {
        std::fprintf(stderr, "ka: failed to run demo binaries\n");
        return 1;
    }
    std::vector<uint8_t> img01, img04;
    int w01 = 0, h01 = 0, w04 = 0, h04 = 0;
    if (!ka_load_png_rgba(tmp_base + "_01.png", img01, w01, h01) ||
        !ka_load_png_rgba(tmp_base + "_04.png", img04, w04, h04))
    {
        std::fprintf(stderr, "ka: failed to load demo PNGs\n");
        return 1;
    }
    check_01(k, img01, w01, h01, 0.0);
    check_02(k);
    check_03(k);
    check_04(k, img04, w04, h04, 0.0);
    check_05(k);

    std::filesystem::remove(tmp_base + "_01.png");
    std::filesystem::remove(tmp_base + "_04.png");

    if (k.failed != 0)
    {
        for (const auto& msg : k.reported) std::fprintf(stderr, "  - %s\n", msg.c_str());
        if (k.failed > static_cast<int>(k.reported.size()))
        {
            std::fprintf(stderr, "  ... %d more suppressed\n",
                         k.failed - static_cast<int>(k.reported.size()));
        }
        std::fprintf(stderr, "t0 known-answer checks FAILED: %d mismatch(es); first: %s\n",
                     k.failed, k.first.c_str());
        return 1;
    }
    std::printf("t0 known-answer checks PASSED (demos 01-05, GPU-free)\n");
    return 0;
}
