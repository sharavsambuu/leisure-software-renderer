// tier1 AD4 — independent known-answer checks for demo 08 (GPU-free).
// Location: .../tier1-classic-shading/08_normal_mapping/normal_mapping_ka.cpp
//
// Runs the real t1_normal_mapping_sw binary and verifies selected pixels
// against analytic oracles derived here from first principles:
//   * flat half: albedo * max(dot((0,0,1), L), 0) + ambient, quantized.
//   * mapped half: full independent chain — hand-computed uv from the
//     documented quad mapping, procedurally regenerated bump texels,
//     8-bit decode round-trip, tangent frame T = normalize(cross(up, N)),
//     B = cross(N, T), Lambert shading.
// No cross-half brightness assumption is made — every sampled pixel is
// checked against its OWN analytic expectation. --prove-wrong-expected
// corrupts one expectation and requires the comparator to catch it.
//
// Run: t1_08_known_answer_checks [--prove-wrong-expected]

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

#include "../common/adventures_frame.hpp"
#include "../common/t1_scenes.hpp"

// Defined in ../../tier0-rasterization-foundations/common/adventures_stb_load.cpp
namespace adventures
{
    bool ka_load_png_rgba(const std::string& path, std::vector<uint8_t>& rgba, int& w, int& h);
}

using namespace adventures;

#ifndef SHS_KA_BIN_DIR
#define SHS_KA_BIN_DIR "."
#endif

namespace
{
    struct Ka
    {
        int         failed = 0;
        std::string first;
    };

    void near(Ka& k, const char* what, double got, double want, double tol)
    {
        if (!(std::abs(got - want) <= tol))
        {
            ++k.failed;
            if (k.first.empty())
            {
                char buf[512];
                std::snprintf(buf, sizeof(buf), "%s: got %.6f want %.6f (tol %.4f)",
                              what, got, want, tol);
                k.first = buf;
            }
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

    // ---- lesson constants (documented in normal_mapping_sw.cpp / .slang) ----
    constexpr double k_bump_strength = 0.08;
    constexpr double k_ambient       = 0.08;
    const glm::dvec3 k_light   = glm::normalize(glm::dvec3(-0.35, 0.5, 0.8));
    const glm::dvec3 k_albedo  = {0.85, 0.87, 0.90};

    // Procedural sine-bump normal map, analytic derivatives (the documented
    // texel formula both twins pin). 16x16, REPEAT.
    void bump_texel(int x, int y, double out[3])
    {
        constexpr double k_two_pi = 6.283185307179586;
        x = ((x % 16) + 16) % 16; // REPEAT wrap
        y = ((y % 16) + 16) % 16;
        const double u = (double(x) + 0.5) / 16.0;
        const double v = (double(y) + 0.5) / 16.0;
        const double dhdu = 4.0 * k_two_pi * 0.5 * std::cos(2.0 * k_two_pi * u) * std::cos(2.0 * k_two_pi * v);
        const double dhdv = -4.0 * k_two_pi * 0.5 * std::sin(2.0 * k_two_pi * u) * std::sin(2.0 * k_two_pi * v);
        const glm::dvec3 n = glm::normalize(glm::dvec3(-k_bump_strength * dhdu, -k_bump_strength * dhdv, 1.0));
        // encoded bytes (decode is byte/255*2-1): encode = n*127.5 + 127.5
        out[0] = std::floor(n.x * 127.5 + 127.5);
        out[1] = std::floor(n.y * 127.5 + 127.5);
        out[2] = std::floor(n.z * 127.5 + 127.5);
    }

    glm::dvec2 uv_at(int px, int py) // full-viewport quad pair, uv tiles 0..4
    {
        const double u = px + 0.5 < 320.0 ? (px + 0.5) / 160.0 : 2.0 + (px + 0.5 - 320.0) / 160.0;
        const double v = 1.0 - (py + 0.5) / 480.0; // framebuffer top row => uv v = 1
        return {u, v};
    }

    // Independent Lambert shading of one mapped pixel: decoded bump normal,
    // tangent frame from the geometric normal, shade, quantize.
    glm::dvec3 expected_mapped_color(int px, int py)
    {
        const glm::dvec2 uv = uv_at(px, py);
        const double fx = uv.x * 16.0 - 0.5;
        const double fy = uv.y * 16.0 - 0.5;
        const int x0 = int(std::floor(fx));
        const int y0 = int(std::floor(fy));
        const double tx = fx - double(x0);
        const double ty = fy - double(y0);
        double t00[3], t10[3], t01[3], t11[3];
        bump_texel(x0, y0, t00);
        bump_texel(x0 + 1, y0, t10);
        bump_texel(x0, y0 + 1, t01);
        bump_texel(x0 + 1, y0 + 1, t11);
        glm::dvec3 n_t{0.0, 0.0, 0.0};
        for (int c = 0; c < 3; ++c)
        {
            const double bytes[4] = {t00[c], t10[c], t01[c], t11[c]};
            const double top = bytes[0] + (bytes[1] - bytes[0]) * tx;
            const double bottom = bytes[2] + (bytes[3] - bytes[2]) * tx;
            // decode: byte/255 -> *2 - 1 (8-bit encoded tangent-space normal)
            n_t[c] = (top + (bottom - top) * ty) / 255.0 * 2.0 - 1.0;
        }
        // tangent frame from the interpolated geometric normal (0,0,1 here)
        const glm::dvec3 n(0.0, 0.0, 1.0);
        const glm::dvec3 t = glm::normalize(glm::cross(glm::dvec3(0.0, 1.0, 0.0), n));
        const glm::dvec3 b = glm::cross(n, t);
        const glm::dvec3 shade = glm::normalize(t * n_t.x + b * n_t.y + n * n_t.z);
        const double diff = glm::max(glm::dot(shade, k_light), 0.0);
        return k_albedo * diff + glm::dvec3(k_ambient);
    }
} // namespace

int main(int argc, char* argv[])
{
    const bool prove_wrong = argc > 1 && std::string(argv[1]) == "--prove-wrong-expected";
    const std::string bin_dir = SHS_KA_BIN_DIR;
    const std::string tmp_png =
        (std::filesystem::temp_directory_path() /
         ("shs_adventures_ka_t1_" + std::to_string(::getpid()) + ".png"))
            .string();

    Ka k;

    if (!run_demo_binary(bin_dir, "t1_normal_mapping_sw", tmp_png))
    {
        std::fprintf(stderr, "ka: failed to run t1_normal_mapping_sw\n");
        return 1;
    }
    std::vector<uint8_t> img;
    int w = 0, h = 0;
    if (!ka_load_png_rgba(tmp_png, img, w, h))
    {
        std::fprintf(stderr, "ka: failed to load %s\n", tmp_png.c_str());
        return 1;
    }

    const double bump = prove_wrong ? 12.0 / 255.0 : 0.0;

    // flat half: geometric normal (0,0,1) => diff = L.z; albedo * diff + ambient
    const double flat_diff = k_light.z;
    const int flat_px[2][2] = {{160, 240}, {80, 120}};
    for (const auto& pxy : flat_px)
    {
        for (int c = 0; c < 3; ++c)
        {
            const double want = (k_albedo[c] * flat_diff + k_ambient) * 255.0 + 0.5 +
                                double(c == 0) * bump * 255.0;
            near(k, "ka08 flat-shading sample",
                 double(pix(img, w, pxy[0], pxy[1], c)), want, 1.0);
        }
    }

    // mapped half: per-pixel independent expectation (no brightness assumption
    // across the halves — each pixel stands on its own analytic value)
    const int mapped_px[4][2] = {{480, 240}, {400, 300}, {560, 200}, {600, 400}};
    for (const auto& pxy : mapped_px)
    {
        const glm::dvec3 want_col = expected_mapped_color(pxy[0], pxy[1]);
        for (int c = 0; c < 3; ++c)
        {
            const double want = want_col[c] * 255.0 + 0.5 + double(c == 0) * bump * 255.0;
            near(k, "ka08 normal-mapped sample",
                 double(pix(img, w, pxy[0], pxy[1], c)), want, 2.0);
        }
    }

    // decoded-normal sanity: regenerated texels must decode back to unit
    // normals with positive z (valid tangent-space bump encoding).
    double tex[3];
    bump_texel(3, 7, tex);
    glm::dvec3 dec((tex[0] / 255.0) * 2.0 - 1.0, (tex[1] / 255.0) * 2.0 - 1.0, (tex[2] / 255.0) * 2.0 - 1.0);
    near(k, "ka08 decoded bump normal is unit length", glm::length(dec), 1.0, 0.02);
    if (dec.z <= 0.0) { ++k.failed; if (k.first.empty()) k.first = "ka08 decoded bump normal z <= 0"; }

    std::filesystem::remove(tmp_png);

    if (prove_wrong)
    {
        if (k.failed > 0)
        {
            std::printf("PROVE OK: corrupted expectation was caught (%d mismatch(es))\n", k.failed);
            return 0;
        }
        std::fprintf(stderr, "PROVE FAILED: corrupted expectation was NOT caught\n");
        return 1;
    }

    if (k.failed != 0)
    {
        std::fprintf(stderr, "t1 08 known-answer checks FAILED: %d mismatch(es); first: %s\n",
                     k.failed, k.first.c_str());
        return 1;
    }
    std::printf("t1 08 known-answer checks PASSED (flat + mapped shading, GPU-free)\n");
    return 0;
}
