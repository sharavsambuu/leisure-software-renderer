#include <cstdint>
#include <cstdio>

#include <glm/glm.hpp>

#include "shs/render/software/rasterizer.hpp"

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; } } while (false)

// R2 (renderer-lib review 2026-09-18): golden pixel-coverage gate for the
// software rasterizer inner loop. Pins one known triangle's exact coverage
// set (every covered pixel, per-row spans, count, positional checksum)
// against BOTH (a) an independent double-precision analytic oracle evaluated
// at each pixel center and (b) literal golden values recorded when the gate
// was introduced. Any future inner-loop change (R3 tiling included) that
// moves even one covered pixel fails this test loudly.
namespace
{
    shs::resources::MeshData make_screen_triangle()
    {
        // Screen-space triangle A(8,8) B(24,8) C(16,24) on a 32x32 target,
        // expressed through the rasterizer's NDC mapping
        //   n = clip.xy / clip.w ;  s = (n * 0.5 + 0.5) * (dim - 1).
        shs::resources::MeshData mesh{};
        const glm::vec2 scr[3] = {{8.0f, 8.0f}, {24.0f, 8.0f}, {16.0f, 24.0f}};
        for (int i = 0; i < 3; ++i)
        {
            const float nx = scr[i].x / 31.0f * 2.0f - 1.0f;
            const float ny = scr[i].y / 31.0f * 2.0f - 1.0f;
            mesh.positions.push_back({nx, ny, 0.0f});
        }
        mesh.indices = {0u, 1u, 2u};
        return mesh;
    }

    // R1 path exercised too: a concrete (non-erased) program.
    const auto& flat_program()
    {
        using namespace shs::render;
        static const auto program = ShaderProgramFn{
            [](const ShaderVertex& v, const ShaderUniforms&) -> VertexOut {
                VertexOut o{};
                o.clip = glm::vec4(v.position, 1.0f);
                return o;
            },
            [](const FragmentIn&, const ShaderUniforms& u) -> FragmentOut {
                FragmentOut o{};
                o.color = ColorF{u.base_color.r, u.base_color.g, u.base_color.b, 1.0f};
                return o;
            }};
        return program;
    }

    // Independent oracle: exact double-precision barycentric coverage at the
    // pixel centers (no incremental stepping, no float accumulation).
    bool oracle_covers(const glm::dvec2& s0, const glm::dvec2& s1, const glm::dvec2& s2, int x, int y)
    {
        const glm::dvec2 p{(double)x + 0.5, (double)y + 0.5};
        const glm::dvec2 e0 = s1 - s0;
        const glm::dvec2 e1 = s2 - s0;
        const double den = e0.x * e1.y - e0.y * e1.x;
        if (den == 0.0) return false;
        const glm::dvec2 v2 = p - s0;
        const double v = (v2.x * e1.y - e1.x * v2.y) / den;
        const double w = (e0.x * v2.y - v2.x * e0.y) / den;
        const double u = 1.0 - v - w;
        return u >= 0.0 && v >= 0.0 && w >= 0.0;
    }
}

int main()
{
    using namespace shs::render;

    const glm::vec2 scr[3] = {{8.0f, 8.0f}, {24.0f, 8.0f}, {16.0f, 24.0f}};
    const glm::dvec2 scr_d[3] = {{8.0, 8.0}, {24.0, 8.0}, {16.0, 24.0}};

    shs::resources::MeshData mesh = make_screen_triangle();
    RT_ColorHDR target(32, 32, ColorF{0.0f, 0.0f, 0.0f, 0.0f});
    ShaderUniforms uniforms{};
    uniforms.base_color = glm::vec3(1.0f, 0.25f, 0.0f);
    RasterizerConfig cfg{};
    cfg.cull_mode = RasterizerCullMode::None;

    const RasterizerStats stats =
        rasterize_mesh(mesh, flat_program(), uniforms, RasterizerTarget{&target, nullptr});
    CHECK(stats.tri_raster == 1);

    // 1) pixel-for-pixel agreement with the independent oracle -----------
    uint64_t count = 0;
    uint64_t checksum = 0;
    for (int y = 0; y < 32; ++y)
    {
        for (int x = 0; x < 32; ++x)
        {
            const ColorF& c = target.color.at(x, y);
            const bool covered =
                (c.r == 1.0f && c.g == 0.25f && c.b == 0.0f && c.a == 1.0f);
            const bool want = oracle_covers(scr_d[0], scr_d[1], scr_d[2], x, y);
            if (covered != want)
            {
                std::fprintf(stderr, "coverage mismatch at (%d,%d): raster=%d oracle=%d\n",
                    x, y, (int)covered, (int)want);
                return 1;
            }
            if (covered)
            {
                ++count;
                checksum += (uint64_t)x * 97u + (uint64_t)y * 31u;
            }
        }
    }

    // 2) literal golden values (recorded 2026-09-18 with the R2 inner loop):
    //    exact rows/columns of the A(8,8) B(24,8) C(16,24) screen triangle.
    CHECK(count == 128u);
    CHECK(checksum == 243288u);
    struct RowSpan { int y, minx, maxx; };
    static constexpr RowSpan kGoldenRows[] = {
        {8, 8, 23}, {9, 9, 22}, {10, 9, 22}, {11, 10, 21}, {12, 10, 21},
        {13, 11, 20}, {14, 11, 20}, {15, 12, 19}, {16, 12, 19}, {17, 13, 18},
        {18, 13, 18}, {19, 14, 17}, {20, 14, 17}, {21, 15, 16}, {22, 15, 16},
    };
    int row_index = 0;
    for (int y = 0; y < 32; ++y)
    {
        int row_min = -1;
        int row_max = -1;
        for (int x = 0; x < 32; ++x)
        {
            if (target.color.at(x, y).r == 1.0f)
            {
                if (row_min < 0) row_min = x;
                row_max = x;
            }
        }
        if (row_min < 0) continue;
        CHECK(row_index < (int)(sizeof(kGoldenRows) / sizeof(kGoldenRows[0])));
        const RowSpan& expect = kGoldenRows[row_index++];
        CHECK(row_min == expect.minx);
        CHECK(row_max == expect.maxx);
    }
    CHECK(row_index == (int)(sizeof(kGoldenRows) / sizeof(kGoldenRows[0])));

    // Background survived outside the triangle.
    CHECK(target.color.at(1, 1).r == 0.0f);
    CHECK(target.color.at(30, 30).r == 0.0f);

    std::fprintf(stderr, "rasterizer coverage golden: %llu pixels, checksum %llu — OK\n",
        (unsigned long long)count, (unsigned long long)checksum);
    return 0;
}
