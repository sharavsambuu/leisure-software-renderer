#include <cmath>
#include <cstdint>
#include <cstdio>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shs/geometry/adapters/jolt/jolt_adapter.hpp"

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; } } while (false)

// Governance review 2026-09-18 todo G3.2: golden-value tests for the math laws.
// Per the G2.3 law-budget norm these tests ARE the named gates:
//   1. Screen/canvas discrete row law (Constitution I, conventions.md §2):
//        row_canvas = (H - 1) - row_screen        (pixel indices)
//        y_canvas   = H - y_screen                (continuous coords)
//      with the rasterizer NDC mapping (ndc*0.5+0.5)*(W-1, H-1) pinned beside
//      them (mixing the two domains is the classic off-by-one the law bans).
//   2. Jolt bridge conjugation (conventions.md §3, jolt_adapter.hpp):
//        to_jph(M) == S·M·S with S = diag(1,1,-1,1)  — and since both
//      directions negate the same slots, the roundtrip is the identity:
//        to_glm(to_jph(M)) == M,  to_jph(to_glm(M)) == M
namespace
{
    // Discrete row law, exactly as the spec states it.
    inline int row_canvas_from_screen(int screen_height, int row_screen)
    {
        return (screen_height - 1) - row_screen;
    }

    // Continuous-coordinate law.
    inline float y_canvas_from_screen(int screen_height, float y_screen)
    {
        return (float)screen_height - y_screen;
    }

    // Rasterizer NDC -> pixel mapping (rasterizer.hpp clip->screen step).
    inline float ndc_to_pixel(float ndc, int dimension)
    {
        return (ndc * 0.5f + 0.5f) * (float)(dimension - 1);
    }

    inline bool feq(float a, float b, float eps = 1e-6f)
    {
        return std::fabs(a - b) <= eps;
    }
}

int main()
{
    // --- 1) screen/canvas discrete row law: golden table (H = 8) -----------
    {
        const int H = 8;
        const int golden[H] = {7, 6, 5, 4, 3, 2, 1, 0};
        for (int row_screen = 0; row_screen < H; ++row_screen)
        {
            CHECK(row_canvas_from_screen(H, row_screen) == golden[row_screen]);
        }
        // Involution: applying the conversion twice is the identity.
        for (int row = 0; row < H; ++row)
        {
            CHECK(row_canvas_from_screen(H, row_canvas_from_screen(H, row)) == row);
        }
        // Degenerate height: a 1-row canvas has exactly one valid row (0).
        CHECK(row_canvas_from_screen(1, 0) == 0);
        // Non-square heights stay involutive.
        for (int H2 : {2, 3, 5, 480, 1080})
        {
            for (int row = 0; row < H2; ++row)
            {
                CHECK(row_canvas_from_screen(H2, row_canvas_from_screen(H2, row)) == row);
            }
        }
    }

    // --- 2) continuous vs discrete domains must never be mixed -------------
    {
        const int H = 480;
        // Continuous: the top edge (y_screen=0) maps to y_canvas = H, OUTSIDE
        // the last row index H-1 — only the discrete law keeps indices in
        // [0, H-1]. Pin the distinction exactly.
        CHECK(y_canvas_from_screen(H, 0.0f) == 480.0f);
        CHECK(y_canvas_from_screen(H, (float)H) == 0.0f);
        // A pixel's CENTER in continuous coords lands on that pixel's canvas
        // row: y_screen center = row + 0.5 => y_canvas = H - row - 0.5, whose
        // floor is exactly (H-1) - row.
        for (int row = 0; row < H; ++row)
        {
            const float center_canvas = y_canvas_from_screen(H, (float)row + 0.5f);
            CHECK(std::floor(center_canvas) == (float)row_canvas_from_screen(H, row));
        }
        // Rasterizer NDC mapping (rasterizer.hpp:308-310) with the y-up LH_NO
        // projection (view_camera.hpp: perspective_lh_no): NDC y=+1 lands on
        // screen row H-1, which is canvas row 0 (top); NDC y=-1 lands on
        // screen row 0, canvas row H-1 (bottom). Screen rows increase upward;
        // canvas rows increase downward — the row law bridges the two.
        const int H2 = 9;
        CHECK((int)ndc_to_pixel(1.0f, H2) == H2 - 1);
        CHECK((int)ndc_to_pixel(-1.0f, H2) == 0);
        CHECK(row_canvas_from_screen(H2, (int)ndc_to_pixel(1.0f, H2)) == 0);
        CHECK(row_canvas_from_screen(H2, (int)ndc_to_pixel(-1.0f, H2)) == H2 - 1);
        // Every canvas row roundtrips: canvas -> screen -> NDC -> screen.
        for (int canvas_row = 0; canvas_row < H2; ++canvas_row)
        {
            const int screen_row = row_canvas_from_screen(H2, canvas_row);
            const float ndc_y = -1.0f + 2.0f * ((float)screen_row / (float)(H2 - 1));
            CHECK((int)ndc_to_pixel(ndc_y, H2) == screen_row);
        }
    }

    // --- 3) Jolt bridge involution: S·M·S with S = diag(1,1,-1,1) ----------
    {
        // Golden spot-check by hand: conjugation by S negates exactly the
        // elements whose row AND column indices touch the Z axis an odd
        // number of times. glm indexing is m[col][row]:
        //   m[0][2], m[1][2], m[3][2] (row 2 of columns 0,1,3) flip;
        //   m[2][0], m[2][1], m[2][3] (column 2, rows 0,1,3) flip;
        //   m[2][2] and everything else keep their sign.
        const glm::mat4 m = glm::mat4(
            glm::vec4( 1.0f,  2.0f,  3.0f,  4.0f),   // column 0
            glm::vec4( 5.0f,  6.0f,  7.0f,  8.0f),   // column 1
            glm::vec4( 9.0f, 10.0f, 11.0f, 12.0f),   // column 2
            glm::vec4(13.0f, 14.0f, 15.0f, 16.0f));  // column 3
        const glm::mat4 S = glm::mat4(
            glm::vec4(1.0f, 0.0f, 0.0f, 0.0f),
            glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
            glm::vec4(0.0f, 0.0f, -1.0f, 0.0f),
            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        const glm::mat4 sms = S * m * S;
        // The bridge roundtrip to_glm(to_jph(m)) is the identity (both sides
        // negate the same slots), so the law to pin against the golden S·M·S
        // is to_jph(m) itself. Read its columns back via GetColumn4.
        const JPH::Mat44 jm = shs::jolt::to_jph(m);
        const glm::mat4 via_bridge{
            glm::vec4(jm.GetColumn4(0).GetX(), jm.GetColumn4(0).GetY(), jm.GetColumn4(0).GetZ(), jm.GetColumn4(0).GetW()),
            glm::vec4(jm.GetColumn4(1).GetX(), jm.GetColumn4(1).GetY(), jm.GetColumn4(1).GetZ(), jm.GetColumn4(1).GetW()),
            glm::vec4(jm.GetColumn4(2).GetX(), jm.GetColumn4(2).GetY(), jm.GetColumn4(2).GetZ(), jm.GetColumn4(2).GetW()),
            glm::vec4(jm.GetColumn4(3).GetX(), jm.GetColumn4(3).GetY(), jm.GetColumn4(3).GetZ(), jm.GetColumn4(3).GetW())};
        for (int c = 0; c < 4; ++c)
        {
            for (int r = 0; r < 4; ++r)
            {
                CHECK(via_bridge[c][r] == sms[c][r]);
            }
        }
        CHECK(via_bridge[0][2] == -3.0f);
        CHECK(via_bridge[2][0] == -9.0f);
        CHECK(via_bridge[2][2] == 11.0f);
        CHECK(via_bridge[2][3] == -12.0f);
        CHECK(via_bridge[3][2] == -15.0f);
        CHECK(via_bridge[0][0] == 1.0f);
        CHECK(via_bridge[3][3] == 16.0f);
    }

    // --- 4) roundtrip involution over heterogeneous transforms -------------
    {
        const glm::mat4 candidates[] = {
            glm::mat4(1.0f),
            glm::translate(glm::mat4(1.0f), glm::vec3(1.5f, -2.0f, 3.25f)),
            glm::rotate(glm::mat4(1.0f), 0.7f, glm::vec3(0.3f, 1.0f, -0.2f)),
            glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 4.0f)),
            glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f))
                * glm::rotate(glm::mat4(1.0f), 1.1f, glm::vec3(0.0f, 1.0f, 0.0f))
                * glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 0.5f)),
        };
        for (const glm::mat4& m : candidates)
        {
            // to_glm(to_jph(M)) == M and to_jph(to_glm(M)) == M — exact float
            // equality is legitimate: the bridge only negates components.
            const glm::mat4 back = shs::jolt::to_glm(shs::jolt::to_jph(m));
            const JPH::Mat44 jph_form = shs::jolt::to_jph(m);
            const JPH::Mat44 forward_back = shs::jolt::to_jph(shs::jolt::to_glm(jph_form));
            for (int c = 0; c < 4; ++c)
            {
                for (int r = 0; r < 4; ++r)
                {
                    CHECK(back[c][r] == m[c][r]);
                }
                CHECK(forward_back.GetColumn4(c) == jph_form.GetColumn4(c));
            }
            // Conjugation is a homomorphism: f(A*B) == f(A)*f(B).
            const glm::mat4 ab = candidates[1] * candidates[3];
            const glm::mat4 f_ab = shs::jolt::to_glm(shs::jolt::to_jph(ab));
            const glm::mat4 f_a_f_b =
                shs::jolt::to_glm(shs::jolt::to_jph(candidates[1]))
                * shs::jolt::to_glm(shs::jolt::to_jph(candidates[3]));
            for (int c = 0; c < 4; ++c)
            {
                for (int r = 0; r < 4; ++r)
                {
                    CHECK(feq(f_ab[c][r], f_a_f_b[c][r]));
                }
            }
        }
        // Quaternion x/y-negation involution (q' = (-qx,-qy,qz,qw), both ways).
        const glm::quat q = glm::quat(0.9f, 0.1f, -0.2f, 0.3f); // (w, x, y, z)
        const JPH::Quat jq = shs::jolt::to_jph(q);
        CHECK(jq.GetX() == -q.x && jq.GetY() == -q.y && jq.GetZ() == q.z && jq.GetW() == q.w);
        const glm::quat qback = shs::jolt::to_glm(jq);
        CHECK(qback.w == q.w && qback.x == q.x && qback.y == q.y && qback.z == q.z);
        const JPH::Quat qforward_back = shs::jolt::to_jph(shs::jolt::to_glm(jq));
        CHECK(qforward_back.GetX() == jq.GetX() && qforward_back.GetY() == jq.GetY()
            && qforward_back.GetZ() == jq.GetZ() && qforward_back.GetW() == jq.GetW());
        // Vec3 z-negation roundtrip.
        const glm::vec3 v{1.0f, -2.0f, 3.0f};
        const JPH::Vec3 jv = shs::jolt::to_jph(v);
        CHECK(jv.GetX() == 1.0f && jv.GetY() == -2.0f && jv.GetZ() == -3.0f);
        CHECK(shs::jolt::to_glm(jv) == v);
    }

    std::printf("math law golden tests: all passed\n");
    return 0;
}
