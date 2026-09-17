#if defined(SHS_TEST_LEGACY_INCLUDE_FIRST)
#include "shs/domains/camera/convention.hpp"
#include "shs/camera/convention.hpp"
#else
#include "shs/camera/convention.hpp"
#include "shs/domains/camera/convention.hpp"
#endif

#include <cmath>
#include <cstdio>

namespace
{
    bool near(float actual, float expected)
    {
        return std::fabs(actual - expected) < 1e-5f;
    }
}

int main()
{
    // Independent convention pins: left-handed forward and [-1, 1] depth.
    const glm::mat4 view = shs::camera::look_at_lh(
        glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    bool ok = true;
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            ok = near(view[column][row], column == row ? 1.0f : 0.0f) && ok;

    const glm::mat4 perspective = shs::camera::perspective_lh_no(1.0f, 1.5f, 1.0f, 11.0f);
    const glm::vec4 near_clip = perspective * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    const glm::vec4 far_clip = perspective * glm::vec4(0.0f, 0.0f, 11.0f, 1.0f);
    ok = near(near_clip.z / near_clip.w, -1.0f) &&
         near(far_clip.z / far_clip.w, 1.0f) && ok;

    const glm::mat4 ortho = shs::camera::ortho_lh_no(-2.0f, 2.0f, -3.0f, 3.0f, 1.0f, 11.0f);
    const glm::vec4 corner = ortho * glm::vec4(2.0f, 3.0f, 11.0f, 1.0f);
    ok = near(corner.x, 1.0f) && near(corner.y, 1.0f) && near(corner.z, 1.0f) && ok;
    std::fprintf(stderr, "[camera-include-compatibility] %s\n", ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}
