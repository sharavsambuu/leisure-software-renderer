#include <cmath>
#include <cstdio>

#include "shs/geometry/geometry.contract.hpp"

// Headless tests for the geometry pod (R4 P3.4: TBN operator pins). The
// identity gateway scaffolding was retired (migration step 4.5): the geometry
// pod owns pure operators only until real intents arrive (empty command
// vocabulary by law §6.1). Links only shs::renderer-values + glm.
namespace
{
    bool near(const glm::vec3& a, const glm::vec3& b, float eps = 1e-5f)
    {
        return std::fabs(a.x - b.x) <= eps && std::fabs(a.y - b.y) <= eps && std::fabs(a.z - b.z) <= eps;
    }

    // +Z normal -> identity frame, orthonormal.
    bool test_frame_identity()
    {
        const shs::geometry::TangentFrame f = shs::geometry::compute_tangent_frame(glm::vec3(0.0f, 0.0f, 1.0f));
        if (!near(f.tangent, glm::vec3(1.0f, 0.0f, 0.0f))) return false;
        if (!near(f.bitangent, glm::vec3(0.0f, 1.0f, 0.0f))) return false;
        if (!near(f.normal, glm::vec3(0.0f, 0.0f, 1.0f))) return false;
        if (std::fabs(glm::dot(f.tangent, f.bitangent)) > 1e-5f) return false;
        if (std::fabs(glm::dot(f.tangent, f.normal)) > 1e-5f) return false;
        return std::fabs(glm::dot(f.bitangent, f.normal)) <= 1e-5f;
    }

    // Tilted normal -> still orthonormal, normal preserved.
    bool test_frame_tilted()
    {
        const glm::vec3 n = glm::normalize(glm::vec3(0.3f, -0.4f, 0.9f));
        const shs::geometry::TangentFrame f = shs::geometry::compute_tangent_frame(n);
        if (!near(f.normal, n)) return false;
        if (std::fabs(glm::length(f.tangent) - 1.0f) > 1e-5f) return false;
        if (std::fabs(glm::length(f.bitangent) - 1.0f) > 1e-5f) return false;
        return std::fabs(glm::dot(f.tangent, f.bitangent)) <= 1e-5f;
    }

    // Flat texel decodes to +Z and perturbs to the geometric normal.
    bool test_perturb_identity()
    {
        const glm::vec3 n_t = shs::geometry::decode_normal_texel(0.5f, 0.5f, 1.0f);
        if (!near(n_t, glm::vec3(0.0f, 0.0f, 1.0f), 0.01f)) return false;
        const shs::geometry::TangentFrame f = shs::geometry::compute_tangent_frame(glm::vec3(0.0f, 0.0f, 1.0f));
        return near(shs::geometry::perturb_normal(f, n_t), glm::vec3(0.0f, 0.0f, 1.0f), 0.01f);
    }

    // Decode corners are exact (byte 0 -> -1, byte 255 -> +1).
    bool test_decode_corners()
    {
        if (shs::geometry::decode_normal_texel(0.0f, 0.0f, 0.0f) != glm::vec3(-1.0f, -1.0f, -1.0f)) return false;
        return shs::geometry::decode_normal_texel(1.0f, 1.0f, 1.0f) == glm::vec3(1.0f, 1.0f, 1.0f);
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_frame_identity() && ok;
    ok = test_frame_tilted() && ok;
    ok = test_perturb_identity() && ok;
    ok = test_decode_corners() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[geometry-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[geometry-tests] all tests passed\n");
    return 0;
}
