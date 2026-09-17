#include <cmath>
#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/geometry/geometry.contract.hpp"
#include "shs/geometry/geometry.gateway.hpp"
#include "shs/core/testing/pod_test_kit.hpp"
#include "identity_step_test.hpp"

// Headless tests for the geometry pod (R4 P3.4: TBN operator pins + identity).
// Links only shs::renderer-values + glm.
namespace
{
    auto run_gateway = [](shs::geometry::GeometryState& s,
                             std::span<const shs::geometry::GeometryCommand> a,
                             const shs::geometry::GeometryContext& in,
                             std::pmr::vector<shs::geometry::GeometryEvent>& e)
    {
        shs::geometry::geometry_gateway(s, a, in, e);
    };

    bool near(const glm::vec3& a, const glm::vec3& b, float eps = 1e-5f)
    {
        return std::fabs(a.x - b.x) <= eps && std::fabs(a.y - b.y) <= eps && std::fabs(a.z - b.z) <= eps;
    }

    // +Z normal -> identity frame, orthonormal.
    bool test_frame_identity()
    {
        const shs::TangentFrame f = shs::compute_tangent_frame(glm::vec3(0.0f, 0.0f, 1.0f));
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
        const shs::TangentFrame f = shs::compute_tangent_frame(n);
        if (!near(f.normal, n)) return false;
        if (std::fabs(glm::length(f.tangent) - 1.0f) > 1e-5f) return false;
        if (std::fabs(glm::length(f.bitangent) - 1.0f) > 1e-5f) return false;
        return std::fabs(glm::dot(f.tangent, f.bitangent)) <= 1e-5f;
    }

    // Flat texel decodes to +Z and perturbs to the geometric normal.
    bool test_perturb_identity()
    {
        const glm::vec3 n_t = shs::decode_normal_texel(0.5f, 0.5f, 1.0f);
        if (!near(n_t, glm::vec3(0.0f, 0.0f, 1.0f), 0.01f)) return false;
        const shs::TangentFrame f = shs::compute_tangent_frame(glm::vec3(0.0f, 0.0f, 1.0f));
        return near(shs::perturb_normal(f, n_t), glm::vec3(0.0f, 0.0f, 1.0f), 0.01f);
    }

    // Decode corners are exact (byte 0 -> -1, byte 255 -> +1).
    bool test_decode_corners()
    {
        if (shs::decode_normal_texel(0.0f, 0.0f, 0.0f) != glm::vec3(-1.0f, -1.0f, -1.0f)) return false;
        return shs::decode_normal_texel(1.0f, 1.0f, 1.0f) == glm::vec3(1.0f, 1.0f, 1.0f);
    }

    bool test_identity_stable()
    {
        return shs::pod_test::empty_log_is_stable<shs::geometry::GeometryState,
            shs::geometry::GeometryCommand, shs::geometry::GeometryContext,
            shs::geometry::GeometryEvent>(
            run_gateway, shs::geometry::GeometryState{}, shs::geometry::GeometryContext{});
    }

    bool test_replay_deterministic()
    {
        const shs::geometry::GeometryState s0{};
        const std::vector<shs::geometry::GeometryCommand> none{};
        return shs::pod_test::replay_is_deterministic<shs::geometry::GeometryState,
            shs::geometry::GeometryCommand, shs::geometry::GeometryContext,
            shs::geometry::GeometryEvent>(
            run_gateway, s0,
            std::span<const shs::geometry::GeometryCommand>{none.data(), none.size()},
            shs::geometry::GeometryContext{});
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_frame_identity() && ok;
    ok = test_frame_tilted() && ok;
    ok = test_perturb_identity() && ok;
    ok = test_decode_corners() && ok;
    ok = identity_step_summary<shs::geometry::GeometryState,
        shs::geometry::GeometryCommand, shs::geometry::GeometryContext,
        shs::geometry::GeometryEvent, shs::geometry::GeometryStep>(shs::geometry::geometry_gateway) && ok;
    ok = test_identity_stable() && ok;
    ok = test_replay_deterministic() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[geometry-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[geometry-tests] all tests passed\n");
    return 0;
}
