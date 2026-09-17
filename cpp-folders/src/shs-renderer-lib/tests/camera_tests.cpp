#include <cmath>
#include <cstdio>

#include "shs/camera/camera.contract.hpp"

// Headless tests for the camera pod (R5a P3.3: builder pins). The identity
// gateway scaffolding was retired (migration step 4.5): the camera pod owns
// pure builders only until real intents arrive (empty command vocabulary
// by law §6.1). Links only shs::renderer-values + glm.
namespace
{
    bool near(float a, float b, float eps = 1e-4f)
    {
        return std::fabs(a - b) <= eps;
    }

    // follow_target: k=1 snaps exactly, k=0 never moves.
    bool test_follow_known_answers()
    {
        shs::CameraRig rig{};
        rig.pos = glm::vec3(0.0f);
        shs::follow_target(rig, glm::vec3(10.0f, 0.0f, 0.0f), glm::vec3(0.0f), 1.0f, 0.016f);
        if (rig.pos != glm::vec3(10.0f, 0.0f, 0.0f)) return false;

        shs::CameraRig rig2{};
        rig2.pos = glm::vec3(1.0f);
        shs::follow_target(rig2, glm::vec3(10.0f, 0.0f, 0.0f), glm::vec3(0.0f), 0.0f, 0.016f);
        return rig2.pos == glm::vec3(1.0f);
    }

    // Light camera: deterministic, looks along -dir, frustum is sane.
    // NOTE: fitted over a sane box. A default (inverted min>max) AABB drives
    // 1e30-scale arithmetic into NaN, which is unobservable even for
    // determinism (NaN != NaN) — degenerate input is caller error, and the
    // builder makes no promise there. An inverted-box assert is R5b work.
    bool test_light_camera_fit()
    {
        shs::AABB box{};
        box.minv = glm::vec3(-5.0f, -2.0f, -5.0f);
        box.maxv = glm::vec3(5.0f, 3.0f, 5.0f);
        const glm::vec3 sun(0.0f, -1.0f, 0.0f);
        const shs::LightCamera a = shs::build_dir_light_camera_aabb(sun, box);
        const shs::LightCamera b = shs::build_dir_light_camera_aabb(sun, box);
        if (a.view != b.view || a.proj != b.proj || a.viewproj != b.viewproj) return false;
        if (a.viewproj != a.proj * a.view) return false;
        return near(a.dir_ws.x, 0.0f) && near(a.dir_ws.y, -1.0f) && near(a.dir_ws.z, 0.0f);
    }

    // View camera chains matrices: viewproj == proj * view, prev preserved.
    bool test_view_chain()
    {
        shs::ViewCamera cam{};
        cam.update_matrices(16.0f / 9.0f);
        if (cam.viewproj != cam.proj * cam.view) return false;
        const glm::mat4 first = cam.viewproj;
        cam.update_matrices(16.0f / 9.0f);
        return cam.prev_viewproj == first && cam.viewproj == first;
    }
} // namespace

int main()
{
    bool ok = true;
    auto run = [&](const char* name, bool result)
    {
        std::fprintf(stderr, "[camera-tests] %s: %s\n", name, result ? "pass" : "FAIL");
        ok = result && ok;
    };

    run("follow_known_answers", test_follow_known_answers());
    run("light_camera_fit", test_light_camera_fit());
    run("view_chain", test_view_chain());

    if (!ok)
    {
        std::fprintf(stderr, "[camera-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[camera-tests] all tests passed\n");
    return 0;
}
