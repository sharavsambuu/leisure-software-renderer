#pragma once
/*
    shs-renderer-lib
    Directional light camera (AABB fit) - Shadow map MVP

    ТАЙЛБАР:
    - Одоохондоо scene AABB дээр суурилж light view/proj-ийг тааруулна
    - Дараа нь frustum slice + texel snapping нэмнэ
*/

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <shs/geom/aabb.hpp>

namespace shs {

struct LightCamera {
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::mat4 viewproj{1.0f};
    glm::vec3 pos_ws{0.0f};
    glm::vec3 dir_ws{0.0f, -1.0f, 0.0f};
};

// Scene AABB -> light orthographic fit
inline LightCamera build_dir_light_camera_aabb(
    const glm::vec3& sun_dir_ws_norm,
    const AABB& scene_aabb_ws,
    float extra_margin = 10.0f
) {
    LightCamera lc{};
    lc.dir_ws = glm::normalize(sun_dir_ws_norm);

    // --- Up vector сонголт (sun_dir бараг up-тэй давхцах үед fallback)
    glm::vec3 up = (std::abs(lc.dir_ws.y) > 0.95f) ? glm::vec3(0,0,1) : glm::vec3(0,1,0);

    // Light "position" is arbitrary for ortho; we just need stable view basis.
    const glm::vec3 c = scene_aabb_ws.center();
    const float scene_radius = glm::length(scene_aabb_ws.extent()) + extra_margin;

    // Light placed back along direction
    lc.pos_ws = c - lc.dir_ws * (scene_radius * 2.0f);

    lc.view = glm::lookAt(lc.pos_ws, c, up);

    // Transform 8 corners into light space to fit ortho bounds
    const glm::vec3 mn = scene_aabb_ws.minv;
    const glm::vec3 mx = scene_aabb_ws.maxv;

    glm::vec3 corners[8] = {
        {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z},
        {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z}, {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z},
    };

    float l =  1e30f, r = -1e30f;
    float b =  1e30f, t = -1e30f;
    float n =  1e30f, f = -1e30f;

    for (int i=0;i<8;i++){
        glm::vec4 p_ls = lc.view * glm::vec4(corners[i], 1.0f);
        l = std::min(l, p_ls.x); r = std::max(r, p_ls.x);
        b = std::min(b, p_ls.y); t = std::max(t, p_ls.y);
        n = std::min(n, p_ls.z); f = std::max(f, p_ls.z);
    }

    // Ortho: note z sign depends on convention; here we keep glm lookAt default (RH view)
    // Expand a bit for safety
    const float m = extra_margin;
    l -= m; r += m; b -= m; t += m;
    n -= m; f += m;

    lc.proj = glm::ortho(l, r, b, t, n, f);
    lc.viewproj = lc.proj * lc.view;
    return lc;
}

} // namespace shs

