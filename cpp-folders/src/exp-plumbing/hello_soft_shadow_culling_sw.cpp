#define SDL_MAIN_HANDLED
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/platform/sdl/sdl_runtime.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/geometry/volumes.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/culling_runtime.hpp>
#include <shs/geometry/culling_software.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/scene/scene_culling.hpp>
#include <shs/camera/light_camera.hpp>
#include <shs/lighting/shadow_sample.hpp>
#include <shs/core/context.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/gfx/rt_shadow.hpp>
#include <shs/camera/camera_math.hpp>
#include <shs/camera/convention.hpp>

using namespace shs;

const int WINDOW_W = 1200;
const int WINDOW_H = 900;
const int CANVAS_W = 1200;
const int CANVAS_H = 900;
const int OCC_W = 300;
const int OCC_H = 225;
const int SHADOW_MAP_W = 1024;
const int SHADOW_MAP_H = 1024;
const int SHADOW_OCC_W = 320;
const int SHADOW_OCC_H = 320;
constexpr float kSunHeightLift = 6.0f;
constexpr float kSunOrbitRadiusScale = 0.70f;
constexpr float kSunMinOrbitRadius = 28.0f;
constexpr float kSunMinHeight = 56.0f;
constexpr float kSunSceneTopOffset = 34.0f;
constexpr float kSunTargetLead = 14.0f;
constexpr float kSunTargetDrop = 16.0f;
constexpr float kShadowStrength = 0.92f;
constexpr float kShadowBiasConst = 0.0008f;
constexpr float kShadowBiasSlope = 0.0016f;
constexpr int kShadowPcfRadius = 2;
constexpr float kShadowPcfStep = 1.0f;
constexpr float kShadowRangeScale = 50.0f;
constexpr float kAmbientBase = 0.22f;
constexpr float kAmbientHemi = 0.12f;
const glm::vec3 kFloorBaseColor(0.30f, 0.30f, 0.35f);

struct ShapeInstance {
    SceneShape shape;
    uint32_t mesh_index = 0;
    glm::vec3 color{1.0f};
    glm::vec3 base_pos{0.0f};
    glm::vec3 base_rot{0.0f};
    glm::vec3 angular_vel{0.0f};
    glm::mat4 model{1.0f};
    bool visible = true;
    bool animated = true;
    bool frustum_visible = true;
    bool occluded = false;
    bool casts_shadow = true;
};

struct FreeCamera {
    glm::vec3 pos{0.0f, 14.0f, -28.0f};
    float yaw = glm::half_pi<float>();
    float pitch = -0.25f;
    float move_speed = 20.0f;
    float look_speed = 0.003f;

    void update(const PlatformInputState& input, float dt) {
        if (input.right_mouse_down) {
            yaw -= input.mouse_dx * look_speed;
            pitch -= input.mouse_dy * look_speed;
            pitch = std::clamp(pitch, -glm::half_pi<float>() + 0.01f, glm::half_pi<float>() - 0.01f);
        }

        glm::vec3 fwd = forward_from_yaw_pitch(yaw, pitch);
        glm::vec3 right = right_from_forward(fwd);
        glm::vec3 up(0.0f, 1.0f, 0.0f);

        float speed = move_speed * (input.boost ? 2.0f : 1.0f);
        if (input.forward) pos += fwd * speed * dt;
        if (input.backward) pos -= fwd * speed * dt;
        if (input.left) pos += right * speed * dt;
        if (input.right) pos -= right * speed * dt;
        if (input.ascend) pos += up * speed * dt;
        if (input.descend) pos -= up * speed * dt;
    }

    glm::mat4 get_view_matrix() const {
        return look_at_lh(pos, pos + forward_from_yaw_pitch(yaw, pitch), glm::vec3(0.0f, 1.0f, 0.0f));
    }
};

void draw_line_rt(RT_ColorLDR& rt, int x0, int y0, int x1, int y1, Color c) {
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    while (true) {
        if (x0 >= 0 && x0 < rt.w && y0 >= 0 && y0 < rt.h) {
            rt.set_rgba(x0, y0, c.r, c.g, c.b, c.a);
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

glm::mat4 compose_model(const glm::vec3& pos, const glm::vec3& rot_euler)
{
    glm::mat4 model(1.0f);
    model = glm::translate(model, pos);
    model = glm::rotate(model, rot_euler.x, glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, rot_euler.y, glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, rot_euler.z, glm::vec3(0.0f, 0.0f, 1.0f));
    return model;
}

float edge_fn(const glm::vec2& a, const glm::vec2& b, const glm::vec2& p)
{
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

bool project_world_to_screen(
    const glm::vec3& world,
    const glm::mat4& vp,
    int canvas_w,
    int canvas_h,
    glm::vec2& out_xy,
    float& out_depth)
{
    const glm::vec4 clip = vp * glm::vec4(world, 1.0f);
    if (clip.w <= 0.001f) return false;
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    if (ndc.z < -1.0f || ndc.z > 1.0f) return false;

    out_xy = glm::vec2(
        (ndc.x + 1.0f) * 0.5f * (float)canvas_w,
        (ndc.y + 1.0f) * 0.5f * (float)canvas_h);
    out_depth = ndc.z * 0.5f + 0.5f;
    return true;
}

void draw_filled_triangle(
    RT_ColorLDR& rt,
    std::vector<float>& depth_buffer,
    const glm::vec2& p0, float z0,
    const glm::vec2& p1, float z1,
    const glm::vec2& p2, float z2,
    Color c)
{
    const float area = edge_fn(p0, p1, p2);
    if (std::abs(area) <= 1e-6f) return;

    const float min_xf = std::min(p0.x, std::min(p1.x, p2.x));
    const float min_yf = std::min(p0.y, std::min(p1.y, p2.y));
    const float max_xf = std::max(p0.x, std::max(p1.x, p2.x));
    const float max_yf = std::max(p0.y, std::max(p1.y, p2.y));

    const int min_x = std::max(0, (int)std::floor(min_xf));
    const int min_y = std::max(0, (int)std::floor(min_yf));
    const int max_x = std::min(rt.w - 1, (int)std::ceil(max_xf));
    const int max_y = std::min(rt.h - 1, (int)std::ceil(max_yf));
    if (min_x > max_x || min_y > max_y) return;

    const bool ccw = area > 0.0f;
    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            const glm::vec2 p((float)x + 0.5f, (float)y + 0.5f);
            const float w0 = edge_fn(p1, p2, p);
            const float w1 = edge_fn(p2, p0, p);
            const float w2 = edge_fn(p0, p1, p);
            const bool inside = ccw ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                                    : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
            if (!inside) continue;

            const float iw0 = w0 / area;
            const float iw1 = w1 / area;
            const float iw2 = w2 / area;
            const float depth = iw0 * z0 + iw1 * z1 + iw2 * z2;
            if (depth < 0.0f || depth > 1.0f) continue;

            const size_t di = (size_t)y * (size_t)rt.w + (size_t)x;
            if (depth < depth_buffer[di]) {
                depth_buffer[di] = depth;
                rt.set_rgba(x, y, c.r, c.g, c.b, c.a);
            }
        }
    }
}

void draw_debug_mesh_wireframe_transformed(
    RT_ColorLDR& rt,
    const DebugMesh& mesh_local,
    const glm::mat4& model,
    const glm::mat4& vp,
    int canvas_w,
    int canvas_h,
    Color line_color)
{
    std::vector<glm::ivec2> projected(mesh_local.vertices.size(), glm::ivec2(-1, -1));
    for (size_t i = 0; i < mesh_local.vertices.size(); ++i)
    {
        const glm::vec3 world = glm::vec3(model * glm::vec4(mesh_local.vertices[i], 1.0f));
        glm::vec2 s{};
        float z = 1.0f;
        if (!project_world_to_screen(world, vp, canvas_w, canvas_h, s, z)) continue;
        projected[i] = glm::ivec2((int)s.x, (int)s.y);
    }

    for (size_t i = 0; i + 2 < mesh_local.indices.size(); i += 3) {
        const uint32_t i0 = mesh_local.indices[i + 0];
        const uint32_t i1 = mesh_local.indices[i + 1];
        const uint32_t i2 = mesh_local.indices[i + 2];
        if (i0 >= projected.size() || i1 >= projected.size() || i2 >= projected.size()) continue;

        const glm::ivec2 v0 = projected[i0];
        const glm::ivec2 v1 = projected[i1];
        const glm::ivec2 v2 = projected[i2];

        if (v0.x >= 0 && v1.x >= 0) draw_line_rt(rt, v0.x, v0.y, v1.x, v1.y, line_color);
        if (v1.x >= 0 && v2.x >= 0) draw_line_rt(rt, v1.x, v1.y, v2.x, v2.y, line_color);
        if (v2.x >= 0 && v0.x >= 0) draw_line_rt(rt, v2.x, v2.y, v0.x, v0.y, line_color);
    }
}

void draw_mesh_blinn_phong_shadowed_transformed(
    RT_ColorLDR& rt,
    std::vector<float>& depth_buffer,
    const DebugMesh& mesh_local,
    const glm::mat4& model,
    const glm::mat4& vp,
    int canvas_w,
    int canvas_h,
    const glm::vec3& camera_pos,
    const glm::vec3& sun_dir_to_scene_ws,
    const glm::vec3& base_color,
    const RT_ShadowDepth& shadow_map,
    const ShadowParams& shadow_params)
{
    // SHS convention: sun_dir_to_scene_ws points from light toward scene.
    const glm::vec3 L = glm::normalize(-sun_dir_to_scene_ws);
    for (size_t i = 0; i + 2 < mesh_local.indices.size(); i += 3) {
        const glm::vec3 lp0 = mesh_local.vertices[mesh_local.indices[i + 0]];
        const glm::vec3 lp1 = mesh_local.vertices[mesh_local.indices[i + 1]];
        const glm::vec3 lp2 = mesh_local.vertices[mesh_local.indices[i + 2]];

        const glm::vec3 p0 = glm::vec3(model * glm::vec4(lp0, 1.0f));
        const glm::vec3 p1 = glm::vec3(model * glm::vec4(lp1, 1.0f));
        const glm::vec3 p2 = glm::vec3(model * glm::vec4(lp2, 1.0f));

        glm::vec2 s0, s1, s2;
        float z0 = 1.0f, z1 = 1.0f, z2 = 1.0f;
        if (!project_world_to_screen(p0, vp, canvas_w, canvas_h, s0, z0)) continue;
        if (!project_world_to_screen(p1, vp, canvas_w, canvas_h, s1, z1)) continue;
        if (!project_world_to_screen(p2, vp, canvas_w, canvas_h, s2, z2)) continue;

        // Mesh winding follows LH + clockwise front faces, so flip RH cross order.
        glm::vec3 n = glm::cross(p2 - p0, p1 - p0);
        const float n2 = glm::dot(n, n);
        if (n2 <= 1e-10f) continue;
        n = glm::normalize(n);

        const float area = edge_fn(s0, s1, s2);
        if (std::abs(area) <= 1e-6f) continue;

        const float min_xf = std::min(s0.x, std::min(s1.x, s2.x));
        const float min_yf = std::min(s0.y, std::min(s1.y, s2.y));
        const float max_xf = std::max(s0.x, std::max(s1.x, s2.x));
        const float max_yf = std::max(s0.y, std::max(s1.y, s2.y));

        const int min_x = std::max(0, (int)std::floor(min_xf));
        const int min_y = std::max(0, (int)std::floor(min_yf));
        const int max_x = std::min(rt.w - 1, (int)std::ceil(max_xf));
        const int max_y = std::min(rt.h - 1, (int)std::ceil(max_yf));
        if (min_x > max_x || min_y > max_y) continue;

        const bool ccw = area > 0.0f;
        for (int y = min_y; y <= max_y; ++y)
        {
            for (int x = min_x; x <= max_x; ++x)
            {
                const glm::vec2 p((float)x + 0.5f, (float)y + 0.5f);
                const float w0 = edge_fn(s1, s2, p);
                const float w1 = edge_fn(s2, s0, p);
                const float w2 = edge_fn(s0, s1, p);
                const bool inside = ccw
                    ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                    : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
                if (!inside) continue;

                const float iw0 = w0 / area;
                const float iw1 = w1 / area;
                const float iw2 = w2 / area;
                const float depth = iw0 * z0 + iw1 * z1 + iw2 * z2;
                if (depth < 0.0f || depth > 1.0f) continue;

                const size_t di = (size_t)y * (size_t)rt.w + (size_t)x;
                if (di >= depth_buffer.size()) continue;
                if (depth >= depth_buffer[di]) continue;

                const glm::vec3 world_pos = iw0 * p0 + iw1 * p1 + iw2 * p2;
                const glm::vec3 V = glm::normalize(camera_pos - world_pos);
                const glm::vec3 H = glm::normalize(L + V);
                const float ndotl = std::max(0.0f, glm::dot(n, L));
                const float ndoth = std::max(0.0f, glm::dot(n, H));

                const float hemi = glm::clamp(n.y * 0.5f + 0.5f, 0.0f, 1.0f);
                const float ambient = kAmbientBase + kAmbientHemi * hemi;
                const float shadow_vis_raw = shadow_visibility_dir(shadow_map, shadow_params, world_pos, ndotl);
                const float shadow_vis = glm::mix(1.0f, shadow_vis_raw, kShadowStrength);
                const float diffuse = 0.72f * ndotl * shadow_vis;
                const float specular = (ndotl > 0.0f) ? (0.35f * std::pow(ndoth, 32.0f) * shadow_vis) : 0.0f;

                glm::vec3 lit = base_color * (ambient + diffuse) + glm::vec3(specular);
                lit = glm::clamp(lit, glm::vec3(0.0f), glm::vec3(1.0f));
                depth_buffer[di] = depth;
                rt.set_rgba(
                    x,
                    y,
                    (uint8_t)std::clamp(lit.r * 255.0f, 0.0f, 255.0f),
                    (uint8_t)std::clamp(lit.g * 255.0f, 0.0f, 255.0f),
                    (uint8_t)std::clamp(lit.b * 255.0f, 0.0f, 255.0f),
                    255);
            }
        }
    }
}

void rasterize_shadow_mesh_transformed(
    RT_ShadowDepth& shadow_map,
    const DebugMesh& mesh_local,
    const glm::mat4& model,
    const glm::mat4& light_vp)
{
    if (shadow_map.w <= 0 || shadow_map.h <= 0 || shadow_map.depth.empty()) return;
    std::span<float> shadow_span(shadow_map.depth.data(), shadow_map.depth.size());
    for (size_t i = 0; i + 2 < mesh_local.indices.size(); i += 3)
    {
        const glm::vec3 lp0 = mesh_local.vertices[mesh_local.indices[i + 0]];
        const glm::vec3 lp1 = mesh_local.vertices[mesh_local.indices[i + 1]];
        const glm::vec3 lp2 = mesh_local.vertices[mesh_local.indices[i + 2]];

        const glm::vec3 p0 = glm::vec3(model * glm::vec4(lp0, 1.0f));
        const glm::vec3 p1 = glm::vec3(model * glm::vec4(lp1, 1.0f));
        const glm::vec3 p2 = glm::vec3(model * glm::vec4(lp2, 1.0f));

        glm::vec2 s0, s1, s2;
        float z0 = 1.0f, z1 = 1.0f, z2 = 1.0f;
        if (!culling_sw::project_world_to_screen(p0, light_vp, shadow_map.w, shadow_map.h, s0, z0)) continue;
        if (!culling_sw::project_world_to_screen(p1, light_vp, shadow_map.w, shadow_map.h, s1, z1)) continue;
        if (!culling_sw::project_world_to_screen(p2, light_vp, shadow_map.w, shadow_map.h, s2, z2)) continue;

        culling_sw::rasterize_depth_triangle(
            shadow_span,
            shadow_map.w,
            shadow_map.h,
            s0, z0,
            s1, z1,
            s2, z2);
    }
}

AABB compute_local_aabb_from_debug_mesh(const DebugMesh& mesh)
{
    AABB out{};
    if (mesh.vertices.empty())
    {
        out.minv = glm::vec3(-0.5f);
        out.maxv = glm::vec3(0.5f);
        return out;
    }
    out.minv = mesh.vertices[0];
    out.maxv = mesh.vertices[0];
    for (const glm::vec3& p : mesh.vertices)
    {
        out.expand(p);
    }
    return out;
}

AABB compute_shadow_caster_bounds_shs(
    const std::vector<ShapeInstance>& instances,
    const std::vector<AABB>& mesh_local_aabbs)
{
    AABB out{};
    bool any = false;
    for (const auto& inst : instances)
    {
        if (!inst.casts_shadow) continue;
        if (inst.mesh_index >= mesh_local_aabbs.size()) continue;
        const AABB box = transform_aabb(mesh_local_aabbs[inst.mesh_index], inst.model);
        if (!any)
        {
            out.minv = box.minv;
            out.maxv = box.maxv;
            any = true;
            continue;
        }
        out.expand(box.minv);
        out.expand(box.maxv);
    }
    if (!any)
    {
        out.minv = glm::vec3(-1.0f);
        out.maxv = glm::vec3(1.0f);
    }
    return out;
}

AABB scale_aabb_about_center(const AABB& src, float scale)
{
    const float s = std::max(scale, 1.0f);
    const glm::vec3 c = src.center();
    const glm::vec3 e = src.extent() * s;
    AABB out{};
    out.minv = c - e;
    out.maxv = c + e;
    return out;
}

enum class DemoShapeKind : uint8_t
{
    Sphere = 0,
    Box = 1,
    Capsule = 2,
    Cylinder = 3,
    TaperedCapsule = 4,
    ConvexHull = 5,
    Mesh = 6,
    ConvexFromMesh = 7,
    PointLightVolume = 8,
    SpotLightVolume = 9,
    RectLightVolume = 10,
    TubeLightVolume = 11
};

float pseudo_random01(uint32_t seed)
{
    uint32_t x = seed;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (float)(x & 0x00ffffffu) / (float)0x01000000u;
}

std::vector<glm::vec3> scaled_custom_hull(float s)
{
    return {
        {-0.8f * s, -0.7f * s, -0.4f * s},
        { 0.9f * s, -0.6f * s, -0.5f * s},
        { 1.0f * s,  0.4f * s, -0.1f * s},
        {-0.7f * s,  0.6f * s, -0.2f * s},
        {-0.3f * s, -0.4f * s,  0.9f * s},
        { 0.4f * s,  0.7f * s,  0.8f * s},
    };
}

MeshData scaled_wedge_mesh(float s)
{
    MeshData wedge_mesh{};
    wedge_mesh.positions = {
        {-0.9f * s, -0.6f * s, -0.6f * s},
        { 0.9f * s, -0.6f * s, -0.6f * s},
        { 0.0f * s,  0.8f * s, -0.6f * s},
        {-0.9f * s, -0.6f * s,  0.6f * s},
        { 0.9f * s, -0.6f * s,  0.6f * s},
        { 0.0f * s,  0.8f * s,  0.6f * s},
    };
    wedge_mesh.indices = {
        0, 1, 2,
        5, 4, 3,
        0, 3, 4, 0, 4, 1,
        1, 4, 5, 1, 5, 2,
        2, 5, 3, 2, 3, 0
    };
    return wedge_mesh;
}

glm::vec3 color_for_demo_shape_kind(DemoShapeKind kind)
{
    switch (kind)
    {
        case DemoShapeKind::Sphere: return {0.95f, 0.35f, 0.35f};
        case DemoShapeKind::Box: return {0.35f, 0.90f, 0.45f};
        case DemoShapeKind::Capsule: return {0.35f, 0.55f, 0.95f};
        case DemoShapeKind::Cylinder: return {0.95f, 0.80f, 0.30f};
        case DemoShapeKind::TaperedCapsule: return {0.80f, 0.40f, 0.95f};
        case DemoShapeKind::ConvexHull: return {0.30f, 0.85f, 0.90f};
        case DemoShapeKind::Mesh: return {0.92f, 0.55f, 0.25f};
        case DemoShapeKind::ConvexFromMesh: return {0.55f, 0.95f, 0.55f};
        case DemoShapeKind::PointLightVolume: return {0.95f, 0.45f, 0.65f};
        case DemoShapeKind::SpotLightVolume: return {0.95f, 0.70f, 0.35f};
        case DemoShapeKind::RectLightVolume: return {0.35f, 0.95f, 0.80f};
        case DemoShapeKind::TubeLightVolume: return {0.70f, 0.65f, 0.95f};
    }
    return {0.9f, 0.9f, 0.9f};
}

JPH::ShapeRefC make_scaled_demo_shape(DemoShapeKind kind, float s)
{
    const float ss = std::max(s, 0.25f);
    switch (kind)
    {
        case DemoShapeKind::Sphere:
            return jolt::make_sphere(1.0f * ss);
        case DemoShapeKind::Box:
            return jolt::make_box(glm::vec3(0.9f, 0.7f, 0.6f) * ss);
        case DemoShapeKind::Capsule:
            return jolt::make_capsule(0.9f * ss, 0.45f * ss);
        case DemoShapeKind::Cylinder:
            return jolt::make_cylinder(0.9f * ss, 0.5f * ss);
        case DemoShapeKind::TaperedCapsule:
            return jolt::make_tapered_capsule(0.9f * ss, 0.25f * ss, 0.65f * ss);
        case DemoShapeKind::ConvexHull:
            return jolt::make_convex_hull(scaled_custom_hull(ss));
        case DemoShapeKind::Mesh:
            return jolt::make_mesh_shape(scaled_wedge_mesh(ss));
        case DemoShapeKind::ConvexFromMesh:
            return jolt::make_convex_hull_from_mesh(scaled_wedge_mesh(ss));
        case DemoShapeKind::PointLightVolume:
            return jolt::make_point_light_volume(1.0f * ss);
        case DemoShapeKind::SpotLightVolume:
            return jolt::make_spot_light_volume(1.8f * ss, glm::radians(28.0f), 20);
        case DemoShapeKind::RectLightVolume:
            return jolt::make_rect_area_light_volume(glm::vec2(0.8f, 0.5f) * ss, 2.0f * ss);
        case DemoShapeKind::TubeLightVolume:
            return jolt::make_tube_area_light_volume(0.9f * ss, 0.35f * ss);
    }
    return jolt::make_sphere(1.0f * ss);
}

DebugMesh make_tessellated_floor_mesh(float half_extent, int subdivisions)
{
    DebugMesh mesh{};
    const int div = std::max(1, subdivisions);
    const int verts_per_row = div + 1;
    const float full = std::max(half_extent, 1.0f) * 2.0f;
    const float step = full / (float)div;

    mesh.vertices.reserve((size_t)verts_per_row * (size_t)verts_per_row);
    mesh.indices.reserve((size_t)div * (size_t)div * 6u);

    for (int z = 0; z <= div; ++z)
    {
        for (int x = 0; x <= div; ++x)
        {
            const float px = -half_extent + (float)x * step;
            const float pz = -half_extent + (float)z * step;
            mesh.vertices.push_back(glm::vec3(px, 0.0f, pz));
        }
    }

    const auto idx_of = [verts_per_row](int x, int z) -> uint32_t {
        return (uint32_t)(z * verts_per_row + x);
    };

    for (int z = 0; z < div; ++z)
    {
        for (int x = 0; x < div; ++x)
        {
            const uint32_t i00 = idx_of(x + 0, z + 0);
            const uint32_t i10 = idx_of(x + 1, z + 0);
            const uint32_t i01 = idx_of(x + 0, z + 1);
            const uint32_t i11 = idx_of(x + 1, z + 1);

            // Keep triangle order consistent with draw_mesh_blinn_phong_shadowed_transformed()
            // normal reconstruction: n = cross(p2 - p0, p1 - p0) should point +Y.
            mesh.indices.push_back(i00);
            mesh.indices.push_back(i10);
            mesh.indices.push_back(i11);

            mesh.indices.push_back(i00);
            mesh.indices.push_back(i11);
            mesh.indices.push_back(i01);
        }
    }

    return mesh;
}

int main() {
    shs::jolt::init_jolt();

    SdlRuntime runtime{
        WindowDesc{"Soft Shadow Culling Demo (Software, All Jolt Shapes)", WINDOW_W, WINDOW_H},
        SurfaceDesc{CANVAS_W, CANVAS_H}
    };
    if (!runtime.valid()) return 1;

    RT_ColorLDR ldr_rt{CANVAS_W, CANVAS_H};
    std::vector<uint8_t> rgba8_staging(CANVAS_W * CANVAS_H * 4);
    std::vector<float> depth_buffer((size_t)CANVAS_W * (size_t)CANVAS_H, 1.0f);
    std::vector<float> occlusion_depth((size_t)OCC_W * (size_t)OCC_H, 1.0f);
    std::vector<float> shadow_occlusion_depth((size_t)SHADOW_OCC_W * (size_t)SHADOW_OCC_H, 1.0f);
    RT_ShadowDepth shadow_map(SHADOW_MAP_W, SHADOW_MAP_H);

    std::vector<ShapeInstance> instances;
    std::vector<DebugMesh> mesh_library;
    std::vector<AABB> mesh_local_aabbs;

    // Large floor
    {
        ShapeInstance floor{};
        floor.shape.shape = jolt::make_box(glm::vec3(120.0f, 0.1f, 120.0f));
        floor.base_pos = glm::vec3(0.0f, -0.2f, 0.0f);
        floor.base_rot = glm::vec3(0.0f);
        floor.model = compose_model(floor.base_pos, floor.base_rot);
        floor.shape.transform = jolt::to_jph(floor.model);
        floor.shape.stable_id = 9000;
        floor.color = kFloorBaseColor;
        floor.animated = false;
        floor.casts_shadow = true;

        floor.mesh_index = static_cast<uint32_t>(mesh_library.size());
        mesh_library.push_back(make_tessellated_floor_mesh(120.0f, 96));
        mesh_local_aabbs.push_back(compute_local_aabb_from_debug_mesh(mesh_library.back()));
        instances.push_back(floor);
    }

    const std::array<DemoShapeKind, 12> shape_kinds = {
        DemoShapeKind::Sphere,
        DemoShapeKind::Box,
        DemoShapeKind::Capsule,
        DemoShapeKind::Cylinder,
        DemoShapeKind::TaperedCapsule,
        DemoShapeKind::ConvexHull,
        DemoShapeKind::Mesh,
        DemoShapeKind::ConvexFromMesh,
        DemoShapeKind::PointLightVolume,
        DemoShapeKind::SpotLightVolume,
        DemoShapeKind::RectLightVolume,
        DemoShapeKind::TubeLightVolume
    };

    uint32_t next_id = 1;
    const int layer_count = 3;
    const int rows_per_layer = 8;
    const int cols_per_row = 10;
    const float col_spacing_x = 5.2f;
    const float row_spacing_z = 4.6f;
    const float layer_spacing_z = 24.0f;
    const float base_y = 1.3f;
    const float layer_y_step = 0.9f;

    for (int layer = 0; layer < layer_count; ++layer)
    {
        const float layer_z = (-0.5f * (float)(layer_count - 1) + (float)layer) * layer_spacing_z;
        for (int row = 0; row < rows_per_layer; ++row)
        {
            const float row_z = layer_z + (-0.5f * (float)(rows_per_layer - 1) + (float)row) * row_spacing_z;
            const float zig = (((row + layer) & 1) != 0) ? (0.42f * col_spacing_x) : 0.0f;
            for (int col = 0; col < cols_per_row; ++col)
            {
                const uint32_t logical_idx =
                    (uint32_t)layer * (uint32_t)(rows_per_layer * cols_per_row) +
                    (uint32_t)row * (uint32_t)cols_per_row +
                    (uint32_t)col;
                const DemoShapeKind kind = shape_kinds[(logical_idx * 7u + 3u) % shape_kinds.size()];
                const float scale = 0.58f + 1.02f * pseudo_random01(logical_idx * 1664525u + 1013904223u);

                ShapeInstance inst{};
                inst.shape.shape = make_scaled_demo_shape(kind, scale);
                inst.mesh_index = static_cast<uint32_t>(mesh_library.size());
                mesh_library.push_back(debug_mesh_from_shape(*inst.shape.shape, JPH::Mat44::sIdentity()));
                mesh_local_aabbs.push_back(compute_local_aabb_from_debug_mesh(mesh_library.back()));

                inst.base_pos = glm::vec3(
                    (-0.5f * (float)(cols_per_row - 1) + (float)col) * col_spacing_x + zig,
                    base_y + layer_y_step * (float)layer + 0.22f * (float)(col % 3),
                    row_z);
                inst.base_rot = glm::vec3(
                    0.21f * pseudo_random01(logical_idx * 279470273u + 1u),
                    0.35f * pseudo_random01(logical_idx * 2246822519u + 7u),
                    0.19f * pseudo_random01(logical_idx * 3266489917u + 11u));
                inst.angular_vel = glm::vec3(
                    0.20f + 0.26f * pseudo_random01(logical_idx * 747796405u + 13u),
                    0.18f + 0.24f * pseudo_random01(logical_idx * 2891336453u + 17u),
                    0.16f + 0.21f * pseudo_random01(logical_idx * 1181783497u + 19u));
                inst.model = compose_model(inst.base_pos, inst.base_rot);
                inst.shape.transform = jolt::to_jph(inst.model);
                inst.shape.stable_id = next_id++;
                inst.color = color_for_demo_shape_kind(kind);
                inst.animated = true;
                inst.casts_shadow = true;
                instances.push_back(std::move(inst));
            }
        }
    }

    // Unit AABB mesh for debug draw (scaled per object world AABB).
    const uint32_t unit_aabb_mesh_index = static_cast<uint32_t>(mesh_library.size());
    {
        AABB unit{};
        unit.minv = glm::vec3(-0.5f);
        unit.maxv = glm::vec3(0.5f);
        mesh_library.push_back(debug_mesh_from_aabb(unit));
        mesh_local_aabbs.push_back(compute_local_aabb_from_debug_mesh(mesh_library.back()));
    }

    SceneElementSet view_cull_scene;
    SceneElementSet shadow_cull_scene;
    view_cull_scene.reserve(instances.size());
    shadow_cull_scene.reserve(instances.size());
    for (size_t i = 0; i < instances.size(); ++i)
    {
        SceneElement view_elem{};
        view_elem.geometry = instances[i].shape;
        view_elem.user_index = static_cast<uint32_t>(i);
        view_elem.visible = instances[i].visible;
        view_elem.frustum_visible = instances[i].frustum_visible;
        view_elem.occluded = instances[i].occluded;
        view_elem.casts_shadow = instances[i].casts_shadow;
        view_cull_scene.add(std::move(view_elem));

        SceneElement shadow_elem{};
        shadow_elem.geometry = instances[i].shape;
        shadow_elem.user_index = static_cast<uint32_t>(i);
        shadow_elem.visible = true;
        shadow_elem.frustum_visible = true;
        shadow_elem.occluded = false;
        shadow_elem.casts_shadow = instances[i].casts_shadow;
        shadow_elem.enabled = instances[i].casts_shadow;
        shadow_cull_scene.add(std::move(shadow_elem));
    }
    SceneCullingContext view_cull_ctx{};
    SceneCullingContext shadow_cull_ctx{};

    FreeCamera camera;
    bool show_aabb_debug = false;
    bool render_lit_surfaces = false;
    bool enable_occlusion = true;
    std::printf("Controls: RMB look, WASD+QE move, Shift boost, B toggle AABB, L toggle debug/lit, F2 toggle occlusion\n");

    auto start_time = std::chrono::steady_clock::now();
    auto last_time = start_time;

    while (true) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        const float time_s = std::chrono::duration<float>(now - start_time).count();
        last_time = now;

        PlatformInputState input{};
        if (!runtime.pump_input(input)) break;
        if (input.quit) break;
        if (input.toggle_bot) show_aabb_debug = !show_aabb_debug;
        if (input.toggle_light_shafts) render_lit_surfaces = !render_lit_surfaces;
        if (input.cycle_cull_mode) enable_occlusion = !enable_occlusion;

        camera.update(input, dt);

        for (auto& inst : instances)
        {
            if (inst.animated)
            {
                const glm::vec3 rot = inst.base_rot + inst.angular_vel * time_s;
                inst.model = compose_model(inst.base_pos, rot);
            }
            inst.shape.transform = jolt::to_jph(inst.model);
            inst.visible = true;
            inst.frustum_visible = true;
            inst.occluded = false;
        }

        auto view_elems = view_cull_scene.elements();
        auto shadow_elems = shadow_cull_scene.elements();
        for (size_t i = 0; i < instances.size(); ++i)
        {
            view_elems[i].geometry = instances[i].shape;
            view_elems[i].visible = true;
            view_elems[i].frustum_visible = true;
            view_elems[i].occluded = false;
            view_elems[i].enabled = true;

            shadow_elems[i].geometry = instances[i].shape;
            shadow_elems[i].visible = true;
            shadow_elems[i].frustum_visible = true;
            shadow_elems[i].occluded = false;
            shadow_elems[i].enabled = instances[i].casts_shadow;
        }

        glm::mat4 view = camera.get_view_matrix();
        glm::mat4 proj = perspective_lh_no(glm::radians(60.0f), (float)CANVAS_W / CANVAS_H, 0.1f, 1000.0f);
        glm::mat4 vp = proj * view;
        const AABB caster_bounds = compute_shadow_caster_bounds_shs(instances, mesh_local_aabbs);
        const AABB shadow_bounds = scale_aabb_about_center(caster_bounds, kShadowRangeScale);
        const glm::vec3 scene_center = caster_bounds.center();
        const float scene_radius = std::max(42.0f, glm::length(caster_bounds.extent()) * 1.8f);
        const float orbit_angle = 0.17f * time_s;
        const float sun_orbit_radius = std::max(kSunMinOrbitRadius, scene_radius * kSunOrbitRadiusScale);
        const float sun_height = std::max(kSunMinHeight, caster_bounds.maxv.y + kSunSceneTopOffset) + kSunHeightLift;
        const glm::vec3 sun_pos_ws = scene_center + glm::vec3(
            std::cos(orbit_angle) * sun_orbit_radius,
            sun_height,
            std::sin(orbit_angle) * sun_orbit_radius);
        const glm::vec3 sun_target_ws = scene_center + glm::vec3(
            -std::sin(orbit_angle) * kSunTargetLead,
            -kSunTargetDrop,
            std::cos(orbit_angle) * kSunTargetLead);
        const glm::vec3 sun_dir_to_scene_ws = glm::normalize(sun_target_ws - sun_pos_ws);

        const LightCamera light_cam = build_dir_light_camera_aabb(
            sun_dir_to_scene_ws,
            shadow_bounds,
            8.0f,
            static_cast<uint32_t>(SHADOW_MAP_W));
        const glm::mat4 light_vp = light_cam.viewproj;
        const Frustum light_frustum = extract_frustum_planes(light_vp);

        shadow_cull_ctx.run_frustum(shadow_cull_scene, light_frustum);
        shadow_cull_ctx.run_software_occlusion(
            shadow_cull_scene,
            enable_occlusion,
            std::span<float>(shadow_occlusion_depth.data(), shadow_occlusion_depth.size()),
            SHADOW_OCC_W,
            SHADOW_OCC_H,
            light_cam.view,
            light_vp,
            [&](const SceneElement& elem, uint32_t, std::span<float> depth_span) {
                if (elem.user_index >= instances.size()) return;
                const ShapeInstance& inst = instances[elem.user_index];
                if (!inst.casts_shadow) return;
                if (inst.mesh_index >= mesh_library.size()) return;
                culling_sw::rasterize_mesh_depth_transformed(
                    depth_span,
                    SHADOW_OCC_W,
                    SHADOW_OCC_H,
                    mesh_library[inst.mesh_index],
                    inst.model,
                    light_vp);
            });
        (void)shadow_cull_ctx.apply_frustum_fallback_if_needed(
            shadow_cull_scene,
            enable_occlusion,
            true,
            0u);

        shadow_map.clear(1.0f);
        const std::vector<uint32_t>& visible_shadow_scene_indices = shadow_cull_ctx.visible_indices();
        for (uint32_t shadow_scene_idx : visible_shadow_scene_indices)
        {
            if (shadow_scene_idx >= shadow_cull_scene.size()) continue;
            const uint32_t idx = shadow_cull_scene[shadow_scene_idx].user_index;
            if (idx >= instances.size()) continue;
            const ShapeInstance& inst = instances[idx];
            if (!inst.casts_shadow) continue;
            if (inst.mesh_index >= mesh_library.size()) continue;
            rasterize_shadow_mesh_transformed(
                shadow_map,
                mesh_library[inst.mesh_index],
                inst.model,
                light_vp);
        }

        const Frustum frustum = extract_frustum_planes(vp);
        view_cull_ctx.run_frustum(view_cull_scene, frustum);
        view_cull_ctx.run_software_occlusion(
            view_cull_scene,
            enable_occlusion,
            std::span<float>(occlusion_depth.data(), occlusion_depth.size()),
            OCC_W,
            OCC_H,
            view,
            vp,
            [&](const SceneElement& elem, uint32_t, std::span<float> depth_span) {
                if (elem.user_index >= instances.size()) return;
                const ShapeInstance& inst = instances[elem.user_index];
                if (inst.mesh_index >= mesh_library.size()) return;
                culling_sw::rasterize_mesh_depth_transformed(
                    depth_span,
                    OCC_W,
                    OCC_H,
                    mesh_library[inst.mesh_index],
                    inst.model,
                    vp);
            });
        (void)view_cull_ctx.apply_frustum_fallback_if_needed(
            view_cull_scene,
            enable_occlusion,
            true,
            0u);

        for (size_t i = 0; i < instances.size(); ++i)
        {
            instances[i].visible = view_elems[i].visible;
            instances[i].frustum_visible = view_elems[i].frustum_visible;
            instances[i].occluded = view_elems[i].occluded;
        }

        const CullingStats& stats = view_cull_ctx.stats();
        const CullingStats& shadow_stats = shadow_cull_ctx.stats();
        const std::vector<uint32_t>& visible_scene_indices = view_cull_ctx.visible_indices();
        std::vector<uint32_t> draw_scene_indices = visible_scene_indices;
        CullingStats display_stats = stats;
        if (!view_cull_scene.empty())
        {
            const uint32_t floor_scene_idx = 0u;
            if (floor_scene_idx < view_elems.size() && view_elems[floor_scene_idx].frustum_visible)
            {
                if (std::find(draw_scene_indices.begin(), draw_scene_indices.end(), floor_scene_idx) == draw_scene_indices.end())
                {
                    draw_scene_indices.push_back(floor_scene_idx);
                    display_stats.visible_count += 1u;
                    if (display_stats.occluded_count > 0u) display_stats.occluded_count -= 1u;
                    normalize_culling_stats(display_stats);
                }
            }
        }
        ShadowParams shadow_params{};
        shadow_params.light_viewproj = light_vp;
        shadow_params.bias_const = kShadowBiasConst;
        shadow_params.bias_slope = kShadowBiasSlope;
        shadow_params.pcf_radius = kShadowPcfRadius;
        shadow_params.pcf_step = kShadowPcfStep;

        ldr_rt.clear({12, 13, 18, 255});
        std::fill(depth_buffer.begin(), depth_buffer.end(), 1.0f);

        for (uint32_t scene_idx : draw_scene_indices) {
            if (scene_idx >= view_cull_scene.size()) continue;
            const uint32_t idx = view_cull_scene[scene_idx].user_index;
            if (idx >= instances.size()) continue;
            const auto& inst = instances[idx];
            if (inst.mesh_index >= mesh_library.size()) continue;
            const DebugMesh& shape_mesh = mesh_library[inst.mesh_index];
            const glm::vec3 base_color = inst.color;
            if (render_lit_surfaces) {
                draw_mesh_blinn_phong_shadowed_transformed(
                    ldr_rt,
                    depth_buffer,
                    shape_mesh,
                    inst.model,
                    vp,
                    CANVAS_W,
                    CANVAS_H,
                    camera.pos,
                    sun_dir_to_scene_ws,
                    base_color,
                    shadow_map,
                    shadow_params);
            } else {
                const Color shape_color{
                    (uint8_t)std::clamp(base_color.r * 255.0f, 0.0f, 255.0f),
                    (uint8_t)std::clamp(base_color.g * 255.0f, 0.0f, 255.0f),
                    (uint8_t)std::clamp(base_color.b * 255.0f, 0.0f, 255.0f),
                    255
                };
                draw_debug_mesh_wireframe_transformed(ldr_rt, shape_mesh, inst.model, vp, CANVAS_W, CANVAS_H, shape_color);
            }

            if (show_aabb_debug && unit_aabb_mesh_index < mesh_library.size()) {
                const AABB box = inst.shape.world_aabb();
                const glm::vec3 center = box.center();
                const glm::vec3 size = glm::max(box.maxv - box.minv, glm::vec3(1e-4f));
                const glm::mat4 aabb_model =
                    glm::translate(glm::mat4(1.0f), center) *
                    glm::scale(glm::mat4(1.0f), size);
                draw_debug_mesh_wireframe_transformed(
                    ldr_rt,
                    mesh_library[unit_aabb_mesh_index],
                    aabb_model,
                    vp,
                    CANVAS_W,
                    CANVAS_H,
                    Color{255, 240, 80, 255});
            }
        }

        for (int y = 0; y < CANVAS_H; ++y) {
            for (int x = 0; x < CANVAS_W; ++x) {
                const auto& src = ldr_rt.color.at(x, CANVAS_H - 1 - y);
                size_t di = (size_t)(y * CANVAS_W + x) * 4;
                rgba8_staging[di + 0] = src.r;
                rgba8_staging[di + 1] = src.g;
                rgba8_staging[di + 2] = src.b;
                rgba8_staging[di + 3] = src.a;
            }
        }

        runtime.upload_rgba8(rgba8_staging.data(), CANVAS_W, CANVAS_H, CANVAS_W * 4);
        runtime.present();

        char title[320];
        std::snprintf(
            title,
            sizeof(title),
            "Soft Shadow Culling Demo (SW) | Scene:%u Frustum:%u Occ:%u Vis:%u | Shadow F:%u O:%u V:%u | Occ:%s | Mode:%s | AABB:%s",
            display_stats.scene_count,
            display_stats.frustum_visible_count,
            display_stats.occluded_count,
            display_stats.visible_count,
            shadow_stats.frustum_visible_count,
            shadow_stats.occluded_count,
            shadow_stats.visible_count,
            enable_occlusion ? "ON" : "OFF",
            render_lit_surfaces ? "Lit" : "Debug",
            show_aabb_debug ? "ON" : "OFF");
        runtime.set_title(title);
        std::printf(
            "Scene:%u Frustum:%u Occ:%u Vis:%u | Shadow F:%u O:%u V:%u | Occ:%s | Mode:%s | AABB:%s\r",
            display_stats.scene_count,
            display_stats.frustum_visible_count,
            display_stats.occluded_count,
            display_stats.visible_count,
            shadow_stats.frustum_visible_count,
            shadow_stats.occluded_count,
            shadow_stats.visible_count,
            enable_occlusion ? "ON " : "OFF",
            render_lit_surfaces ? "Lit  " : "Debug",
            show_aabb_debug ? "ON " : "OFF");
        std::fflush(stdout);
    }

    std::printf("\n");
    shs::jolt::shutdown_jolt();
    return 0;
}
