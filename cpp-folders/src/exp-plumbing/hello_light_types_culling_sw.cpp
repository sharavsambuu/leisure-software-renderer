#define SDL_MAIN_HANDLED

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/camera/camera_math.hpp>
#include <shs/camera/convention.hpp>
#include <shs/core/context.hpp>
#include <shs/geometry/culling_software.hpp>
#include <shs/geometry/culling_runtime.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/geometry/volumes.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/lighting/light_culling_runtime.hpp>
#include <shs/lighting/light_runtime.hpp>
#include <shs/platform/sdl/sdl_runtime.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/scene/scene_culling.hpp>
#include <shs/scene/scene_elements.hpp>

using namespace shs;

namespace
{
constexpr int kWindowW = 1200;
constexpr int kWindowH = 900;
constexpr int kCanvasW = 1200;
constexpr int kCanvasH = 900;
constexpr int kOccW = 320;
constexpr int kOccH = 240;
constexpr int kLightOccW = 240;
constexpr int kLightOccH = 180;
constexpr uint32_t kMaxLightsPerObject = kLightSelectionCapacity;
constexpr uint32_t kLightBinTileSize = 32u;
constexpr uint32_t kLightClusterDepthSlices = 16u;
constexpr float kCameraNear = 0.1f;
constexpr float kCameraFar = 1000.0f;
constexpr float kAmbientBase = 0.22f;
constexpr float kAmbientHemi = 0.12f;
constexpr bool kLightOcclusionDefault = false;

struct ShapeInstance
{
    SceneShape shape;
    uint32_t mesh_index = 0;
    glm::vec3 color{1.0f};
    glm::vec3 base_pos{0.0f};
    glm::vec3 base_rot{0.0f};
    glm::vec3 angular_vel{0.0f};
    glm::mat4 model{1.0f};
    bool visible = true;
    bool frustum_visible = true;
    bool occluded = false;
    bool animated = true;
};

struct FreeCamera
{
    glm::vec3 pos{0.0f, 14.0f, -28.0f};
    float yaw = glm::half_pi<float>();
    float pitch = -0.25f;
    float move_speed = 20.0f;
    float look_speed = 0.003f;
    static constexpr float kMouseSpikeThreshold = 180.0f;
    static constexpr float kMouseDeltaClamp = 70.0f;

    void update(const PlatformInputState& input, float dt)
    {
        if (input.right_mouse_down || input.left_mouse_down)
        {
            float mdx = input.mouse_dx;
            float mdy = input.mouse_dy;
            // WSL2 relative-mode can produce one-frame spikes.
            if (std::abs(mdx) > kMouseSpikeThreshold || std::abs(mdy) > kMouseSpikeThreshold)
            {
                mdx = 0.0f;
                mdy = 0.0f;
            }
            mdx = std::clamp(mdx, -kMouseDeltaClamp, kMouseDeltaClamp);
            mdy = std::clamp(mdy, -kMouseDeltaClamp, kMouseDeltaClamp);
            yaw -= mdx * look_speed;
            pitch -= mdy * look_speed;
            pitch = std::clamp(pitch, -glm::half_pi<float>() + 0.01f, glm::half_pi<float>() - 0.01f);
        }

        const glm::vec3 fwd = forward_from_yaw_pitch(yaw, pitch);
        const glm::vec3 right = right_from_forward(fwd);
        const glm::vec3 up(0.0f, 1.0f, 0.0f);

        const float speed = move_speed * (input.boost ? 2.0f : 1.0f);
        if (input.forward) pos += fwd * speed * dt;
        if (input.backward) pos -= fwd * speed * dt;
        if (input.left) pos += right * speed * dt;
        if (input.right) pos -= right * speed * dt;
        if (input.ascend) pos += up * speed * dt;
        if (input.descend) pos -= up * speed * dt;
    }

    glm::mat4 get_view_matrix() const
    {
        return look_at_lh(pos, pos + forward_from_yaw_pitch(yaw, pitch), glm::vec3(0.0f, 1.0f, 0.0f));
    }
};

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

uint8_t to_u8(float v)
{
    return static_cast<uint8_t>(std::clamp(v * 255.0f, 0.0f, 255.0f));
}

float pseudo_random01(uint32_t seed)
{
    uint32_t x = seed;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00ffffffu) / static_cast<float>(0x01000000u);
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

void draw_line_rt(RT_ColorLDR& rt, int x0, int y0, int x1, int y1, Color c)
{
    int dx = std::abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    while (true)
    {
        if (x0 >= 0 && x0 < rt.w && y0 >= 0 && y0 < rt.h)
        {
            rt.set_rgba(x0, y0, c.r, c.g, c.b, c.a);
        }
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
    }
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
        (ndc.x + 1.0f) * 0.5f * static_cast<float>(canvas_w),
        (ndc.y + 1.0f) * 0.5f * static_cast<float>(canvas_h));
    out_depth = ndc.z * 0.5f + 0.5f;
    return true;
}

void draw_filled_triangle(
    RT_ColorLDR& rt,
    std::vector<float>& depth_buffer,
    const glm::vec2& p0,
    float z0,
    const glm::vec2& p1,
    float z1,
    const glm::vec2& p2,
    float z2,
    Color c)
{
    const float area = edge_fn(p0, p1, p2);
    if (std::abs(area) <= 1e-6f) return;

    const float min_xf = std::min(p0.x, std::min(p1.x, p2.x));
    const float min_yf = std::min(p0.y, std::min(p1.y, p2.y));
    const float max_xf = std::max(p0.x, std::max(p1.x, p2.x));
    const float max_yf = std::max(p0.y, std::max(p1.y, p2.y));

    const int min_x = std::max(0, static_cast<int>(std::floor(min_xf)));
    const int min_y = std::max(0, static_cast<int>(std::floor(min_yf)));
    const int max_x = std::min(rt.w - 1, static_cast<int>(std::ceil(max_xf)));
    const int max_y = std::min(rt.h - 1, static_cast<int>(std::ceil(max_yf)));
    if (min_x > max_x || min_y > max_y) return;

    const bool ccw = area > 0.0f;
    for (int y = min_y; y <= max_y; ++y)
    {
        for (int x = min_x; x <= max_x; ++x)
        {
            const glm::vec2 p(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
            const float w0 = edge_fn(p1, p2, p);
            const float w1 = edge_fn(p2, p0, p);
            const float w2 = edge_fn(p0, p1, p);
            const bool inside = ccw
                ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
            if (!inside) continue;

            const float iw0 = w0 / area;
            const float iw1 = w1 / area;
            const float iw2 = w2 / area;
            const float depth = iw0 * z0 + iw1 * z1 + iw2 * z2;
            if (depth < 0.0f || depth > 1.0f) continue;

            const size_t di = static_cast<size_t>(y) * static_cast<size_t>(rt.w) + static_cast<size_t>(x);
            if (depth < depth_buffer[di])
            {
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
        projected[i] = glm::ivec2(static_cast<int>(s.x), static_cast<int>(s.y));
    }

    for (size_t i = 0; i + 2 < mesh_local.indices.size(); i += 3)
    {
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

AABB compute_scene_bounds(
    const std::vector<ShapeInstance>& instances,
    const std::vector<AABB>& mesh_local_aabbs,
    bool animated_only)
{
    AABB out{};
    bool any = false;
    for (const ShapeInstance& inst : instances)
    {
        if (animated_only && !inst.animated) continue;
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
        out.minv = glm::vec3(-10.0f);
        out.maxv = glm::vec3(10.0f);
    }

    return out;
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
    const float step = full / static_cast<float>(div);

    mesh.vertices.reserve(static_cast<size_t>(verts_per_row) * static_cast<size_t>(verts_per_row));
    mesh.indices.reserve(static_cast<size_t>(div) * static_cast<size_t>(div) * 6u);

    for (int z = 0; z <= div; ++z)
    {
        for (int x = 0; x <= div; ++x)
        {
            const float px = -half_extent + static_cast<float>(x) * step;
            const float pz = -half_extent + static_cast<float>(z) * step;
            mesh.vertices.push_back(glm::vec3(px, 0.0f, pz));
        }
    }

    const auto idx_of = [verts_per_row](int x, int z) -> uint32_t {
        return static_cast<uint32_t>(z * verts_per_row + x);
    };

    for (int z = 0; z < div; ++z)
    {
        for (int x = 0; x < div; ++x)
        {
            const uint32_t i00 = idx_of(x + 0, z + 0);
            const uint32_t i10 = idx_of(x + 1, z + 0);
            const uint32_t i01 = idx_of(x + 0, z + 1);
            const uint32_t i11 = idx_of(x + 1, z + 1);

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

void draw_mesh_multi_light_transformed(
    RT_ColorLDR& rt,
    std::vector<float>& depth_buffer,
    const DebugMesh& mesh_local,
    const glm::mat4& model,
    const glm::mat4& vp,
    int canvas_w,
    int canvas_h,
    const glm::vec3& camera_pos,
    const glm::vec3& base_color,
    const std::vector<LightInstance>& lights,
    const LightSelection& selection)
{
    for (size_t i = 0; i + 2 < mesh_local.indices.size(); i += 3)
    {
        const glm::vec3 lp0 = mesh_local.vertices[mesh_local.indices[i + 0]];
        const glm::vec3 lp1 = mesh_local.vertices[mesh_local.indices[i + 1]];
        const glm::vec3 lp2 = mesh_local.vertices[mesh_local.indices[i + 2]];

        const glm::vec3 p0 = glm::vec3(model * glm::vec4(lp0, 1.0f));
        const glm::vec3 p1 = glm::vec3(model * glm::vec4(lp1, 1.0f));
        const glm::vec3 p2 = glm::vec3(model * glm::vec4(lp2, 1.0f));

        glm::vec2 s0, s1, s2;
        float z0 = 1.0f;
        float z1 = 1.0f;
        float z2 = 1.0f;
        if (!project_world_to_screen(p0, vp, canvas_w, canvas_h, s0, z0)) continue;
        if (!project_world_to_screen(p1, vp, canvas_w, canvas_h, s1, z1)) continue;
        if (!project_world_to_screen(p2, vp, canvas_w, canvas_h, s2, z2)) continue;

        glm::vec3 n = glm::cross(p2 - p0, p1 - p0);
        const float n2 = glm::dot(n, n);
        if (n2 <= 1e-10f) continue;
        n *= 1.0f / std::sqrt(n2);

        const glm::vec3 centroid = (p0 + p1 + p2) * (1.0f / 3.0f);
        const glm::vec3 V = normalize_or(camera_pos - centroid, glm::vec3(0.0f, 0.0f, 1.0f));

        const float hemi = 0.5f + 0.5f * std::clamp(n.y, -1.0f, 1.0f);
        glm::vec3 lit = base_color * (kAmbientBase + kAmbientHemi * hemi);

        for (uint32_t si = 0; si < selection.count; ++si)
        {
            const uint32_t light_idx = selection.indices[si];
            if (light_idx >= lights.size()) continue;

            const LightInstance& light = lights[light_idx];
            const LightContribution contrib = light.model->sample(light.props, centroid, n, V);
            lit += base_color * contrib.diffuse + contrib.specular;
        }

        lit = glm::clamp(lit, glm::vec3(0.0f), glm::vec3(1.0f));
        const Color c{to_u8(lit.r), to_u8(lit.g), to_u8(lit.b), 255};
        draw_filled_triangle(rt, depth_buffer, s0, z0, s1, z1, s2, z2, c);
    }
}

void sync_instances_to_scene(SceneElementSet& scene, const std::vector<ShapeInstance>& instances)
{
    auto elems = scene.elements();
    for (size_t i = 0; i < instances.size() && i < elems.size(); ++i)
    {
        elems[i].geometry = instances[i].shape;
        elems[i].visible = true;
        elems[i].frustum_visible = true;
        elems[i].occluded = false;
        elems[i].enabled = true;
    }
}

void sync_lights_to_scene(SceneElementSet& scene, const std::vector<LightInstance>& lights)
{
    auto elems = scene.elements();
    for (size_t i = 0; i < lights.size() && i < elems.size(); ++i)
    {
        elems[i].geometry = lights[i].volume;
        elems[i].visible = true;
        elems[i].frustum_visible = true;
        elems[i].occluded = false;
        elems[i].enabled = true;
    }
}

} // namespace

int main()
{
    shs::jolt::init_jolt();

    SdlRuntime runtime{
        WindowDesc{"Light Types + Culling Demo (Software)", kWindowW, kWindowH},
        SurfaceDesc{kCanvasW, kCanvasH}
    };
    if (!runtime.valid()) return 1;

    RT_ColorLDR ldr_rt{kCanvasW, kCanvasH};
    std::vector<uint8_t> rgba8_staging(static_cast<size_t>(kCanvasW) * static_cast<size_t>(kCanvasH) * 4u);
    std::vector<float> depth_buffer(static_cast<size_t>(kCanvasW) * static_cast<size_t>(kCanvasH), 1.0f);
    std::vector<float> occlusion_depth(static_cast<size_t>(kOccW) * static_cast<size_t>(kOccH), 1.0f);
    std::vector<float> light_occlusion_depth(static_cast<size_t>(kLightOccW) * static_cast<size_t>(kLightOccH), 1.0f);

    std::vector<ShapeInstance> instances{};
    std::vector<DebugMesh> mesh_library{};
    std::vector<AABB> mesh_local_aabbs{};

    {
        ShapeInstance floor{};
        floor.shape.shape = jolt::make_box(glm::vec3(90.0f, 0.1f, 90.0f));
        floor.base_pos = glm::vec3(0.0f, -0.2f, 0.0f);
        floor.base_rot = glm::vec3(0.0f);
        floor.model = compose_model(floor.base_pos, floor.base_rot);
        floor.shape.transform = jolt::to_jph(floor.model);
        floor.shape.stable_id = 9000;
        floor.color = glm::vec3(0.44f, 0.44f, 0.46f);
        floor.animated = false;

        floor.mesh_index = static_cast<uint32_t>(mesh_library.size());
        mesh_library.push_back(make_tessellated_floor_mesh(90.0f, 80));
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

    uint32_t next_shape_id = 1;
    const int layer_count = 2;
    const int rows_per_layer = 6;
    const int cols_per_row = 8;
    const float col_spacing_x = 5.1f;
    const float row_spacing_z = 4.8f;
    const float layer_spacing_z = 19.0f;
    const float base_y = 1.3f;
    const float layer_y_step = 0.85f;

    for (int layer = 0; layer < layer_count; ++layer)
    {
        const float layer_z = (-0.5f * static_cast<float>(layer_count - 1) + static_cast<float>(layer)) * layer_spacing_z;
        for (int row = 0; row < rows_per_layer; ++row)
        {
            const float row_z = layer_z + (-0.5f * static_cast<float>(rows_per_layer - 1) + static_cast<float>(row)) * row_spacing_z;
            const float zig = (((row + layer) & 1) != 0) ? (0.44f * col_spacing_x) : 0.0f;
            for (int col = 0; col < cols_per_row; ++col)
            {
                const uint32_t logical_idx =
                    static_cast<uint32_t>(layer * rows_per_layer * cols_per_row + row * cols_per_row + col);
                const DemoShapeKind kind = shape_kinds[(logical_idx * 7u + 3u) % shape_kinds.size()];
                const float scale = 0.58f + 0.90f * pseudo_random01(logical_idx * 1664525u + 1013904223u);

                ShapeInstance inst{};
                inst.shape.shape = make_scaled_demo_shape(kind, scale);
                inst.mesh_index = static_cast<uint32_t>(mesh_library.size());
                mesh_library.push_back(debug_mesh_from_shape(*inst.shape.shape, JPH::Mat44::sIdentity()));
                mesh_local_aabbs.push_back(compute_local_aabb_from_debug_mesh(mesh_library.back()));

                inst.base_pos = glm::vec3(
                    (-0.5f * static_cast<float>(cols_per_row - 1) + static_cast<float>(col)) * col_spacing_x + zig,
                    base_y + layer_y_step * static_cast<float>(layer) + 0.22f * static_cast<float>(col % 3),
                    row_z);
                inst.base_rot = glm::vec3(
                    0.21f * pseudo_random01(logical_idx * 279470273u + 1u),
                    0.35f * pseudo_random01(logical_idx * 2246822519u + 7u),
                    0.19f * pseudo_random01(logical_idx * 3266489917u + 11u));
                inst.angular_vel = glm::vec3(
                    0.16f + 0.20f * pseudo_random01(logical_idx * 747796405u + 13u),
                    0.14f + 0.24f * pseudo_random01(logical_idx * 2891336453u + 17u),
                    0.12f + 0.18f * pseudo_random01(logical_idx * 1181783497u + 19u));
                inst.model = compose_model(inst.base_pos, inst.base_rot);
                inst.shape.transform = jolt::to_jph(inst.model);
                inst.shape.stable_id = next_shape_id++;
                inst.color = color_for_demo_shape_kind(kind);
                inst.animated = true;
                instances.push_back(std::move(inst));
            }
        }
    }

    const uint32_t unit_aabb_mesh_index = static_cast<uint32_t>(mesh_library.size());
    {
        AABB unit{};
        unit.minv = glm::vec3(-0.5f);
        unit.maxv = glm::vec3(0.5f);
        mesh_library.push_back(debug_mesh_from_aabb(unit));
        mesh_local_aabbs.push_back(compute_local_aabb_from_debug_mesh(mesh_library.back()));
    }

    const AABB dynamic_scene_bounds = compute_scene_bounds(instances, mesh_local_aabbs, true);
    const glm::vec3 dynamic_center = dynamic_scene_bounds.center();
    const glm::vec3 dynamic_extent = glm::max(dynamic_scene_bounds.extent(), glm::vec3(10.0f));

    const PointLightModel point_model{};
    const SpotLightModel spot_model{};
    const RectAreaLightModel rect_model{};
    const TubeAreaLightModel tube_model{};

    std::vector<LightInstance> lights{};
    std::vector<DebugMesh> light_mesh_library{};
    uint32_t next_light_id = 50000;

    const std::array<const ILightModel*, 4> light_models = {
        &point_model,
        &spot_model,
        &rect_model,
        &tube_model
    };
    const std::array<glm::vec3, 10> light_palette = {
        glm::vec3(0.98f, 0.45f, 0.50f),
        glm::vec3(0.45f, 0.82f, 1.00f),
        glm::vec3(0.55f, 1.00f, 0.60f),
        glm::vec3(1.00f, 0.85f, 0.48f),
        glm::vec3(0.92f, 0.52f, 1.00f),
        glm::vec3(1.00f, 0.62f, 0.40f),
        glm::vec3(0.62f, 0.78f, 1.00f),
        glm::vec3(0.90f, 1.00f, 0.60f),
        glm::vec3(1.00f, 0.58f, 0.78f),
        glm::vec3(0.60f, 0.98f, 0.96f)
    };

    const uint32_t lights_per_type = 5;
    for (uint32_t type_i = 0; type_i < light_models.size(); ++type_i)
    {
        for (uint32_t li = 0; li < lights_per_type; ++li)
        {
            const uint32_t light_index = type_i * lights_per_type + li;
            const float r0 = pseudo_random01(light_index * 747796405u + 13u);
            const float r1 = pseudo_random01(light_index * 2891336453u + 17u);
            const float r2 = pseudo_random01(light_index * 1181783497u + 19u);
            const float r3 = pseudo_random01(light_index * 2246822519u + 23u);
            const float r4 = pseudo_random01(light_index * 3266489917u + 29u);
            const float r5 = pseudo_random01(light_index * 668265263u + 31u);

            LightInstance light{};
            light.model = light_models[type_i];
            light.props.color = light_palette[(light_index * 3u + type_i) % light_palette.size()] * (0.82f + 0.30f * r0);
            light.props.flags = LightFlagsDefault;

            switch (light.model->type())
            {
                case LightType::Point:
                    light.props.range = 8.5f + 5.0f * r1;
                    light.props.intensity = 1.9f + 1.1f * r2;
                    light.props.attenuation_model = LightAttenuationModel::Smooth;
                    light.props.attenuation_power = 1.25f;
                    break;
                case LightType::Spot:
                    light.props.range = 11.0f + 6.0f * r1;
                    light.props.intensity = 2.3f + 1.4f * r2;
                    light.props.inner_angle_rad = glm::radians(11.0f + 8.0f * r3);
                    light.props.outer_angle_rad = light.props.inner_angle_rad + glm::radians(10.0f + 14.0f * r4);
                    light.props.attenuation_model = LightAttenuationModel::Smooth;
                    light.props.attenuation_power = 1.30f;
                    break;
                case LightType::RectArea:
                    light.props.range = 12.0f + 7.0f * r1;
                    light.props.intensity = 1.6f + 1.0f * r2;
                    light.props.rect_half_extents = glm::vec2(0.65f + 0.65f * r3, 0.35f + 0.45f * r4);
                    light.props.attenuation_model = LightAttenuationModel::InverseSquare;
                    light.props.attenuation_bias = 0.16f;
                    light.props.attenuation_power = 1.0f;
                    break;
                case LightType::TubeArea:
                    light.props.range = 11.0f + 6.0f * r1;
                    light.props.intensity = 1.7f + 1.1f * r2;
                    light.props.tube_half_length = 0.95f + 1.00f * r3;
                    light.props.tube_radius = 0.20f + 0.25f * r4;
                    light.props.attenuation_model = LightAttenuationModel::InverseSquare;
                    light.props.attenuation_bias = 0.14f;
                    light.props.attenuation_power = 1.0f;
                    break;
                default:
                    break;
            }

            light.motion.orbit_center = dynamic_center + glm::vec3(
                (r0 - 0.5f) * dynamic_extent.x * 0.75f,
                2.4f + 3.4f * r1,
                (r2 - 0.5f) * dynamic_extent.z * 0.75f);
            light.motion.aim_center = dynamic_center + glm::vec3(
                (r3 - 0.5f) * dynamic_extent.x * 0.35f,
                1.2f + 1.2f * r4,
                (r5 - 0.5f) * dynamic_extent.z * 0.35f);
            light.motion.orbit_axis = normalize_or(glm::vec3(r2 - 0.5f, 1.0f, r3 - 0.5f), glm::vec3(0.0f, 1.0f, 0.0f));
            light.motion.radial_axis = normalize_or(glm::vec3(r4 - 0.5f, 0.2f * (r0 - 0.5f), r5 - 0.5f), glm::vec3(1.0f, 0.0f, 0.0f));
            light.motion.orbit_radius = 3.8f + 7.2f * r4;
            light.motion.orbit_speed = 0.22f + 0.88f * r5;
            light.motion.orbit_phase = glm::two_pi<float>() * r3;
            light.motion.vertical_amplitude = 0.4f + 1.6f * r2;
            light.motion.vertical_speed = 0.8f + 1.4f * r1;
            light.motion.direction_lead = 0.18f + 0.54f * r0;
            light.motion.vertical_aim_bias = -0.08f - 0.18f * r5;

            update_light_motion(light, 0.0f);
            light.volume_model = light.model->volume_model_matrix(light.props);
            light.volume.shape = light.model->create_volume_shape(light.props);
            light.volume.transform = jolt::to_jph(light.volume_model);
            light.volume.stable_id = next_light_id++;
            light.packed = light.model->pack_for_culling(light.props);

            light.mesh_index = static_cast<uint32_t>(light_mesh_library.size());
            light_mesh_library.push_back(debug_mesh_from_shape(*light.volume.shape, JPH::Mat44::sIdentity()));
            lights.push_back(std::move(light));
        }
    }

    SceneElementSet view_cull_scene{};
    view_cull_scene.reserve(instances.size());
    for (size_t i = 0; i < instances.size(); ++i)
    {
        SceneElement elem{};
        elem.geometry = instances[i].shape;
        elem.user_index = static_cast<uint32_t>(i);
        elem.visible = true;
        elem.frustum_visible = true;
        elem.occluded = false;
        elem.enabled = true;
        view_cull_scene.add(std::move(elem));
    }

    SceneElementSet light_cull_scene{};
    light_cull_scene.reserve(lights.size());
    for (size_t i = 0; i < lights.size(); ++i)
    {
        SceneElement elem{};
        elem.geometry = lights[i].volume;
        elem.user_index = static_cast<uint32_t>(i);
        elem.visible = true;
        elem.frustum_visible = true;
        elem.occluded = false;
        elem.enabled = true;
        light_cull_scene.add(std::move(elem));
    }

    SceneCullingContext view_cull_ctx{};
    SceneCullingContext light_cull_ctx{};

    FreeCamera camera{};
    bool show_aabb_debug = false;
    bool render_lit_surfaces = true;
    bool draw_light_volumes = true;
    bool enable_scene_occlusion = true;
    bool enable_light_occlusion = kLightOcclusionDefault;
    bool freeze_lights = false;
    LightCullingMode light_culling_mode = LightCullingMode::Clustered;
    LightObjectCullMode light_object_cull_mode = LightObjectCullMode::VolumeAabb;

    bool mouse_drag_held = false;
    std::printf(
        "Controls: LMB/RMB drag look, WASD+QE move, Shift boost | "
        "L lit/debug, B AABB, F1 light volumes, F2 scene occlusion, F3 light occlusion, F4 light/object culling, F5 freeze lights, F6 light bin mode\n");

    auto start_time = std::chrono::steady_clock::now();
    auto last_time = start_time;

    while (true)
    {
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - last_time).count();
        const float time_s = std::chrono::duration<float>(now - start_time).count();
        last_time = now;

        PlatformInputState input{};
        if (!runtime.pump_input(input)) break;
        if (input.quit) break;

        if (input.toggle_bot) show_aabb_debug = !show_aabb_debug;
        if (input.toggle_light_shafts) render_lit_surfaces = !render_lit_surfaces;
        if (input.cycle_debug_view) draw_light_volumes = !draw_light_volumes;
        if (input.cycle_cull_mode) enable_scene_occlusion = !enable_scene_occlusion;
        if (input.toggle_front_face) enable_light_occlusion = !enable_light_occlusion;
        if (input.toggle_shading_model) light_object_cull_mode = next_light_object_cull_mode(light_object_cull_mode);
        if (input.toggle_sky_mode) freeze_lights = !freeze_lights;
        if (input.toggle_follow_camera) light_culling_mode = next_light_culling_mode(light_culling_mode);

        const bool look_drag = input.right_mouse_down || input.left_mouse_down;
        if (look_drag != mouse_drag_held)
        {
            mouse_drag_held = look_drag;
            runtime.set_relative_mouse_mode(mouse_drag_held);
            input.mouse_dx = 0.0f;
            input.mouse_dy = 0.0f;
        }

        camera.update(input, dt);

        for (ShapeInstance& inst : instances)
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

        if (!freeze_lights)
        {
            for (LightInstance& light : lights)
            {
                update_light_motion(light, time_s);
            }
        }

        for (LightInstance& light : lights)
        {
            light.volume_model = light.model->volume_model_matrix(light.props);
            light.volume.transform = jolt::to_jph(light.volume_model);
            light.packed = light.model->pack_for_culling(light.props);
            light.visible = true;
            light.frustum_visible = true;
            light.occluded = false;
        }

        sync_instances_to_scene(view_cull_scene, instances);
        sync_lights_to_scene(light_cull_scene, lights);

        const glm::mat4 view = camera.get_view_matrix();
        const glm::mat4 proj = perspective_lh_no(
            glm::radians(60.0f),
            static_cast<float>(kCanvasW) / static_cast<float>(kCanvasH),
            kCameraNear,
            kCameraFar);
        const glm::mat4 vp = proj * view;

        const Frustum frustum = extract_frustum_planes(vp);

        view_cull_ctx.run_frustum(view_cull_scene, frustum);
        view_cull_ctx.run_software_occlusion(
            view_cull_scene,
            enable_scene_occlusion,
            std::span<float>(occlusion_depth.data(), occlusion_depth.size()),
            kOccW,
            kOccH,
            view,
            vp,
            [&](const SceneElement& elem, uint32_t, std::span<float> depth_span) {
                if (elem.user_index >= instances.size()) return;
                const ShapeInstance& inst = instances[elem.user_index];
                if (inst.mesh_index >= mesh_library.size()) return;
                culling_sw::rasterize_mesh_depth_transformed(
                    depth_span,
                    kOccW,
                    kOccH,
                    mesh_library[inst.mesh_index],
                    inst.model,
                    vp);
            });
        (void)view_cull_ctx.apply_frustum_fallback_if_needed(
            view_cull_scene,
            enable_scene_occlusion,
            true,
            0u);

        light_cull_ctx.run_frustum(light_cull_scene, frustum);
        light_cull_ctx.run_software_occlusion(
            light_cull_scene,
            enable_light_occlusion,
            std::span<float>(light_occlusion_depth.data(), light_occlusion_depth.size()),
            kLightOccW,
            kLightOccH,
            view,
            vp,
            [&](const SceneElement& elem, uint32_t, std::span<float> depth_span) {
                if (elem.user_index >= lights.size()) return;
                const LightInstance& light = lights[elem.user_index];
                if (light.mesh_index >= light_mesh_library.size()) return;
                culling_sw::rasterize_mesh_depth_transformed(
                    depth_span,
                    kLightOccW,
                    kLightOccH,
                    light_mesh_library[light.mesh_index],
                    light.volume_model,
                    vp);
            });
        (void)light_cull_ctx.apply_frustum_fallback_if_needed(
            light_cull_scene,
            enable_light_occlusion,
            true,
            0u);

        {
            auto view_elems = view_cull_scene.elements();
            for (size_t i = 0; i < instances.size() && i < view_elems.size(); ++i)
            {
                instances[i].visible = view_elems[i].visible;
                instances[i].frustum_visible = view_elems[i].frustum_visible;
                instances[i].occluded = view_elems[i].occluded;
            }

            auto light_elems = light_cull_scene.elements();
            for (size_t i = 0; i < lights.size() && i < light_elems.size(); ++i)
            {
                lights[i].visible = light_elems[i].visible;
                lights[i].frustum_visible = light_elems[i].frustum_visible;
                lights[i].occluded = light_elems[i].occluded;
            }
        }

        const CullingStats& object_stats = view_cull_ctx.stats();
        const CullingStats& light_stats = light_cull_ctx.stats();

        const std::vector<uint32_t>& visible_scene_indices = view_cull_ctx.visible_indices();
        std::vector<uint32_t> draw_scene_indices = visible_scene_indices;
        CullingStats draw_stats = object_stats;

        if (!view_cull_scene.empty())
        {
            const uint32_t floor_scene_idx = 0u;
            const auto view_elems = view_cull_scene.elements();
            if (floor_scene_idx < view_elems.size() && view_elems[floor_scene_idx].frustum_visible)
            {
                if (std::find(draw_scene_indices.begin(), draw_scene_indices.end(), floor_scene_idx) == draw_scene_indices.end())
                {
                    draw_scene_indices.push_back(floor_scene_idx);
                    draw_stats.visible_count += 1u;
                    if (draw_stats.occluded_count > 0u) draw_stats.occluded_count -= 1u;
                    normalize_culling_stats(draw_stats);
                }
            }
        }

        const std::vector<uint32_t>& visible_light_scene_indices = light_cull_ctx.visible_indices();
        LightBinCullingConfig light_bin_cfg{};
        light_bin_cfg.mode = light_culling_mode;
        light_bin_cfg.tile_size = kLightBinTileSize;
        light_bin_cfg.cluster_depth_slices = kLightClusterDepthSlices;
        light_bin_cfg.z_near = kCameraNear;
        light_bin_cfg.z_far = kCameraFar;

        TileViewDepthRange tile_depth_range{};
        std::span<const float> tile_min_depth{};
        std::span<const float> tile_max_depth{};
        if (light_culling_mode == LightCullingMode::TiledDepthRange)
        {
            tile_depth_range = build_tile_view_depth_range_from_scene(
                std::span<const uint32_t>(draw_scene_indices.data(), draw_scene_indices.size()),
                view_cull_scene,
                view,
                vp,
                static_cast<uint32_t>(kCanvasW),
                static_cast<uint32_t>(kCanvasH),
                kLightBinTileSize,
                kCameraNear,
                kCameraFar);

            if (tile_depth_range.valid())
            {
                tile_min_depth = std::span<const float>(tile_depth_range.min_view_depth.data(), tile_depth_range.min_view_depth.size());
                tile_max_depth = std::span<const float>(tile_depth_range.max_view_depth.data(), tile_depth_range.max_view_depth.size());
            }
        }

        const LightBinCullingData light_bin_data = build_light_bin_culling(
            std::span<const uint32_t>(visible_light_scene_indices.data(), visible_light_scene_indices.size()),
            light_cull_scene,
            vp,
            static_cast<uint32_t>(kCanvasW),
            static_cast<uint32_t>(kCanvasH),
            light_bin_cfg,
            tile_min_depth,
            tile_max_depth);

        ldr_rt.clear({12, 13, 18, 255});
        std::fill(depth_buffer.begin(), depth_buffer.end(), 1.0f);

        uint64_t light_links_total = 0;
        uint32_t max_lights_linked = 0;
        uint64_t light_candidates_total = 0;
        uint32_t max_light_candidates = 0;
        std::vector<uint32_t> light_candidate_scene_scratch{};

        for (const uint32_t scene_idx : draw_scene_indices)
        {
            if (scene_idx >= view_cull_scene.size()) continue;
            const uint32_t obj_idx = view_cull_scene[scene_idx].user_index;
            if (obj_idx >= instances.size()) continue;

            const ShapeInstance& inst = instances[obj_idx];
            if (inst.mesh_index >= mesh_library.size()) continue;

            const AABB world_box = inst.shape.world_aabb();
            const std::span<const uint32_t> candidate_light_scene_indices =
                gather_light_scene_candidates_for_aabb(
                    light_bin_data,
                    world_box,
                    view,
                    vp,
                    light_candidate_scene_scratch);

            light_candidates_total += candidate_light_scene_indices.size();
            max_light_candidates = std::max(max_light_candidates, static_cast<uint32_t>(candidate_light_scene_indices.size()));

            const LightSelection selection = collect_object_lights(
                world_box,
                candidate_light_scene_indices,
                light_cull_scene,
                lights,
                light_object_cull_mode);

            light_links_total += selection.count;
            max_lights_linked = std::max(max_lights_linked, selection.count);

            if (render_lit_surfaces)
            {
                draw_mesh_multi_light_transformed(
                    ldr_rt,
                    depth_buffer,
                    mesh_library[inst.mesh_index],
                    inst.model,
                    vp,
                    kCanvasW,
                    kCanvasH,
                    camera.pos,
                    inst.color,
                    lights,
                    selection);
            }
            else
            {
                const Color shape_color{to_u8(inst.color.r), to_u8(inst.color.g), to_u8(inst.color.b), 255};
                draw_debug_mesh_wireframe_transformed(
                    ldr_rt,
                    mesh_library[inst.mesh_index],
                    inst.model,
                    vp,
                    kCanvasW,
                    kCanvasH,
                    shape_color);
            }

            if (show_aabb_debug && unit_aabb_mesh_index < mesh_library.size())
            {
                const glm::vec3 center = world_box.center();
                const glm::vec3 size = glm::max(world_box.maxv - world_box.minv, glm::vec3(1e-4f));
                const glm::mat4 aabb_model =
                    glm::translate(glm::mat4(1.0f), center) *
                    glm::scale(glm::mat4(1.0f), size);
                draw_debug_mesh_wireframe_transformed(
                    ldr_rt,
                    mesh_library[unit_aabb_mesh_index],
                    aabb_model,
                    vp,
                    kCanvasW,
                    kCanvasH,
                    Color{255, 240, 80, 255});
            }
        }

        if (draw_light_volumes)
        {
            for (const uint32_t light_scene_idx : visible_light_scene_indices)
            {
                if (light_scene_idx >= light_cull_scene.size()) continue;
                const uint32_t light_idx = light_cull_scene[light_scene_idx].user_index;
                if (light_idx >= lights.size()) continue;
                const LightInstance& light = lights[light_idx];
                if (light.mesh_index >= light_mesh_library.size()) continue;

                const glm::vec3 lc = glm::clamp(light.props.color * 1.05f, glm::vec3(0.0f), glm::vec3(1.0f));
                draw_debug_mesh_wireframe_transformed(
                    ldr_rt,
                    light_mesh_library[light.mesh_index],
                    light.volume_model,
                    vp,
                    kCanvasW,
                    kCanvasH,
                    Color{to_u8(lc.r), to_u8(lc.g), to_u8(lc.b), 255});
            }
        }

        for (int y = 0; y < kCanvasH; ++y)
        {
            for (int x = 0; x < kCanvasW; ++x)
            {
                const auto& src = ldr_rt.color.at(x, kCanvasH - 1 - y);
                const size_t di = static_cast<size_t>(y * kCanvasW + x) * 4u;
                rgba8_staging[di + 0] = src.r;
                rgba8_staging[di + 1] = src.g;
                rgba8_staging[di + 2] = src.b;
                rgba8_staging[di + 3] = src.a;
            }
        }

        runtime.upload_rgba8(rgba8_staging.data(), kCanvasW, kCanvasH, kCanvasW * 4);
        runtime.present();

        const float avg_lights_per_obj =
            draw_scene_indices.empty() ? 0.0f : static_cast<float>(light_links_total) / static_cast<float>(draw_scene_indices.size());
        const float avg_candidates_per_obj =
            draw_scene_indices.empty() ? 0.0f : static_cast<float>(light_candidates_total) / static_cast<float>(draw_scene_indices.size());

        char title[560];
        std::snprintf(
            title,
            sizeof(title),
            "Light Types Culling (SW) | Obj F:%u O:%u V:%u | Light F:%u O:%u V:%u | Cand %.2f (max %u) | L/Obj %.2f (max %u) | LMode:%s | LCull:%s | Occ:%s/%s | Vol:%s | %s",
            draw_stats.frustum_visible_count,
            draw_stats.occluded_count,
            draw_stats.visible_count,
            light_stats.frustum_visible_count,
            light_stats.occluded_count,
            light_stats.visible_count,
            avg_candidates_per_obj,
            max_light_candidates,
            avg_lights_per_obj,
            max_lights_linked,
            light_culling_mode_name(light_culling_mode),
            light_object_cull_mode_name(light_object_cull_mode),
            enable_scene_occlusion ? "ON" : "OFF",
            enable_light_occlusion ? "ON" : "OFF",
            draw_light_volumes ? "ON" : "OFF",
            render_lit_surfaces ? "Lit" : "Debug");
        runtime.set_title(title);

        std::printf(
            "Obj F:%u O:%u V:%u | Light F:%u O:%u V:%u | Cand:%4.2f max:%u | L/Obj:%4.2f max:%u | LMode:%s | LCull:%s | Occ:%s/%s | Vol:%s | Mode:%s\r",
            draw_stats.frustum_visible_count,
            draw_stats.occluded_count,
            draw_stats.visible_count,
            light_stats.frustum_visible_count,
            light_stats.occluded_count,
            light_stats.visible_count,
            avg_candidates_per_obj,
            max_light_candidates,
            avg_lights_per_obj,
            max_lights_linked,
            light_culling_mode_name(light_culling_mode),
            light_object_cull_mode_name(light_object_cull_mode),
            enable_scene_occlusion ? "ON " : "OFF",
            enable_light_occlusion ? "ON " : "OFF",
            draw_light_volumes ? "ON " : "OFF",
            render_lit_surfaces ? "Lit  " : "Debug");
        std::fflush(stdout);
    }

    std::printf("\n");
    runtime.set_relative_mouse_mode(false);
    shs::jolt::shutdown_jolt();
    return 0;
}
