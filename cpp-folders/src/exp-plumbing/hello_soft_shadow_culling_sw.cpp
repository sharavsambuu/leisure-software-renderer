#define SDL_MAIN_HANDLED
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
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
#include <shs/scene/scene_instance.hpp>
#include <shs/camera/light_camera.hpp>
#include <shs/lighting/shadow_sample.hpp>
#include <shs/core/context.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/gfx/rt_shadow.hpp>
#include <shs/pipeline/render_composition_presets.hpp>
#include <shs/pipeline/render_path_executor.hpp>
#include <shs/pipeline/render_technique_presets.hpp>
#include <shs/sw_render/debug_draw.hpp>
#include <shs/input/value_actions.hpp>
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
constexpr bool kShadowOcclusionDefault = false;
constexpr float kSunHeightLift = 2.5f;
constexpr float kSunOrbitRadiusScale = 0.90f;
constexpr float kSunMinOrbitRadius = 28.0f;
constexpr float kSunMinHeight = 24.0f;
constexpr float kSunSceneTopOffset = 14.0f;
constexpr float kSunTargetLead = 18.0f;
constexpr float kSunTargetDrop = 8.0f;
constexpr float kShadowStrength = 1.0f;
constexpr float kShadowBiasConst = 0.00035f;
constexpr float kShadowBiasSlope = 0.0009f;
constexpr int kShadowPcfRadius = 1;
constexpr float kShadowPcfStep = 1.0f;
constexpr float kShadowRangeScale = 1.35f;
const glm::vec3 kSunTint(1.08f, 1.00f, 0.92f);
const glm::vec3 kSkyTint(0.60f, 0.68f, 0.92f);
const glm::vec3 kFloorBaseColor(0.58f, 0.56f, 0.52f);

struct SurfaceLightingParams
{
    float ambient_base = 0.10f;
    float ambient_hemi = 0.07f;
    float diffuse_gain = 1.05f;
    float specular_gain = 0.52f;
    float specular_power = 40.0f;
};

SurfaceLightingParams make_surface_lighting_params(RenderTechniquePreset preset)
{
    SurfaceLightingParams params{};
    if (preset == RenderTechniquePreset::PBR)
    {
        params.ambient_base = 0.16f;
        params.ambient_hemi = 0.11f;
        params.diffuse_gain = 0.92f;
        params.specular_gain = 0.24f;
        params.specular_power = 22.0f;
    }
    return params;
}



struct FreeCamera {
    glm::vec3 pos{0.0f, 14.0f, -28.0f};
    float yaw = glm::half_pi<float>();
    float pitch = -0.25f;
    float move_speed = 20.0f;
    float look_speed = 0.003f;
    static constexpr float kMouseSpikeThreshold = 180.0f;
    static constexpr float kMouseDeltaClamp = 70.0f;

    void update(const PlatformInputState& input, float dt) {
        if (input.right_mouse_down || input.left_mouse_down) {
            float mdx = input.mouse_dx;
            float mdy = input.mouse_dy;
            // WSL2 relative-mode occasionally reports large one-frame spikes.
            if (std::abs(mdx) > kMouseSpikeThreshold || std::abs(mdy) > kMouseSpikeThreshold) {
                mdx = 0.0f;
                mdy = 0.0f;
            }
            mdx = std::clamp(mdx, -kMouseDeltaClamp, kMouseDeltaClamp);
            mdy = std::clamp(mdy, -kMouseDeltaClamp, kMouseDeltaClamp);
            yaw -= mdx * look_speed;
            pitch -= mdy * look_speed;
            pitch = std::clamp(pitch, -glm::half_pi<float>() + 0.01f, glm::half_pi<float>() - 0.01f);
        }

        glm::vec3 fwd = forward_from_yaw_pitch(yaw, pitch);
        glm::vec3 right = right_from_forward(fwd);
        glm::vec3 up(0.0f, 1.0f, 0.0f);

        float speed = move_speed * (input.boost ? 2.0f : 1.0f);
        if (input.forward) pos += fwd * speed * dt;
        if (input.backward) pos -= fwd * speed * dt;
        if (input.left) pos -= right * speed * dt;
        if (input.right) pos += right * speed * dt;
        if (input.ascend) pos += up * speed * dt;
        if (input.descend) pos -= up * speed * dt;
    }

    glm::mat4 get_view_matrix() const {
        return look_at_lh(pos, pos + forward_from_yaw_pitch(yaw, pitch), glm::vec3(0.0f, 1.0f, 0.0f));
    }
};

InputState make_runtime_input_state(const PlatformInputState& in)
{
    InputState out{};
    out.forward = in.forward;
    out.backward = in.backward;
    out.left = in.left;
    out.right = in.right;
    out.ascend = in.ascend;
    out.descend = in.descend;
    out.boost = in.boost;
    out.look_active = in.right_mouse_down || in.left_mouse_down;
    float mdx = in.mouse_dx;
    float mdy = in.mouse_dy;
    if (std::abs(mdx) > FreeCamera::kMouseSpikeThreshold || std::abs(mdy) > FreeCamera::kMouseSpikeThreshold)
    {
        mdx = 0.0f;
        mdy = 0.0f;
    }
    mdx = std::clamp(mdx, -FreeCamera::kMouseDeltaClamp, FreeCamera::kMouseDeltaClamp);
    mdy = std::clamp(mdy, -FreeCamera::kMouseDeltaClamp, FreeCamera::kMouseDeltaClamp);
    out.look_dx = -mdx;
    out.look_dy = mdy;
    out.quit = in.quit;
    return out;
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
    const SurfaceLightingParams& lighting_params,
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
        if (!debug_draw::project_world_to_screen(p0, vp, canvas_w, canvas_h, s0, z0)) continue;
        if (!debug_draw::project_world_to_screen(p1, vp, canvas_w, canvas_h, s1, z1)) continue;
        if (!debug_draw::project_world_to_screen(p2, vp, canvas_w, canvas_h, s2, z2)) continue;

        // Mesh winding follows LH + clockwise front faces, so flip RH cross order.
        glm::vec3 n = glm::cross(p2 - p0, p1 - p0);
        const float n2 = glm::dot(n, n);
        if (n2 <= 1e-10f) continue;
        n = glm::normalize(n);

        const float area = debug_draw::edge_fn(s0, s1, s2);
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
                const float w0 = debug_draw::edge_fn(s1, s2, p);
                const float w1 = debug_draw::edge_fn(s2, s0, p);
                const float w2 = debug_draw::edge_fn(s0, s1, p);
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
                const float ambient = lighting_params.ambient_base + lighting_params.ambient_hemi * hemi;
                const float shadow_vis_raw = shadow_visibility_dir(shadow_map, shadow_params, world_pos, ndotl);
                const float shadow_vis = glm::mix(1.0f, shadow_vis_raw, kShadowStrength);
                const float diffuse = lighting_params.diffuse_gain * ndotl * shadow_vis;
                const float specular = (ndotl > 0.0f)
                    ? (lighting_params.specular_gain * std::pow(ndoth, lighting_params.specular_power) * shadow_vis)
                    : 0.0f;

                glm::vec3 lit = base_color * (ambient * kSkyTint + diffuse * kSunTint) + (specular * kSunTint);
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
    const std::vector<SceneInstance>& instances,
    const std::vector<AABB>& mesh_local_aabbs)
{
    AABB out{};
    bool any = false;
    for (const auto& inst : instances)
    {
        if (!inst.casts_shadow) continue;
        if (inst.user_index >= mesh_local_aabbs.size()) continue;
        const AABB box = transform_aabb(mesh_local_aabbs[inst.user_index], jolt::to_glm(inst.geometry.transform));
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
            return jolt::make_spot_light_volume(1.2f * ss, glm::radians(28.0f), 20);
        case DemoShapeKind::RectLightVolume:
            // For general visualization scaling, use a very small attenuation bound
            // so the shape draws reasonably as a panel rather than a giant cube.
            return jolt::make_rect_area_light_volume(glm::vec2(0.8f, 0.5f) * ss, 0.1f * ss);
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

    Context ctx{};
    auto backend_result = create_render_backend("software");
    std::vector<std::unique_ptr<IRenderBackend>> backend_keepalive{};
    backend_keepalive.reserve(1 + backend_result.auxiliary_backends.size());
    if (backend_result.backend)
    {
        backend_keepalive.push_back(std::move(backend_result.backend));
    }
    for (auto& b : backend_result.auxiliary_backends)
    {
        if (b) backend_keepalive.push_back(std::move(b));
    }
    if (backend_keepalive.empty() || !backend_keepalive[0])
    {
        std::fprintf(stderr, "[render-path][sw] Failed to create software backend.\n");
        shs::jolt::shutdown_jolt();
        return 1;
    }
    ctx.set_primary_backend(backend_keepalive[0].get());
    for (size_t i = 1; i < backend_keepalive.size(); ++i)
    {
        if (backend_keepalive[i]) ctx.register_backend(backend_keepalive[i].get());
    }
    if (!backend_result.note.empty())
    {
        std::fprintf(stderr, "[shs] %s\n", backend_result.note.c_str());
    }

    RenderPathExecutor render_path_executor{};
    RenderPathExecutionPlan active_render_plan{};
    std::string active_render_path_name = "safe_default";
    std::string active_render_mode_name = "safe";
    TechniqueMode active_render_mode = TechniqueMode::Forward;
    bool active_render_path_valid = false;
    RenderTechniquePreset active_render_technique_preset = RenderTechniquePreset::BlinnPhong;
    RenderCompositionRecipe active_composition_recipe = make_builtin_render_composition_recipe(
        RenderPathPreset::Forward,
        active_render_technique_preset,
        "composition_sw");
    std::vector<RenderCompositionRecipe> composition_cycle_order{};
    size_t active_composition_index = 0u;

    RT_ColorLDR ldr_rt{CANVAS_W, CANVAS_H};
    std::vector<uint8_t> rgba8_staging(CANVAS_W * CANVAS_H * 4);
    std::vector<float> depth_buffer((size_t)CANVAS_W * (size_t)CANVAS_H, 1.0f);
    std::vector<float> occlusion_depth((size_t)OCC_W * (size_t)OCC_H, 1.0f);
    std::vector<float> shadow_occlusion_depth((size_t)SHADOW_OCC_W * (size_t)SHADOW_OCC_H, 1.0f);
    RT_ShadowDepth shadow_map(SHADOW_MAP_W, SHADOW_MAP_H);

    std::vector<SceneInstance> instances;
    std::vector<DebugMesh> mesh_library;
    std::vector<AABB> mesh_local_aabbs;

    // Large floor
    {
        SceneInstance floor{};
        floor.geometry.shape = jolt::make_box(glm::vec3(120.0f, 0.1f, 120.0f));
        floor.anim.base_pos = glm::vec3(0.0f, -0.2f, 0.0f);
        floor.anim.base_rot = glm::vec3(0.0f);
        floor.geometry.transform = jolt::to_jph(compose_model(floor.anim.base_pos, floor.anim.base_rot));
        floor.geometry.stable_id = 9000;
        floor.tint_color = kFloorBaseColor;
        floor.anim.animated = false;
        floor.casts_shadow = true;

        floor.user_index = static_cast<uint32_t>(mesh_library.size());
        mesh_library.push_back(debug_mesh_from_shape(*floor.geometry.shape, JPH::Mat44::sIdentity()));
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

    struct ShapeType
    {
        JPH::ShapeRefC shape;
        uint32_t mesh_index;
        glm::vec3 color;
    };
    std::vector<ShapeType> shape_types;
    shape_types.reserve(shape_kinds.size());

    for (size_t i = 0; i < shape_kinds.size(); ++i)
    {
        const DemoShapeKind kind = shape_kinds[i];
        const float scale = 0.58f + 1.02f * pseudo_random01(i * 1664525u + 1013904223u);
        ShapeType st{};
        st.shape = make_scaled_demo_shape(kind, scale);
        st.mesh_index = static_cast<uint32_t>(mesh_library.size());
        mesh_library.push_back(debug_mesh_from_shape(*st.shape, JPH::Mat44::sIdentity()));
        mesh_local_aabbs.push_back(compute_local_aabb_from_debug_mesh(mesh_library.back()));
        st.color = color_for_demo_shape_kind(kind);
        shape_types.push_back(std::move(st));
    }

    const int copies_per_type = 10;
    const float spacing_x = 5.2f;
    const float spacing_z = 4.6f;

    for (size_t t = 0; t < shape_types.size(); ++t) {
        for (int c = 0; c < copies_per_type; ++c) {
            SceneInstance inst{};
            inst.geometry.shape = shape_types[t].shape;
            inst.user_index = shape_types[t].mesh_index;
            inst.anim.base_pos = glm::vec3(
                (-0.5f * (copies_per_type - 1) + (float)c) * spacing_x,
                1.25f + 0.25f * (float)(c % 3),
                (-0.5f * (float)(shape_types.size() - 1) + (float)t) * spacing_z);
            inst.anim.base_rot = glm::vec3(
                0.17f * (float)c,
                0.23f * (float)t,
                0.11f * (float)(c + (int)t));
            inst.anim.angular_vel = glm::vec3(
                0.30f + 0.07f * (float)((c + (int)t) % 5),
                0.42f + 0.06f * (float)(c % 4),
                0.36f + 0.05f * (float)((int)t % 6));
            inst.geometry.transform = jolt::to_jph(compose_model(inst.anim.base_pos, inst.anim.base_rot));
            inst.geometry.stable_id = next_id++;
            inst.tint_color = shape_types[t].color;
            inst.anim.animated = true;
            inst.casts_shadow = true;
            instances.push_back(std::move(inst));
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
        view_elem.geometry = instances[i].geometry;
        view_elem.user_index = static_cast<uint32_t>(i);
        view_elem.visible = instances[i].visible;
        view_elem.frustum_visible = instances[i].frustum_visible;
        view_elem.occluded = instances[i].occluded;
        view_elem.casts_shadow = instances[i].casts_shadow;
        view_cull_scene.add(std::move(view_elem));

        SceneElement shadow_elem{};
        shadow_elem.geometry = instances[i].geometry;
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
    bool render_lit_surfaces = true;
    bool enable_occlusion = true;
    bool enable_shadow_occlusion_culling = kShadowOcclusionDefault;
    bool enable_shadows = true;
    bool mouse_drag_held = false;
    SurfaceLightingParams active_lighting_params = make_surface_lighting_params(active_render_technique_preset);

    const auto plan_has_pass = [](const RenderPathExecutionPlan& plan, PassId pass_id) -> bool
    {
        if (!pass_id_is_standard(pass_id)) return false;
        for (const auto& pass : plan.pass_chain)
        {
            if (pass.pass_id == pass_id || parse_pass_id(pass.id) == pass_id) return true;
        }
        return false;
    };

    const auto reset_culling_history = [&]()
    {
        view_cull_ctx.clear();
        shadow_cull_ctx.clear();

        auto view_elems = view_cull_scene.elements();
        for (auto& elem : view_elems) elem.occluded = false;

        auto shadow_elems = shadow_cull_scene.elements();
        for (auto& elem : shadow_elems) elem.occluded = false;
    };

    const auto apply_render_technique_preset = [&](RenderTechniquePreset preset)
    {
        active_render_technique_preset = preset;
        active_lighting_params = make_surface_lighting_params(active_render_technique_preset);
    };

    const auto apply_safe_runtime_defaults = [&]()
    {
        active_render_plan = RenderPathExecutionPlan{};
        active_render_path_name = "safe_default";
        active_render_mode_name = "safe";
        active_render_mode = TechniqueMode::Forward;
        active_render_path_valid = false;
        show_aabb_debug = false;
        render_lit_surfaces = true;
        enable_occlusion = true;
        enable_shadow_occlusion_culling = kShadowOcclusionDefault;
        enable_shadows = true;
        reset_culling_history();
    };

    const auto apply_render_path_by_index = [&](size_t index) -> bool
    {
        if (!render_path_executor.has_recipes()) return false;

        active_render_path_valid = render_path_executor.apply_index(index, ctx, nullptr);
        const RenderPathRecipe& recipe = render_path_executor.active_recipe();
        const RenderPathExecutionPlan& plan = render_path_executor.active_plan();
        active_render_plan = plan;
        active_render_path_name = recipe.name.empty() ? std::string("unnamed_path") : recipe.name;
        active_render_mode = plan.technique_mode;
        active_render_mode_name = technique_mode_name(plan.technique_mode);

        for (const auto& w : plan.warnings)
        {
            std::fprintf(stderr, "[render-path][sw][warn] %s\n", w.c_str());
        }
        for (const auto& e : plan.errors)
        {
            std::fprintf(stderr, "[render-path][sw][error] %s\n", e.c_str());
        }

        if (!plan.valid)
        {
            return false;
        }

        show_aabb_debug = recipe.runtime_defaults.debug_aabb;
        render_lit_surfaces = recipe.runtime_defaults.lit_mode;
        const bool has_depth_prepass = plan_has_pass(plan, PassId::DepthPrepass);
        enable_occlusion = recipe.runtime_defaults.view_occlusion_enabled && has_depth_prepass;
        enable_shadow_occlusion_culling = recipe.runtime_defaults.shadow_occlusion_enabled && has_depth_prepass;
        enable_shadows = recipe.wants_shadows &&
            recipe.runtime_defaults.enable_shadows &&
            plan_has_pass(plan, PassId::ShadowMap);
        reset_culling_history();

        std::fprintf(stderr, "[render-path][sw] Using recipe '%s' (%s), passes:%zu.\n",
            active_render_path_name.c_str(),
            active_render_mode_name.c_str(),
            plan.pass_chain.size());
        return active_render_path_valid;
    };

    const auto refresh_active_composition_recipe = [&]()
    {
        const RenderPathPreset active_path_preset = render_path_preset_for_mode(active_render_mode);
        active_composition_recipe = make_builtin_render_composition_recipe(
            active_path_preset,
            active_render_technique_preset,
            "composition_sw");
        for (size_t i = 0; i < composition_cycle_order.size(); ++i)
        {
            const auto& c = composition_cycle_order[i];
            if (c.path_preset == active_path_preset &&
                c.technique_preset == active_render_technique_preset)
            {
                active_composition_index = i;
                active_composition_recipe = c;
                break;
            }
        }
    };

    const auto apply_render_composition_by_index = [&](size_t index) -> bool
    {
        if (composition_cycle_order.empty() || !render_path_executor.has_recipes()) return false;
        const size_t resolved = index % composition_cycle_order.size();
        const RenderCompositionRecipe& composition = composition_cycle_order[resolved];
        const size_t path_index = render_path_executor.find_recipe_index_by_mode(
            render_path_preset_mode(composition.path_preset));
        apply_render_technique_preset(composition.technique_preset);
        if (!apply_render_path_by_index(path_index))
        {
            refresh_active_composition_recipe();
            return false;
        }
        active_composition_index = resolved;
        active_composition_recipe = composition;
        return true;
    };

    composition_cycle_order = make_default_render_composition_recipes("composition_sw");

    const bool have_builtin_paths =
        render_path_executor.register_builtin_presets(RenderBackendType::Software, "sw_path");
    if (!have_builtin_paths || !render_path_executor.has_recipes())
    {
        std::fprintf(stderr, "[render-path][sw] Built-in presets unavailable. Switching to safe defaults.\n");
        apply_safe_runtime_defaults();
    }
    else
    {
        const size_t preferred_path_index =
            render_path_executor.find_recipe_index_by_mode(TechniqueMode::Forward);
        if (!apply_render_path_by_index(preferred_path_index))
        {
            std::fprintf(stderr, "[render-path][sw] Initial recipe apply failed. Switching to safe defaults.\n");
            apply_safe_runtime_defaults();
        }
    }
    refresh_active_composition_recipe();

    std::printf("Controls: LMB/RMB drag look, WASD+QE move, Shift boost, B toggle AABB, L toggle debug/lit, F1 toggle occlusion, F2 cycle render path, F3 cycle composition, F4 cycle shading, F5 toggle shadow occlusion\n");

    auto start_time = std::chrono::steady_clock::now();
    auto last_time = start_time;
    RuntimeState runtime_state{};
    runtime_state.camera.pos = camera.pos;
    runtime_state.camera.yaw = camera.yaw;
    runtime_state.camera.pitch = camera.pitch;
    std::vector<RuntimeAction> runtime_actions{};

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
        if (input.cycle_debug_view) enable_occlusion = !enable_occlusion;
        if (input.cycle_cull_mode)
        {
            if (!apply_render_path_by_index(render_path_executor.active_index() + 1u))
            {
                std::fprintf(stderr, "[render-path][sw] Path cycle failed. Switching to safe defaults.\n");
                apply_safe_runtime_defaults();
            }
            refresh_active_composition_recipe();
        }
        if (input.toggle_front_face)
        {
            if (!apply_render_composition_by_index(active_composition_index + 1u))
            {
                std::fprintf(stderr, "[render-path][sw] Composition cycle failed. Switching to safe defaults.\n");
                apply_safe_runtime_defaults();
                refresh_active_composition_recipe();
            }
        }
        if (input.toggle_shading_model)
        {
            apply_render_technique_preset(next_render_technique_preset(active_render_technique_preset));
            refresh_active_composition_recipe();
        }
        if (input.toggle_sky_mode)
        {
            enable_shadow_occlusion_culling = !enable_shadow_occlusion_culling;
            shadow_cull_ctx.clear();
            auto shadow_elems_reset = shadow_cull_scene.elements();
            for (auto& elem : shadow_elems_reset) elem.occluded = false;
        }
        const bool look_drag = input.right_mouse_down || input.left_mouse_down;
        if (look_drag != mouse_drag_held) {
            mouse_drag_held = look_drag;
            runtime.set_relative_mouse_mode(mouse_drag_held);
            input.mouse_dx = 0.0f;
            input.mouse_dy = 0.0f;
        }

        runtime_actions.clear();
        emit_human_actions(
            make_runtime_input_state(input),
            runtime_actions,
            camera.move_speed,
            2.0f,
            camera.look_speed);
        runtime_state = reduce_runtime_state(runtime_state, runtime_actions, dt);
        if (runtime_state.quit_requested) break;
        camera.pos = runtime_state.camera.pos;
        camera.yaw = runtime_state.camera.yaw;
        camera.pitch = runtime_state.camera.pitch;

        for (auto& inst : instances)
        {
            if (inst.anim.animated)
            {
                const glm::vec3 rot = inst.anim.base_rot + inst.anim.angular_vel * time_s;
                inst.geometry.transform = jolt::to_jph(compose_model(inst.anim.base_pos, rot));
            }
            inst.visible = true;
            inst.frustum_visible = true;
            inst.occluded = false;
        }

        auto view_elems = view_cull_scene.elements();
        auto shadow_elems = shadow_cull_scene.elements();
        for (size_t i = 0; i < instances.size(); ++i)
        {
            view_elems[i].geometry = instances[i].geometry;
            view_elems[i].visible = true;
            view_elems[i].frustum_visible = true;
            view_elems[i].occluded = false;
            view_elems[i].enabled = true;

            shadow_elems[i].geometry = instances[i].geometry;
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
        shadow_map.clear(1.0f);
        if (enable_shadows)
        {
            shadow_cull_ctx.run_frustum(shadow_cull_scene, light_frustum);
            const bool enable_shadow_occlusion = enable_occlusion && enable_shadow_occlusion_culling;
            shadow_cull_ctx.run_software_occlusion(
                shadow_cull_scene,
                enable_shadow_occlusion,
                std::span<float>(shadow_occlusion_depth.data(), shadow_occlusion_depth.size()),
                SHADOW_OCC_W,
                SHADOW_OCC_H,
                light_cam.view,
                light_vp,
                [&](const SceneElement& elem, uint32_t, std::span<float> depth_span) {
                    if (elem.user_index >= instances.size()) return;
                    const SceneInstance& inst = instances[elem.user_index];
                    if (!inst.casts_shadow) return;
                    if (inst.user_index >= mesh_library.size()) return;
                    culling_sw::rasterize_mesh_depth_transformed(
                        depth_span,
                        SHADOW_OCC_W,
                        SHADOW_OCC_H,
                        mesh_library[inst.user_index],
                        jolt::to_glm(inst.geometry.transform),
                        light_vp);
                });
            (void)shadow_cull_ctx.apply_frustum_fallback_if_needed(
                shadow_cull_scene,
                enable_shadow_occlusion,
                true,
                0u);

            const std::vector<uint32_t>& visible_shadow_scene_indices = shadow_cull_ctx.visible_indices();
            for (uint32_t shadow_scene_idx : visible_shadow_scene_indices)
            {
                if (shadow_scene_idx >= shadow_cull_scene.size()) continue;
                const uint32_t idx = shadow_cull_scene[shadow_scene_idx].user_index;
                if (idx >= instances.size()) continue;
                const SceneInstance& inst = instances[idx];
                if (!inst.casts_shadow) continue;
                if (inst.user_index >= mesh_library.size()) continue;
                rasterize_shadow_mesh_transformed(
                    shadow_map,
                    mesh_library[inst.user_index],
                    jolt::to_glm(inst.geometry.transform),
                    light_vp);
            }
        }
        else
        {
            shadow_cull_ctx.clear();
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
                const SceneInstance& inst = instances[elem.user_index];
                if (inst.user_index >= mesh_library.size()) return;
                culling_sw::rasterize_mesh_depth_transformed(
                    depth_span,
                    OCC_W,
                    OCC_H,
                    mesh_library[inst.user_index],
                    jolt::to_glm(inst.geometry.transform),
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
            if (inst.user_index >= mesh_library.size()) continue;
            const DebugMesh& shape_mesh = mesh_library[inst.user_index];
            const glm::vec3 base_color = inst.tint_color;
            if (render_lit_surfaces) {
                draw_mesh_blinn_phong_shadowed_transformed(
                    ldr_rt,
                    depth_buffer,
                    shape_mesh,
                    jolt::to_glm(inst.geometry.transform),
                    vp,
                    CANVAS_W,
                    CANVAS_H,
                    camera.pos,
                    sun_dir_to_scene_ws,
                    base_color,
                    active_lighting_params,
                    shadow_map,
                    shadow_params);
            } else {
                const Color shape_color{
                    (uint8_t)std::clamp(base_color.r * 255.0f, 0.0f, 255.0f),
                    (uint8_t)std::clamp(base_color.g * 255.0f, 0.0f, 255.0f),
                    (uint8_t)std::clamp(base_color.b * 255.0f, 0.0f, 255.0f),
                    255
                };
                debug_draw::draw_debug_mesh_wireframe_transformed(ldr_rt, shape_mesh, jolt::to_glm(inst.geometry.transform), vp, CANVAS_W, CANVAS_H, shape_color);
            }

            if (show_aabb_debug && unit_aabb_mesh_index < mesh_library.size()) {
                const AABB box = inst.geometry.world_aabb();
                const glm::vec3 center = box.center();
                const glm::vec3 size = glm::max(box.maxv - box.minv, glm::vec3(1e-4f));
                const glm::mat4 aabb_model =
                    glm::translate(glm::mat4(1.0f), center) *
                    glm::scale(glm::mat4(1.0f), size);
                debug_draw::draw_debug_mesh_wireframe_transformed(
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

        const size_t comp_total = composition_cycle_order.size();
        const size_t comp_slot = comp_total > 0u ? ((active_composition_index % comp_total) + 1u) : 0u;
        char title[320];
        std::snprintf(
            title,
            sizeof(title),
            "Soft Shadow Culling Demo (SW) | Path[F2]:%s (%s/%s,p%zu) | Comp[F3]:%s(%zu/%zu) | Tech[F4]:%s | Scene:%u Frustum:%u Occ:%u Vis:%u | Shadow F:%u O:%u V:%u | Occ[F1]:%s | SOcc[F5]:%s | Shadows:%s | Mode[L]:%s | AABB[B]:%s",
            active_render_path_name.c_str(),
            active_render_mode_name.c_str(),
            active_render_path_valid ? "ok" : "fallback",
            active_render_plan.pass_chain.size(),
            active_composition_recipe.name.c_str(),
            comp_slot,
            comp_total,
            render_technique_preset_name(active_render_technique_preset),
            display_stats.scene_count,
            display_stats.frustum_visible_count,
            display_stats.occluded_count,
            display_stats.visible_count,
            shadow_stats.frustum_visible_count,
            shadow_stats.occluded_count,
            shadow_stats.visible_count,
            enable_occlusion ? "ON" : "OFF",
            enable_shadow_occlusion_culling ? "ON" : "OFF",
            enable_shadows ? "ON" : "OFF",
            render_lit_surfaces ? "Lit" : "Debug",
            show_aabb_debug ? "ON" : "OFF");
        runtime.set_title(title);
        std::printf(
            "Path:%s(%s/%s,p%zu) | Comp:%s(%zu/%zu) | Tech:%s | Scene:%u Frustum:%u Occ:%u Vis:%u | Shadow F:%u O:%u V:%u | Occ:%s | SOcc:%s | Shadows:%s | Mode:%s | AABB:%s\r",
            active_render_path_name.c_str(),
            active_render_mode_name.c_str(),
            active_render_path_valid ? "ok" : "fallback",
            active_render_plan.pass_chain.size(),
            active_composition_recipe.name.c_str(),
            comp_slot,
            comp_total,
            render_technique_preset_name(active_render_technique_preset),
            display_stats.scene_count,
            display_stats.frustum_visible_count,
            display_stats.occluded_count,
            display_stats.visible_count,
            shadow_stats.frustum_visible_count,
            shadow_stats.occluded_count,
            shadow_stats.visible_count,
            enable_occlusion ? "ON " : "OFF",
            enable_shadow_occlusion_culling ? "ON " : "OFF",
            enable_shadows ? "ON " : "OFF",
            render_lit_surfaces ? "Lit  " : "Debug",
            show_aabb_debug ? "ON " : "OFF");
        std::fflush(stdout);
    }

    std::printf("\n");
    runtime.set_relative_mouse_mode(false);
    shs::jolt::shutdown_jolt();
    return 0;
}
