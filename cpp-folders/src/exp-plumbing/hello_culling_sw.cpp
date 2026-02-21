#define SDL_MAIN_HANDLED
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include <shs/platform/sdl/sdl_runtime.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/geometry/volumes.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/culling_runtime.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/core/context.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/camera/camera_math.hpp>
#include <shs/camera/convention.hpp>

using namespace shs;

const int WINDOW_W = 1200;
const int WINDOW_H = 900;
const int CANVAS_W = 1200;
const int CANVAS_H = 900;
const glm::vec3 kSunLightDirWs = glm::normalize(glm::vec3(0.20f, -1.0f, 0.16f));

struct ShapeInstance {
    SceneShape shape;
    glm::vec3 color;
    glm::vec3 base_pos{0.0f};
    glm::vec3 base_rot{0.0f};
    glm::vec3 angular_vel{0.0f};
    bool visible = true;
    bool animated = true;
};

struct FreeCamera {
    glm::vec3 pos{0.0f, 14.0f, -28.0f};
    float yaw = glm::half_pi<float>(); // Pointing towards +Z
    float pitch = -0.25f;
    float move_speed = 20.0f;
    float look_speed = 0.003f;
    static constexpr float kMouseSpikeThreshold = 240.0f;
    static constexpr float kMouseDeltaClamp = 90.0f;

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
            // Invert yaw delta to match SHS LH (looking right = yaw decrease)
            yaw -= mdx * look_speed;
            pitch -= mdy * look_speed;
            pitch = std::clamp(pitch, -glm::half_pi<float>() + 0.01f, glm::half_pi<float>() - 0.01f);
        }

        glm::vec3 fwd = forward_from_yaw_pitch(yaw, pitch);
        glm::vec3 right = right_from_forward(fwd);
        glm::vec3 up(0, 1, 0);

        float speed = move_speed * (input.boost ? 2.0f : 1.0f);
        if (input.forward) pos += fwd * speed * dt;
        if (input.backward) pos -= fwd * speed * dt;
        if (input.left) pos += right * speed * dt;   // 'right' vector points Left in LH
        if (input.right) pos -= right * speed * dt;  // so subtract to move Right
        if (input.ascend) pos += up * speed * dt;
        if (input.descend) pos -= up * speed * dt;
    }

    glm::mat4 get_view_matrix() const {
        return look_at_lh(pos, pos + forward_from_yaw_pitch(yaw, pitch), glm::vec3(0, 1, 0));
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

void draw_debug_mesh_wireframe(
    RT_ColorLDR& rt,
    const DebugMesh& mesh,
    const glm::mat4& vp,
    int canvas_w,
    int canvas_h,
    Color line_color)
{
    auto project = [&](const glm::vec3& p) -> glm::ivec2 {
        glm::vec4 clip = vp * glm::vec4(p, 1.0f);
        if (clip.w <= 0.001f) return {-1, -1};
        glm::vec3 ndc = glm::vec3(clip) / clip.w;
        if (ndc.z < -1.0f || ndc.z > 1.0f) return {-1, -1};
        return {
            (int)((ndc.x + 1.0f) * 0.5f * (float)canvas_w),
            (int)((ndc.y + 1.0f) * 0.5f * (float)canvas_h)
        };
    };

    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const glm::vec3 p0 = mesh.vertices[mesh.indices[i + 0]];
        const glm::vec3 p1 = mesh.vertices[mesh.indices[i + 1]];
        const glm::vec3 p2 = mesh.vertices[mesh.indices[i + 2]];

        const glm::ivec2 v0 = project(p0);
        const glm::ivec2 v1 = project(p1);
        const glm::ivec2 v2 = project(p2);

        if (v0.x >= 0 && v1.x >= 0) draw_line_rt(rt, v0.x, v0.y, v1.x, v1.y, line_color);
        if (v1.x >= 0 && v2.x >= 0) draw_line_rt(rt, v1.x, v1.y, v2.x, v2.y, line_color);
        if (v2.x >= 0 && v0.x >= 0) draw_line_rt(rt, v2.x, v2.y, v0.x, v0.y, line_color);
    }
}

float edge_fn(const glm::vec2& a, const glm::vec2& b, const glm::vec2& p)
{
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

bool project_clip_to_screen(
    const glm::vec3& p,
    const glm::mat4& vp,
    int canvas_w,
    int canvas_h,
    glm::vec2& out_xy,
    float& out_depth)
{
    const glm::vec4 clip = vp * glm::vec4(p, 1.0f);
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

void draw_mesh_blinn_phong(
    RT_ColorLDR& rt,
    std::vector<float>& depth_buffer,
    const DebugMesh& mesh,
    const glm::mat4& vp,
    int canvas_w,
    int canvas_h,
    const glm::vec3& camera_pos,
    const glm::vec3& light_dir_ws,
    const glm::vec3& base_color)
{
    const glm::vec3 L = glm::normalize(-light_dir_ws);
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const glm::vec3 p0 = mesh.vertices[mesh.indices[i + 0]];
        const glm::vec3 p1 = mesh.vertices[mesh.indices[i + 1]];
        const glm::vec3 p2 = mesh.vertices[mesh.indices[i + 2]];

        glm::vec2 s0, s1, s2;
        float z0 = 1.0f, z1 = 1.0f, z2 = 1.0f;
        if (!project_clip_to_screen(p0, vp, canvas_w, canvas_h, s0, z0)) continue;
        if (!project_clip_to_screen(p1, vp, canvas_w, canvas_h, s1, z1)) continue;
        if (!project_clip_to_screen(p2, vp, canvas_w, canvas_h, s2, z2)) continue;

        // Mesh winding follows LH + clockwise front faces, so flip RH cross order.
        glm::vec3 n = glm::cross(p2 - p0, p1 - p0);
        const float n2 = glm::dot(n, n);
        if (n2 <= 1e-10f) continue;
        n = glm::normalize(n);

        const glm::vec3 centroid = (p0 + p1 + p2) * (1.0f / 3.0f);
        const glm::vec3 V = glm::normalize(camera_pos - centroid);
        const glm::vec3 H = glm::normalize(L + V);

        const float ndotl = std::max(0.0f, glm::dot(n, L));
        const float ndoth = std::max(0.0f, glm::dot(n, H));
        const float ambient = 0.18f;
        const float diffuse = 0.72f * ndotl;
        const float specular = (ndotl > 0.0f) ? (0.35f * std::pow(ndoth, 32.0f)) : 0.0f;

        glm::vec3 lit = base_color * (ambient + diffuse) + glm::vec3(specular);
        lit = glm::clamp(lit, glm::vec3(0.0f), glm::vec3(1.0f));
        const Color c{
            (uint8_t)std::clamp(lit.r * 255.0f, 0.0f, 255.0f),
            (uint8_t)std::clamp(lit.g * 255.0f, 0.0f, 255.0f),
            (uint8_t)std::clamp(lit.b * 255.0f, 0.0f, 255.0f),
            255
        };

        draw_filled_triangle(rt, depth_buffer, s0, z0, s1, z1, s2, z2, c);
    }
}

int main() {
    shs::jolt::init_jolt();

    SdlRuntime runtime{
        WindowDesc{"Culling & Debug Draw Demo (Software, All Jolt Shapes)", WINDOW_W, WINDOW_H},
        SurfaceDesc{CANVAS_W, CANVAS_H}
    };
    if (!runtime.valid()) return 1;

    RT_ColorLDR ldr_rt{CANVAS_W, CANVAS_H};
    std::vector<uint8_t> rgba8_staging(CANVAS_W * CANVAS_H * 4);
    std::vector<float> depth_buffer((size_t)CANVAS_W * (size_t)CANVAS_H, 1.0f);

    std::vector<ShapeInstance> instances;

    // Large floor
    {
        ShapeInstance floor{};
        floor.shape.shape = jolt::make_box(glm::vec3(50.0f, 0.1f, 50.0f));
        floor.base_pos = glm::vec3(0.0f, -0.2f, 0.0f);
        floor.base_rot = glm::vec3(0.0f);
        floor.shape.transform = jolt::to_jph(compose_model(floor.base_pos, floor.base_rot));
        floor.shape.stable_id = 9000;
        floor.color = glm::vec3(0.18f, 0.18f, 0.22f);
        floor.animated = false;
        instances.push_back(floor);
    }

    // Custom convex hull vertices
    const std::vector<glm::vec3> custom_hull_verts = {
        {-0.8f, -0.7f, -0.4f},
        { 0.9f, -0.6f, -0.5f},
        { 1.0f,  0.4f, -0.1f},
        {-0.7f,  0.6f, -0.2f},
        {-0.3f, -0.4f,  0.9f},
        { 0.4f,  0.7f,  0.8f},
    };

    // Custom mesh shape (triangular prism / wedge-like)
    MeshData wedge_mesh{};
    wedge_mesh.positions = {
        {-0.9f, -0.6f, -0.6f}, // 0
        { 0.9f, -0.6f, -0.6f}, // 1
        { 0.0f,  0.8f, -0.6f}, // 2
        {-0.9f, -0.6f,  0.6f}, // 3
        { 0.9f, -0.6f,  0.6f}, // 4
        { 0.0f,  0.8f,  0.6f}, // 5
    };
    wedge_mesh.indices = {
        0, 1, 2, // back
        5, 4, 3, // front
        0, 3, 4, 0, 4, 1, // bottom
        1, 4, 5, 1, 5, 2, // right
        2, 5, 3, 2, 3, 0  // left
    };

    const JPH::ShapeRefC sphere_shape = jolt::make_sphere(1.0f);
    const JPH::ShapeRefC box_shape = jolt::make_box(glm::vec3(0.9f, 0.7f, 0.6f));
    const JPH::ShapeRefC capsule_shape = jolt::make_capsule(0.9f, 0.45f);
    const JPH::ShapeRefC cylinder_shape = jolt::make_cylinder(0.9f, 0.5f);
    const JPH::ShapeRefC tapered_capsule_shape = jolt::make_tapered_capsule(0.9f, 0.25f, 0.65f);
    const JPH::ShapeRefC convex_hull_shape = jolt::make_convex_hull(custom_hull_verts);
    const JPH::ShapeRefC mesh_shape = jolt::make_mesh_shape(wedge_mesh);
    const JPH::ShapeRefC convex_from_mesh_shape = jolt::make_convex_hull_from_mesh(wedge_mesh);
    const JPH::ShapeRefC point_light_volume_shape = jolt::make_point_light_volume(1.0f);
    const JPH::ShapeRefC spot_light_volume_shape = jolt::make_spot_light_volume(1.2f, glm::radians(28.0f), 20);
    // For general visualization scaling, use a very small attenuation bound
    // so the shape draws reasonably as a panel rather than a giant cube.
    const JPH::ShapeRefC rect_light_volume_shape = jolt::make_rect_area_light_volume(glm::vec2(0.8f, 0.5f), 0.1f);
    const JPH::ShapeRefC tube_light_volume_shape = jolt::make_tube_area_light_volume(0.9f, 0.35f);

    struct ShapeTypeDef {
        JPH::ShapeRefC shape;
        glm::vec3 color;
    };

    const std::vector<ShapeTypeDef> shape_types = {
        {sphere_shape,             {0.95f, 0.35f, 0.35f}},
        {box_shape,                {0.35f, 0.90f, 0.45f}},
        {capsule_shape,            {0.35f, 0.55f, 0.95f}},
        {cylinder_shape,           {0.95f, 0.80f, 0.30f}},
        {tapered_capsule_shape,    {0.80f, 0.40f, 0.95f}},
        {convex_hull_shape,        {0.30f, 0.85f, 0.90f}},
        {mesh_shape,               {0.92f, 0.55f, 0.25f}},
        {convex_from_mesh_shape,   {0.55f, 0.95f, 0.55f}},
        {point_light_volume_shape, {0.95f, 0.45f, 0.65f}},
        {spot_light_volume_shape,  {0.95f, 0.70f, 0.35f}},
        {rect_light_volume_shape,  {0.35f, 0.95f, 0.80f}},
        {tube_light_volume_shape,  {0.70f, 0.65f, 0.95f}},
    };

    uint32_t next_id = 0;
    const int copies_per_type = 6;
    const float spacing_x = 5.6f;
    const float spacing_z = 4.8f;
    for (size_t t = 0; t < shape_types.size(); ++t) {
        for (int c = 0; c < copies_per_type; ++c) {
            ShapeInstance inst{};
            inst.shape.shape = shape_types[t].shape;
            inst.base_pos = glm::vec3(
                (-0.5f * (copies_per_type - 1) + (float)c) * spacing_x,
                1.25f + 0.25f * (float)(c % 3),
                (-0.5f * (float)(shape_types.size() - 1) + (float)t) * spacing_z);
            inst.base_rot = glm::vec3(
                0.17f * (float)c,
                0.23f * (float)t,
                0.11f * (float)(c + (int)t));
            inst.angular_vel = glm::vec3(
                0.30f + 0.07f * (float)((c + (int)t) % 5),
                0.42f + 0.06f * (float)(c % 4),
                0.36f + 0.05f * (float)((int)t % 6));
            inst.shape.transform = jolt::to_jph(compose_model(inst.base_pos, inst.base_rot));
            inst.shape.stable_id = next_id++;
            inst.color = shape_types[t].color;
            inst.animated = true;
            instances.push_back(std::move(inst));
        }
    }

    FreeCamera camera;
    bool show_aabb_debug = false;
    bool render_lit_surfaces = false;
    bool mouse_drag_held = false;
    std::printf("Controls: LMB/RMB drag look, WASD+QE move, Shift boost, B toggle AABB, L toggle debug/lit\n");

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
        const bool look_drag = input.right_mouse_down || input.left_mouse_down;
        if (look_drag != mouse_drag_held) {
            mouse_drag_held = look_drag;
            runtime.set_relative_mouse_mode(mouse_drag_held);
            input.mouse_dx = 0.0f;
            input.mouse_dy = 0.0f;
        }

        camera.update(input, dt);

        // Animate rotations for all non-floor shapes.
        for (auto& inst : instances)
        {
            if (!inst.animated) continue;
            const glm::vec3 rot = inst.base_rot + inst.angular_vel * time_s;
            inst.shape.transform = jolt::to_jph(compose_model(inst.base_pos, rot));
        }

        glm::mat4 view = camera.get_view_matrix();
        glm::mat4 proj = perspective_lh_no(glm::radians(60.0f), (float)CANVAS_W / CANVAS_H, 0.1f, 1000.0f);
        glm::mat4 vp = proj * view;

        const Frustum frustum = extract_frustum_planes(vp);
        const CullingResultEx frustum_result = run_frustum_culling(
            std::span<const ShapeInstance>(instances.data(), instances.size()),
            frustum,
            [](const ShapeInstance& inst) -> const SceneShape& { return inst.shape; });

        for (auto& inst : instances) inst.visible = false;
        for (uint32_t idx : frustum_result.visible_indices)
        {
            if (idx < instances.size()) instances[idx].visible = true;
        }
        const CullingStats stats = frustum_result.stats;

        ldr_rt.clear({12, 13, 18, 255});
        std::fill(depth_buffer.begin(), depth_buffer.end(), 1.0f);

        for (const auto& inst : instances) {
            if (!inst.visible) continue;

            const DebugMesh shape_mesh = debug_mesh_from_scene_shape(inst.shape);
            const glm::vec3 base_color = inst.color;
            if (render_lit_surfaces) {
                draw_mesh_blinn_phong(
                    ldr_rt,
                    depth_buffer,
                    shape_mesh,
                    vp,
                    CANVAS_W,
                    CANVAS_H,
                    camera.pos,
                    kSunLightDirWs,
                    base_color);
            } else {
                const Color shape_color{
                    (uint8_t)std::clamp(base_color.r * 255.0f, 0.0f, 255.0f),
                    (uint8_t)std::clamp(base_color.g * 255.0f, 0.0f, 255.0f),
                    (uint8_t)std::clamp(base_color.b * 255.0f, 0.0f, 255.0f),
                    255
                };
                draw_debug_mesh_wireframe(ldr_rt, shape_mesh, vp, CANVAS_W, CANVAS_H, shape_color);
            }

            if (show_aabb_debug) {
                const DebugMesh aabb_mesh = debug_mesh_from_aabb(inst.shape.world_aabb());
                draw_debug_mesh_wireframe(ldr_rt, aabb_mesh, vp, CANVAS_W, CANVAS_H, Color{255, 240, 80, 255});
            }
        }

        // Convert LDR to RGBA8 for presentation (with Y-flip for SDL)
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

        char title[256];
        std::snprintf(
            title,
            sizeof(title),
            "Culling Demo (SW) | Scene:%u Visible:%u Culled:%u | Mode:%s | AABB:%s",
            stats.scene_count,
            stats.visible_count,
            stats.culled_count,
            render_lit_surfaces ? "Lit" : "Debug",
            show_aabb_debug ? "ON" : "OFF");
        runtime.set_title(title);
        std::printf("Scene:%u Visible:%u Culled:%u | Mode:%s | AABB debug: %s\r",
            stats.scene_count,
            stats.visible_count,
            stats.culled_count,
            render_lit_surfaces ? "Lit  " : "Debug",
            show_aabb_debug ? "ON " : "OFF");
        std::fflush(stdout);
    }

    std::printf("\n");
    runtime.set_relative_mouse_mode(false);
    shs::jolt::shutdown_jolt();
    return 0;
}
