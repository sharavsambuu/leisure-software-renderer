#define SDL_MAIN_HANDLED
#include <algorithm>
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
#include <shs/scene/scene_instance.hpp>
#include <shs/core/context.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/sw_render/debug_draw.hpp>
#include <shs/camera/camera_math.hpp>
#include <shs/camera/convention.hpp>

using namespace shs;

const int WINDOW_W = 1200;
const int WINDOW_H = 900;
const int CANVAS_W = 1200;
const int CANVAS_H = 900;
const int OCC_W = 300;
const int OCC_H = 225;
const glm::vec3 kSunLightDirWs = glm::normalize(glm::vec3(0.20f, -1.0f, 0.16f));



struct FreeCamera {
    glm::vec3 pos{0.0f, 14.0f, -28.0f};
    float yaw = glm::half_pi<float>();
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
        if (input.left) pos += right * speed * dt;
        if (input.right) pos -= right * speed * dt;
        if (input.ascend) pos += up * speed * dt;
        if (input.descend) pos -= up * speed * dt;
    }

    glm::mat4 get_view_matrix() const {
        return look_at_lh(pos, pos + forward_from_yaw_pitch(yaw, pitch), glm::vec3(0.0f, 1.0f, 0.0f));
    }
};



glm::mat4 compose_model(const glm::vec3& pos, const glm::vec3& rot_euler)
{
    glm::mat4 model(1.0f);
    model = glm::translate(model, pos);
    model = glm::rotate(model, rot_euler.x, glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, rot_euler.y, glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, rot_euler.z, glm::vec3(0.0f, 0.0f, 1.0f));
    return model;
}



int main() {
    shs::jolt::init_jolt();

    SdlRuntime runtime{
        WindowDesc{"Occlusion + Frustum Culling Demo (Software, All Jolt Shapes)", WINDOW_W, WINDOW_H},
        SurfaceDesc{CANVAS_W, CANVAS_H}
    };
    if (!runtime.valid()) return 1;

    RT_ColorLDR ldr_rt{CANVAS_W, CANVAS_H};
    std::vector<uint8_t> rgba8_staging(CANVAS_W * CANVAS_H * 4);
    std::vector<float> depth_buffer((size_t)CANVAS_W * (size_t)CANVAS_H, 1.0f);
    std::vector<float> occlusion_depth((size_t)OCC_W * (size_t)OCC_H, 1.0f);

    std::vector<SceneInstance> instances;
    std::vector<DebugMesh> mesh_library;

    // Large floor
    {
        SceneInstance floor{};
        floor.geometry.shape = jolt::make_box(glm::vec3(50.0f, 0.1f, 50.0f));
        floor.anim.base_pos = glm::vec3(0.0f, -0.2f, 0.0f);
        floor.anim.base_rot = glm::vec3(0.0f);
        floor.geometry.transform = jolt::to_jph(compose_model(floor.anim.base_pos, floor.anim.base_rot));
        floor.geometry.stable_id = 9000;
        floor.tint_color = glm::vec3(0.18f, 0.18f, 0.22f);
        floor.anim.animated = false;

        floor.user_index = static_cast<uint32_t>(mesh_library.size());
        mesh_library.push_back(debug_mesh_from_shape(*floor.geometry.shape, JPH::Mat44::sIdentity()));
        instances.push_back(floor);
    }

    const std::vector<glm::vec3> custom_hull_verts = {
        {-0.8f, -0.7f, -0.4f},
        { 0.9f, -0.6f, -0.5f},
        { 1.0f,  0.4f, -0.1f},
        {-0.7f,  0.6f, -0.2f},
        {-0.3f, -0.4f,  0.9f},
        { 0.4f,  0.7f,  0.8f},
    };

    MeshData wedge_mesh{};
    wedge_mesh.positions = {
        {-0.9f, -0.6f, -0.6f},
        { 0.9f, -0.6f, -0.6f},
        { 0.0f,  0.8f, -0.6f},
        {-0.9f, -0.6f,  0.6f},
        { 0.9f, -0.6f,  0.6f},
        { 0.0f,  0.8f,  0.6f},
    };
    wedge_mesh.indices = {
        0, 1, 2,
        5, 4, 3,
        0, 3, 4, 0, 4, 1,
        1, 4, 5, 1, 5, 2,
        2, 5, 3, 2, 3, 0
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
    const JPH::ShapeRefC rect_light_volume_shape = jolt::make_rect_area_light_volume(glm::vec2(0.8f, 0.5f), 0.1f);
    const JPH::ShapeRefC tube_light_volume_shape = jolt::make_tube_area_light_volume(0.9f, 0.35f);

    struct ShapeTypeDef {
        JPH::ShapeRefC shape;
        glm::vec3 color;
        uint32_t mesh_index = 0;
    };

    std::vector<ShapeTypeDef> shape_types = {
        {sphere_shape,             {0.95f, 0.35f, 0.35f}, 0u},
        {box_shape,                {0.35f, 0.90f, 0.45f}, 0u},
        {capsule_shape,            {0.35f, 0.55f, 0.95f}, 0u},
        {cylinder_shape,           {0.95f, 0.80f, 0.30f}, 0u},
        {tapered_capsule_shape,    {0.80f, 0.40f, 0.95f}, 0u},
        {convex_hull_shape,        {0.30f, 0.85f, 0.90f}, 0u},
        {mesh_shape,               {0.92f, 0.55f, 0.25f}, 0u},
        {convex_from_mesh_shape,   {0.55f, 0.95f, 0.55f}, 0u},
        {point_light_volume_shape, {0.95f, 0.45f, 0.65f}, 0u},
        {spot_light_volume_shape,  {0.95f, 0.70f, 0.35f}, 0u},
        {rect_light_volume_shape,  {0.35f, 0.95f, 0.80f}, 0u},
        {tube_light_volume_shape,  {0.70f, 0.65f, 0.95f}, 0u},
    };

    for (auto& t : shape_types)
    {
        t.mesh_index = static_cast<uint32_t>(mesh_library.size());
        mesh_library.push_back(debug_mesh_from_shape(*t.shape, JPH::Mat44::sIdentity()));
    }

    uint32_t next_id = 0;
    const int copies_per_type = 6;
    const float spacing_x = 5.6f;
    const float spacing_z = 4.8f;
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
    }

    SceneElementSet cull_scene;
    cull_scene.reserve(instances.size());
    for (size_t i = 0; i < instances.size(); ++i)
    {
        SceneElement elem{};
        elem.geometry = instances[i].geometry;
        elem.user_index = static_cast<uint32_t>(i);
        elem.visible = instances[i].visible;
        elem.frustum_visible = instances[i].frustum_visible;
        elem.occluded = instances[i].occluded;
        elem.casts_shadow = true;
        cull_scene.add(std::move(elem));
    }
    SceneCullingContext cull_ctx{};

    FreeCamera camera;
    bool show_aabb_debug = false;
    bool render_lit_surfaces = false;
    bool enable_occlusion = true;
    bool mouse_drag_held = false;
    std::printf("Controls: LMB/RMB drag look, WASD+QE move, Shift boost, B toggle AABB, L toggle debug/lit, F2 toggle occlusion\n");

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
        const bool look_drag = input.right_mouse_down || input.left_mouse_down;
        if (look_drag != mouse_drag_held) {
            mouse_drag_held = look_drag;
            runtime.set_relative_mouse_mode(mouse_drag_held);
            input.mouse_dx = 0.0f;
            input.mouse_dy = 0.0f;
        }

        camera.update(input, dt);

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

        auto cull_elems = cull_scene.elements();
        for (size_t i = 0; i < instances.size(); ++i)
        {
            cull_elems[i].geometry = instances[i].geometry;
            cull_elems[i].visible = true;
            cull_elems[i].frustum_visible = true;
            cull_elems[i].occluded = false;
        }

        glm::mat4 view = camera.get_view_matrix();
        glm::mat4 proj = perspective_lh_no(glm::radians(60.0f), (float)CANVAS_W / CANVAS_H, 0.1f, 1000.0f);
        glm::mat4 vp = proj * view;

        const Frustum frustum = extract_frustum_planes(vp);
        cull_ctx.run_frustum(cull_scene, frustum);
        cull_ctx.run_software_occlusion(
            cull_scene,
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

        for (size_t i = 0; i < instances.size(); ++i)
        {
            instances[i].visible = cull_elems[i].visible;
            instances[i].frustum_visible = cull_elems[i].frustum_visible;
            instances[i].occluded = cull_elems[i].occluded;
        }

        const CullingStats& stats = cull_ctx.stats();
        const std::vector<uint32_t>& visible_scene_indices = cull_ctx.visible_indices();

        ldr_rt.clear({12, 13, 18, 255});
        std::fill(depth_buffer.begin(), depth_buffer.end(), 1.0f);

        for (uint32_t scene_idx : visible_scene_indices) {
            if (scene_idx >= cull_scene.size()) continue;
            const uint32_t idx = cull_scene[scene_idx].user_index;
            if (idx >= instances.size()) continue;
            const auto& inst = instances[idx];
            if (inst.user_index >= mesh_library.size()) continue;
            const DebugMesh& shape_mesh = mesh_library[inst.user_index];
            const glm::vec3 base_color = inst.tint_color;
            if (render_lit_surfaces) {
                debug_draw::draw_mesh_blinn_phong_transformed(
                    ldr_rt,
                    depth_buffer,
                    shape_mesh,
                    jolt::to_glm(inst.geometry.transform),
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

        char title[320];
        std::snprintf(
            title,
            sizeof(title),
            "Occlusion Culling Demo (SW) | Scene:%u Frustum:%u Occluded:%u Visible:%u | Occ:%s | Mode:%s | AABB:%s",
            stats.scene_count,
            stats.frustum_visible_count,
            stats.occluded_count,
            stats.visible_count,
            enable_occlusion ? "ON" : "OFF",
            render_lit_surfaces ? "Lit" : "Debug",
            show_aabb_debug ? "ON" : "OFF");
        runtime.set_title(title);
        std::printf(
            "Scene:%u Frustum:%u Occluded:%u Visible:%u | Occ:%s | Mode:%s | AABB:%s\r",
            stats.scene_count,
            stats.frustum_visible_count,
            stats.occluded_count,
            stats.visible_count,
            enable_occlusion ? "ON " : "OFF",
            render_lit_surfaces ? "Lit  " : "Debug",
            show_aabb_debug ? "ON " : "OFF");
        std::fflush(stdout);
    }

    std::printf("\n");
    runtime.set_relative_mouse_mode(false);
    shs::jolt::shutdown_jolt();
    return 0;
}
