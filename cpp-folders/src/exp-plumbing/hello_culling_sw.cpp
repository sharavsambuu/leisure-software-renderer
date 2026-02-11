#define SDL_MAIN_HANDLED
#include <chrono>
#include <cstdio>
#include <vector>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include <shs/platform/sdl/sdl_runtime.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/geometry/volumes.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/core/context.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/camera/camera_math.hpp>
#include <shs/camera/convention.hpp>

using namespace shs;

const int WINDOW_W = 1024;
const int WINDOW_H = 768;
const int CANVAS_W = 640;
const int CANVAS_H = 480;

struct ShapeInstance {
    SceneShape shape;
    glm::vec3 color;
    bool visible = true;
};

struct FreeCamera {
    glm::vec3 pos{0.0f, 15.0f, -35.0f};
    float yaw = glm::half_pi<float>(); // Pointing towards +Z
    float pitch = -0.3f;
    float move_speed = 20.0f;
    float look_speed = 0.003f;

    void update(const PlatformInputState& input, float dt) {
        if (input.right_mouse_down) {
            // Invert yaw delta to match SHS LH (looking right = yaw decrease)
            yaw -= input.mouse_dx * look_speed;
            pitch -= input.mouse_dy * look_speed;
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

int main() {
    shs::jolt::init_jolt();

    SdlRuntime runtime{
        WindowDesc{"Culling & Debug Draw Demo (Software)", WINDOW_W, WINDOW_H},
        SurfaceDesc{CANVAS_W, CANVAS_H}
    };
    if (!runtime.valid()) return 1;

    RT_ColorLDR ldr_rt{CANVAS_W, CANVAS_H};
    std::vector<uint8_t> rgba8_staging(CANVAS_W * CANVAS_H * 4);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> jitter(-1.5f, 1.5f);
    std::uniform_int_distribution<int> type_dist(0, 4);

    std::vector<ShapeInstance> instances;

    // 1. Large floor
    instances.push_back({
        SceneShape{
            jolt::make_box(glm::vec3(50.0f, 0.1f, 50.0f)),
            jolt::to_jph(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.1f, 0.0f))),
            999
        },
        glm::vec3(0.2f, 0.2f, 0.25f)
    });

    // 2. Jittered Grid placement
    const int GRID_SIZE = 12;
    const float GRID_SPACING = 4.0f;
    for (int y = 0; y < GRID_SIZE; ++y) {
        for (int x = 0; x < GRID_SIZE; ++x) {
            float px = (x - GRID_SIZE / 2) * GRID_SPACING + jitter(rng);
            float pz = (y - GRID_SIZE / 2) * GRID_SPACING + jitter(rng);
            glm::vec3 pos(px, 1.0f, pz);

            JPH::ShapeRefC shape;
            int type = type_dist(rng);
            if (type == 0) shape = jolt::make_sphere(1.0f);
            else if (type == 1) shape = jolt::make_box(glm::vec3(0.7f, 1.0f, 0.7f));
            else if (type == 2) shape = jolt::make_capsule(1.0f, 0.5f);
            else if (type == 3) shape = jolt::make_cylinder(1.0f, 0.5f);
            else shape = jolt::make_tapered_capsule(1.0f, 0.3f, 0.7f);

            instances.push_back({
                SceneShape{
                    shape,
                    jolt::to_jph(glm::translate(glm::mat4(1.0f), pos)),
                    (uint32_t)(y * GRID_SIZE + x)
                },
                glm::vec3(0.4f + 0.6f * (float)rand()/RAND_MAX, 0.4f + 0.6f * (float)rand()/RAND_MAX, 0.4f + 0.6f * (float)rand()/RAND_MAX)
            });
        }
    }

    FreeCamera camera;
    glm::vec3 global_light_dir = glm::normalize(glm::vec3(0.5f, -1.0f, 0.4f));

    auto last_time = std::chrono::steady_clock::now();
    
    while (true) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;

        PlatformInputState input{};
        if (!runtime.pump_input(input)) break;
        if (input.quit) break;

        camera.update(input, dt);
        
        glm::mat4 view = camera.get_view_matrix();
        glm::mat4 proj = perspective_lh_no(glm::radians(60.0f), (float)CANVAS_W / CANVAS_H, 0.1f, 1000.0f);
        glm::mat4 vp = proj * view;

        // Extract frustum
        Frustum frustum = extract_frustum_planes(vp);

        // Cull
        for (auto& inst : instances) {
            CullClass cc = classify_vs_frustum(inst.shape, frustum);
            inst.visible = (cc != CullClass::Outside);
        }

        // Render (Software)
        ldr_rt.clear({15, 15, 20, 255});

        auto project = [&](const glm::vec3& p) -> glm::ivec2 {
            glm::vec4 clip = vp * glm::vec4(p, 1.0f);
            if (clip.w <= 0.001f) return {-1, -1};
            glm::vec3 ndc = glm::vec3(clip) / clip.w;
            if (ndc.z < -1.0f || ndc.z > 1.0f) return {-1, -1};
            return {
                (int)((ndc.x + 1.0f) * 0.5f * CANVAS_W),
                (int)((1.0f - ndc.y) * 0.5f * CANVAS_H)
            };
        };

        for (const auto& inst : instances) {
            if (!inst.visible) continue;

            DebugMesh mesh = debug_mesh_from_scene_shape(inst.shape);
            glm::vec3 base_color = inst.color;

            for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                glm::vec3 p0 = mesh.vertices[mesh.indices[i+0]];
                glm::vec3 p1 = mesh.vertices[mesh.indices[i+1]];
                glm::vec3 p2 = mesh.vertices[mesh.indices[i+2]];

                // Calculate facet normal for shading
                glm::vec3 edge1 = p1 - p0;
                glm::vec3 edge2 = p2 - p0;
                glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

                // Basic Lambertian shading for wireframe depth
                float diffuse = std::max(0.2f, glm::dot(-normal, global_light_dir));
                glm::vec3 final_color = base_color * diffuse;
                Color line_color{(uint8_t)(final_color.r * 255), (uint8_t)(final_color.g * 255), (uint8_t)(final_color.b * 255), 255};

                glm::ivec2 v0 = project(p0);
                glm::ivec2 v1 = project(p1);
                glm::ivec2 v2 = project(p2);

                // Simple check: if at least two points are on screen, draw edges
                if (v0.x >= 0 && v1.x >= 0) draw_line_rt(ldr_rt, v0.x, v0.y, v1.x, v1.y, line_color);
                if (v1.x >= 0 && v2.x >= 0) draw_line_rt(ldr_rt, v1.x, v1.y, v2.x, v2.y, line_color);
                if (v2.x >= 0 && v0.x >= 0) draw_line_rt(ldr_rt, v2.x, v2.y, v0.x, v0.y, line_color);
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
    }

    shs::jolt::shutdown_jolt();
    return 0;
}
