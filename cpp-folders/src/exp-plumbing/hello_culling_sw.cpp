#define SDL_MAIN_HANDLED
#include <chrono>
#include <cstdio>
#include <vector>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/platform/sdl/sdl_runtime.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/geometry/volumes.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/core/context.hpp>
#include <shs/gfx/rt_types.hpp>

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
        WindowDesc{"Culling Demo (Software)", WINDOW_W, WINDOW_H},
        SurfaceDesc{CANVAS_W, CANVAS_H}
    };
    if (!runtime.valid()) return 1;

    RT_ColorLDR ldr_rt{CANVAS_W, CANVAS_H};
    std::vector<uint8_t> rgba8_staging(CANVAS_W * CANVAS_H * 4);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);
    std::uniform_int_distribution<int> type_dist(0, 3);

    std::vector<ShapeInstance> instances;

    // 1. Large floor
    instances.push_back({
        SceneShape{
            jolt::make_box(glm::vec3(50.0f, 0.1f, 50.0f)),
            jolt::to_jph(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, 0.0f))),
            999
        },
        glm::vec3(0.3f, 0.3f, 0.35f)
    });

    // 2. Random shapes
    for (int i = 0; i < 100; ++i) {
        glm::vec3 pos(dist(rng), 0.5f, dist(rng));
        JPH::ShapeRefC shape;
        int type = type_dist(rng);
        if (type == 0) shape = jolt::make_sphere(0.8f);
        else if (type == 1) shape = jolt::make_box(glm::vec3(0.6f));
        else if (type == 2) shape = jolt::make_capsule(0.5f, 0.4f);
        else shape = jolt::make_cylinder(0.6f, 0.4f);

        instances.push_back({
            SceneShape{
                shape,
                jolt::to_jph(glm::translate(glm::mat4(1.0f), pos)),
                (uint32_t)i
            },
            glm::vec3(0.5f + 0.5f * (float)rand()/RAND_MAX, 0.5f + 0.5f * (float)rand()/RAND_MAX, 0.5f + 0.5f * (float)rand()/RAND_MAX)
        });
    }

    auto start_time = std::chrono::steady_clock::now();
    bool running = true;

    while (running) {
        PlatformInputState input{};
        if (!runtime.pump_input(input)) break;
        if (input.quit) break;

        auto now = std::chrono::steady_clock::now();
        float time = std::chrono::duration<float>(now - start_time).count();

        // Automatic camera movement (orbit)
        float cam_radius = 35.0f;
        float cam_x = std::cos(time * 0.2f) * cam_radius;
        float cam_z = std::sin(time * 0.2f) * cam_radius;
        glm::vec3 cam_pos(cam_x, 15.0f, cam_z);
        glm::vec3 target(0.0f, 0.0f, 0.0f);
        
        glm::mat4 view = glm::lookAt(cam_pos, target, glm::vec3(0, 1, 0));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f), (float)CANVAS_W / CANVAS_H, 0.1f, 1000.0f);
        glm::mat4 vp = proj * view;

        // Extract frustum
        Frustum frustum = extract_frustum_planes(vp);

        // Cull
        uint32_t visible_count = 0;
        for (auto& inst : instances) {
            CullClass cc = classify_vs_frustum(inst.shape, frustum);
            inst.visible = (cc != CullClass::Outside);
            if (inst.visible) visible_count++;
        }

        // Render (Software)
        ldr_rt.clear({20, 20, 25, 255});

        auto project = [&](const glm::vec3& p) -> glm::ivec2 {
            glm::vec4 clip = vp * glm::vec4(p, 1.0f);
            if (clip.w <= 0.0f) return {-1, -1};
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
            Color c{(uint8_t)(inst.color.r * 255), (uint8_t)(inst.color.g * 255), (uint8_t)(inst.color.b * 255), 255};
            
            for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                glm::vec3 p0 = mesh.vertices[mesh.indices[i+0]];
                glm::vec3 p1 = mesh.vertices[mesh.indices[i+1]];
                glm::vec3 p2 = mesh.vertices[mesh.indices[i+2]];

                glm::ivec2 v0 = project(p0);
                glm::ivec2 v1 = project(p1);
                glm::ivec2 v2 = project(p2);

                if (v0.x >= 0 && v1.x >= 0) draw_line_rt(ldr_rt, v0.x, v0.y, v1.x, v1.y, c);
                if (v1.x >= 0 && v2.x >= 0) draw_line_rt(ldr_rt, v1.x, v1.y, v2.x, v2.y, c);
                if (v2.x >= 0 && v0.x >= 0) draw_line_rt(ldr_rt, v2.x, v2.y, v0.x, v0.y, c);
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
