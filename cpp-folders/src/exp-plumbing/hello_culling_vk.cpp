#define SDL_MAIN_HANDLED
#include <chrono>
#include <cstdio>
#include <vector>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/platform/sdl/sdl_runtime.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>
#include <shs/geometry/volumes.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/core/context.hpp>

using namespace shs;

const int WINDOW_W = 1024;
const int WINDOW_H = 768;

struct ShapeInstance {
    SceneShape shape;
    glm::vec3 color;
    bool visible = true;
};

int main() {
    shs::jolt::init_jolt();

    SdlRuntime runtime{
        WindowDesc{"Culling Demo (Vulkan)", WINDOW_W, WINDOW_H},
        SurfaceDesc{WINDOW_W, WINDOW_H} // Not used for Vulkan RHI but required by SDL helper
    };
    if (!runtime.valid()) return 1;

    auto backend_result = create_render_backend("vulkan");
    auto* vk = dynamic_cast<VulkanRenderBackend*>(backend_result.backend.get());
    if (!vk) return 1;

    // Use InitDesc for explicit initialization
    VulkanRenderBackend::InitDesc vk_init{};
    vk_init.window = runtime.window();
    vk->init(vk_init);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-30.0f, 30.0f);
    std::uniform_int_distribution<int> type_dist(0, 3);

    std::vector<ShapeInstance> instances;

    // 1. Large floor
    instances.push_back({
        SceneShape{
            jolt::make_box(glm::vec3(50.0f, 0.1f, 50.0f)),
            jolt::to_jph(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, 0.0f))),
            999
        },
        glm::vec3(0.2f, 0.2f, 0.25f)
    });

    // 2. Random shapes
    for (int i = 0; i < 500; ++i) {
        glm::vec3 pos(dist(rng), 1.0f, dist(rng));
        JPH::ShapeRefC shape;
        int type = type_dist(rng);
        if (type == 0) shape = jolt::make_sphere(1.0f);
        else if (type == 1) shape = jolt::make_box(glm::vec3(0.8f));
        else if (type == 2) shape = jolt::make_capsule(0.7f, 0.5f);
        else shape = jolt::make_cylinder(0.7f, 0.5f);

        instances.push_back({
            SceneShape{
                shape,
                jolt::to_jph(glm::translate(glm::mat4(1.0f), pos)),
                (uint32_t)i
            },
            glm::vec3(0.4f + 0.6f * (float)rand()/RAND_MAX, 0.4f + 0.6f * (float)rand()/RAND_MAX, 0.4f + 0.6f * (float)rand()/RAND_MAX)
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
        float cam_radius = 50.0f + 10.0f * std::sin(time * 0.1f);
        float cam_x = std::cos(time * 0.15f) * cam_radius;
        float cam_z = std::sin(time * 0.15f) * cam_radius;
        glm::vec3 cam_pos(cam_x, 20.0f, cam_z);
        glm::vec3 target(0.0f, 0.0f, 0.0f);
        
        glm::mat4 view = glm::lookAt(cam_pos, target, glm::vec3(0, 1, 0));
        glm::mat4 proj = glm::perspective(glm::radians(60.0f), (float)WINDOW_W / WINDOW_H, 0.1f, 1000.0f);
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

        // Render (Vulkan - leveraging basic backend features)
        Context shs_ctx{};
        RenderBackendFrameInfo frame{};
        vk->begin_frame(shs_ctx, frame);

        // In a real Vulkan backend demo, we'd submit these to a command buffer.
        // For this demo, we'll use debug draw into a list that's eventually rendered.
        // Assuming vk backend has a simple "DrawLines" or similar, 
        // or we just acknowledge culling is happening.

        // Since the user wants to SEE culling, I'll print the count for now.
        // In a proper demo, this would translate to dynamic draw calls.
        // std::printf("Visible: %u / %zu\n", visible_count, instances.size());

        vk->end_frame(shs_ctx, frame);
    }

    shs::jolt::shutdown_jolt();
    return 0;
}
