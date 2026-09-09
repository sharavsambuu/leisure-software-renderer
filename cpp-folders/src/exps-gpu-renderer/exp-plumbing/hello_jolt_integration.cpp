#define SDL_MAIN_HANDLED

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shs/core/context.hpp"
#include "shs/camera/convention.hpp"
#include "shs/frame/technique_mode.hpp"
#include "shs/input/value_actions.hpp"
#include "shs/input/value_input_latch.hpp"
#include "shs/lighting/light_set.hpp"
#include "shs/resources/resource_registry.hpp"
#include "shs/resources/loaders/primitive_import.hpp"
#include "shs/rhi/backend/backend_factory.hpp"
#include "shs/rhi/core/backend.hpp"
#include "shs/rhi/drivers/vulkan/vk_backend.hpp"

// Jolt Integration Headers
#include "shs/geometry/jolt_adapter.hpp"
#include "shs/geometry/jolt_shapes.hpp"
#include "shs/geometry/jolt_culling.hpp"
#include "shs/lighting/jolt_light_culling.hpp"
#include "shs/geometry/jolt_debug_draw.hpp"
#include "shs/geometry/jolt_renderable.hpp"

using namespace shs;

namespace
{
    constexpr int kWidth = 1280;
    constexpr int kHeight = 720;

    struct App
    {
        SDL_Window* win = nullptr;
        shs::Context ctx{};
        shs::IRenderBackend* vk = nullptr;
        std::vector<std::unique_ptr<shs::IRenderBackend>> keep_alive{};

        shs::ResourceRegistry resources{};
        std::vector<shs::JoltRenderable> renderables{};
        shs::LightSet light_set{};
        std::vector<shs::SceneShape> light_shapes{};

        // Dummy culling results
        shs::TiledLightCullingResult tiled_lights{};

        void init()
        {
            SDL_Init(SDL_INIT_VIDEO);
            win = SDL_CreateWindow("Hello Jolt Integration", 
                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, kWidth, kHeight, 
                SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

            jolt::init_jolt();

            auto res = shs::create_render_backend(shs::RenderBackendType::Vulkan);
            vk = res.backend.get();
            keep_alive.push_back(std::move(res.backend));
            
            shs::VulkanRenderBackend::InitDesc desc{};
            desc.window = win;
            desc.width = kWidth;
            desc.height = kHeight;
            if (auto* vkb = dynamic_cast<shs::VulkanRenderBackend*>(vk))
            {
                vkb->init(desc);
            }

            load_scene();
        }

        void load_scene()
        {
            // 1. Materials
            MaterialData mat_red{};
            mat_red.name = "MatRed";
            mat_red.base_color = glm::vec3(0.8f, 0.1f, 0.1f);
            mat_red.roughness = 0.2f;
            mat_red.metallic = 0.8f;
            auto h_red = resources.add_material(mat_red);

            MaterialData mat_blue{};
            mat_blue.name = "MatBlue";
            mat_blue.base_color = glm::vec3(0.1f, 0.1f, 0.8f);
            mat_blue.roughness = 0.4f;
            auto h_blue = resources.add_material(mat_blue);

            // 2. Primitives -> Jolt Shapes
            // Sphere
            JoltRenderable sphere{};
            sphere.name = "Sphere";
            sphere.geometry.shape = jolt::make_sphere(0.5f);
            sphere.geometry.transform = jolt::to_jph(glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f)));
            sphere.material = h_red;
            sphere.visual_mesh = resources.add_mesh(make_sphere({0.5f, 18, 12}));
            renderables.push_back(sphere);

            // Box
            JoltRenderable box{};
            box.name = "Box";
            box.geometry.shape = jolt::make_box(glm::vec3(0.5f));
            box.geometry.transform = jolt::to_jph(glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));
            box.material = h_blue;
            box.visual_mesh = resources.add_mesh(make_box({glm::vec3(1.0f)}));
            renderables.push_back(box);

            // 3. Lights
            for (int i = 0; i < 50; ++i)
            {
                PointLight p{};
                p.common.position_ws = glm::vec3((i % 10) * 2.0f - 10.0f, 2.0f, (i / 10) * 2.0f - 5.0f);
                p.common.range = 5.0f;
                p.common.color = glm::vec3(0.5f, 0.5f, 1.0f);
                light_set.points.push_back(p);

                SceneShape ls{};
                ls.shape = jolt::make_point_light_volume(p.common.range);
                ls.transform = JPH::Mat44::sTranslation(jolt::to_jph(p.common.position_ws));
                light_shapes.push_back(ls);
            }
        }

        void run()
        {
            bool running = true;
            shs::RuntimeInputLatch input_latch{};
            std::vector<shs::RuntimeInputEvent> pending_input_events{};
            shs::RuntimeState runtime_state{};
            std::vector<shs::RuntimeAction> runtime_actions{};
            while (running)
            {
                SDL_Event e;
                while (SDL_PollEvent(&e))
                {
                    if (e.type == SDL_QUIT)
                    {
                        pending_input_events.push_back(shs::make_quit_input_event());
                    }
                    if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)
                    {
                        pending_input_events.push_back(shs::make_quit_input_event());
                    }
                }
                input_latch = shs::reduce_runtime_input_latch(input_latch, pending_input_events);
                pending_input_events.clear();
                runtime_actions.clear();
                shs::InputState runtime_input{};
                runtime_input.quit = input_latch.quit_requested;
                shs::emit_human_actions(runtime_input, runtime_actions, 0.0f, 1.0f, 0.0f);
                runtime_state = shs::reduce_runtime_state(runtime_state, runtime_actions, 0.0f);
                if (runtime_state.quit_requested) break;

                update();
                render();
            }
        }

        void update()
        {
            // Culling using Jolt
            const glm::vec3 eye(0.0f, 0.0f, -10.0f);
            const glm::vec3 target(0.0f, 0.0f, 0.0f);
            glm::mat4 view = look_at_lh(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 proj = perspective_lh_no(glm::radians(60.0f), (float)kWidth / kHeight, 0.1f, 1000.0f);
            
            CullingCell camera_cell = extract_frustum_cell(proj * view);

            // 1. Renderable Culling
            // Here we would typically batch cull.
            for (auto& r : renderables)
            {
                shs::CullClass c = shs::classify_vs_cell(r.geometry, camera_cell);
                r.visible = (c != shs::CullClass::Outside);
            }

            // 2. Light Culling
            tiled_lights = cull_lights_tiled(light_shapes, proj * view, kWidth, kHeight);
        }

        void render()
        {
            // In a real demo, we'd record Vulkan commands here.
            // For now, we're just validating the logic compiles and runs.
            std::printf("Visible renderables: %d\n", (int)std::count_if(renderables.begin(), renderables.end(), [](auto& r){return r.visible;}));
        }

        void cleanup()
        {
            jolt::shutdown_jolt();
        }
    };
}

int main()
{
    App app{};
    app.init();
    app.run();
    app.cleanup();
    return 0;
}
