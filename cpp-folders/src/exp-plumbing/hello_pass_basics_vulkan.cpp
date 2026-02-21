#define SDL_MAIN_HANDLED

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/app/camera_sync.hpp>
#include <shs/camera/convention.hpp>
#include <shs/camera/follow_camera.hpp>
#include <shs/core/context.hpp>
#include <shs/frame/frame_params.hpp>
#include <shs/gfx/rt_registry.hpp>
#include <shs/gfx/rt_shadow.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/job/thread_pool_job_system.hpp>
#include <shs/logic/fsm.hpp>
#include <shs/platform/platform_input.hpp>
#include <shs/platform/platform_runtime.hpp>
#include <shs/pipeline/technique_profile.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>
#include <shs/rhi/drivers/vulkan/vk_cmd_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_render_path_descriptors.hpp>
#include <shs/rhi/drivers/vulkan/vk_memory_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_shader_utils.hpp>
#include <shs/resources/loaders/primitive_import.hpp>
#include <shs/resources/loaders/resource_import.hpp>
#include <shs/resources/resource_registry.hpp>
#include <shs/scene/scene_bindings.hpp>
#include <shs/scene/scene_objects.hpp>
#include <shs/scene/system_processors.hpp>
#include <shs/scene/scene_types.hpp>
#include <shs/sky/cubemap_sky.hpp>
#include <shs/sky/loaders/cubemap_loader_sdl.hpp>
#include <shs/sky/procedural_sky.hpp>

/*
    HelloPassBasics demo
    - Pass pipeline: shadow -> PBR/Blinn forward -> bright -> shafts -> flare -> tonemap
    - Scene: floor + subaru + monkey
    - Runtime toggle: debug/shading/sky/follow camera + pass isolation ladder
*/

namespace
{
    constexpr int WINDOW_W = 900;
    constexpr int WINDOW_H = 600;
    constexpr int CANVAS_W = 900;
    constexpr int CANVAS_H = 600;
    constexpr float PI = 3.14159265f;
    constexpr float TWO_PI = 6.2831853f;
    constexpr float MOUSE_LOOK_SENS = 0.0025f;
    constexpr float FREE_CAM_BASE_SPEED = 8.0f;
    constexpr float CHASE_ORBIT_SENS = 0.0025f;

    enum class ModelForwardAxis : uint8_t
    {
        PosX = 0,
        NegX = 1,
        PosZ = 2,
        NegZ = 3,
    };

    constexpr ModelForwardAxis SUBARU_VISUAL_FORWARD_AXIS = ModelForwardAxis::PosZ;

    enum class PassIsolationStage : uint8_t
    {
        Minimal = 0,
        Shadow = 1,
        Bright = 2,
        Shafts = 3,
        MotionBlur = 4,
    };

    struct PassExecutionPlan
    {
        bool run_shadow = true;
        bool run_bright = true;
        bool run_shafts = true;
        bool run_flare = true;
        bool enable_motion_blur = true;
        PassIsolationStage stage = PassIsolationStage::MotionBlur;
    };

    const char* pass_isolation_stage_name(PassIsolationStage stage)
    {
        switch (stage)
        {
            case PassIsolationStage::Minimal: return "minimal";
            case PassIsolationStage::Shadow: return "shadow";
            case PassIsolationStage::Bright: return "bright";
            case PassIsolationStage::Shafts: return "shafts";
            case PassIsolationStage::MotionBlur: return "motion_blur";
        }
        return "unknown";
    }

    PassIsolationStage step_pass_isolation_stage(PassIsolationStage stage, int delta)
    {
        constexpr int kMin = static_cast<int>(PassIsolationStage::Minimal);
        constexpr int kMax = static_cast<int>(PassIsolationStage::MotionBlur);
        int idx = static_cast<int>(stage) + delta;
        idx = std::clamp(idx, kMin, kMax);
        return static_cast<PassIsolationStage>(idx);
    }

    PassExecutionPlan make_pass_execution_plan(
        PassIsolationStage stage,
        bool user_shadow_enabled,
        bool user_light_shafts_enabled,
        bool user_motion_blur_enabled,
        bool profile_shadow_enabled,
        bool profile_motion_blur_enabled)
    {
        const bool allow_shadow = static_cast<int>(stage) >= static_cast<int>(PassIsolationStage::Shadow);
        const bool allow_bright = static_cast<int>(stage) >= static_cast<int>(PassIsolationStage::Bright);
        const bool allow_shafts = static_cast<int>(stage) >= static_cast<int>(PassIsolationStage::Shafts);
        const bool allow_motion_blur = static_cast<int>(stage) >= static_cast<int>(PassIsolationStage::MotionBlur);

        PassExecutionPlan plan{};
        plan.stage = stage;
        plan.run_shadow = allow_shadow && user_shadow_enabled && profile_shadow_enabled;
        plan.run_bright = allow_bright;
        plan.run_shafts = allow_shafts && user_light_shafts_enabled;
        // Flare-ийг shafts toggle-той хамт ажиллуулж, bright pass бэлэн үед л гүйцэтгэнэ.
        plan.run_flare = plan.run_shafts && plan.run_bright;
        plan.enable_motion_blur = allow_motion_blur && user_motion_blur_enabled && profile_motion_blur_enabled;
        return plan;
    }

    bool profile_has_pass(const shs::TechniqueProfile& profile, const char* pass_id)
    {
        for (const auto& p : profile.passes)
        {
            if (p.id == pass_id) return true;
        }
        return false;
    }

    const std::array<shs::TechniqueMode, 5>& known_technique_modes()
    {
        static const std::array<shs::TechniqueMode, 5> modes = {
            shs::TechniqueMode::Forward,
            shs::TechniqueMode::ForwardPlus,
            shs::TechniqueMode::Deferred,
            shs::TechniqueMode::TiledDeferred,
            shs::TechniqueMode::ClusteredForward
        };
        return modes;
    }

    class SdlVulkanRuntime
    {
    public:
        SdlVulkanRuntime(const shs::WindowDesc& win, const shs::SurfaceDesc& surface)
            : surface_w_(surface.width)
            , surface_h_(surface.height)
        {
            if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) return;

            const int img_flags = IMG_INIT_PNG | IMG_INIT_JPG;
            if ((IMG_Init(img_flags) & img_flags) == 0) return;

            window_ = SDL_CreateWindow(
                win.title.c_str(),
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                win.width,
                win.height,
                SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN
            );
            if (!window_) return;
            valid_ = true;
        }

        ~SdlVulkanRuntime()
        {
            if (vk_ && vk_->device() != VK_NULL_HANDLE)
            {
                (void)vkDeviceWaitIdle(vk_->device());
            }
            if (window_) SDL_DestroyWindow(window_);
            IMG_Quit();
            SDL_Quit();
        }

        bool valid() const { return valid_; }

        bool bind_vulkan_backend(shs::VulkanRenderBackend* backend, const char* app_name)
        {
            if (!valid_ || !backend || !window_) return false;
            vk_ = backend;

            int dw = 0;
            int dh = 0;
            SDL_Vulkan_GetDrawableSize(window_, &dw, &dh);
            if (dw <= 0 || dh <= 0)
            {
                dw = surface_w_ > 0 ? surface_w_ : WINDOW_W;
                dh = surface_h_ > 0 ? surface_h_ : WINDOW_H;
            }

            shs::VulkanRenderBackend::InitDesc init{};
            init.window = window_;
            init.width = dw;
            init.height = dh;
            init.enable_validation = true;
            init.app_name = app_name ? app_name : "HelloPassBasicsVulkan";
            return vk_->init(init);
        }

        bool pump_input(shs::PlatformInputState& out)
        {
            out = shs::PlatformInputState{};

            SDL_Event e{};
            while (SDL_PollEvent(&e))
            {
                if (e.type == SDL_QUIT) out.quit = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) out.quit = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_l) out.toggle_light_shafts = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_b) out.toggle_bot = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F1) out.cycle_debug_view = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F2) out.cycle_cull_mode = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F3) out.toggle_front_face = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F4) out.toggle_shading_model = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F5) out.toggle_sky_mode = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F6) out.toggle_follow_camera = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F7) out.toggle_fxaa = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_m) out.toggle_motion_blur = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_LEFTBRACKET) out.step_pass_isolation_prev = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_RIGHTBRACKET) out.step_pass_isolation_next = true;
                if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_RIGHT) out.right_mouse_down = true;
                if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_RIGHT) out.right_mouse_up = true;
                if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) out.left_mouse_down = true;
                if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) out.left_mouse_up = true;
                if (e.type == SDL_MOUSEMOTION)
                {
                    out.mouse_dx += (float)e.motion.xrel;
                    out.mouse_dy += (float)e.motion.yrel;
                }
                if (vk_ &&
                    e.type == SDL_WINDOWEVENT &&
                    (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED || e.window.event == SDL_WINDOWEVENT_RESIZED))
                {
                    vk_->request_resize(e.window.data1, e.window.data2);
                }
            }

            const uint8_t* ks = SDL_GetKeyboardState(nullptr);
            out.forward = ks[SDL_SCANCODE_W] != 0;
            out.backward = ks[SDL_SCANCODE_S] != 0;
            out.left = ks[SDL_SCANCODE_A] != 0;
            out.right = ks[SDL_SCANCODE_D] != 0;
            out.descend = ks[SDL_SCANCODE_Q] != 0;
            out.ascend = ks[SDL_SCANCODE_E] != 0;
            out.boost = ks[SDL_SCANCODE_LSHIFT] != 0;
            return !out.quit;
        }

        void set_relative_mouse_mode(bool enabled)
        {
            SDL_SetRelativeMouseMode(enabled ? SDL_TRUE : SDL_FALSE);
        }

        void set_title(const std::string& title)
        {
            if (window_) SDL_SetWindowTitle(window_, title.c_str());
        }

    private:
        bool valid_ = false;
        int surface_w_ = 0;
        int surface_h_ = 0;
        SDL_Window* window_ = nullptr;
        shs::VulkanRenderBackend* vk_ = nullptr;
    };

    class VulkanSceneRenderer
    {
    public:
        explicit VulkanSceneRenderer(shs::VulkanRenderBackend* backend)
            : vk_(backend)
        {}

        ~VulkanSceneRenderer()
        {
            shutdown();
        }

        bool init()
        {
            if (!vk_) return false;
            if (vk_->device() == VK_NULL_HANDLE) return false;
            if (!create_upload_command_pool()) return false;
            if (!create_descriptor_resources()) return false;
            if (!ensure_white_texture()) return false;
            if (!ensure_offscreen_resources(vk_->swapchain_extent().width, vk_->swapchain_extent().height)) return false;
            if (!ensure_pipelines(shs::CullMode::Back, true)) return false;
            return true;
        }

        void shutdown()
        {
            if (!vk_) return;
            VkDevice dev = vk_->device();
            if (dev == VK_NULL_HANDLE) return;

            (void)vkDeviceWaitIdle(dev);

            for (auto& [_, mesh] : meshes_)
            {
                destroy_mesh(mesh);
            }
            meshes_.clear();

            for (auto& [_, tex] : textures_)
            {
                destroy_texture(tex);
            }
            textures_.clear();

            for (auto& [_, obj] : objects_)
            {
                if (obj.ubo != VK_NULL_HANDLE) vkDestroyBuffer(dev, obj.ubo, nullptr);
                if (obj.umem != VK_NULL_HANDLE) vkFreeMemory(dev, obj.umem, nullptr);
            }
            objects_.clear();
            prev_models_.clear();

            destroy_texture(white_texture_);
            destroy_texture(sky_texture_);
            last_sky_model_ = nullptr;

            destroy_offscreen_resources();
            destroy_pipelines();

            if (sampler_linear_repeat_ != VK_NULL_HANDLE)
            {
                vkDestroySampler(dev, sampler_linear_repeat_, nullptr);
                sampler_linear_repeat_ = VK_NULL_HANDLE;
            }
            if (sampler_linear_clamp_ != VK_NULL_HANDLE)
            {
                vkDestroySampler(dev, sampler_linear_clamp_, nullptr);
                sampler_linear_clamp_ = VK_NULL_HANDLE;
            }
            if (sampler_sky_ != VK_NULL_HANDLE)
            {
                vkDestroySampler(dev, sampler_sky_, nullptr);
                sampler_sky_ = VK_NULL_HANDLE;
            }
            if (sampler_shadow_ != VK_NULL_HANDLE)
            {
                vkDestroySampler(dev, sampler_shadow_, nullptr);
                sampler_shadow_ = VK_NULL_HANDLE;
            }

            if (descriptor_pool_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(dev, descriptor_pool_, nullptr);
                descriptor_pool_ = VK_NULL_HANDLE;
            }

            if (scene_obj_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(dev, scene_obj_layout_, nullptr);
                scene_obj_layout_ = VK_NULL_HANDLE;
            }
            if (bindless_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(dev, bindless_layout_, nullptr);
                bindless_layout_ = VK_NULL_HANDLE;
            }
            if (bindless_pool_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(dev, bindless_pool_, nullptr);
                bindless_pool_ = VK_NULL_HANDLE;
            }
            if (scene_shadow_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(dev, scene_shadow_layout_, nullptr);
                scene_shadow_layout_ = VK_NULL_HANDLE;
            }
            if (single_tex_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(dev, single_tex_layout_, nullptr);
                single_tex_layout_ = VK_NULL_HANDLE;
            }
            if (shafts_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(dev, shafts_layout_, nullptr);
                shafts_layout_ = VK_NULL_HANDLE;
            }
            if (composite_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(dev, composite_layout_, nullptr);
                composite_layout_ = VK_NULL_HANDLE;
            }

            if (upload_cmd_pool_ != VK_NULL_HANDLE)
            {
                vkDestroyCommandPool(dev, upload_cmd_pool_, nullptr);
                upload_cmd_pool_ = VK_NULL_HANDLE;
            }

            prev_viewproj_ = glm::mat4(1.0f);
            offscreen_w_ = 0;
            offscreen_h_ = 0;
            pipeline_gen_ = 0;
        }

        bool render(
            shs::Context& ctx,
            const shs::Scene& scene,
            const shs::FrameParams& fp,
            const shs::ResourceRegistry& resources,
            const PassExecutionPlan& pass_plan,
            bool enable_fxaa
        )
        {
            if (!vk_ || vk_->device() == VK_NULL_HANDLE) return false;

            const VkExtent2D ex = vk_->swapchain_extent();
            if (ex.width == 0 || ex.height == 0) return false;

            shs::RenderBackendFrameInfo frame{};
            frame.frame_index = ctx.frame_index;
            frame.width = static_cast<int>(ex.width);
            frame.height = static_cast<int>(ex.height);

            shs::VulkanRenderBackend::FrameInfo fi{};
            if (!vk_->begin_frame(ctx, frame, fi))
            {
                SDL_Delay(2);
                return false;
            }

            const auto submit_noop_frame = [&]() {
                VkCommandBufferBeginInfo begin_info{};
                begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if (vkBeginCommandBuffer(fi.cmd, &begin_info) != VK_SUCCESS) return;
                if (vkEndCommandBuffer(fi.cmd) != VK_SUCCESS) return;
                vk_->end_frame(fi);
                ctx.frame_index++;
            };

            if (!ensure_offscreen_resources(fi.extent.width, fi.extent.height))
            {
                submit_noop_frame();
                return false;
            }
            if (!ensure_pipelines(fp.cull_mode, fp.front_face_ccw))
            {
                submit_noop_frame();
                return false;
            }
            if (!ensure_white_texture())
            {
                submit_noop_frame();
                return false;
            }
            if (!ensure_sky_texture(scene))
            {
                submit_noop_frame();
                return false;
            }
            if (!update_static_descriptor_sets())
            {
                submit_noop_frame();
                return false;
            }

            for (uint32_t i = 0; i < static_cast<uint32_t>(scene.items.size()); ++i)
            {
                const auto& item = scene.items[i];
                if (!item.visible) continue;
                const shs::MaterialData* mat = resources.get_material((shs::MaterialAssetHandle)item.mat);
                const shs::TextureAssetHandle base_tex_h = mat ? mat->base_color_tex : 0;
                VkDescriptorSet preload_set = VK_NULL_HANDLE;
                if (!ensure_object_descriptor(object_key(item, i), base_tex_h, preload_set, resources))
                {
                    submit_noop_frame();
                    return false;
                }
            }

            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(fi.cmd, &bi) != VK_SUCCESS) return false;

            const LightMatrices light = compute_light_matrices(scene);
            const glm::vec2 sun_uv = compute_sun_uv(scene);

            record_shadow_pass(fi.cmd, scene, resources, light, pass_plan.run_shadow);
            record_scene_pass(fi.cmd, scene, resources, light, fp);
            if (pass_plan.run_bright) record_bright_pass(fi.cmd);
            else clear_post_target(fi.cmd, bright_fb_);
            if (pass_plan.run_shafts) record_shafts_pass(fi.cmd, sun_uv, fp);
            else clear_post_target(fi.cmd, shafts_fb_);
            if (pass_plan.run_flare) record_flare_pass(fi.cmd, sun_uv, fp);
            else clear_post_target(fi.cmd, flare_fb_);
            barrier_color_write_to_shader_read(fi.cmd, scene_hdr_.image);
            barrier_color_write_to_shader_read(fi.cmd, velocity_.image);
            barrier_color_write_to_shader_read(fi.cmd, shafts_.image);
            barrier_color_write_to_shader_read(fi.cmd, flare_.image);
            record_composite_pass(fi.cmd, fp);
            barrier_color_write_to_shader_read(fi.cmd, composite_.image);
            record_fxaa_to_swapchain(fi.cmd, fi, enable_fxaa);

            if (vkEndCommandBuffer(fi.cmd) != VK_SUCCESS) return false;

            prev_viewproj_ = scene.cam.viewproj;
            vk_->end_frame(fi);
            ctx.frame_index++;
            return true;
        }

    private:
        static constexpr uint32_t kShadowMapSize = 2048;

        struct Vertex
        {
            glm::vec3 pos{0.0f};
            glm::vec3 normal{0.0f, 1.0f, 0.0f};
            glm::vec2 uv{0.0f, 0.0f};
        };

        struct ObjectUBO
        {
            glm::mat4 mvp{1.0f};
            glm::mat4 prev_mvp{1.0f};
            glm::mat4 model{1.0f};
            glm::mat4 light_mvp{1.0f};
            glm::vec4 base_color_metallic{1.0f, 1.0f, 1.0f, 0.0f};
            glm::vec4 roughness_ao_emissive_hastex{0.6f, 1.0f, 0.0f, 0.0f};
            glm::vec4 camera_pos_sun_intensity{0.0f, 0.0f, 0.0f, 1.0f};
            glm::vec4 sun_color_pad{1.0f, 0.97f, 0.92f, 0.0f};
            glm::vec4 sun_dir_ws_pad{0.0f, -1.0f, 0.0f, 0.0f};
            glm::vec4 shadow_params{1.0f, 0.0008f, 0.0015f, 1.0f}; // x=strength,y=bias_const,z=bias_slope,w=pcf_step
            glm::uvec4 extra_indices{0u, 0u, 0u, 0u}; // x=texture_index, y,z,w=pad
        };

        struct ShadowPush
        {
            glm::mat4 light_mvp{1.0f};
        };

        struct BrightPush
        {
            float threshold = 1.0f;
            float intensity = 1.0f;
            float knee = 0.5f;
            float pad = 0.0f;
        };

        struct ShaftsPush
        {
            glm::vec2 sun_uv{0.5f, 0.5f};
            float intensity = 0.5f;
            float density = 0.9f;
            float decay = 0.95f;
            float weight = 0.45f;
            float exposure = 1.0f;
            int steps = 40;
        };

        struct FlarePush
        {
            glm::vec2 sun_uv{0.5f, 0.5f};
            float intensity = 0.55f;
            float halo_intensity = 0.35f;
            float chroma_shift = 0.8f;
            int ghosts = 3;
        };

        struct CompositePush
        {
            glm::vec2 inv_size{1.0f, 1.0f};
            float mb_strength = 1.0f;
            float shafts_strength = 1.0f;
            float flare_strength = 1.0f;
            int mb_samples = 10;
            float exposure = 1.35f;
            float gamma = 2.2f;
        };

        struct FxaaPush
        {
            glm::vec2 inv_size{1.0f, 1.0f};
            float enable_fxaa = 1.0f;
            float _pad0 = 0.0f;
        };

        struct GpuMesh
        {
            VkBuffer vb = VK_NULL_HANDLE;
            VkDeviceMemory vmem = VK_NULL_HANDLE;
            VkBuffer ib = VK_NULL_HANDLE;
            VkDeviceMemory imem = VK_NULL_HANDLE;
            uint32_t index_count = 0;
        };

        struct GpuTexture
        {
            VkImage image = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkDescriptorSet set = VK_NULL_HANDLE;
            VkFormat format = VK_FORMAT_UNDEFINED;
            int w = 0;
            int h = 0;
        };

        struct GpuObject
        {
            VkBuffer ubo = VK_NULL_HANDLE;
            VkDeviceMemory umem = VK_NULL_HANDLE;
            VkDescriptorSet set = VK_NULL_HANDLE;
            shs::TextureAssetHandle bound_tex = 0;
            bool has_bound_tex = false;
        };

        struct Target
        {
            VkImage image = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkFormat format = VK_FORMAT_UNDEFINED;
        };

        struct LightMatrices
        {
            glm::mat4 view{1.0f};
            glm::mat4 proj{1.0f};
            glm::mat4 viewproj{1.0f};
        };

        bool load_shader_module(const char* path, VkShaderModule& out_shader) const
        {
            out_shader = VK_NULL_HANDLE;
            std::vector<char> code{};
            if (!shs::vk_try_read_binary_file(path, code))
            {
                return false;
            }
            return shs::vk_try_create_shader_module(vk_->device(), code, out_shader);
        }

        uint64_t object_key(const shs::RenderItem& item, uint32_t draw_index) const
        {
            if (item.object_id != 0) return item.object_id;
            uint64_t k = (uint64_t(item.mesh) << 32) ^ uint64_t(item.mat);
            k ^= (uint64_t(draw_index) + 0x9e3779b97f4a7c15ull + (k << 6) + (k >> 2));
            return k;
        }

        glm::mat4 build_model_matrix(const shs::Transform& tr) const
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, tr.pos);
            model = glm::rotate(model, tr.rot_euler.x, glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::rotate(model, tr.rot_euler.y, glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::rotate(model, tr.rot_euler.z, glm::vec3(0.0f, 0.0f, 1.0f));
            model = glm::scale(model, tr.scl);
            return model;
        }

        LightMatrices compute_light_matrices(const shs::Scene& scene) const
        {
            glm::vec3 center(0.0f);
            if (!scene.items.empty())
            {
                for (const auto& item : scene.items) center += item.tr.pos;
                center /= static_cast<float>(scene.items.size());
            }
            else
            {
                center = scene.cam.pos + glm::normalize(scene.cam.target - scene.cam.pos) * 10.0f;
            }

            float radius = 20.0f;
            for (const auto& item : scene.items)
            {
                radius = std::max(radius, glm::length(item.tr.pos - center) + 10.0f);
            }

            const glm::vec3 light_dir = glm::normalize(scene.sun.dir_ws);
            const glm::vec3 light_pos = center - light_dir * (radius * 2.0f);
            glm::vec3 up(0.0f, 1.0f, 0.0f);
            if (std::abs(glm::dot(up, -light_dir)) > 0.98f) up = glm::vec3(1.0f, 0.0f, 0.0f);

            LightMatrices out{};
            out.view = shs::look_at_lh(light_pos, center, up);
            // Stabilize shadow projection by snapping light-space center to texel grid.
            const float world_units_per_texel = (2.0f * radius) / static_cast<float>(kShadowMapSize);
            const glm::vec4 center_ls4 = out.view * glm::vec4(center, 1.0f);
            const glm::vec2 center_ls(center_ls4.x, center_ls4.y);
            const glm::vec2 snapped_ls = glm::round(center_ls / world_units_per_texel) * world_units_per_texel;
            const glm::vec2 delta = snapped_ls - center_ls;
            out.view = glm::translate(glm::mat4(1.0f), glm::vec3(delta.x, delta.y, 0.0f)) * out.view;
            out.proj = shs::ortho_lh_no(-radius, radius, -radius, radius, 0.1f, radius * 4.5f);

            glm::mat4 clip(1.0f);
            clip[2][2] = 0.5f;
            clip[3][2] = 0.5f;

            out.viewproj = clip * out.proj * out.view;
            return out;
        }

        glm::vec2 compute_sun_uv(const shs::Scene& scene) const
        {
            // Use camera-rotation-only transform for directional sun to avoid
            // translation-induced parallax jitter.
            const glm::vec3 sun_dir_ws = -glm::normalize(scene.sun.dir_ws);
            const glm::vec3 sun_dir_vs = glm::mat3(scene.cam.view) * sun_dir_ws;
            if (sun_dir_vs.z <= 1e-5f) return glm::vec2(-10.0f, -10.0f);

            const glm::vec4 clip = scene.cam.proj * glm::vec4(sun_dir_vs, 1.0f);
            if (std::abs(clip.w) < 1e-6f || clip.w <= 0.0f) return glm::vec2(-10.0f, -10.0f);
            const glm::vec2 ndc = glm::vec2(clip.x, clip.y) / clip.w;
            return ndc * 0.5f + glm::vec2(0.5f, 0.5f);
        }

        void destroy_mesh(GpuMesh& mesh)
        {
            if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
            VkDevice dev = vk_->device();
            if (mesh.vb != VK_NULL_HANDLE) vkDestroyBuffer(dev, mesh.vb, nullptr);
            if (mesh.vmem != VK_NULL_HANDLE) vkFreeMemory(dev, mesh.vmem, nullptr);
            if (mesh.ib != VK_NULL_HANDLE) vkDestroyBuffer(dev, mesh.ib, nullptr);
            if (mesh.imem != VK_NULL_HANDLE) vkFreeMemory(dev, mesh.imem, nullptr);
            mesh = GpuMesh{};
        }

        void destroy_texture(GpuTexture& tex)
        {
            if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
            VkDevice dev = vk_->device();
            if (tex.view != VK_NULL_HANDLE) vkDestroyImageView(dev, tex.view, nullptr);
            if (tex.image != VK_NULL_HANDLE) vkDestroyImage(dev, tex.image, nullptr);
            if (tex.memory != VK_NULL_HANDLE) vkFreeMemory(dev, tex.memory, nullptr);
            tex = GpuTexture{};
        }

        void destroy_target(Target& t)
        {
            if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
            VkDevice dev = vk_->device();
            if (t.view != VK_NULL_HANDLE) vkDestroyImageView(dev, t.view, nullptr);
            if (t.image != VK_NULL_HANDLE) vkDestroyImage(dev, t.image, nullptr);
            if (t.memory != VK_NULL_HANDLE) vkFreeMemory(dev, t.memory, nullptr);
            t = Target{};
        }

        bool create_buffer(
            VkDeviceSize size,
            VkBufferUsageFlags usage,
            VkMemoryPropertyFlags props,
            VkBuffer& out_buffer,
            VkDeviceMemory& out_memory
        )
        {
            return shs::vk_create_buffer(
                vk_->device(),
                vk_->physical_device(),
                size,
                usage,
                props,
                out_buffer,
                out_memory);
        }

        bool create_image(
            uint32_t w,
            uint32_t h,
            VkFormat format,
            VkImageUsageFlags usage,
            VkImageAspectFlags aspect,
            Target& out
        )
        {
            out = Target{};
            out.format = format;

            VkImageCreateInfo ii{};
            ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ii.imageType = VK_IMAGE_TYPE_2D;
            ii.extent.width = w;
            ii.extent.height = h;
            ii.extent.depth = 1;
            ii.mipLevels = 1;
            ii.arrayLayers = 1;
            ii.format = format;
            ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ii.usage = usage;
            ii.samples = VK_SAMPLE_COUNT_1_BIT;
            ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateImage(vk_->device(), &ii, nullptr, &out.image) != VK_SUCCESS) return false;

            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(vk_->device(), out.image, &req);
            const uint32_t mt = shs::vk_find_memory_type(
                vk_->physical_device(),
                req.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (mt == UINT32_MAX)
            {
                vkDestroyImage(vk_->device(), out.image, nullptr);
                out.image = VK_NULL_HANDLE;
                return false;
            }

            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = mt;
            if (vkAllocateMemory(vk_->device(), &ai, nullptr, &out.memory) != VK_SUCCESS)
            {
                vkDestroyImage(vk_->device(), out.image, nullptr);
                out.image = VK_NULL_HANDLE;
                return false;
            }
            if (vkBindImageMemory(vk_->device(), out.image, out.memory, 0) != VK_SUCCESS)
            {
                vkFreeMemory(vk_->device(), out.memory, nullptr);
                vkDestroyImage(vk_->device(), out.image, nullptr);
                out.memory = VK_NULL_HANDLE;
                out.image = VK_NULL_HANDLE;
                return false;
            }

            VkImageViewCreateInfo iv{};
            iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            iv.image = out.image;
            iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            iv.format = format;
            iv.subresourceRange.aspectMask = aspect;
            iv.subresourceRange.baseMipLevel = 0;
            iv.subresourceRange.levelCount = 1;
            iv.subresourceRange.baseArrayLayer = 0;
            iv.subresourceRange.layerCount = 1;
            if (vkCreateImageView(vk_->device(), &iv, nullptr, &out.view) != VK_SUCCESS)
            {
                vkFreeMemory(vk_->device(), out.memory, nullptr);
                vkDestroyImage(vk_->device(), out.image, nullptr);
                out.memory = VK_NULL_HANDLE;
                out.image = VK_NULL_HANDLE;
                return false;
            }
            return true;
        }

        bool create_upload_command_pool()
        {
            if (upload_cmd_pool_ != VK_NULL_HANDLE) return true;
            VkCommandPoolCreateInfo cp{};
            cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cp.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            cp.queueFamilyIndex = vk_->graphics_queue_family_index();
            return vkCreateCommandPool(vk_->device(), &cp, nullptr, &upload_cmd_pool_) == VK_SUCCESS;
        }

        VkCommandBuffer begin_one_time_commands()
        {
            if (upload_cmd_pool_ == VK_NULL_HANDLE) return VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = upload_cmd_pool_;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;

            VkCommandBuffer cmd = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(vk_->device(), &ai, &cmd) != VK_SUCCESS) return VK_NULL_HANDLE;

            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
            {
                vkFreeCommandBuffers(vk_->device(), upload_cmd_pool_, 1, &cmd);
                return VK_NULL_HANDLE;
            }
            return cmd;
        }

        bool end_one_time_commands(VkCommandBuffer cmd)
        {
            if (cmd == VK_NULL_HANDLE) return false;
            if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
            {
                vkFreeCommandBuffers(vk_->device(), upload_cmd_pool_, 1, &cmd);
                return false;
            }

            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            if (vkQueueSubmit(vk_->graphics_queue(), 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
            {
                vkFreeCommandBuffers(vk_->device(), upload_cmd_pool_, 1, &cmd);
                return false;
            }
            if (vkQueueWaitIdle(vk_->graphics_queue()) != VK_SUCCESS)
            {
                vkFreeCommandBuffers(vk_->device(), upload_cmd_pool_, 1, &cmd);
                return false;
            }
            vkFreeCommandBuffers(vk_->device(), upload_cmd_pool_, 1, &cmd);
            return true;
        }



        void transition_color_image(
            VkCommandBuffer cmd,
            VkImage image,
            VkImageLayout old_layout,
            VkImageLayout new_layout
        )
        {
            if (!vk_ || cmd == VK_NULL_HANDLE || image == VK_NULL_HANDLE) return;
            VkPipelineStageFlags2 src_stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            VkPipelineStageFlags2 dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            VkAccessFlags2 src_access = 0;
            VkAccessFlags2 dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT;

            if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            {
                src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                dst_access = VK_ACCESS_2_SHADER_READ_BIT;
                src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                dst_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            }

            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.baseMipLevel = 0;
            range.levelCount = 1;
            range.baseArrayLayer = 0;
            range.layerCount = 1;

            vk_->transition_image_layout(cmd, image, old_layout, new_layout, range, src_stage, src_access, dst_stage, dst_access);
        }

        void barrier_color_write_to_shader_read(VkCommandBuffer cmd, VkImage image)
        {
            if (!vk_ || cmd == VK_NULL_HANDLE || image == VK_NULL_HANDLE) return;
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.baseMipLevel = 0;
            range.levelCount = 1;
            range.baseArrayLayer = 0;
            range.layerCount = 1;
            vk_->transition_image_layout(
                cmd, image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                range,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT);
        }

        bool allocate_single_descriptor(VkDescriptorSetLayout layout, VkDescriptorSet& out_set)
        {
            out_set = VK_NULL_HANDLE;
            if (descriptor_pool_ == VK_NULL_HANDLE || layout == VK_NULL_HANDLE) return false;
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptor_pool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &layout;
            return vkAllocateDescriptorSets(vk_->device(), &ai, &out_set) == VK_SUCCESS;
        }

        bool create_descriptor_resources()
        {
            VkDevice dev = vk_->device();

            if (scene_obj_layout_ == VK_NULL_HANDLE)
            {
                VkDescriptorSetLayoutBinding b[1]{};
                b[0].binding = 1;
                b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                b[0].descriptorCount = 1;
                b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

                VkDescriptorSetLayoutCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                ci.bindingCount = 1;
                ci.pBindings = b;
                if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &scene_obj_layout_) != VK_SUCCESS) return false;
            }

            if (bindless_layout_ == VK_NULL_HANDLE)
            {
                shs::vk_create_bindless_descriptor_set_layout(dev, 4096, &bindless_layout_);
                shs::vk_create_bindless_descriptor_pool(dev, 4096, &bindless_pool_);
            }

            if (scene_shadow_layout_ == VK_NULL_HANDLE)
            {
                VkDescriptorSetLayoutBinding b[2]{};
                b[0].binding = 0;
                b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                b[0].descriptorCount = 1;
                b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                // PBR scene shader дахь environment-aware IBL sampling-д sky map дамжуулна.
                b[1].binding = 1;
                b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                b[1].descriptorCount = 1;
                b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                VkDescriptorSetLayoutCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                ci.bindingCount = 2;
                ci.pBindings = b;
                if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &scene_shadow_layout_) != VK_SUCCESS) return false;
            }

            if (single_tex_layout_ == VK_NULL_HANDLE)
            {
                VkDescriptorSetLayoutBinding b{};
                b.binding = 0;
                b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                b.descriptorCount = 1;
                b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                VkDescriptorSetLayoutCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                ci.bindingCount = 1;
                ci.pBindings = &b;
                if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &single_tex_layout_) != VK_SUCCESS) return false;
            }

            if (shafts_layout_ == VK_NULL_HANDLE)
            {
                VkDescriptorSetLayoutBinding b[2]{};
                b[0].binding = 0;
                b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                b[0].descriptorCount = 1;
                b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                b[1].binding = 1;
                b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                b[1].descriptorCount = 1;
                b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                VkDescriptorSetLayoutCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                ci.bindingCount = 2;
                ci.pBindings = b;
                if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &shafts_layout_) != VK_SUCCESS) return false;
            }

            if (composite_layout_ == VK_NULL_HANDLE)
            {
                VkDescriptorSetLayoutBinding b[4]{};
                for (uint32_t i = 0; i < 4; ++i)
                {
                    b[i].binding = i;
                    b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    b[i].descriptorCount = 1;
                    b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
                }

                VkDescriptorSetLayoutCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                ci.bindingCount = 4;
                ci.pBindings = b;
                if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &composite_layout_) != VK_SUCCESS) return false;
            }

            if (descriptor_pool_ == VK_NULL_HANDLE)
            {
                VkDescriptorPoolSize sizes[] = {
                    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 200},
                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 200}};
                VkDescriptorPoolCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                ci.poolSizeCount = 2;
                ci.pPoolSizes = sizes;
                ci.maxSets = 400;
                ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                if (vkCreateDescriptorPool(dev, &ci, nullptr, &descriptor_pool_) != VK_SUCCESS) return false;
            }

            if (bindless_set_ == VK_NULL_HANDLE)
            {
                VkDescriptorSetVariableDescriptorCountAllocateInfoEXT count_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT};
                uint32_t max_binding = 4096;
                count_info.descriptorSetCount = 1;
                count_info.pDescriptorCounts = &max_binding;
        
                VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                ai.pNext = &count_info;
                ai.descriptorPool = bindless_pool_;
                ai.descriptorSetCount = 1;
                ai.pSetLayouts = &bindless_layout_;
                if (vkAllocateDescriptorSets(vk_->device(), &ai, &bindless_set_) != VK_SUCCESS) return false;
            }

            if (sampler_linear_repeat_ == VK_NULL_HANDLE)
            {
                VkSamplerCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                ci.magFilter = VK_FILTER_LINEAR;
                ci.minFilter = VK_FILTER_LINEAR;
                ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                if (vkCreateSampler(dev, &ci, nullptr, &sampler_linear_repeat_) != VK_SUCCESS) return false;
            }

            if (sampler_linear_clamp_ == VK_NULL_HANDLE)
            {
                VkSamplerCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                ci.magFilter = VK_FILTER_LINEAR;
                ci.minFilter = VK_FILTER_LINEAR;
                ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                if (vkCreateSampler(dev, &ci, nullptr, &sampler_linear_clamp_) != VK_SUCCESS) return false;
            }

            if (sampler_sky_ == VK_NULL_HANDLE)
            {
                VkSamplerCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                ci.magFilter = VK_FILTER_LINEAR;
                ci.minFilter = VK_FILTER_LINEAR;
                ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                if (vkCreateSampler(dev, &ci, nullptr, &sampler_sky_) != VK_SUCCESS) return false;
            }

            if (sampler_shadow_ == VK_NULL_HANDLE)
            {
                VkSamplerCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                ci.magFilter = VK_FILTER_LINEAR;
                ci.minFilter = VK_FILTER_LINEAR;
                ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
                if (vkCreateSampler(dev, &ci, nullptr, &sampler_shadow_) != VK_SUCCESS) return false;
            }

            if (shadow_set_ == VK_NULL_HANDLE && !allocate_single_descriptor(scene_shadow_layout_, shadow_set_)) return false;
            if (sky_set_ == VK_NULL_HANDLE && !allocate_single_descriptor(single_tex_layout_, sky_set_)) return false;
            if (bright_set_ == VK_NULL_HANDLE && !allocate_single_descriptor(single_tex_layout_, bright_set_)) return false;
            if (flare_set_ == VK_NULL_HANDLE && !allocate_single_descriptor(single_tex_layout_, flare_set_)) return false;
            if (fxaa_set_ == VK_NULL_HANDLE && !allocate_single_descriptor(single_tex_layout_, fxaa_set_)) return false;
            if (shafts_set_ == VK_NULL_HANDLE && !allocate_single_descriptor(shafts_layout_, shafts_set_)) return false;
            if (composite_set_ == VK_NULL_HANDLE && !allocate_single_descriptor(composite_layout_, composite_set_)) return false;

            return true;
        }

        bool update_static_descriptor_sets()
        {
            if (!vk_ || vk_->device() == VK_NULL_HANDLE) return false;
            if (shadow_depth_.view == VK_NULL_HANDLE || scene_hdr_.view == VK_NULL_HANDLE ||
                scene_depth_.view == VK_NULL_HANDLE || bright_.view == VK_NULL_HANDLE ||
                shafts_.view == VK_NULL_HANDLE || flare_.view == VK_NULL_HANDLE || composite_.view == VK_NULL_HANDLE)
            {
                return false;
            }

            VkDescriptorImageInfo shadow_info{};
            shadow_info.sampler = sampler_shadow_;
            shadow_info.imageView = shadow_depth_.view;
            shadow_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo sky_info{};
            sky_info.sampler = (sampler_sky_ != VK_NULL_HANDLE) ? sampler_sky_ : sampler_linear_clamp_;
            sky_info.imageView = (sky_texture_.view != VK_NULL_HANDLE) ? sky_texture_.view : white_texture_.view;
            sky_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo scene_info{};
            scene_info.sampler = sampler_linear_clamp_;
            scene_info.imageView = scene_hdr_.view;
            scene_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo velocity_info{};
            velocity_info.sampler = sampler_linear_clamp_;
            velocity_info.imageView = velocity_.view;
            velocity_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo depth_info{};
            depth_info.sampler = sampler_linear_clamp_;
            depth_info.imageView = scene_depth_.view;
            depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo bright_info{};
            bright_info.sampler = sampler_linear_clamp_;
            bright_info.imageView = bright_.view;
            bright_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo shafts_info{};
            shafts_info.sampler = sampler_linear_clamp_;
            shafts_info.imageView = shafts_.view;
            shafts_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo flare_info{};
            flare_info.sampler = sampler_linear_clamp_;
            flare_info.imageView = flare_.view;
            flare_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo composite_info{};
            composite_info.sampler = sampler_linear_clamp_;
            composite_info.imageView = composite_.view;
            composite_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet w[13]{};
            uint32_t n = 0;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = shadow_set_;
            w[n].dstBinding = 0;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &shadow_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = shadow_set_;
            w[n].dstBinding = 1;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &sky_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = sky_set_;
            w[n].dstBinding = 0;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &sky_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = bright_set_;
            w[n].dstBinding = 0;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &scene_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = flare_set_;
            w[n].dstBinding = 0;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &bright_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = fxaa_set_;
            w[n].dstBinding = 0;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &composite_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = shafts_set_;
            w[n].dstBinding = 0;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &bright_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = shafts_set_;
            w[n].dstBinding = 1;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &depth_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = composite_set_;
            w[n].dstBinding = 0;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &scene_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = composite_set_;
            w[n].dstBinding = 1;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &velocity_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = composite_set_;
            w[n].dstBinding = 2;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &shafts_info;
            ++n;

            w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[n].dstSet = composite_set_;
            w[n].dstBinding = 3;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[n].pImageInfo = &flare_info;
            ++n;

            vkUpdateDescriptorSets(vk_->device(), n, w, 0, nullptr);
            return true;
        }

        bool create_texture_from_rgba(const uint8_t* rgba, int w, int h, GpuTexture& out_tex)
        {
            if (!rgba || w <= 0 || h <= 0) return false;
            const size_t bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 4u;

            VkBuffer staging = VK_NULL_HANDLE;
            VkDeviceMemory staging_mem = VK_NULL_HANDLE;
            const VkMemoryPropertyFlags host_visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if (!create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_visible, staging, staging_mem)) return false;

            void* mapped = nullptr;
            if (vkMapMemory(vk_->device(), staging_mem, 0, bytes, 0, &mapped) != VK_SUCCESS)
            {
                vkDestroyBuffer(vk_->device(), staging, nullptr);
                vkFreeMemory(vk_->device(), staging_mem, nullptr);
                return false;
            }
            std::memcpy(mapped, rgba, bytes);
            vkUnmapMemory(vk_->device(), staging_mem);

            Target t{};
            if (!create_image(
                    static_cast<uint32_t>(w),
                    static_cast<uint32_t>(h),
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    t))
            {
                vkDestroyBuffer(vk_->device(), staging, nullptr);
                vkFreeMemory(vk_->device(), staging_mem, nullptr);
                return false;
            }

            VkCommandBuffer cmd = begin_one_time_commands();
            if (cmd == VK_NULL_HANDLE)
            {
                destroy_target(t);
                vkDestroyBuffer(vk_->device(), staging, nullptr);
                vkFreeMemory(vk_->device(), staging_mem, nullptr);
                return false;
            }

            transition_color_image(cmd, t.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            VkBufferImageCopy copy{};
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.mipLevel = 0;
            copy.imageSubresource.baseArrayLayer = 0;
            copy.imageSubresource.layerCount = 1;
            copy.imageExtent.width = static_cast<uint32_t>(w);
            copy.imageExtent.height = static_cast<uint32_t>(h);
            copy.imageExtent.depth = 1;
            vkCmdCopyBufferToImage(cmd, staging, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

            transition_color_image(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            if (!end_one_time_commands(cmd))
            {
                destroy_target(t);
                vkDestroyBuffer(vk_->device(), staging, nullptr);
                vkFreeMemory(vk_->device(), staging_mem, nullptr);
                return false;
            }

            vkDestroyBuffer(vk_->device(), staging, nullptr);
            vkFreeMemory(vk_->device(), staging_mem, nullptr);

            out_tex.image = t.image;
            out_tex.memory = t.memory;
            out_tex.view = t.view;
            out_tex.format = VK_FORMAT_R8G8B8A8_UNORM;
            out_tex.w = w;
            out_tex.h = h;
            out_tex.set = VK_NULL_HANDLE;
            return true;
        }

        bool ensure_white_texture()
        {
            if (white_texture_.view != VK_NULL_HANDLE) return true;
            const uint8_t white[] = {255u, 255u, 255u, 255u};
            if (!create_texture_from_rgba(white, 1, 1, white_texture_)) return false;

            if (bindless_set_ != VK_NULL_HANDLE && sampler_linear_repeat_ != VK_NULL_HANDLE)
            {
                bindless_indices_[0] = 0; // AssetHandle 0 -> bindless index 0
                next_bindless_index_ = std::max(next_bindless_index_, 1u);
                shs::vk_update_bindless_texture(vk_->device(), bindless_set_, 0, sampler_linear_repeat_, white_texture_.view);
            }
            return true;
        }

        bool ensure_sky_texture(const shs::Scene& scene)
        {
            if (!scene.sky)
            {
                if (sky_texture_.image != VK_NULL_HANDLE) destroy_texture(sky_texture_);
                sky_texture_ = GpuTexture{};
                last_sky_model_ = nullptr;
                return true;
            }

            if (scene.sky == last_sky_model_ && sky_texture_.view != VK_NULL_HANDLE) return true;

            if (sky_texture_.image != VK_NULL_HANDLE)
            {
                destroy_texture(sky_texture_);
            }
            sky_texture_ = GpuTexture{};

            constexpr int sky_w = 1024;
            constexpr int sky_h = 512;
            std::vector<uint8_t> rgba(static_cast<size_t>(sky_w) * static_cast<size_t>(sky_h) * 4u, 0u);

            for (int y = 0; y < sky_h; ++y)
            {
                const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(sky_h);
                const float lat = (0.5f - v) * PI;
                const float sin_lat = std::sin(lat);
                const float cos_lat = std::cos(lat);
                for (int x = 0; x < sky_w; ++x)
                {
                    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(sky_w);
                    const float lon = (u - 0.5f) * TWO_PI;
                    const glm::vec3 dir{
                        cos_lat * std::cos(lon),
                        sin_lat,
                        cos_lat * std::sin(lon)
                    };

                    glm::vec3 c = scene.sky->sample(dir);
                    c = glm::max(c, glm::vec3(0.0f));
                    // Sky texture is sampled as linear UNORM in Vulkan.
                    // Keep values linear in [0,1] (no extra gamma-encoding)
                    // to avoid over-bright/double-curve sky output.
                    c = c / (glm::vec3(1.0f) + c);
                    c = glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f));

                    const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(sky_w) + static_cast<size_t>(x)) * 4u;
                    rgba[idx + 0] = static_cast<uint8_t>(std::lround(c.r * 255.0f));
                    rgba[idx + 1] = static_cast<uint8_t>(std::lround(c.g * 255.0f));
                    rgba[idx + 2] = static_cast<uint8_t>(std::lround(c.b * 255.0f));
                    rgba[idx + 3] = 255u;
                }
            }

            if (!create_texture_from_rgba(rgba.data(), sky_w, sky_h, sky_texture_))
            {
                sky_texture_ = white_texture_;
                return false;
            }

            last_sky_model_ = scene.sky;
            return true;
        }

        bool ensure_mesh_uploaded(shs::MeshAssetHandle mesh_h, const shs::MeshData& mesh)
        {
            if (meshes_.find(mesh_h) != meshes_.end()) return true;

            std::vector<Vertex> verts{};
            verts.resize(mesh.positions.size());
            for (size_t i = 0; i < mesh.positions.size(); ++i)
            {
                verts[i].pos = mesh.positions[i];
                verts[i].normal = (i < mesh.normals.size()) ? mesh.normals[i] : glm::vec3(0.0f, 1.0f, 0.0f);
                verts[i].uv = (i < mesh.uvs.size()) ? mesh.uvs[i] : glm::vec2(0.0f, 0.0f);
            }

            std::vector<uint32_t> indices{};
            if (!mesh.indices.empty())
            {
                indices = mesh.indices;
            }
            else
            {
                indices.resize(mesh.positions.size());
                for (uint32_t i = 0; i < static_cast<uint32_t>(indices.size()); ++i) indices[i] = i;
            }

            if (verts.empty() || indices.empty()) return false;

            GpuMesh gm{};
            const VkDeviceSize vb_size = sizeof(Vertex) * verts.size();
            const VkDeviceSize ib_size = sizeof(uint32_t) * indices.size();
            const VkMemoryPropertyFlags host_visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            if (!create_buffer(vb_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, host_visible, gm.vb, gm.vmem)) return false;
            if (!create_buffer(ib_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, host_visible, gm.ib, gm.imem))
            {
                destroy_mesh(gm);
                return false;
            }

            void* mapped = nullptr;
            if (vkMapMemory(vk_->device(), gm.vmem, 0, vb_size, 0, &mapped) != VK_SUCCESS)
            {
                destroy_mesh(gm);
                return false;
            }
            std::memcpy(mapped, verts.data(), static_cast<size_t>(vb_size));
            vkUnmapMemory(vk_->device(), gm.vmem);

            if (vkMapMemory(vk_->device(), gm.imem, 0, ib_size, 0, &mapped) != VK_SUCCESS)
            {
                destroy_mesh(gm);
                return false;
            }
            std::memcpy(mapped, indices.data(), static_cast<size_t>(ib_size));
            vkUnmapMemory(vk_->device(), gm.imem);

            gm.index_count = static_cast<uint32_t>(indices.size());
            meshes_[mesh_h] = gm;
            return true;
        }

        bool ensure_object_descriptor(
            uint64_t key,
            shs::TextureAssetHandle tex_h,
            VkDescriptorSet& out_set,
            const shs::ResourceRegistry& resources)
        {
            auto it = objects_.find(key);
            if (it == objects_.end())
            {
                GpuObject obj{};
                if (!create_buffer(
                        sizeof(ObjectUBO),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        obj.ubo,
                        obj.umem))
                {
                    return false;
                }

                if (!allocate_single_descriptor(scene_obj_layout_, obj.set))
                {
                    vkDestroyBuffer(vk_->device(), obj.ubo, nullptr);
                    vkFreeMemory(vk_->device(), obj.umem, nullptr);
                    return false;
                }

                objects_.emplace(key, obj);
                it = objects_.find(key);
            }

            GpuObject& obj = it->second;

            GpuTexture* tex = &white_texture_;
            bool has_tex = false;
            if (tex_h != 0)
            {
                const auto it_tex = textures_.find(tex_h);
                if (it_tex != textures_.end())
                {
                    tex = &textures_.at(tex_h);
                    has_tex = true;
                }
                else
                {
                    const shs::Texture2DData* src = resources.get_texture(tex_h);
                    if (src && src->valid())
                    {
                        std::vector<uint8_t> rgba(static_cast<size_t>(src->w) * static_cast<size_t>(src->h) * 4u, 0u);
                        for (int y = 0; y < src->h; ++y)
                        {
                            for (int x = 0; x < src->w; ++x)
                            {
                                const shs::Color c = src->at(x, y);
                                const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(src->w) + static_cast<size_t>(x)) * 4u;
                                rgba[idx + 0] = c.r;
                                rgba[idx + 1] = c.g;
                                rgba[idx + 2] = c.b;
                                rgba[idx + 3] = c.a;
                            }
                        }

                        GpuTexture gt{};
                        if (create_texture_from_rgba(rgba.data(), src->w, src->h, gt))
                        {
                            textures_[tex_h] = gt;
                            tex = &textures_.at(tex_h);
                            has_tex = true;
                            
                            if (bindless_set_ != VK_NULL_HANDLE && sampler_linear_repeat_ != VK_NULL_HANDLE)
                            {
                                uint32_t b_idx = next_bindless_index_++;
                                bindless_indices_[tex_h] = b_idx;
                                shs::vk_update_bindless_texture(vk_->device(), bindless_set_, b_idx, sampler_linear_repeat_, gt.view);
                            }
                        }
                    }
                }
            }

            const shs::TextureAssetHandle desired_tex = has_tex ? tex_h : 0;
            const bool tex_changed = (!obj.has_bound_tex) || (obj.bound_tex != desired_tex);
            if (tex_changed || obj.bound_tex == 0)
            {
                VkDescriptorBufferInfo buf{};
                buf.buffer = obj.ubo;
                buf.offset = 0;
                buf.range = sizeof(ObjectUBO);

                VkWriteDescriptorSet w[1]{};
                w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[0].dstSet = obj.set;
                w[0].dstBinding = 1;
                w[0].descriptorCount = 1;
                w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w[0].pBufferInfo = &buf;

                vkUpdateDescriptorSets(vk_->device(), 1, w, 0, nullptr);

                obj.bound_tex = desired_tex;
                obj.has_bound_tex = true;
            }

            out_set = obj.set;
            return true;
        }

        bool update_object_ubo(uint64_t key, const ObjectUBO& ubo)
        {
            auto it = objects_.find(key);
            if (it == objects_.end()) return false;

            void* mapped = nullptr;
            if (vkMapMemory(vk_->device(), it->second.umem, 0, sizeof(ObjectUBO), 0, &mapped) != VK_SUCCESS) return false;
            std::memcpy(mapped, &ubo, sizeof(ObjectUBO));
            vkUnmapMemory(vk_->device(), it->second.umem);
            return true;
        }

        void destroy_render_pass_and_fb(VkRenderPass& rp, VkFramebuffer& fb)
        {
            if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
            VkDevice dev = vk_->device();
            if (fb != VK_NULL_HANDLE)
            {
                vkDestroyFramebuffer(dev, fb, nullptr);
                fb = VK_NULL_HANDLE;
            }
            if (rp != VK_NULL_HANDLE)
            {
                vkDestroyRenderPass(dev, rp, nullptr);
                rp = VK_NULL_HANDLE;
            }
        }

        void destroy_offscreen_resources()
        {
            if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
            VkDevice dev = vk_->device();

            if (shadow_fb_ != VK_NULL_HANDLE) { vkDestroyFramebuffer(dev, shadow_fb_, nullptr); shadow_fb_ = VK_NULL_HANDLE; }
            if (scene_fb_ != VK_NULL_HANDLE) { vkDestroyFramebuffer(dev, scene_fb_, nullptr); scene_fb_ = VK_NULL_HANDLE; }
            if (bright_fb_ != VK_NULL_HANDLE) { vkDestroyFramebuffer(dev, bright_fb_, nullptr); bright_fb_ = VK_NULL_HANDLE; }
            if (shafts_fb_ != VK_NULL_HANDLE) { vkDestroyFramebuffer(dev, shafts_fb_, nullptr); shafts_fb_ = VK_NULL_HANDLE; }
            if (flare_fb_ != VK_NULL_HANDLE) { vkDestroyFramebuffer(dev, flare_fb_, nullptr); flare_fb_ = VK_NULL_HANDLE; }
            if (composite_fb_ != VK_NULL_HANDLE) { vkDestroyFramebuffer(dev, composite_fb_, nullptr); composite_fb_ = VK_NULL_HANDLE; }

            if (shadow_render_pass_ != VK_NULL_HANDLE) { vkDestroyRenderPass(dev, shadow_render_pass_, nullptr); shadow_render_pass_ = VK_NULL_HANDLE; }
            if (scene_render_pass_ != VK_NULL_HANDLE) { vkDestroyRenderPass(dev, scene_render_pass_, nullptr); scene_render_pass_ = VK_NULL_HANDLE; }
            if (post_render_pass_ != VK_NULL_HANDLE) { vkDestroyRenderPass(dev, post_render_pass_, nullptr); post_render_pass_ = VK_NULL_HANDLE; }

            destroy_target(shadow_depth_);
            destroy_target(scene_hdr_);
            destroy_target(velocity_);
            destroy_target(scene_depth_);
            destroy_target(bright_);
            destroy_target(shafts_);
            destroy_target(flare_);
            destroy_target(composite_);

            offscreen_w_ = 0;
            offscreen_h_ = 0;
        }

        bool create_shadow_pass_resources(uint32_t w, uint32_t h)
        {
            const VkFormat depth_fmt = (vk_->depth_format() != VK_FORMAT_UNDEFINED) ? vk_->depth_format() : VK_FORMAT_D32_SFLOAT;

            if (!create_image(
                    kShadowMapSize,
                    kShadowMapSize,
                    depth_fmt,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    shadow_depth_))
            {
                return false;
            }

            VkAttachmentDescription depth{};
            depth.format = depth_fmt;
            depth.samples = VK_SAMPLE_COUNT_1_BIT;
            depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkAttachmentReference depth_ref{};
            depth_ref.attachment = 0;
            depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.pDepthStencilAttachment = &depth_ref;

            VkSubpassDependency deps[2]{};
            deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            deps[0].dstSubpass = 0;
            deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            deps[1].srcSubpass = 0;
            deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            VkRenderPassCreateInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            rp.attachmentCount = 1;
            rp.pAttachments = &depth;
            rp.subpassCount = 1;
            rp.pSubpasses = &sub;
            rp.dependencyCount = 2;
            rp.pDependencies = deps;
            if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &shadow_render_pass_) != VK_SUCCESS) return false;

            VkFramebufferCreateInfo fb{};
            fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb.renderPass = shadow_render_pass_;
            fb.attachmentCount = 1;
            fb.pAttachments = &shadow_depth_.view;
            fb.width = kShadowMapSize;
            fb.height = kShadowMapSize;
            fb.layers = 1;
            if (vkCreateFramebuffer(vk_->device(), &fb, nullptr, &shadow_fb_) != VK_SUCCESS) return false;

            (void)w;
            (void)h;
            return true;
        }

        bool create_scene_pass_resources(uint32_t w, uint32_t h)
        {
            const VkFormat hdr_fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
            const VkFormat vel_fmt = VK_FORMAT_R16G16_SFLOAT;
            const VkFormat depth_fmt = (vk_->depth_format() != VK_FORMAT_UNDEFINED) ? vk_->depth_format() : VK_FORMAT_D32_SFLOAT;

            if (!create_image(
                    w,
                    h,
                    hdr_fmt,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    scene_hdr_))
            {
                return false;
            }

            if (!create_image(
                    w,
                    h,
                    vel_fmt,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    velocity_))
            {
                return false;
            }

            if (!create_image(
                    w,
                    h,
                    depth_fmt,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    scene_depth_))
            {
                return false;
            }

            VkAttachmentDescription att[3]{};
            att[0].format = hdr_fmt;
            att[0].samples = VK_SAMPLE_COUNT_1_BIT;
            att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            att[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            att[1].format = vel_fmt;
            att[1].samples = VK_SAMPLE_COUNT_1_BIT;
            att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            att[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            att[2].format = depth_fmt;
            att[2].samples = VK_SAMPLE_COUNT_1_BIT;
            att[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            att[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            att[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            att[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkAttachmentReference color_refs[2]{};
            color_refs[0].attachment = 0;
            color_refs[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_refs[1].attachment = 1;
            color_refs[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentReference depth_ref{};
            depth_ref.attachment = 2;
            depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 2;
            sub.pColorAttachments = color_refs;
            sub.pDepthStencilAttachment = &depth_ref;

            VkSubpassDependency deps[2]{};
            deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            deps[0].dstSubpass = 0;
            deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            deps[1].srcSubpass = 0;
            deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            VkRenderPassCreateInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            rp.attachmentCount = 3;
            rp.pAttachments = att;
            rp.subpassCount = 1;
            rp.pSubpasses = &sub;
            rp.dependencyCount = 2;
            rp.pDependencies = deps;
            if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &scene_render_pass_) != VK_SUCCESS) return false;

            VkImageView views[3] = {scene_hdr_.view, velocity_.view, scene_depth_.view};
            VkFramebufferCreateInfo fb{};
            fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb.renderPass = scene_render_pass_;
            fb.attachmentCount = 3;
            fb.pAttachments = views;
            fb.width = w;
            fb.height = h;
            fb.layers = 1;
            if (vkCreateFramebuffer(vk_->device(), &fb, nullptr, &scene_fb_) != VK_SUCCESS) return false;

            return true;
        }

        bool create_post_pass_resources(uint32_t w, uint32_t h)
        {
            const VkFormat hdr_fmt = VK_FORMAT_R16G16B16A16_SFLOAT;

            if (!create_image(w, h, hdr_fmt,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    bright_)) return false;

            if (!create_image(w, h, hdr_fmt,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    shafts_)) return false;

            if (!create_image(w, h, hdr_fmt,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    flare_)) return false;

            if (!create_image(w, h, hdr_fmt,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    composite_)) return false;

            VkAttachmentDescription color{};
            color.format = hdr_fmt;
            color.samples = VK_SAMPLE_COUNT_1_BIT;
            color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            color.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkAttachmentReference color_ref{};
            color_ref.attachment = 0;
            color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1;
            sub.pColorAttachments = &color_ref;

            VkSubpassDependency deps[2]{};
            deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            deps[0].dstSubpass = 0;
            deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            deps[1].srcSubpass = 0;
            deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            VkRenderPassCreateInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            rp.attachmentCount = 1;
            rp.pAttachments = &color;
            rp.subpassCount = 1;
            rp.pSubpasses = &sub;
            rp.dependencyCount = 2;
            rp.pDependencies = deps;
            if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &post_render_pass_) != VK_SUCCESS) return false;

            auto create_fb = [&](VkImageView view, VkFramebuffer& out_fb) -> bool {
                VkFramebufferCreateInfo fb{};
                fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                fb.renderPass = post_render_pass_;
                fb.attachmentCount = 1;
                fb.pAttachments = &view;
                fb.width = w;
                fb.height = h;
                fb.layers = 1;
                return vkCreateFramebuffer(vk_->device(), &fb, nullptr, &out_fb) == VK_SUCCESS;
            };

            if (!create_fb(bright_.view, bright_fb_)) return false;
            if (!create_fb(shafts_.view, shafts_fb_)) return false;
            if (!create_fb(flare_.view, flare_fb_)) return false;
            if (!create_fb(composite_.view, composite_fb_)) return false;

            return true;
        }

        bool ensure_offscreen_resources(uint32_t w, uint32_t h)
        {
            if (w == 0 || h == 0) return false;
            if (offscreen_w_ == w && offscreen_h_ == h && shadow_fb_ != VK_NULL_HANDLE && scene_fb_ != VK_NULL_HANDLE && post_render_pass_ != VK_NULL_HANDLE)
            {
                return true;
            }

            destroy_pipelines();
            destroy_offscreen_resources();

            if (!create_shadow_pass_resources(w, h)) return false;
            if (!create_scene_pass_resources(w, h)) return false;
            if (!create_post_pass_resources(w, h)) return false;

            offscreen_w_ = w;
            offscreen_h_ = h;
            return true;
        }

        VkCullModeFlags to_vk_cull(shs::CullMode mode) const
        {
            switch (mode)
            {
                case shs::CullMode::None: return VK_CULL_MODE_NONE;
                case shs::CullMode::Front: return VK_CULL_MODE_FRONT_BIT;
                case shs::CullMode::Back:
                default: return VK_CULL_MODE_BACK_BIT;
            }
        }

        bool create_pipeline_layout(
            VkPipelineLayout& out,
            const VkDescriptorSetLayout* set_layouts,
            uint32_t set_count,
            VkShaderStageFlags push_stage,
            uint32_t push_size)
        {
            out = VK_NULL_HANDLE;
            VkPushConstantRange pcr{};
            pcr.stageFlags = push_stage;
            pcr.offset = 0;
            pcr.size = push_size;

            VkPipelineLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            ci.setLayoutCount = set_count;
            ci.pSetLayouts = set_layouts;
            if (push_size > 0)
            {
                ci.pushConstantRangeCount = 1;
                ci.pPushConstantRanges = &pcr;
            }
            return vkCreatePipelineLayout(vk_->device(), &ci, nullptr, &out) == VK_SUCCESS;
        }

        bool create_shadow_pipeline(shs::CullMode cull_mode, bool front_face_ccw)
        {
            VkShaderModule vs = VK_NULL_HANDLE;
            if (!load_shader_module(SHS_VK_PB_SHADOW_VERT_SPV, vs)) return false;

            if (!create_pipeline_layout(shadow_pipeline_layout_, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, sizeof(ShadowPush)))
            {
                vkDestroyShaderModule(vk_->device(), vs, nullptr);
                return false;
            }

            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
            stage.module = vs;
            stage.pName = "main";

            VkVertexInputBindingDescription binding{};
            binding.binding = 0;
            binding.stride = sizeof(Vertex);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription attr{};
            attr.location = 0;
            attr.binding = 0;
            attr.format = VK_FORMAT_R32G32B32_SFLOAT;
            attr.offset = offsetof(Vertex, pos);

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount = 1;
            vi.pVertexBindingDescriptions = &binding;
            vi.vertexAttributeDescriptionCount = 1;
            vi.pVertexAttributeDescriptions = &attr;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = to_vk_cull(cull_mode);
            rs.frontFace = front_face_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
            rs.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineDynamicStateCreateInfo dyn{};
            VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 2;
            dyn.pDynamicStates = dyn_states;

            VkGraphicsPipelineCreateInfo gp{};
            gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gp.stageCount = 1;
            gp.pStages = &stage;
            gp.pVertexInputState = &vi;
            gp.pInputAssemblyState = &ia;
            gp.pViewportState = &vp;
            gp.pRasterizationState = &rs;
            gp.pMultisampleState = &ms;
            gp.pDepthStencilState = &ds;
            gp.pDynamicState = &dyn;
            gp.layout = shadow_pipeline_layout_;
            gp.renderPass = shadow_render_pass_;
            gp.subpass = 0;

            const VkResult res = vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp, nullptr, &shadow_pipeline_);
            vkDestroyShaderModule(vk_->device(), vs, nullptr);
            return res == VK_SUCCESS;
        }

        bool create_scene_pipeline(shs::CullMode cull_mode, bool front_face_ccw)
        {
            VkShaderModule vs = VK_NULL_HANDLE;
            VkShaderModule fs = VK_NULL_HANDLE;
            if (!load_shader_module(SHS_VK_PB_SCENE_VERT_SPV, vs)) return false;
            if (!load_shader_module(SHS_VK_PB_SCENE_FRAG_SPV, fs))
            {
                vkDestroyShaderModule(vk_->device(), vs, nullptr);
                return false;
            }

            VkDescriptorSetLayout sets[3] = {scene_obj_layout_, scene_shadow_layout_, bindless_layout_};
            if (!create_pipeline_layout(scene_pipeline_layout_, sets, 3, 0, 0))
            {
                vkDestroyShaderModule(vk_->device(), vs, nullptr);
                vkDestroyShaderModule(vk_->device(), fs, nullptr);
                return false;
            }

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vs;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fs;
            stages[1].pName = "main";

            VkVertexInputBindingDescription binding{};
            binding.binding = 0;
            binding.stride = sizeof(Vertex);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription attrs[3]{};
            attrs[0].location = 0;
            attrs[0].binding = 0;
            attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            attrs[0].offset = offsetof(Vertex, pos);
            attrs[1].location = 1;
            attrs[1].binding = 0;
            attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attrs[1].offset = offsetof(Vertex, normal);
            attrs[2].location = 2;
            attrs[2].binding = 0;
            attrs[2].format = VK_FORMAT_R32G32_SFLOAT;
            attrs[2].offset = offsetof(Vertex, uv);

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount = 1;
            vi.pVertexBindingDescriptions = &binding;
            vi.vertexAttributeDescriptionCount = 3;
            vi.pVertexAttributeDescriptions = attrs;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = to_vk_cull(cull_mode);
            rs.frontFace = front_face_ccw ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendAttachmentState cba[2]{};
            for (int i = 0; i < 2; ++i)
            {
                cba[i].colorWriteMask =
                    VK_COLOR_COMPONENT_R_BIT |
                    VK_COLOR_COMPONENT_G_BIT |
                    VK_COLOR_COMPONENT_B_BIT |
                    VK_COLOR_COMPONENT_A_BIT;
                cba[i].blendEnable = VK_FALSE;
            }

            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 2;
            cb.pAttachments = cba;

            VkPipelineDynamicStateCreateInfo dyn{};
            VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 2;
            dyn.pDynamicStates = dyn_states;

            VkGraphicsPipelineCreateInfo gp{};
            gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gp.stageCount = 2;
            gp.pStages = stages;
            gp.pVertexInputState = &vi;
            gp.pInputAssemblyState = &ia;
            gp.pViewportState = &vp;
            gp.pRasterizationState = &rs;
            gp.pMultisampleState = &ms;
            gp.pDepthStencilState = &ds;
            gp.pColorBlendState = &cb;
            gp.pDynamicState = &dyn;
            gp.layout = scene_pipeline_layout_;
            gp.renderPass = scene_render_pass_;
            gp.subpass = 0;

            const VkResult res = vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp, nullptr, &scene_pipeline_);
            vkDestroyShaderModule(vk_->device(), vs, nullptr);
            vkDestroyShaderModule(vk_->device(), fs, nullptr);
            return res == VK_SUCCESS;
        }

        bool create_fullscreen_pipeline(
            const char* frag_path,
            VkRenderPass render_pass,
            VkDescriptorSetLayout set_layout,
            VkShaderStageFlags push_stage,
            uint32_t push_size,
            VkPipelineLayout& out_layout,
            VkPipeline& out_pipeline,
            uint32_t color_attachment_count = 1)
        {
            VkShaderModule vs = VK_NULL_HANDLE;
            VkShaderModule fs = VK_NULL_HANDLE;
            if (!load_shader_module(SHS_VK_PB_POST_VERT_SPV, vs)) return false;
            if (!load_shader_module(frag_path, fs))
            {
                vkDestroyShaderModule(vk_->device(), vs, nullptr);
                return false;
            }

            if (!create_pipeline_layout(out_layout, &set_layout, 1, push_stage, push_size))
            {
                vkDestroyShaderModule(vk_->device(), vs, nullptr);
                vkDestroyShaderModule(vk_->device(), fs, nullptr);
                return false;
            }

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vs;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fs;
            stages[1].pName = "main";

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable = VK_FALSE;
            ds.depthWriteEnable = VK_FALSE;
            ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;

            std::array<VkPipelineColorBlendAttachmentState, 2> cba{};
            cba[0].colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
            cba[0].blendEnable = VK_FALSE;
            cba[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
            cba[1].blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = std::clamp<uint32_t>(color_attachment_count, 1u, 2u);
            cb.pAttachments = cba.data();

            VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 2;
            dyn.pDynamicStates = dyn_states;

            VkGraphicsPipelineCreateInfo gp{};
            gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gp.stageCount = 2;
            gp.pStages = stages;
            gp.pVertexInputState = &vi;
            gp.pInputAssemblyState = &ia;
            gp.pViewportState = &vp;
            gp.pRasterizationState = &rs;
            gp.pMultisampleState = &ms;
            gp.pDepthStencilState = &ds;
            gp.pColorBlendState = &cb;
            gp.pDynamicState = &dyn;
            gp.layout = out_layout;
            gp.renderPass = render_pass;
            gp.subpass = 0;

            const VkResult res = vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp, nullptr, &out_pipeline);
            vkDestroyShaderModule(vk_->device(), vs, nullptr);
            vkDestroyShaderModule(vk_->device(), fs, nullptr);
            return res == VK_SUCCESS;
        }

        bool ensure_pipelines(shs::CullMode cull_mode, bool front_face_ccw)
        {
            if (shadow_render_pass_ == VK_NULL_HANDLE || scene_render_pass_ == VK_NULL_HANDLE || post_render_pass_ == VK_NULL_HANDLE) return false;
            if (pipeline_gen_ == vk_->swapchain_generation() &&
                shadow_pipeline_ != VK_NULL_HANDLE &&
                scene_pipeline_ != VK_NULL_HANDLE &&
                sky_pipeline_ != VK_NULL_HANDLE &&
                bright_pipeline_ != VK_NULL_HANDLE &&
                shafts_pipeline_ != VK_NULL_HANDLE &&
                flare_pipeline_ != VK_NULL_HANDLE &&
                composite_pipeline_ != VK_NULL_HANDLE &&
                fxaa_pipeline_ != VK_NULL_HANDLE &&
                last_cull_mode_ == cull_mode &&
                last_front_face_ccw_ == front_face_ccw)
            {
                return true;
            }

            destroy_pipelines();

            if (!create_shadow_pipeline(cull_mode, front_face_ccw)) return false;
            if (!create_scene_pipeline(cull_mode, front_face_ccw)) return false;

            if (!create_fullscreen_pipeline(
                    SHS_VK_PB_SKY_FRAG_SPV,
                    scene_render_pass_,
                    single_tex_layout_,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    sizeof(glm::mat4),
                    sky_pipeline_layout_,
                    sky_pipeline_,
                    2u)) return false;

            if (!create_fullscreen_pipeline(
                    SHS_VK_PB_BRIGHT_FRAG_SPV,
                    post_render_pass_,
                    single_tex_layout_,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    sizeof(BrightPush),
                    bright_pipeline_layout_,
                    bright_pipeline_)) return false;

            if (!create_fullscreen_pipeline(
                    SHS_VK_PB_SHAFTS_FRAG_SPV,
                    post_render_pass_,
                    shafts_layout_,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    sizeof(ShaftsPush),
                    shafts_pipeline_layout_,
                    shafts_pipeline_)) return false;

            if (!create_fullscreen_pipeline(
                    SHS_VK_PB_FLARE_FRAG_SPV,
                    post_render_pass_,
                    single_tex_layout_,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    sizeof(FlarePush),
                    flare_pipeline_layout_,
                    flare_pipeline_)) return false;

            if (!create_fullscreen_pipeline(
                    SHS_VK_PB_COMPOSITE_FRAG_SPV,
                    post_render_pass_,
                    composite_layout_,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    sizeof(CompositePush),
                    composite_pipeline_layout_,
                    composite_pipeline_)) return false;

            if (!create_fullscreen_pipeline(
                    SHS_VK_PB_FXAA_FRAG_SPV,
                    vk_->render_pass(),
                    single_tex_layout_,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    sizeof(FxaaPush),
                    fxaa_pipeline_layout_,
                    fxaa_pipeline_)) return false;

            pipeline_gen_ = vk_->swapchain_generation();
            last_cull_mode_ = cull_mode;
            last_front_face_ccw_ = front_face_ccw;
            return true;
        }

        void destroy_pipelines()
        {
            if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
            VkDevice dev = vk_->device();

            auto destroy_pipeline = [&](VkPipeline& p) {
                if (p != VK_NULL_HANDLE) { vkDestroyPipeline(dev, p, nullptr); p = VK_NULL_HANDLE; }
            };
            auto destroy_layout = [&](VkPipelineLayout& l) {
                if (l != VK_NULL_HANDLE) { vkDestroyPipelineLayout(dev, l, nullptr); l = VK_NULL_HANDLE; }
            };

            destroy_pipeline(shadow_pipeline_);
            destroy_layout(shadow_pipeline_layout_);

            destroy_pipeline(scene_pipeline_);
            destroy_layout(scene_pipeline_layout_);

            destroy_pipeline(sky_pipeline_);
            destroy_layout(sky_pipeline_layout_);

            destroy_pipeline(bright_pipeline_);
            destroy_layout(bright_pipeline_layout_);

            destroy_pipeline(shafts_pipeline_);
            destroy_layout(shafts_pipeline_layout_);

            destroy_pipeline(flare_pipeline_);
            destroy_layout(flare_pipeline_layout_);

            destroy_pipeline(composite_pipeline_);
            destroy_layout(composite_pipeline_layout_);

            destroy_pipeline(fxaa_pipeline_);
            destroy_layout(fxaa_pipeline_layout_);

            pipeline_gen_ = 0;
        }

        void cmd_set_viewport_scissor(VkCommandBuffer cmd, uint32_t w, uint32_t h, bool flip_y) const
        {
            shs::vk_cmd_set_viewport_scissor(cmd, w, h, flip_y);
        }

        void begin_render_pass(
            VkCommandBuffer cmd,
            VkRenderPass rp,
            VkFramebuffer fb,
            uint32_t w,
            uint32_t h,
            const VkClearValue* clears,
            uint32_t clear_count)
        {
            VkRenderPassBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            bi.renderPass = rp;
            bi.framebuffer = fb;
            bi.renderArea.offset = {0, 0};
            bi.renderArea.extent = {w, h};
            bi.clearValueCount = clear_count;
            bi.pClearValues = clears;
            vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
        }

        void draw_fullscreen_triangle(VkCommandBuffer cmd)
        {
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }

        void record_shadow_pass(
            VkCommandBuffer cmd,
            const shs::Scene& scene,
            const shs::ResourceRegistry& resources,
            const LightMatrices& light,
            bool enable_shadow_casters)
        {
            VkClearValue clear{};
            clear.depthStencil = {1.0f, 0};
            begin_render_pass(cmd, shadow_render_pass_, shadow_fb_, kShadowMapSize, kShadowMapSize, &clear, 1);

            cmd_set_viewport_scissor(cmd, kShadowMapSize, kShadowMapSize, false);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);

            if (!enable_shadow_casters)
            {
                vkCmdEndRenderPass(cmd);
                return;
            }

            for (const auto& item : scene.items)
            {
                if (!item.visible || !item.casts_shadow) continue;

                const shs::MeshData* mesh_data = resources.get_mesh((shs::MeshAssetHandle)item.mesh);
                if (!mesh_data || mesh_data->positions.empty()) continue;
                if (!ensure_mesh_uploaded((shs::MeshAssetHandle)item.mesh, *mesh_data)) continue;

                const auto it_mesh = meshes_.find((shs::MeshAssetHandle)item.mesh);
                if (it_mesh == meshes_.end()) continue;
                const GpuMesh& gm = it_mesh->second;
                if (gm.index_count == 0) continue;

                const glm::mat4 model = build_model_matrix(item.tr);
                ShadowPush pc{};
                pc.light_mvp = light.viewproj * model;

                VkDeviceSize vb_offset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &gm.vb, &vb_offset);
                vkCmdBindIndexBuffer(cmd, gm.ib, 0, VK_INDEX_TYPE_UINT32);
                vkCmdPushConstants(cmd, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &pc);
                vkCmdDrawIndexed(cmd, gm.index_count, 1, 0, 0, 0);

            }

            vkCmdEndRenderPass(cmd);
        }

        void record_scene_pass(
            VkCommandBuffer cmd,
            const shs::Scene& scene,
            const shs::ResourceRegistry& resources,
            const LightMatrices& light,
            const shs::FrameParams& fp)
        {
            VkClearValue clears[3]{};
            clears[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            clears[1].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
            clears[2].depthStencil = {1.0f, 0};
            begin_render_pass(cmd, scene_render_pass_, scene_fb_, offscreen_w_, offscreen_h_, clears, 3);

            cmd_set_viewport_scissor(cmd, offscreen_w_, offscreen_h_, true);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline_layout_, 0, 1, &sky_set_, 0, nullptr);
            glm::mat4 sky_view = scene.cam.view;
            sky_view[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            const glm::mat4 inv_vp = glm::inverse(scene.cam.proj * sky_view);
            vkCmdPushConstants(cmd, sky_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::mat4), &inv_vp);
            draw_fullscreen_triangle(cmd);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scene_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scene_pipeline_layout_, 1, 1, &shadow_set_, 0, nullptr);

            uint32_t draw_index = 0;
            for (const auto& item : scene.items)
            {
                const uint64_t key = object_key(item, draw_index);
                const glm::mat4 model = build_model_matrix(item.tr);

                if (!item.visible)
                {
                    prev_models_[key] = model;
                    ++draw_index;
                    continue;
                }

                const shs::MeshData* mesh_data = resources.get_mesh((shs::MeshAssetHandle)item.mesh);
                if (!mesh_data || mesh_data->positions.empty())
                {
                    prev_models_[key] = model;
                    ++draw_index;
                    continue;
                }
                if (!ensure_mesh_uploaded((shs::MeshAssetHandle)item.mesh, *mesh_data))
                {
                    prev_models_[key] = model;
                    ++draw_index;
                    continue;
                }

                const auto it_mesh = meshes_.find((shs::MeshAssetHandle)item.mesh);
                if (it_mesh == meshes_.end())
                {
                    prev_models_[key] = model;
                    ++draw_index;
                    continue;
                }
                const GpuMesh& gm = it_mesh->second;
                if (gm.index_count == 0)
                {
                    prev_models_[key] = model;
                    ++draw_index;
                    continue;
                }

                const shs::MaterialData* mat = resources.get_material((shs::MaterialAssetHandle)item.mat);
                const shs::TextureAssetHandle tex_h = mat ? mat->base_color_tex : 0;

                VkDescriptorSet obj_set = VK_NULL_HANDLE;
                if (!ensure_object_descriptor(key, tex_h, obj_set, resources))
                {
                    prev_models_[key] = model;
                    ++draw_index;
                    continue;
                }
                glm::mat4 prev_model = model;
                const auto it_prev = prev_models_.find(key);
                if (it_prev != prev_models_.end()) prev_model = it_prev->second;

                ObjectUBO ubo{};
                ubo.mvp = scene.cam.viewproj * model;
                ubo.prev_mvp = prev_viewproj_ * prev_model;
                ubo.model = model;
                ubo.light_mvp = light.viewproj * model;
                ubo.base_color_metallic = glm::vec4(
                    mat ? mat->base_color : glm::vec3(0.75f, 0.75f, 0.78f),
                    mat ? mat->metallic : 0.0f);
                ubo.roughness_ao_emissive_hastex = glm::vec4(
                    mat ? mat->roughness : 0.6f,
                    mat ? mat->ao : 1.0f,
                    mat ? mat->emissive_intensity : 0.0f,
                    tex_h != 0 ? 1.0f : 0.0f);
                ubo.camera_pos_sun_intensity = glm::vec4(scene.cam.pos, scene.sun.intensity);
                ubo.sun_color_pad = glm::vec4(scene.sun.color, 0.0f);
                ubo.sun_dir_ws_pad = glm::vec4(scene.sun.dir_ws, static_cast<float>(std::max(fp.pass.shadow.pcf_radius, 0)));
                ubo.shadow_params = glm::vec4(
                    fp.pass.shadow.enable ? fp.pass.shadow.strength : 0.0f,
                    fp.pass.shadow.bias_const,
                    fp.pass.shadow.bias_slope,
                    fp.pass.shadow.pcf_step);
                    
                uint32_t b_idx = 0;
                auto it_b = bindless_indices_.find(tex_h);
                if (it_b != bindless_indices_.end()) b_idx = it_b->second;
                ubo.extra_indices = glm::uvec4(b_idx, 0, 0, 0);

                if (!update_object_ubo(key, ubo))
                {
                    prev_models_[key] = model;
                    ++draw_index;
                    continue;
                }

                VkDeviceSize vb_offset = 0;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scene_pipeline_layout_, 0, 1, &obj_set, 0, nullptr);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scene_pipeline_layout_, 2, 1, &bindless_set_, 0, nullptr);
                vkCmdBindVertexBuffers(cmd, 0, 1, &gm.vb, &vb_offset);
                vkCmdBindIndexBuffer(cmd, gm.ib, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, gm.index_count, 1, 0, 0, 0);

                prev_models_[key] = model;
                ++draw_index;
            }

            vkCmdEndRenderPass(cmd);
            (void)fp;
        }

        void record_bright_pass(VkCommandBuffer cmd)
        {
            VkClearValue clear{};
            clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            begin_render_pass(cmd, post_render_pass_, bright_fb_, offscreen_w_, offscreen_h_, &clear, 1);
            cmd_set_viewport_scissor(cmd, offscreen_w_, offscreen_h_, true);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bright_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bright_pipeline_layout_, 0, 1, &bright_set_, 0, nullptr);

            BrightPush pc{};
            pc.threshold = 1.0f;
            pc.intensity = 1.0f;
            pc.knee = 0.5f;
            vkCmdPushConstants(cmd, bright_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(BrightPush), &pc);
            draw_fullscreen_triangle(cmd);
            vkCmdEndRenderPass(cmd);
        }

        void clear_post_target(VkCommandBuffer cmd, VkFramebuffer fb)
        {
            VkClearValue clear{};
            clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            begin_render_pass(cmd, post_render_pass_, fb, offscreen_w_, offscreen_h_, &clear, 1);
            vkCmdEndRenderPass(cmd);
        }

        void record_shafts_pass(VkCommandBuffer cmd, const glm::vec2& sun_uv, const shs::FrameParams& fp)
        {
            VkClearValue clear{};
            clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            begin_render_pass(cmd, post_render_pass_, shafts_fb_, offscreen_w_, offscreen_h_, &clear, 1);
            cmd_set_viewport_scissor(cmd, offscreen_w_, offscreen_h_, true);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shafts_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shafts_pipeline_layout_, 0, 1, &shafts_set_, 0, nullptr);

            ShaftsPush pc{};
            pc.sun_uv = sun_uv;
            // hello_pbr_light_shafts.cpp default-tuning-тай ойролцоо утгууд.
            pc.intensity = fp.pass.light_shafts.enable ? 0.22f : 0.0f;
            pc.density = fp.pass.light_shafts.density;
            pc.decay = fp.pass.light_shafts.decay;
            pc.weight = fp.pass.light_shafts.weight;
            pc.exposure = 1.0f;
            pc.steps = std::max(fp.pass.light_shafts.steps, 1);
            vkCmdPushConstants(cmd, shafts_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShaftsPush), &pc);
            draw_fullscreen_triangle(cmd);
            vkCmdEndRenderPass(cmd);
        }

        void record_flare_pass(VkCommandBuffer cmd, const glm::vec2& sun_uv, const shs::FrameParams& fp)
        {
            VkClearValue clear{};
            clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            begin_render_pass(cmd, post_render_pass_, flare_fb_, offscreen_w_, offscreen_h_, &clear, 1);
            cmd_set_viewport_scissor(cmd, offscreen_w_, offscreen_h_, true);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, flare_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, flare_pipeline_layout_, 0, 1, &flare_set_, 0, nullptr);

            FlarePush pc{};
            pc.sun_uv = sun_uv;
            // Sensitive боловч overbloom болохооргүйгээр flare-ийг даруухан барина.
            pc.intensity = fp.pass.light_shafts.enable ? 0.34f : 0.0f;
            pc.halo_intensity = 0.18f;
            pc.chroma_shift = 1.15f;
            pc.ghosts = 4;
            vkCmdPushConstants(cmd, flare_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FlarePush), &pc);
            draw_fullscreen_triangle(cmd);
            vkCmdEndRenderPass(cmd);
        }

        void record_composite_pass(VkCommandBuffer cmd, const shs::FrameParams& fp)
        {
            VkClearValue clear{};
            clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            begin_render_pass(cmd, post_render_pass_, composite_fb_, offscreen_w_, offscreen_h_, &clear, 1);
            cmd_set_viewport_scissor(cmd, offscreen_w_, offscreen_h_, true);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composite_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composite_pipeline_layout_, 0, 1, &composite_set_, 0, nullptr);

            CompositePush pc{};
            pc.inv_size = glm::vec2(1.0f / float(offscreen_w_), 1.0f / float(offscreen_h_));
            pc.mb_strength = fp.pass.motion_blur.enable ? fp.pass.motion_blur.strength : 0.0f;
            pc.shafts_strength = fp.pass.light_shafts.enable ? 1.0f : 0.0f;
            pc.flare_strength = fp.pass.light_shafts.enable ? 0.95f : 0.0f;
            pc.mb_samples = std::max(fp.pass.motion_blur.samples, 1);
            pc.exposure = std::max(0.0001f, fp.pass.tonemap.exposure);
            pc.gamma = std::max(0.001f, fp.pass.tonemap.gamma);
            vkCmdPushConstants(cmd, composite_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(CompositePush), &pc);
            draw_fullscreen_triangle(cmd);
            vkCmdEndRenderPass(cmd);
        }

        void record_fxaa_to_swapchain(VkCommandBuffer cmd, const shs::VulkanRenderBackend::FrameInfo& fi, bool enable_fxaa)
        {
            VkClearValue clear[2]{};
            clear[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            clear[1].depthStencil = {1.0f, 0};

            VkRenderPassBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            bi.renderPass = fi.render_pass;
            bi.framebuffer = fi.framebuffer;
            bi.renderArea.offset = {0, 0};
            bi.renderArea.extent = fi.extent;
            bi.clearValueCount = vk_->has_depth_attachment() ? 2u : 1u;
            bi.pClearValues = clear;
            vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);

            cmd_set_viewport_scissor(cmd, fi.extent.width, fi.extent.height, true);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fxaa_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fxaa_pipeline_layout_, 0, 1, &fxaa_set_, 0, nullptr);

            FxaaPush pc{};
            pc.inv_size = glm::vec2(1.0f / float(offscreen_w_), 1.0f / float(offscreen_h_));
            pc.enable_fxaa = enable_fxaa ? 1.0f : 0.0f;
            vkCmdPushConstants(cmd, fxaa_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FxaaPush), &pc);
            draw_fullscreen_triangle(cmd);

            vkCmdEndRenderPass(cmd);
        }

    private:
        shs::VulkanRenderBackend* vk_ = nullptr;

        VkCommandPool upload_cmd_pool_ = VK_NULL_HANDLE;

        VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout scene_obj_layout_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout bindless_layout_ = VK_NULL_HANDLE;
        VkDescriptorPool bindless_pool_ = VK_NULL_HANDLE;
        VkDescriptorSet bindless_set_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout scene_shadow_layout_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout single_tex_layout_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout shafts_layout_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout composite_layout_ = VK_NULL_HANDLE;

        VkSampler sampler_linear_repeat_ = VK_NULL_HANDLE;
        VkSampler sampler_linear_clamp_ = VK_NULL_HANDLE;
        VkSampler sampler_sky_ = VK_NULL_HANDLE;
        VkSampler sampler_shadow_ = VK_NULL_HANDLE;

        VkDescriptorSet shadow_set_ = VK_NULL_HANDLE;
        VkDescriptorSet sky_set_ = VK_NULL_HANDLE;
        VkDescriptorSet bright_set_ = VK_NULL_HANDLE;
        VkDescriptorSet shafts_set_ = VK_NULL_HANDLE;
        VkDescriptorSet flare_set_ = VK_NULL_HANDLE;
        VkDescriptorSet composite_set_ = VK_NULL_HANDLE;
        VkDescriptorSet fxaa_set_ = VK_NULL_HANDLE;

        Target shadow_depth_{};
        Target scene_hdr_{};
        Target velocity_{};
        Target scene_depth_{};
        Target bright_{};
        Target shafts_{};
        Target flare_{};
        Target composite_{};

        VkRenderPass shadow_render_pass_ = VK_NULL_HANDLE;
        VkRenderPass scene_render_pass_ = VK_NULL_HANDLE;
        VkRenderPass post_render_pass_ = VK_NULL_HANDLE;

        VkFramebuffer shadow_fb_ = VK_NULL_HANDLE;
        VkFramebuffer scene_fb_ = VK_NULL_HANDLE;
        VkFramebuffer bright_fb_ = VK_NULL_HANDLE;
        VkFramebuffer shafts_fb_ = VK_NULL_HANDLE;
        VkFramebuffer flare_fb_ = VK_NULL_HANDLE;
        VkFramebuffer composite_fb_ = VK_NULL_HANDLE;

        VkPipelineLayout shadow_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline shadow_pipeline_ = VK_NULL_HANDLE;

        VkPipelineLayout scene_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline scene_pipeline_ = VK_NULL_HANDLE;

        VkPipelineLayout sky_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline sky_pipeline_ = VK_NULL_HANDLE;

        VkPipelineLayout bright_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline bright_pipeline_ = VK_NULL_HANDLE;

        VkPipelineLayout shafts_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline shafts_pipeline_ = VK_NULL_HANDLE;

        VkPipelineLayout flare_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline flare_pipeline_ = VK_NULL_HANDLE;

        VkPipelineLayout composite_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline composite_pipeline_ = VK_NULL_HANDLE;

        VkPipelineLayout fxaa_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline fxaa_pipeline_ = VK_NULL_HANDLE;

        std::unordered_map<shs::TextureAssetHandle, uint32_t> bindless_indices_;
        uint32_t next_bindless_index_ = 0;
        std::unordered_map<shs::MeshAssetHandle, GpuMesh> meshes_{};
        std::unordered_map<shs::TextureAssetHandle, GpuTexture> textures_{};
        std::unordered_map<uint64_t, GpuObject> objects_{};
        std::unordered_map<uint64_t, glm::mat4> prev_models_{};

        GpuTexture white_texture_{};
        GpuTexture sky_texture_{};
        const shs::ISkyModel* last_sky_model_ = nullptr;

        glm::mat4 prev_viewproj_{1.0f};

        uint32_t offscreen_w_ = 0;
        uint32_t offscreen_h_ = 0;

        uint64_t pipeline_gen_ = 0;
        shs::CullMode last_cull_mode_ = shs::CullMode::Back;
        bool last_front_face_ccw_ = true;
    };
    float lerp_angle_rad(float a, float b, float t)
    {
        float d = b - a;
        while (d > PI) d -= TWO_PI;
        while (d < -PI) d += TWO_PI;
        return a + d * t;
    }

    float visual_yaw_from_world_forward(const glm::vec3& fwd_ws, ModelForwardAxis axis)
    {
        glm::vec2 d{fwd_ws.x, fwd_ws.z};
        const float len = glm::length(d);
        if (len <= 1e-6f) return 0.0f;
        d /= len;
        switch (axis)
        {
            case ModelForwardAxis::PosX: return std::atan2(d.y, d.x);
            case ModelForwardAxis::NegX: return std::atan2(-d.y, -d.x);
            case ModelForwardAxis::PosZ: return std::atan2(d.x, d.y);
            case ModelForwardAxis::NegZ: return std::atan2(-d.x, -d.y);
        }
        return 0.0f;
    }

    glm::vec3 world_forward_from_visual_yaw(float visual_yaw, ModelForwardAxis axis)
    {
        switch (axis)
        {
            case ModelForwardAxis::PosX: return glm::vec3(std::cos(visual_yaw), 0.0f, std::sin(visual_yaw));
            case ModelForwardAxis::NegX: return glm::vec3(-std::cos(visual_yaw), 0.0f, -std::sin(visual_yaw));
            case ModelForwardAxis::PosZ: return glm::vec3(std::sin(visual_yaw), 0.0f, std::cos(visual_yaw));
            case ModelForwardAxis::NegZ: return glm::vec3(-std::sin(visual_yaw), 0.0f, -std::cos(visual_yaw));
        }
        return glm::vec3(1.0f, 0.0f, 0.0f);
    }

    // Subaru машинд deterministic төлөвт автомат жолоодлого (Cruise/Turn/Recover/Idle) хэрэгжүүлнэ.
    class SubaruCruiseSystem final : public shs::ILogicSystem
    {
    public:
        enum class DriveState : uint8_t
        {
            Cruise = 0,
            Turn = 1,
            Recover = 2,
            Idle = 3,
        };

        struct FsmContext
        {
            SubaruCruiseSystem& self;
            shs::SceneObject& obj;
        };

        SubaruCruiseSystem(
            std::string object_name,
            float area_half_extent,
            float y_level,
            float cruise_speed = 6.5f,
            float max_turn_rate_rad = 1.9f,
            ModelForwardAxis visual_forward_axis = ModelForwardAxis::PosX,
            float visual_yaw_offset_rad = 0.0f,
            uint32_t seed = 0xC0FFEEu
        )
            : object_name_(std::move(object_name))
            , area_half_extent_(area_half_extent)
            , y_level_(y_level)
            , cruise_speed_(cruise_speed)
            , max_turn_rate_rad_(max_turn_rate_rad)
            , visual_forward_axis_(visual_forward_axis)
            , visual_yaw_offset_rad_(visual_yaw_offset_rad)
            , rng_(seed)
            , area_dist_(-area_half_extent_ * 0.90f, area_half_extent_ * 0.90f)
            , unit_dist_(0.0f, 1.0f)
            , turn_rate_dist_(0.95f, 1.80f)
            , cruise_yaw_bias_dist_(-0.46f, 0.46f)
            , speed_jitter_(0.82f, 1.18f)
        {
            configure_fsm();
        }

        const char* state_name() const
        {
            switch (current_state())
            {
                case DriveState::Cruise: return "Cruise";
                case DriveState::Turn: return "Turn";
                case DriveState::Recover: return "Recover";
                case DriveState::Idle: return "Idle";
            }
            return "Unknown";
        }

        float state_progress() const
        {
            if (!fsm_.started()) return 0.0f;
            if (state_duration_ <= 1e-6f) return 1.0f;
            return std::clamp(fsm_.state_time() / state_duration_, 0.0f, 1.0f);
        }

        glm::vec3 heading_ws() const
        {
            if (!initialized_) return glm::vec3(1.0f, 0.0f, 0.0f);
            return glm::normalize(glm::vec3(std::cos(current_yaw_), 0.0f, std::sin(current_yaw_)));
        }

        void tick(shs::LogicSystemContext& ctx) override
        {
            if (!ctx.objects) return;
            auto* obj = ctx.objects->find(object_name_);
            if (!obj) return;
            const float dt = std::max(0.0f, ctx.dt);
            if (dt <= 1e-6f) return;

            if (!initialized_)
            {
                // Эхний чиглэлийг model-ийн yaw-аас coordinate convention дагуу сэргээнэ.
                obj->tr.pos.y = y_level_;
                const glm::vec3 seed_fwd = world_forward_from_visual_yaw(
                    obj->tr.rot_euler.y - visual_yaw_offset_rad_,
                    visual_forward_axis_
                );
                current_yaw_ = std::atan2(seed_fwd.z, seed_fwd.x);
                current_speed_ = cruise_speed_;

                FsmContext fsm_ctx{*this, *obj};
                (void)fsm_.start(DriveState::Cruise, fsm_ctx);
                initialized_ = true;
            }

            obj->tr.pos.y = y_level_;

            desired_yaw_ = current_yaw_;
            desired_speed_ = cruise_speed_;
            FsmContext fsm_ctx{*this, *obj};
            fsm_.tick(fsm_ctx, dt);

            const float edge_ratio = boundary_ratio(obj->tr.pos);
            apply_boundary_steer(obj->tr.pos, desired_yaw_, desired_speed_);

            float dy = desired_yaw_ - current_yaw_;
            while (dy > PI) dy -= TWO_PI;
            while (dy < -PI) dy += TWO_PI;
            const float max_step = max_turn_rate_rad_ * dt;
            dy = std::clamp(dy, -max_step, max_step);
            current_yaw_ += dy;

            const float speed_lerp_t = 1.0f - std::exp(-dt * 6.0f);
            current_speed_ = glm::mix(current_speed_, desired_speed_, speed_lerp_t);

            const glm::vec3 fwd{std::cos(current_yaw_), 0.0f, std::sin(current_yaw_)};
            const float speed_scale = 1.0f - edge_ratio * 0.35f;
            obj->tr.pos += fwd * (current_speed_ * speed_scale * dt);
            obj->tr.pos.x = std::clamp(obj->tr.pos.x, -area_half_extent_, area_half_extent_);
            obj->tr.pos.z = std::clamp(obj->tr.pos.z, -area_half_extent_, area_half_extent_);
            obj->tr.pos.y = y_level_;
            obj->tr.rot_euler.y = visual_yaw_from_world_forward(fwd, visual_forward_axis_) + visual_yaw_offset_rad_;
        }

    private:
        DriveState current_state() const
        {
            const auto s = fsm_.current_state();
            return s.has_value() ? *s : DriveState::Cruise;
        }

        void configure_fsm()
        {
            using StateCallbacks = typename shs::StateMachine<DriveState, FsmContext>::StateCallbacks;

            fsm_.add_state(DriveState::Cruise, StateCallbacks{
                [this](FsmContext& fctx) { on_enter_state(DriveState::Cruise, fctx.obj.tr.pos); },
                [this](FsmContext&, float dt, float) { update_cruise(dt); },
                {}
            });
            fsm_.add_state(DriveState::Turn, StateCallbacks{
                [this](FsmContext& fctx) { on_enter_state(DriveState::Turn, fctx.obj.tr.pos); },
                [this](FsmContext&, float dt, float) { update_turn(dt); },
                {}
            });
            fsm_.add_state(DriveState::Recover, StateCallbacks{
                [this](FsmContext& fctx) { on_enter_state(DriveState::Recover, fctx.obj.tr.pos); },
                [this](FsmContext& fctx, float, float) { update_recover(fctx.obj); },
                {}
            });
            fsm_.add_state(DriveState::Idle, StateCallbacks{
                [this](FsmContext& fctx) { on_enter_state(DriveState::Idle, fctx.obj.tr.pos); },
                [this](FsmContext&, float, float) { update_idle(); },
                {}
            });

            // Нэг төлөвийн хугацаа дуусахад тухайн төлөв дээр урьдчилан тооцсон дараагийн төлөв рүү шилжинэ.
            fsm_.add_transition(DriveState::Cruise, DriveState::Idle, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Idle;
            });
            fsm_.add_transition(DriveState::Cruise, DriveState::Turn, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Turn;
            });
            fsm_.add_transition(DriveState::Turn, DriveState::Recover, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Recover;
            });
            fsm_.add_transition(DriveState::Recover, DriveState::Idle, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Idle;
            });
            fsm_.add_transition(DriveState::Recover, DriveState::Cruise, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Cruise;
            });
            fsm_.add_transition(DriveState::Idle, DriveState::Cruise, [this](const FsmContext&, float elapsed) {
                return elapsed >= state_duration_ && timeout_next_state_ == DriveState::Cruise;
            });
        }

        float rand01()
        {
            return unit_dist_(rng_);
        }

        float rand_range(float lo, float hi)
        {
            return lo + (hi - lo) * rand01();
        }

        float boundary_ratio(const glm::vec3& p) const
        {
            const float edge = std::max(std::abs(p.x), std::abs(p.z));
            return std::clamp((edge - area_half_extent_ * 0.66f) / (area_half_extent_ * 0.34f), 0.0f, 1.0f);
        }

        void apply_boundary_steer(const glm::vec3& p, float& desired_yaw, float& desired_speed)
        {
            const float edge_ratio = boundary_ratio(p);
            if (edge_ratio <= 0.0f) return;

            glm::vec2 to_center{-p.x, -p.z};
            const float len = glm::length(to_center);
            if (len > 1e-6f)
            {
                to_center /= len;
                const float center_yaw = std::atan2(to_center.y, to_center.x);
                const float steer_w = std::clamp(edge_ratio * (current_state() == DriveState::Recover ? 1.0f : 0.74f), 0.0f, 1.0f);
                desired_yaw = lerp_angle_rad(desired_yaw, center_yaw, steer_w);
            }
            desired_speed *= (1.0f - edge_ratio * 0.28f);

            // Ирмэгт хэт ойртох үед Recover рүү шууд request өгч буцааж төв рүү эргүүлнэ.
            if (edge_ratio > 0.92f && current_state() != DriveState::Recover)
            {
                fsm_.request_transition(DriveState::Recover);
            }
        }

        void pick_recover_target(const glm::vec3& current_pos)
        {
            for (int i = 0; i < 24; ++i)
            {
                const glm::vec3 c{area_dist_(rng_), y_level_, area_dist_(rng_)};
                if (glm::length(glm::vec2(c.x - current_pos.x, c.z - current_pos.z)) > area_half_extent_ * 0.24f)
                {
                    recover_target_ = c;
                    return;
                }
            }
            recover_target_ = glm::vec3(area_dist_(rng_), y_level_, area_dist_(rng_));
        }

        float duration_for_state(DriveState s)
        {
            switch (s)
            {
                case DriveState::Cruise: return rand_range(2.6f, 5.6f);
                case DriveState::Turn: return rand_range(0.55f, 1.65f);
                case DriveState::Recover: return rand_range(1.0f, 2.2f);
                case DriveState::Idle: return rand_range(0.25f, 0.95f);
            }
            return 1.0f;
        }

        DriveState timeout_next_for_state(DriveState s)
        {
            switch (s)
            {
                case DriveState::Cruise: return (rand01() < 0.16f) ? DriveState::Idle : DriveState::Turn;
                case DriveState::Turn: return DriveState::Recover;
                case DriveState::Recover: return (rand01() < 0.20f) ? DriveState::Idle : DriveState::Cruise;
                case DriveState::Idle: return DriveState::Cruise;
            }
            return DriveState::Cruise;
        }

        void on_enter_state(DriveState s, const glm::vec3& pos)
        {
            state_duration_ = duration_for_state(s);
            timeout_next_state_ = timeout_next_for_state(s);
            switch (s)
            {
                case DriveState::Cruise:
                    cruise_turn_rate_ = cruise_yaw_bias_dist_(rng_);
                    cruise_target_speed_ = cruise_speed_ * speed_jitter_(rng_);
                    break;
                case DriveState::Turn:
                {
                    const float sign = (rand01() < 0.5f) ? -1.0f : 1.0f;
                    turn_rate_ = turn_rate_dist_(rng_) * sign;
                    break;
                }
                case DriveState::Recover:
                    pick_recover_target(pos);
                    break;
                case DriveState::Idle:
                    break;
            }
        }

        void update_cruise(float dt)
        {
            desired_yaw_ = current_yaw_ + cruise_turn_rate_ * dt;
            desired_speed_ = cruise_target_speed_;
        }

        void update_turn(float dt)
        {
            desired_yaw_ = current_yaw_ + turn_rate_ * dt;
            desired_speed_ = cruise_speed_ * 0.76f;
        }

        void update_recover(const shs::SceneObject& obj)
        {
            const glm::vec3 to_goal = recover_target_ - obj.tr.pos;
            const glm::vec2 to_goal_xz{to_goal.x, to_goal.z};
            const float len = glm::length(to_goal_xz);
            if (len > 1e-5f)
            {
                const glm::vec2 d = to_goal_xz / len;
                desired_yaw_ = std::atan2(d.y, d.x);
            }
            desired_speed_ = cruise_speed_ * 0.92f;
            if (len < area_half_extent_ * 0.10f)
            {
                fsm_.request_transition(timeout_next_state_);
            }
        }

        void update_idle()
        {
            desired_yaw_ = current_yaw_;
            desired_speed_ = 0.0f;
        }

        std::string object_name_{};
        float area_half_extent_ = 16.0f;
        float y_level_ = 0.0f;
        float cruise_speed_ = 6.5f;
        float max_turn_rate_rad_ = 1.9f;
        ModelForwardAxis visual_forward_axis_ = ModelForwardAxis::PosX;
        float visual_yaw_offset_rad_ = 0.0f;
        float current_speed_ = 0.0f;
        float current_yaw_ = 0.0f;
        bool initialized_ = false;

        float state_duration_ = 1.0f;
        DriveState timeout_next_state_ = DriveState::Cruise;
        float desired_yaw_ = 0.0f;
        float desired_speed_ = 0.0f;

        float cruise_turn_rate_ = 0.0f;
        float cruise_target_speed_ = 6.5f;
        float turn_rate_ = 0.0f;
        glm::vec3 recover_target_{0.0f};

        shs::StateMachine<DriveState, FsmContext> fsm_{};
        std::mt19937 rng_{};
        std::uniform_real_distribution<float> area_dist_{-12.0f, 12.0f};
        std::uniform_real_distribution<float> unit_dist_{0.0f, 1.0f};
        std::uniform_real_distribution<float> turn_rate_dist_{0.95f, 1.80f};
        std::uniform_real_distribution<float> cruise_yaw_bias_dist_{-0.46f, 0.46f};
        std::uniform_real_distribution<float> speed_jitter_{0.82f, 1.18f};
    };

    // Follow mode асаалттай үед камерыг машины араас зөөлөн дагуулах логик систем.
    class FollowCameraSystem final : public shs::ILogicSystem
    {
    public:
        FollowCameraSystem(
            shs::CameraRig* rig,
            bool* enabled,
            std::string target_name,
            float follow_distance,
            float follow_height,
            float look_ahead,
            float smoothing
        )
            : rig_(rig)
            , enabled_(enabled)
            , target_name_(std::move(target_name))
            , follow_distance_(follow_distance)
            , follow_height_(follow_height)
            , look_ahead_(look_ahead)
            , smoothing_(smoothing)
        {}

        void tick(shs::LogicSystemContext& ctx) override
        {
            if (!rig_ || !enabled_ || !(*enabled_) || !ctx.objects) return;
            const auto* target = ctx.objects->find(target_name_);
            if (!target) return;
            const glm::vec3 fwd{std::cos(target->tr.rot_euler.y), 0.0f, std::sin(target->tr.rot_euler.y)};
            // Камерын хүссэн байрлалыг объектын ар ба дээд талд тооцоолоод smooth байдлаар дөхүүлнэ.
            const glm::vec3 desired_cam = target->tr.pos - fwd * follow_distance_ + glm::vec3(0.0f, follow_height_, 0.0f);
            follow_target(*rig_, desired_cam, glm::vec3(0.0f), smoothing_, ctx.dt);

            const glm::vec3 look_point = target->tr.pos + fwd * look_ahead_ + glm::vec3(0.0f, 0.8f, 0.0f);
            const glm::vec3 v = look_point - rig_->pos;
            const float len = glm::length(v);
            if (len > 1e-6f)
            {
                const glm::vec3 d = v / len;
                const float target_yaw = std::atan2(d.z, d.x);
                const float target_pitch = std::asin(glm::clamp(d.y, -1.0f, 1.0f));
                const float t = std::clamp(smoothing_ * ctx.dt * 8.0f, 0.0f, 1.0f);

                float dy = target_yaw - rig_->yaw;
                while (dy > 3.14159265f) dy -= 6.2831853f;
                while (dy < -3.14159265f) dy += 6.2831853f;
                rig_->yaw += dy * t;
                rig_->pitch = glm::mix(rig_->pitch, target_pitch, t);
            }
        }

    private:
        shs::CameraRig* rig_ = nullptr;
        bool* enabled_ = nullptr;
        std::string target_name_{};
        float follow_distance_ = 8.0f;
        float follow_height_ = 3.0f;
        float look_ahead_ = 3.0f;
        float smoothing_ = 0.18f;
    };

    // Monkey объектод эргэлт + босоо чиглэлийн жижиг савлалт өгнө.
    class MonkeyWiggleSystem final : public shs::ILogicSystem
    {
    public:
        MonkeyWiggleSystem(std::string object_name, float spin_rps, float bob_amp, float bob_hz)
            : object_name_(std::move(object_name))
            , spin_rps_(spin_rps)
            , bob_amp_(bob_amp)
            , bob_hz_(bob_hz)
        {}

        void tick(shs::LogicSystemContext& ctx) override
        {
            if (!ctx.objects) return;
            auto* obj = ctx.objects->find(object_name_);
            if (!obj) return;
            if (!base_captured_)
            {
                base_pos_ = obj->tr.pos;
                base_captured_ = true;
            }

            time_ += std::max(0.0f, ctx.dt);
            obj->tr.rot_euler.y += (2.0f * 3.14159265f) * spin_rps_ * std::max(0.0f, ctx.dt);
            obj->tr.pos = base_pos_;
            obj->tr.pos.y += std::sin(time_ * (2.0f * 3.14159265f) * bob_hz_) * bob_amp_;
        }

    private:
        std::string object_name_{};
        float spin_rps_ = 0.25f;
        float bob_amp_ = 0.2f;
        float bob_hz_ = 1.7f;
        bool base_captured_ = false;
        float time_ = 0.0f;
        glm::vec3 base_pos_{0.0f};
    };
}

int main()
{
    // SDL runtime: Vulkan window + input.
    SdlVulkanRuntime runtime{
        shs::WindowDesc{"HelloPassBasicsVulkan", WINDOW_W, WINDOW_H},
        shs::SurfaceDesc{CANVAS_W, CANVAS_H}
    };
    if (!runtime.valid()) return 1;

    shs::Context ctx{};
    const char* backend_env = std::getenv("SHS_RENDER_BACKEND");
    auto backend_result = shs::create_render_backend(backend_env ? backend_env : "vulkan");
    std::vector<std::unique_ptr<shs::IRenderBackend>> backend_keepalive{};
    backend_keepalive.reserve(1 + backend_result.auxiliary_backends.size());
    if (backend_result.backend) backend_keepalive.push_back(std::move(backend_result.backend));
    for (auto& b : backend_result.auxiliary_backends)
    {
        if (b) backend_keepalive.push_back(std::move(b));
    }
    if (backend_keepalive.empty()) return 1;

    for (size_t i = 0; i < backend_keepalive.size(); ++i)
    {
        if (backend_keepalive[i]) ctx.register_backend(backend_keepalive[i].get());
    }
    if (!backend_result.note.empty())
    {
        std::fprintf(stderr, "[shs] %s\n", backend_result.note.c_str());
    }

    auto* vk_backend = dynamic_cast<shs::VulkanRenderBackend*>(ctx.backend(shs::RenderBackendType::Vulkan));
    if (!vk_backend)
    {
        std::fprintf(stderr, "Fatal: Vulkan backend is not available in this build/configuration.\n");
        return 1;
    }
    ctx.set_primary_backend(vk_backend);
    if (!runtime.bind_vulkan_backend(vk_backend, "HelloPassBasicsVulkan"))
    {
        std::fprintf(stderr, "Fatal: Vulkan backend init_sdl failed.\n");
        return 1;
    }

    // Рендерийн parallel хэсгүүдэд ашиглагдах thread pool.
    shs::ThreadPoolJobSystem jobs{std::max(1u, std::thread::hardware_concurrency())};
    ctx.job_system = &jobs;

    shs::ResourceRegistry resources{};
    shs::LogicSystemProcessor logic_systems{};
    VulkanSceneRenderer gpu_renderer{vk_backend};
    if (!gpu_renderer.init())
    {
        std::fprintf(stderr, "Fatal: HelloPassBasicsVulkan GPU renderer init failed.\n");
        return 1;
    }

    shs::Scene scene{};
    scene.resources = &resources;
    scene.sun.dir_ws = glm::normalize(glm::vec3(0.4668f, -0.3487f, 0.8127f));
    scene.sun.color = glm::vec3(1.00f, 0.96f, 0.90f);
    scene.sun.intensity = 1.30f;
    // Cubemap default; хэрэв cubemap уншигдахгүй бол procedural sky fallback.
    shs::ProceduralSky procedural_sky{scene.sun.dir_ws};
    const shs::CubemapData sky_cm = shs::load_cubemap_sdl_folder("./assets/images/skybox/water_scene", true);
    shs::CubemapSky cubemap_sky{sky_cm, 1.0f};
    bool use_cubemap_sky = sky_cm.valid();
    scene.sky = use_cubemap_sky ? static_cast<const shs::ISkyModel*>(&cubemap_sky)
                                : static_cast<const shs::ISkyModel*>(&procedural_sky);

    const float plane_extent = 64.0f;
    const shs::MeshAssetHandle plane_h = shs::import_plane_primitive(resources, shs::PlaneDesc{plane_extent, plane_extent, 32, 32}, "plane");
    shs::MeshAssetHandle subaru_h = shs::import_mesh_assimp(resources, "./assets/obj/subaru/SUBARU_1.rawobj", "subaru_mesh");
    const bool subaru_loaded = (subaru_h != 0);
    if (!subaru_loaded) subaru_h = shs::import_box_primitive(resources, shs::BoxDesc{glm::vec3(2.4f, 1.1f, 4.8f), 2, 1, 2}, "subaru_fallback");
    const shs::TextureAssetHandle subaru_albedo_h = shs::import_texture_sdl(resources, "./assets/obj/subaru/SUBARU1_M.bmp", "subaru_albedo", true);
    shs::MeshAssetHandle monkey_h = shs::import_mesh_assimp(resources, "./assets/obj/monkey/monkey.rawobj", "monkey_mesh");
    if (monkey_h == 0) monkey_h = shs::import_sphere_primitive(resources, shs::SphereDesc{1.0f, 28, 18}, "monkey_fallback");
    const glm::vec3 car_scale = subaru_loaded ? glm::vec3(0.020f) : glm::vec3(1.0f);

    // Scene материалууд: plastic floor, textured subaru, gold monkey.
    const shs::MaterialAssetHandle floor_mat_h = resources.add_material(
        shs::MaterialData{"mat_floor_plastic", glm::vec3(0.42f, 0.44f, 0.48f), 0.0f, 0.96f, 1.0f},
        "mat_floor"
    );
    const shs::MaterialAssetHandle subaru_mat_h = resources.add_material(
        shs::MaterialData{"mat_subaru", glm::vec3(1.0f), 0.28f, 0.44f, 1.0f, glm::vec3(0.0f), 0.0f, subaru_albedo_h, 0, 0, 0},
        "mat_subaru"
    );
    const shs::MaterialAssetHandle monkey_mat_h = resources.add_material(
        shs::MaterialData{"mat_monkey_gold", glm::vec3(1.000f, 0.766f, 0.336f), 1.00f, 0.14f, 1.0f},
        "mat_monkey_gold"
    );

    shs::SceneObjectSet objects{};
    objects.add(shs::SceneObject{
        "floor",
        (shs::MeshHandle)plane_h,
        (shs::MaterialHandle)floor_mat_h,
        shs::Transform{glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f), glm::vec3(1.0f)},
        true,
        false
    });
    objects.add(shs::SceneObject{
        "subaru",
        (shs::MeshHandle)subaru_h,
        (shs::MaterialHandle)subaru_mat_h,
        shs::Transform{glm::vec3(0.0f, -0.95f, 0.0f), glm::vec3(0.0f), car_scale},
        true,
        true
    });
    objects.add(shs::SceneObject{
        "monkey",
        (shs::MeshHandle)monkey_h,
        (shs::MaterialHandle)monkey_mat_h,
        shs::Transform{glm::vec3(0.0f, 1.45f, 0.0f), glm::vec3(0.0f), glm::vec3(1.05f)},
        true,
        true
    });
    objects.sync_to_scene(scene);

    // Frame-level render тохиргоонууд.
    shs::FrameParams fp{};
    fp.w = CANVAS_W;
    fp.h = CANVAS_H;
    fp.debug_view = shs::DebugViewMode::Final;
    fp.cull_mode = shs::CullMode::None;
    fp.shading_model = shs::ShadingModel::PBRMetalRough;
    fp.pass.tonemap.exposure = 1.35f;
    fp.pass.tonemap.gamma = 2.2f;
    fp.exposure = fp.pass.tonemap.exposure;
    fp.gamma = fp.pass.tonemap.gamma;
    fp.pass.shadow.enable = true;
    fp.pass.shadow.pcf_radius = 1;
    fp.pass.shadow.pcf_step = 1.0f;
    fp.pass.shadow.strength = 0.80f;
    fp.pass.light_shafts.enable = true;
    fp.pass.light_shafts.steps = 28;
    fp.pass.light_shafts.density = 0.85f;
    fp.pass.light_shafts.weight = 0.26f;
    fp.pass.light_shafts.decay = 0.95f;
    fp.pass.motion_vectors.enable = true;
    fp.pass.motion_blur.enable = true;
    fp.pass.motion_blur.samples = 12;
    fp.pass.motion_blur.strength = 0.85f;
    fp.pass.motion_blur.max_velocity_px = 20.0f;
    fp.pass.motion_blur.min_velocity_px = 0.30f;
    fp.pass.motion_blur.depth_reject = 0.10f;

    shs::TechniqueMode active_technique_mode = shs::TechniqueMode::Forward;
    size_t technique_cycle_index = 0;
    bool user_shadow_enabled = fp.pass.shadow.enable;
    bool user_light_shafts_enabled = fp.pass.light_shafts.enable;
    bool user_motion_blur_enabled = fp.pass.motion_blur.enable;
    bool user_fxaa_enabled = true;
    PassIsolationStage pass_isolation_stage = PassIsolationStage::MotionBlur;
    PassExecutionPlan pass_plan{};
    auto apply_technique_composition = [&]() {
        const shs::TechniqueProfile profile = shs::make_default_technique_profile(active_technique_mode);
        fp.technique.mode = active_technique_mode;
        fp.technique.depth_prepass = profile_has_pass(profile, "depth_prepass");
        fp.technique.light_culling =
            profile_has_pass(profile, "light_culling") ||
            profile_has_pass(profile, "cluster_light_assign");

        const bool profile_shadow = profile_has_pass(profile, "shadow_map");
        const bool profile_motion_blur = profile_has_pass(profile, "motion_blur");
        pass_plan = make_pass_execution_plan(
            pass_isolation_stage,
            user_shadow_enabled,
            user_light_shafts_enabled,
            user_motion_blur_enabled,
            profile_shadow,
            profile_motion_blur
        );
        fp.pass.shadow.enable = pass_plan.run_shadow;
        fp.pass.light_shafts.enable = pass_plan.run_shafts;
        fp.pass.motion_blur.enable = pass_plan.enable_motion_blur;
        fp.pass.motion_vectors.enable = fp.pass.motion_vectors.enable || fp.pass.motion_blur.enable;
    };
    apply_technique_composition();

    shs::CameraRig cam{};
    cam.pos = glm::vec3(0.0f, 6.0f, -16.0f);
    cam.yaw = glm::radians(90.0f);
    cam.pitch = glm::radians(-16.0f);
    // Follow mode default асаалттай.
    bool follow_camera = true;
    // Free болон chase камерыг тусад нь хадгалж, эцсийн камераа blend хийж гаргана.
    shs::CameraRig free_cam = cam;
    shs::CameraRig chase_cam = cam;
    float follow_blend = follow_camera ? 1.0f : 0.0f;
    bool drag_look = false;
    bool left_mouse_held = false;
    bool right_mouse_held = false;
    const float chase_dist = 9.5f;
    const float chase_height = 1.0f;
    const float chase_look_ahead = 3.5f;
    const float chase_smoothing = 0.16f;
    const float mode_blend_speed = 6.0f;
    float chase_orbit_yaw = 0.0f;
    float chase_orbit_pitch = glm::radians(20.0f);
    glm::vec3 chase_forward{1.0f, 0.0f, 0.0f};
    glm::vec3 prev_subaru_pos{0.0f};
    bool has_prev_subaru_pos = false;
    auto& subaru_ai = logic_systems.add_system<SubaruCruiseSystem>(
        "subaru",
        plane_extent * 0.48f,
        -0.95f,
        6.8f,
        1.9f,
        SUBARU_VISUAL_FORWARD_AXIS,
        0.0f
    );
    logic_systems.add_system<MonkeyWiggleSystem>("monkey", 0.32f, 0.22f, 1.9f);

    if (const auto* subaru_init = objects.find("subaru"))
    {
        prev_subaru_pos = subaru_init->tr.pos;
        has_prev_subaru_pos = true;
        chase_forward = world_forward_from_visual_yaw(subaru_init->tr.rot_euler.y, SUBARU_VISUAL_FORWARD_AXIS);
    }

    bool running = true;
    auto prev = std::chrono::steady_clock::now();
    float time_s = 0.0f;
    int frames = 0;
    float fps_accum = 0.0f;
    float logic_ms_accum = 0.0f;
    float render_ms_accum = 0.0f;
    float smoothed_dt = 1.0f / 60.0f;

    // Main loop: input -> logic -> scene/camera sync -> render -> present.
    while (running)
    {
        const auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - prev).count();
        prev = now;
        if (dt > 0.1f) dt = 0.1f;
        smoothed_dt = glm::mix(smoothed_dt, dt, 0.15f);
        dt = std::clamp(smoothed_dt, 1.0f / 240.0f, 1.0f / 20.0f);
        time_s += dt;
        fp.dt = dt;
        fp.time = time_s;

        shs::PlatformInputState pin{};
        if (!runtime.pump_input(pin)) break;
        if (pin.quit) running = false;
        // F1: debug view цикл.
        if (pin.cycle_debug_view)
        {
            const int next = (((int)fp.debug_view) + 1) % 4;
            fp.debug_view = (shs::DebugViewMode)next;
        }
        if (pin.cycle_cull_mode)
        {
            switch (fp.cull_mode)
            {
                case shs::CullMode::None: fp.cull_mode = shs::CullMode::Back; break;
                case shs::CullMode::Back: fp.cull_mode = shs::CullMode::Front; break;
                case shs::CullMode::Front:
                default: fp.cull_mode = shs::CullMode::None; break;
            }
        }
        if (pin.toggle_front_face)
        {
            fp.front_face_ccw = !fp.front_face_ccw;
        }
        // F4: PBR <-> BlinnPhong солих.
        if (pin.toggle_shading_model)
        {
            fp.shading_model = (fp.shading_model == shs::ShadingModel::PBRMetalRough)
                ? shs::ShadingModel::BlinnPhong
                : shs::ShadingModel::PBRMetalRough;
        }
        // B: technique composition цикл.
        if (pin.toggle_bot)
        {
            const auto& modes = known_technique_modes();
            technique_cycle_index = (technique_cycle_index + 1u) % modes.size();
            active_technique_mode = modes[technique_cycle_index];
            apply_technique_composition();
        }
        // L: light shafts user preference on/off.
        if (pin.toggle_light_shafts)
        {
            user_light_shafts_enabled = !user_light_shafts_enabled;
            apply_technique_composition();
        }
        // M: motion blur on/off.
        if (pin.toggle_motion_blur)
        {
            user_motion_blur_enabled = !user_motion_blur_enabled;
            apply_technique_composition();
        }
        // F7: FXAA final pass on/off (present path isolation).
        if (pin.toggle_fxaa)
        {
            user_fxaa_enabled = !user_fxaa_enabled;
        }
        // [ / ]: pass isolation ladder алхам алхмаар буцаах/урагшлуулах.
        if (pin.step_pass_isolation_prev)
        {
            pass_isolation_stage = step_pass_isolation_stage(pass_isolation_stage, -1);
            apply_technique_composition();
        }
        if (pin.step_pass_isolation_next)
        {
            pass_isolation_stage = step_pass_isolation_stage(pass_isolation_stage, +1);
            apply_technique_composition();
        }
        // F5: cubemap/procedural sky солих.
        if (pin.toggle_sky_mode)
        {
            if (sky_cm.valid()) use_cubemap_sky = !use_cubemap_sky;
        }
        // F6: camera follow mode toggle.
        if (pin.toggle_follow_camera)
        {
            const bool prev = follow_camera;
            follow_camera = !follow_camera;
            if (prev && !follow_camera)
            {
                // Chase -> Free: одоогийн харагдаж буй камераас free горим эхэлнэ.
                free_cam = cam;
            }
            else if (!prev && follow_camera)
            {
                // Free -> Chase: blend-г таслахгүй байлгахын тулд chase camera-г одоогийн байрлалаас эхлүүлнэ.
                chase_cam = cam;
            }
        }

        // Mouse hold төлөвийг SDL-ээс шууд уншиж drag-look/relative mode-ыг тогтвортой болгоно.
        const uint32_t mouse_state = SDL_GetMouseState(nullptr, nullptr);
        left_mouse_held = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
        const bool right_now = (mouse_state & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
        if (right_now != right_mouse_held)
        {
            right_mouse_held = right_now;
            runtime.set_relative_mouse_mode(right_mouse_held);
        }
        drag_look = left_mouse_held || right_mouse_held;

        // Left/Right drag хийхэд 2 горимд хоёуланд нь камер эргэлдэнэ.
        if (drag_look)
        {
            if (follow_camera)
            {
                chase_orbit_yaw -= pin.mouse_dx * CHASE_ORBIT_SENS;
                chase_orbit_pitch = std::clamp(
                    chase_orbit_pitch + pin.mouse_dy * CHASE_ORBIT_SENS,
                    glm::radians(5.0f),
                    glm::radians(70.0f)
                );
            }
            else
            {
                free_cam.yaw += pin.mouse_dx * MOUSE_LOOK_SENS;
                free_cam.pitch = std::clamp(
                    free_cam.pitch - pin.mouse_dy * MOUSE_LOOK_SENS,
                    glm::radians(-85.0f),
                    glm::radians(85.0f)
                );
            }
        }

        // Free camera хөдөлгөөн (WASD + QE).
        if (!follow_camera)
        {
            const float move_speed = FREE_CAM_BASE_SPEED * (pin.boost ? 2.5f : 1.0f) * dt;
            glm::vec3 fwd = free_cam.forward();
            fwd.y = 0.0f;
            const float fwd_len = glm::length(fwd);
            if (fwd_len > 1e-6f) fwd /= fwd_len;
            const glm::vec3 right = free_cam.right();
            if (pin.forward) free_cam.pos += fwd * move_speed;
            if (pin.backward) free_cam.pos -= fwd * move_speed;
            if (pin.right) free_cam.pos += right * move_speed;
            if (pin.left) free_cam.pos -= right * move_speed;
            if (pin.ascend) free_cam.pos.y += move_speed;
            if (pin.descend) free_cam.pos.y -= move_speed;
        }

        // Logic systems ажиллуулна (subaru cruise, follow camera, monkey wiggle).
        const auto t_logic0 = std::chrono::steady_clock::now();
        shs::LogicSystemContext logic_ctx{};
        logic_ctx.dt = dt;
        logic_ctx.time = time_s;
        logic_ctx.objects = &objects;
        logic_ctx.scene = &scene;
        logic_ctx.frame = &fp;
        logic_systems.tick(logic_ctx);
        const auto t_logic1 = std::chrono::steady_clock::now();
        logic_ms_accum += std::chrono::duration<float, std::milli>(t_logic1 - t_logic0).count();

        // Subaru-ийн transform-аас chase camera зорилтот байрлал/чиглэлийг frame бүр шинэчилнэ.
        if (const auto* subaru = objects.find("subaru"))
        {
            // Chase чиглэлийг model yaw бус, бодит хөдөлгөөний вектороос тооцно.
            glm::vec3 move = subaru->tr.pos - prev_subaru_pos;
            move.y = 0.0f;
            const float move_len = glm::length(move);
            if (has_prev_subaru_pos && move_len > 1e-4f)
            {
                const glm::vec3 move_dir = move / move_len;
                const float t_dir = 1.0f - std::exp(-std::max(0.0f, dt) * 10.0f);
                chase_forward = glm::normalize(glm::mix(chase_forward, move_dir, t_dir));
            }
            else
            {
                // Машин бараг зогссон үед AI-ийн одоогийн heading-ийг fallback чиглэл болгон авна.
                const glm::vec3 fallback_fwd = subaru_ai.heading_ws();
                chase_forward = glm::normalize(glm::mix(chase_forward, fallback_fwd, 0.08f));
            }
            prev_subaru_pos = subaru->tr.pos;
            has_prev_subaru_pos = true;

            const float car_yaw = std::atan2(chase_forward.z, chase_forward.x);
            const float orbit_yaw = car_yaw + 3.14159265f + chase_orbit_yaw;
            const float orbit_pitch = std::clamp(chase_orbit_pitch, glm::radians(5.0f), glm::radians(70.0f));
            const float cp = std::cos(orbit_pitch);
            const glm::vec3 orbit_dir{
                cp * std::cos(orbit_yaw),
                std::sin(orbit_pitch),
                cp * std::sin(orbit_yaw)
            };
            const glm::vec3 focus = subaru->tr.pos + glm::vec3(0.0f, chase_height, 0.0f);
            const glm::vec3 desired_cam = focus + orbit_dir * chase_dist;
            follow_target(chase_cam, desired_cam, glm::vec3(0.0f), chase_smoothing, dt);

            const glm::vec3 look_point = subaru->tr.pos + chase_forward * chase_look_ahead + glm::vec3(0.0f, 0.8f, 0.0f);
            const glm::vec3 v = look_point - chase_cam.pos;
            const float len = glm::length(v);
            if (len > 1e-6f)
            {
                const glm::vec3 d = v / len;
                const float target_yaw = std::atan2(d.z, d.x);
                const float target_pitch = std::asin(glm::clamp(d.y, -1.0f, 1.0f));
                const float rot_t = std::clamp(chase_smoothing * dt * 8.0f, 0.0f, 1.0f);
                chase_cam.yaw = lerp_angle_rad(chase_cam.yaw, target_yaw, rot_t);
                chase_cam.pitch = glm::mix(chase_cam.pitch, target_pitch, rot_t);
            }
        }

        // Камерын mode шилжилтийг тасралтгүй, зөөлөн blend-ээр шийднэ.
        const float target_blend = follow_camera ? 1.0f : 0.0f;
        const float blend_t = 1.0f - std::exp(-mode_blend_speed * std::max(0.0f, dt));
        follow_blend = glm::mix(follow_blend, target_blend, blend_t);
        cam.pos = glm::mix(free_cam.pos, chase_cam.pos, follow_blend);
        cam.yaw = lerp_angle_rad(free_cam.yaw, chase_cam.yaw, follow_blend);
        cam.pitch = glm::mix(free_cam.pitch, chase_cam.pitch, follow_blend);

        // Logic-оор шинэчлэгдсэн object/camera төлөвийг render scene рүү sync хийнэ.
        objects.sync_to_scene(scene);
        const VkExtent2D cam_extent = vk_backend->swapchain_extent();
        if (cam_extent.width > 0 && cam_extent.height > 0)
        {
            fp.w = static_cast<int>(cam_extent.width);
            fp.h = static_cast<int>(cam_extent.height);
        }
        const float cam_aspect = (fp.h > 0) ? (static_cast<float>(fp.w) / static_cast<float>(fp.h))
                                            : ((float)CANVAS_W / (float)CANVAS_H);
        shs::sync_camera_to_scene(cam, scene, cam_aspect);
        procedural_sky.set_sun_direction(scene.sun.dir_ws);
        scene.sky = use_cubemap_sky ? static_cast<const shs::ISkyModel*>(&cubemap_sky)
                                    : static_cast<const shs::ISkyModel*>(&procedural_sky);

        // Vulkan GPU draw (scene meshes -> swapchain).
        const auto t_render0 = std::chrono::steady_clock::now();
        if (!gpu_renderer.render(ctx, scene, fp, resources, pass_plan, user_fxaa_enabled))
        {
            SDL_Delay(2);
        }
        const auto t_render1 = std::chrono::steady_clock::now();
        render_ms_accum += std::chrono::duration<float, std::milli>(t_render1 - t_render0).count();

        // Богино хугацааны FPS/telemetry-ийг title дээр шинэчилнэ.
        frames++;
        fps_accum += dt;
        if (fps_accum >= 0.25f)
        {
            const int fps = (int)std::lround((float)frames / fps_accum);
            std::string title = "HelloPassBasicsVulkan | FPS: " + std::to_string(fps)
                + " | backend: " + std::string(ctx.active_backend_name())
                + " | dbg[F1]: " + std::to_string((int)fp.debug_view)
                + " | tech[B]: " + std::string(shs::technique_mode_name(active_technique_mode))
                + " | shade[F4]: " + (fp.shading_model == shs::ShadingModel::PBRMetalRough ? "PBR" : "Blinn")
                + " | cull[F2]: " + std::to_string((int)fp.cull_mode)
                + " | front[F3]: " + std::string(fp.front_face_ccw ? "CCW" : "CW")
                + " | sky[F5]: " + (use_cubemap_sky ? "cubemap" : "procedural")
                + " | follow[F6]: " + (follow_camera ? "on" : "off")
                + " | ai: " + std::string(subaru_ai.state_name())
                + "(" + std::to_string((int)std::lround(subaru_ai.state_progress() * 100.0f)) + "%)"
                + " | isolate[[/]]: " + std::string(pass_isolation_stage_name(pass_plan.stage))
                + " | shadow: " + std::string(fp.pass.shadow.enable ? "on" : "off")
                + " | bright: " + std::string(pass_plan.run_bright ? "on" : "off")
                + " | shafts[L]: " + (fp.pass.light_shafts.enable ? "on" : "off")
                + " | flare: " + std::string(pass_plan.run_flare ? "on" : "off")
                + " | mblur[M]: " + (fp.pass.motion_blur.enable ? "on" : "off")
                + " | fxaa[F7]: " + std::string(user_fxaa_enabled ? "on" : "off")
                + " | logic: " + std::to_string((int)std::lround(logic_ms_accum / std::max(1, frames))) + "ms"
                + " | render: " + std::to_string((int)std::lround(render_ms_accum / std::max(1, frames))) + "ms"
                + " | path: gpu-draw(composed)";
            runtime.set_title(title);
            frames = 0;
            fps_accum = 0.0f;
            logic_ms_accum = 0.0f;
            render_ms_accum = 0.0f;
        }
    }

    return 0;
}
