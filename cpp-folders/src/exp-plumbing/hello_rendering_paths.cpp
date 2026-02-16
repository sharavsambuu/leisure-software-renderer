#define SDL_MAIN_HANDLED

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/core/context.hpp>
#include <shs/camera/camera_math.hpp>
#include <shs/camera/convention.hpp>
#include <shs/camera/light_camera.hpp>
#include <shs/frame/technique_mode.hpp>
#include <shs/geometry/jolt_adapter.hpp>
#include <shs/geometry/culling_software.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/job/thread_pool_job_system.hpp>
#include <shs/job/wait_group.hpp>
#include <shs/lighting/light_culling_mode.hpp>
#include <shs/lighting/light_runtime.hpp>
#include <shs/lighting/light_set.hpp>
#include <shs/lighting/shadow_technique.hpp>
#include <shs/pipeline/render_path_compiler.hpp>
#include <shs/pipeline/render_path_recipe.hpp>
#include <shs/pipeline/render_path_registry.hpp>
#include <shs/pipeline/technique_profile.hpp>
#include <shs/resources/loaders/primitive_import.hpp>
#include <shs/resources/resource_registry.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>
#include <shs/rhi/drivers/vulkan/vk_cmd_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_frame_ownership.hpp>
#include <shs/rhi/drivers/vulkan/vk_memory_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_shader_utils.hpp>

namespace
{
constexpr int kDefaultW = 1280;
constexpr int kDefaultH = 720;
constexpr uint32_t kTileSize = 16;
constexpr uint32_t kMaxLightsPerTile = 128;
constexpr uint32_t kMaxLights = 768;
constexpr uint32_t kDefaultLightCount = 384;
constexpr int kSceneOccW = 320;
constexpr int kSceneOccH = 180;
constexpr int kLightOccW = 320;
constexpr int kLightOccH = 180;
constexpr float kTechniqueSwitchPeriodSec = 8.0f;
constexpr uint32_t kClusterZSlices = 16;
constexpr float kShadowNearZ = 0.05f;
constexpr uint32_t kSunShadowMapSize = 2048;
constexpr uint32_t kLocalShadowMapSize = 1024;
constexpr uint32_t kMaxSpotShadowMaps = 8;
constexpr uint32_t kMaxPointShadowLights = 2;
constexpr uint32_t kPointShadowFaceCount = 6;
constexpr uint32_t kMaxLocalShadowLayers = kMaxSpotShadowMaps + (kMaxPointShadowLights * kPointShadowFaceCount);
constexpr uint32_t kWorkerPoolRingSize = 2;
constexpr const char* kAppName = "HelloRenderingPaths";

struct Vertex
{
    glm::vec3 pos{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
};

struct DrawPush
{
    glm::mat4 model{1.0f};
    glm::vec4 base_color{1.0f};
    glm::vec4 material_params{0.0f, 0.5f, 1.0f, 0.0f}; // x: metallic, y: roughness, z: ao
};

struct ShadowPush
{
    glm::mat4 light_view_proj{1.0f};
    glm::mat4 model{1.0f};
};

struct alignas(16) CameraUBO
{
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::mat4 view_proj{1.0f};
    glm::vec4 camera_pos_time{0.0f};
    glm::vec4 sun_dir_intensity{0.0f, -1.0f, 0.0f, 1.0f};
    glm::uvec4 screen_tile_lightcount{0u}; // x: width, y: height, z: tiles_x, w: light_count
    glm::uvec4 params{0u};                  // x: tiles_y, y: max_per_tile, z: tile_size, w: culling_mode
    glm::uvec4 culling_params{0u};          // x: cluster_z_slices, y: lighting_technique
    glm::vec4 depth_params{0.1f, 260.0f, 0.0f, 0.0f}; // x: near, y: far
    glm::vec4 exposure_gamma{1.0f, 2.2f, 0.0f, 0.0f};
    glm::mat4 sun_shadow_view_proj{1.0f};
    glm::vec4 sun_shadow_params{1.0f, 0.0008f, 0.0015f, 2.0f}; // x: strength, y: bias_const, z: bias_slope, w: pcf_radius
    glm::vec4 sun_shadow_filter{1.0f, 1.0f, 0.0f, 0.0f};       // x: pcf_step, y: enabled
};

struct alignas(16) ShadowLightGPU
{
    glm::mat4 light_view_proj{1.0f};
    glm::vec4 position_range{0.0f}; // xyz: light pos, w: range/far
    glm::vec4 shadow_params{0.0f};  // x: strength, y: bias_const, z: bias_slope, w: pcf_radius
    glm::uvec4 meta{0u};            // x: ShadowTechnique, y: layer base, z: reserved, w: enabled
};
static_assert(sizeof(ShadowLightGPU) % 16 == 0, "ShadowLightGPU must be std430 compatible");

struct Instance
{
    enum class MeshKind : uint8_t
    {
        Sphere = 0,
        Box = 1,
        Cone = 2,
        Capsule = 3,
        Cylinder = 4
    };

    glm::vec3 base_pos{0.0f};
    glm::vec4 base_color{1.0f};
    glm::vec3 base_rot{0.0f};
    glm::vec3 rot_speed{0.0f};
    float scale = 1.0f;
    float phase = 0.0f;
    float metallic = 0.08f;
    float roughness = 0.36f;
    float ao = 1.0f;
    MeshKind mesh_kind = MeshKind::Sphere;
};

struct GpuBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

struct FrameResources
{
    GpuBuffer camera_buffer{};
    GpuBuffer light_buffer{};
    GpuBuffer shadow_light_buffer{};
    GpuBuffer tile_counts_buffer{};
    GpuBuffer tile_indices_buffer{};
    GpuBuffer tile_depth_ranges_buffer{};
    VkDescriptorSet global_set = VK_NULL_HANDLE;
};

struct DepthTarget
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t w = 0;
    uint32_t h = 0;
};

struct LayeredDepthTarget
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView sampled_view = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    std::vector<VkImageView> layer_views{};
    std::vector<VkFramebuffer> framebuffers{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t w = 0;
    uint32_t h = 0;
    uint32_t layers = 0;
};

struct WorkerPool
{
    std::array<VkCommandPool, kWorkerPoolRingSize> pools{VK_NULL_HANDLE, VK_NULL_HANDLE};
};

struct LocalShadowCaster
{
    uint32_t light_index = 0;
    shs::ShadowTechnique technique = shs::ShadowTechnique::None;
    uint32_t layer_base = 0;
    glm::vec3 position_ws{0.0f};
    float range = 1.0f;
    glm::vec3 direction_ws{0.0f, -1.0f, 0.0f};
    float outer_angle_rad = glm::radians(35.0f);
    float strength = 1.0f;
};

struct FreeCamera
{
    glm::vec3 pos{0.0f, 13.0f, -38.0f};
    float yaw = glm::half_pi<float>();
    float pitch = -0.22f;
    float move_speed = 20.0f;
    float look_speed = 0.003f;
    static constexpr float kMouseSpikeThreshold = 240.0f;
    static constexpr float kMouseDeltaClamp = 90.0f;

    void update(
        bool move_forward,
        bool move_backward,
        bool move_left,
        bool move_right,
        bool move_up,
        bool move_down,
        bool boost,
        bool left_mouse_down,
        bool right_mouse_down,
        float mouse_dx,
        float mouse_dy,
        float dt)
    {
        if (left_mouse_down || right_mouse_down)
        {
            float mdx = mouse_dx;
            float mdy = mouse_dy;
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

        const glm::vec3 fwd = shs::forward_from_yaw_pitch(yaw, pitch);
        const glm::vec3 right = shs::right_from_forward(fwd);
        const glm::vec3 up(0.0f, 1.0f, 0.0f);
        const float speed = move_speed * (boost ? 2.0f : 1.0f);
        if (move_forward) pos += fwd * speed * dt;
        if (move_backward) pos -= fwd * speed * dt;
        if (move_left) pos += right * speed * dt;
        if (move_right) pos -= right * speed * dt;
        if (move_up) pos += up * speed * dt;
        if (move_down) pos -= up * speed * dt;
    }

    glm::mat4 view_matrix() const
    {
        return shs::look_at_lh(pos, pos + shs::forward_from_yaw_pitch(yaw, pitch), glm::vec3(0.0f, 1.0f, 0.0f));
    }
};

enum class DebugVolumeMeshKind : uint8_t
{
    Sphere = 0,
    Cone = 1,
    Box = 2
};

struct LightVolumeDebugDraw
{
    DebugVolumeMeshKind mesh = DebugVolumeMeshKind::Sphere;
    glm::mat4 model{1.0f};
    glm::vec4 color{1.0f};
};

enum class VulkanCullerBackend : uint8_t
{
    GpuCompute = 0,
    Disabled = 1
};

enum class LightingTechnique : uint32_t
{
    PBR = 0u,
    BlinnPhong = 1u
};

const char* lighting_technique_name(LightingTechnique tech)
{
    switch (tech)
    {
        case LightingTechnique::PBR: return "pbr";
        case LightingTechnique::BlinnPhong: return "blinn";
    }
    return "pbr";
}

const char* vulkan_culler_backend_name(VulkanCullerBackend backend)
{
    switch (backend)
    {
        case VulkanCullerBackend::GpuCompute: return "gpu";
        case VulkanCullerBackend::Disabled: return "off";
    }
    return "gpu";
}

glm::vec3 safe_perp_axis(const glm::vec3& v)
{
    if (std::abs(v.y) < 0.9f) return glm::vec3(0.0f, 1.0f, 0.0f);
    return glm::vec3(0.0f, 0.0f, 1.0f);
}

void basis_from_axis(
    const glm::vec3& axis_y,
    glm::vec3& out_x,
    glm::vec3& out_y,
    glm::vec3& out_z)
{
    out_y = shs::normalize_or(axis_y, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 up_hint = safe_perp_axis(out_y);
    out_x = shs::normalize_or(glm::cross(up_hint, out_y), glm::vec3(1.0f, 0.0f, 0.0f));
    out_z = shs::normalize_or(glm::cross(out_y, out_x), glm::vec3(0.0f, 0.0f, 1.0f));
}

glm::mat4 model_from_basis_and_scale(
    const glm::vec3& position,
    const glm::vec3& axis_x,
    const glm::vec3& axis_y,
    const glm::vec3& axis_z,
    const glm::vec3& scale_xyz)
{
    glm::mat4 m(1.0f);
    m[0] = glm::vec4(axis_x * scale_xyz.x, 0.0f);
    m[1] = glm::vec4(axis_y * scale_xyz.y, 0.0f);
    m[2] = glm::vec4(axis_z * scale_xyz.z, 0.0f);
    m[3] = glm::vec4(position, 1.0f);
    return m;
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

shs::LightCullingMode default_culling_mode_for_technique(shs::TechniqueMode mode)
{
    switch (mode)
    {
        case shs::TechniqueMode::ForwardPlus:
            return shs::LightCullingMode::Tiled;
        case shs::TechniqueMode::TiledDeferred:
            return shs::LightCullingMode::TiledDepthRange;
        case shs::TechniqueMode::ClusteredForward:
            return shs::LightCullingMode::Clustered;
        case shs::TechniqueMode::Forward:
        case shs::TechniqueMode::Deferred:
        default:
            return shs::LightCullingMode::None;
    }
}

shs::RenderPathRenderingTechnique default_render_path_technique_for_mode(shs::TechniqueMode mode)
{
    switch (mode)
    {
        case shs::TechniqueMode::Forward:
            return shs::RenderPathRenderingTechnique::ForwardLit;
        case shs::TechniqueMode::ForwardPlus:
        case shs::TechniqueMode::ClusteredForward:
            return shs::RenderPathRenderingTechnique::ForwardPlus;
        case shs::TechniqueMode::Deferred:
        case shs::TechniqueMode::TiledDeferred:
            return shs::RenderPathRenderingTechnique::Deferred;
        default:
            return shs::RenderPathRenderingTechnique::ForwardPlus;
    }
}

shs::RenderPathRecipe make_default_stress_vk_recipe(shs::TechniqueMode mode)
{
    shs::RenderPathRecipe recipe{};
    recipe.name = std::string("stress_vk_") + shs::technique_mode_name(mode);
    recipe.backend = shs::RenderBackendType::Vulkan;
    recipe.light_volume_provider = shs::RenderPathLightVolumeProvider::JoltShapeVolumes;
    recipe.view_culling = shs::RenderPathCullingMode::FrustumAndOptionalOcclusion;
    recipe.shadow_culling = shs::RenderPathCullingMode::FrustumAndOptionalOcclusion;
    recipe.render_technique = default_render_path_technique_for_mode(mode);
    recipe.technique_mode = mode;
    recipe.runtime_defaults.view_occlusion_enabled = true;
    recipe.runtime_defaults.shadow_occlusion_enabled = false;
    recipe.runtime_defaults.debug_aabb = false;
    recipe.runtime_defaults.lit_mode = true;
    recipe.runtime_defaults.enable_shadows = true;
    recipe.wants_shadows = true;
    recipe.strict_validation = true;

    const shs::TechniqueProfile profile = shs::make_default_technique_profile(mode);
    recipe.pass_chain.reserve(profile.passes.size());
    for (const auto& p : profile.passes)
    {
        recipe.pass_chain.push_back(shs::RenderPathPassEntry{p.id, p.required});
    }
    return recipe;
}

LightingTechnique next_lighting_technique(LightingTechnique tech)
{
    switch (tech)
    {
        case LightingTechnique::PBR:
            return LightingTechnique::BlinnPhong;
        case LightingTechnique::BlinnPhong:
        default:
            return LightingTechnique::PBR;
    }
}

class HelloRenderingPathsApp
{
public:
    ~HelloRenderingPathsApp()
    {
        cleanup();
    }

    void run()
    {
        shs::jolt::init_jolt();
        init_sdl();
        init_backend();
        configure_vulkan_culler_backend_from_env();
        init_jobs();
        init_scene_data();
        init_gpu_resources();
        print_controls();
        main_loop();
    }

    void cleanup()
    {
        if (cleaned_up_) return;
        cleaned_up_ = true;

        if (vk_) vk_->wait_idle();

        destroy_pipelines();
        destroy_depth_target();
        destroy_layered_depth_target(sun_shadow_target_);
        destroy_layered_depth_target(local_shadow_target_);

        destroy_worker_pools();
        if (jobs_) jobs_.reset();
        
        destroy_buffer(vertex_buffer_);
        destroy_buffer(index_buffer_);
        destroy_buffer(floor_vertex_buffer_);
        destroy_buffer(floor_index_buffer_);
        destroy_buffer(cone_vertex_buffer_);
        destroy_buffer(cone_index_buffer_);
        destroy_buffer(box_vertex_buffer_);
        destroy_buffer(box_index_buffer_);
        destroy_buffer(sphere_line_index_buffer_);
        destroy_buffer(cone_line_index_buffer_);
        destroy_buffer(box_line_index_buffer_);
        destroy_buffer(capsule_vertex_buffer_);
        destroy_buffer(capsule_index_buffer_);
        destroy_buffer(cylinder_vertex_buffer_);
        destroy_buffer(cylinder_index_buffer_);

        for (auto& fr : frame_resources_)
        {
            destroy_buffer(fr.camera_buffer);
            destroy_buffer(fr.light_buffer);
            destroy_buffer(fr.shadow_light_buffer);
            destroy_buffer(fr.tile_counts_buffer);
            destroy_buffer(fr.tile_depth_ranges_buffer);
            destroy_buffer(fr.tile_indices_buffer);
            fr.global_set = VK_NULL_HANDLE;
        }

        if (vk_ && vk_->device() != VK_NULL_HANDLE)
        {
            if (depth_sampler_ != VK_NULL_HANDLE)
            {
                vkDestroySampler(vk_->device(), depth_sampler_, nullptr);
                depth_sampler_ = VK_NULL_HANDLE;
            }
            if (descriptor_pool_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(vk_->device(), descriptor_pool_, nullptr);
                descriptor_pool_ = VK_NULL_HANDLE;
            }
            if (global_set_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(vk_->device(), global_set_layout_, nullptr);
                global_set_layout_ = VK_NULL_HANDLE;
            }
        }

        keep_.clear();
        vk_ = nullptr;

        if (win_)
        {
            SDL_DestroyWindow(win_);
            win_ = nullptr;
        }
        if (sdl_ready_)
        {
            SDL_Quit();
            sdl_ready_ = false;
        }

        shs::jolt::shutdown_jolt();
    }

private:
    void print_controls() const
    {
        std::fprintf(stderr, "\n[%s] Controls\n", kAppName);
        std::fprintf(stderr, "  Esc        : quit\n");
        std::fprintf(stderr, "  F1         : toggle recording mode (inline / MT-secondary)\n");
        std::fprintf(stderr, "  F2         : cycle rendering path (Forward/Forward+/Deferred/TiledDeferred/ClusteredForward)\n");
        std::fprintf(stderr, "  Shift+F2   : cycle lighting technique (PBR/Blinn)\n");
        std::fprintf(stderr, "  Tab        : cycle rendering path (alias)\n");
        std::fprintf(stderr, "  F6         : toggle Vulkan culler backend (gpu / disabled)\n");
        std::fprintf(stderr, "  F7         : toggle light debug wireframe draw\n");
        std::fprintf(stderr, "  F11        : toggle auto lighting-technique switching\n");
        std::fprintf(stderr, "  F12        : toggle directional (sun) shadow contribution\n");
        std::fprintf(stderr, "  Drag LMB/RMB: free-look camera (WSL spike-filtered)\n");
        std::fprintf(stderr, "  W/A/S/D + Q/E: move camera, Shift: boost\n");
        std::fprintf(stderr, "  1/2        : orbit radius scale -/+\n");
        std::fprintf(stderr, "  3/4        : light height bias -/+\n");
        std::fprintf(stderr, "  5/6        : light range scale -/+\n");
        std::fprintf(stderr, "  7/8        : light intensity scale -/+\n");
        std::fprintf(stderr, "  9/0        : sun shadow strength -/+ (when F12 is on)\n");
        std::fprintf(stderr, "  R          : reset light tuning\n");
        std::fprintf(stderr, "  +/-        : decrease/increase active light count\n");
        std::fprintf(stderr, "  Title bar  : shows lighting-technique, render-path, culling mode, rejections, and frame ms\n\n");
    }

    void init_sdl()
    {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
        {
            throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
        }
        sdl_ready_ = true;

        win_ = SDL_CreateWindow(
            kAppName,
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            kDefaultW,
            kDefaultH,
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE
        );
        if (!win_)
        {
            throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
        }
    }

    void init_backend()
    {
        shs::RenderBackendCreateResult created = shs::create_render_backend(shs::RenderBackendType::Vulkan);
        if (!created.note.empty()) std::fprintf(stderr, "[shs] %s\n", created.note.c_str());
        if (!created.backend) throw std::runtime_error("Backend factory did not return a backend");

        keep_.push_back(std::move(created.backend));
        for (auto& aux : created.auxiliary_backends)
        {
            if (aux) keep_.push_back(std::move(aux));
        }
        for (const auto& b : keep_)
        {
            ctx_.register_backend(b.get());
        }

        if (created.active != shs::RenderBackendType::Vulkan)
        {
            throw std::runtime_error("Vulkan backend is not active");
        }

        vk_ = dynamic_cast<shs::VulkanRenderBackend*>(ctx_.backend(shs::RenderBackendType::Vulkan));
        if (!vk_)
        {
            throw std::runtime_error("Failed to acquire Vulkan backend instance");
        }

        int dw = 0;
        int dh = 0;
        SDL_Vulkan_GetDrawableSize(win_, &dw, &dh);
        if (dw <= 0 || dh <= 0)
        {
            dw = kDefaultW;
            dh = kDefaultH;
        }

        shs::VulkanRenderBackend::InitDesc init{};
        init.window = win_;
        init.width = dw;
        init.height = dh;
        init.enable_validation = true;
        init.app_name = kAppName;
        if (!vk_->init(init))
        {
            throw std::runtime_error("Vulkan backend init_sdl failed");
        }

        ctx_.set_primary_backend(vk_);
        std::fprintf(stderr, "[shs] active backend: %s\n", ctx_.active_backend_name());
    }

    void init_jobs()
    {
        const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
        worker_count_ = std::clamp<uint32_t>(hc, 1u, 8u);
        jobs_ = std::make_unique<shs::ThreadPoolJobSystem>(worker_count_);
    }

    static shs::AABB compute_local_aabb_from_positions(const std::vector<glm::vec3>& positions)
    {
        shs::AABB out{};
        if (positions.empty())
        {
            out.minv = glm::vec3(-0.5f);
            out.maxv = glm::vec3(0.5f);
            return out;
        }
        for (const glm::vec3& p : positions) out.expand(p);
        return out;
    }

    static shs::AABB compute_local_aabb_from_vertices(const std::vector<Vertex>& vertices)
    {
        shs::AABB out{};
        if (vertices.empty())
        {
            out.minv = glm::vec3(-0.5f);
            out.maxv = glm::vec3(0.5f);
            return out;
        }
        for (const Vertex& v : vertices) out.expand(v.pos);
        return out;
    }

    static void make_tessellated_floor_geometry(
        float half_extent,
        int subdivisions,
        std::vector<Vertex>& out_vertices,
        std::vector<uint32_t>& out_indices)
    {
        const int div = std::max(1, subdivisions);
        const int verts_per_row = div + 1;
        const float full = std::max(half_extent, 1.0f) * 2.0f;
        const float step = full / static_cast<float>(div);

        out_vertices.clear();
        out_indices.clear();
        out_vertices.reserve(static_cast<size_t>(verts_per_row) * static_cast<size_t>(verts_per_row));
        out_indices.reserve(static_cast<size_t>(div) * static_cast<size_t>(div) * 6u);

        for (int z = 0; z <= div; ++z)
        {
            for (int x = 0; x <= div; ++x)
            {
                const float px = -half_extent + static_cast<float>(x) * step;
                const float pz = -half_extent + static_cast<float>(z) * step;
                Vertex v{};
                v.pos = glm::vec3(px, 0.0f, pz);
                v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
                out_vertices.push_back(v);
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

                out_indices.push_back(i00);
                out_indices.push_back(i10);
                out_indices.push_back(i11);

                out_indices.push_back(i00);
                out_indices.push_back(i11);
                out_indices.push_back(i01);
            }
        }
    }

    static shs::DebugMesh make_debug_mesh_from_vertex_index_data(
        const std::vector<Vertex>& verts,
        const std::vector<uint32_t>& indices)
    {
        shs::DebugMesh mesh{};
        mesh.vertices.reserve(verts.size());
        for (const Vertex& v : verts)
        {
            mesh.vertices.push_back(v.pos);
        }
        mesh.indices = indices;
        return mesh;
    }

    static std::vector<uint32_t> make_line_indices_from_triangles(const std::vector<uint32_t>& tri_indices)
    {
        std::vector<uint32_t> out{};
        out.reserve((tri_indices.size() / 3u) * 6u);
        for (size_t i = 0; i + 2 < tri_indices.size(); i += 3)
        {
            const uint32_t a = tri_indices[i + 0];
            const uint32_t b = tri_indices[i + 1];
            const uint32_t c = tri_indices[i + 2];
            out.push_back(a); out.push_back(b);
            out.push_back(b); out.push_back(c);
            out.push_back(c); out.push_back(a);
        }
        return out;
    }

    static std::vector<Vertex> make_vertices_with_normals_from_debug_mesh(const shs::DebugMesh& mesh)
    {
        std::vector<Vertex> verts(mesh.vertices.size());
        for (size_t i = 0; i < mesh.vertices.size(); ++i)
        {
            verts[i].pos = mesh.vertices[i];
            verts[i].normal = glm::vec3(0.0f, 1.0f, 0.0f);
        }

        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3)
        {
            const uint32_t i0 = mesh.indices[i + 0];
            const uint32_t i1 = mesh.indices[i + 1];
            const uint32_t i2 = mesh.indices[i + 2];
            if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) continue;

            const glm::vec3 p0 = verts[i0].pos;
            const glm::vec3 p1 = verts[i1].pos;
            const glm::vec3 p2 = verts[i2].pos;
            glm::vec3 n = glm::cross(p2 - p0, p1 - p0);
            const float n2 = glm::dot(n, n);
            if (n2 <= 1e-12f) n = glm::vec3(0.0f, 1.0f, 0.0f);
            else n *= (1.0f / std::sqrt(n2));

            verts[i0].normal += n;
            verts[i1].normal += n;
            verts[i2].normal += n;
        }

        for (auto& v : verts)
        {
            const float n2 = glm::dot(v.normal, v.normal);
            if (n2 <= 1e-12f) v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            else v.normal *= (1.0f / std::sqrt(n2));
        }
        return verts;
    }

    const shs::AABB& local_aabb_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_local_aabb_;
            case Instance::MeshKind::Cone: return cone_local_aabb_;
            case Instance::MeshKind::Capsule: return capsule_local_aabb_;
            case Instance::MeshKind::Cylinder: return cylinder_local_aabb_;
            case Instance::MeshKind::Sphere:
            default: return sphere_local_aabb_;
        }
    }

    const shs::Sphere& local_bound_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_local_bound_;
            case Instance::MeshKind::Cone: return cone_local_bound_;
            case Instance::MeshKind::Capsule: return capsule_local_bound_;
            case Instance::MeshKind::Cylinder: return cylinder_local_bound_;
            case Instance::MeshKind::Sphere:
            default: return sphere_local_bound_;
        }
    }

    const JPH::ShapeRefC& cull_shape_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_shape_jolt_;
            case Instance::MeshKind::Cone: return cone_shape_jolt_;
            case Instance::MeshKind::Capsule: return capsule_shape_jolt_;
            case Instance::MeshKind::Cylinder: return cylinder_shape_jolt_;
            case Instance::MeshKind::Sphere:
            default: return sphere_shape_jolt_;
        }
    }

    const shs::DebugMesh& occluder_mesh_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_occluder_mesh_;
            case Instance::MeshKind::Cone: return cone_occluder_mesh_;
            case Instance::MeshKind::Capsule: return capsule_occluder_mesh_;
            case Instance::MeshKind::Cylinder: return cylinder_occluder_mesh_;
            case Instance::MeshKind::Sphere:
            default: return sphere_occluder_mesh_;
        }
    }

    const GpuBuffer& vertex_buffer_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_vertex_buffer_;
            case Instance::MeshKind::Cone: return cone_vertex_buffer_;
            case Instance::MeshKind::Capsule: return capsule_vertex_buffer_;
            case Instance::MeshKind::Cylinder: return cylinder_vertex_buffer_;
            case Instance::MeshKind::Sphere:
            default: return vertex_buffer_;
        }
    }

    const GpuBuffer& index_buffer_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_index_buffer_;
            case Instance::MeshKind::Cone: return cone_index_buffer_;
            case Instance::MeshKind::Capsule: return capsule_index_buffer_;
            case Instance::MeshKind::Cylinder: return cylinder_index_buffer_;
            case Instance::MeshKind::Sphere:
            default: return index_buffer_;
        }
    }

    uint32_t index_count_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return static_cast<uint32_t>(box_indices_.size());
            case Instance::MeshKind::Cone: return static_cast<uint32_t>(cone_indices_.size());
            case Instance::MeshKind::Capsule: return static_cast<uint32_t>(capsule_indices_.size());
            case Instance::MeshKind::Cylinder: return static_cast<uint32_t>(cylinder_indices_.size());
            case Instance::MeshKind::Sphere:
            default: return static_cast<uint32_t>(indices_.size());
        }
    }

    void init_scene_data()
    {
        shs::ResourceRegistry resources{};
        const shs::MeshAssetHandle sphere_h = shs::import_sphere_primitive(resources, shs::SphereDesc{0.5f, 18, 12}, "fplus_sphere");
        const shs::MeshAssetHandle cone_h = shs::import_cone_primitive(resources, shs::ConeDesc{1.0f, 1.0f, 20, 1, false}, "fplus_light_cone");
        const shs::MeshAssetHandle box_h = shs::import_box_primitive(resources, shs::BoxDesc{glm::vec3(1.0f), 1, 1, 1}, "fplus_light_box");

        const shs::MeshData* sphere_mesh = resources.get_mesh(sphere_h);
        if (!sphere_mesh || sphere_mesh->empty())
        {
            throw std::runtime_error("Failed to generate sphere primitive mesh");
        }
        const shs::MeshData* cone_mesh = resources.get_mesh(cone_h);
        if (!cone_mesh || cone_mesh->empty())
        {
            throw std::runtime_error("Failed to generate cone primitive mesh");
        }
        const shs::MeshData* box_mesh = resources.get_mesh(box_h);
        if (!box_mesh || box_mesh->empty())
        {
            throw std::runtime_error("Failed to generate box primitive mesh");
        }

        const JPH::ShapeRefC capsule_debug_shape = shs::jolt::make_capsule(0.92f, 0.42f);
        const JPH::ShapeRefC cylinder_debug_shape = shs::jolt::make_cylinder(0.90f, 0.46f);
        const shs::DebugMesh capsule_debug_mesh = shs::debug_mesh_from_shape(*capsule_debug_shape, JPH::Mat44::sIdentity());
        const shs::DebugMesh cylinder_debug_mesh = shs::debug_mesh_from_shape(*cylinder_debug_shape, JPH::Mat44::sIdentity());
        if (capsule_debug_mesh.vertices.empty() || capsule_debug_mesh.indices.empty())
        {
            throw std::runtime_error("Failed to build capsule debug mesh");
        }
        if (cylinder_debug_mesh.vertices.empty() || cylinder_debug_mesh.indices.empty())
        {
            throw std::runtime_error("Failed to build cylinder debug mesh");
        }

        sphere_local_aabb_ = compute_local_aabb_from_positions(sphere_mesh->positions);
        make_tessellated_floor_geometry(90.0f, 80, floor_vertices_, floor_indices_);
        floor_local_aabb_ = compute_local_aabb_from_vertices(floor_vertices_);
        cone_local_aabb_ = compute_local_aabb_from_positions(cone_mesh->positions);
        box_local_aabb_ = compute_local_aabb_from_positions(box_mesh->positions);
        capsule_local_aabb_ = compute_local_aabb_from_positions(capsule_debug_mesh.vertices);
        cylinder_local_aabb_ = compute_local_aabb_from_positions(cylinder_debug_mesh.vertices);
        sphere_local_bound_ = shs::sphere_from_aabb(sphere_local_aabb_);
        cone_local_bound_ = shs::sphere_from_aabb(cone_local_aabb_);
        box_local_bound_ = shs::sphere_from_aabb(box_local_aabb_);
        capsule_local_bound_ = shs::sphere_from_aabb(capsule_local_aabb_);
        cylinder_local_bound_ = shs::sphere_from_aabb(cylinder_local_aabb_);
        sphere_shape_jolt_ = shs::jolt::make_sphere(sphere_local_bound_.radius);
        box_shape_jolt_ = shs::jolt::make_box(box_local_aabb_.extent());
        cone_shape_jolt_ = shs::jolt::make_convex_hull(cone_mesh->positions);
        capsule_shape_jolt_ = capsule_debug_shape;
        cylinder_shape_jolt_ = cylinder_debug_shape;

        vertices_.clear();
        vertices_.reserve(sphere_mesh->positions.size());
        for (size_t i = 0; i < sphere_mesh->positions.size(); ++i)
        {
            Vertex v{};
            v.pos = sphere_mesh->positions[i];
            if (i < sphere_mesh->normals.size()) v.normal = sphere_mesh->normals[i];
            vertices_.push_back(v);
        }
        indices_ = sphere_mesh->indices;

        floor_model_ = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.25f, 0.0f));
        floor_material_color_ = glm::vec4(120.0f / 255.0f, 122.0f / 255.0f, 128.0f / 255.0f, 1.0f);
        // PBR plastic floor material.
        floor_material_params_ = glm::vec4(0.0f, 0.62f, 1.0f, 0.0f);

        cone_vertices_.clear();
        cone_vertices_.reserve(cone_mesh->positions.size());
        for (size_t i = 0; i < cone_mesh->positions.size(); ++i)
        {
            Vertex v{};
            v.pos = cone_mesh->positions[i];
            if (i < cone_mesh->normals.size()) v.normal = cone_mesh->normals[i];
            cone_vertices_.push_back(v);
        }
        cone_indices_ = cone_mesh->indices;
        cone_line_indices_ = make_line_indices_from_triangles(cone_indices_);
        cone_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(cone_vertices_, cone_indices_);

        box_vertices_.clear();
        box_vertices_.reserve(box_mesh->positions.size());
        for (size_t i = 0; i < box_mesh->positions.size(); ++i)
        {
            Vertex v{};
            v.pos = box_mesh->positions[i];
            if (i < box_mesh->normals.size()) v.normal = box_mesh->normals[i];
            box_vertices_.push_back(v);
        }
        box_indices_ = box_mesh->indices;
        box_line_indices_ = make_line_indices_from_triangles(box_indices_);

        capsule_vertices_ = make_vertices_with_normals_from_debug_mesh(capsule_debug_mesh);
        capsule_indices_ = capsule_debug_mesh.indices;
        cylinder_vertices_ = make_vertices_with_normals_from_debug_mesh(cylinder_debug_mesh);
        cylinder_indices_ = cylinder_debug_mesh.indices;

        sphere_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(vertices_, indices_);
        sphere_line_indices_ = make_line_indices_from_triangles(indices_);
        box_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(box_vertices_, box_indices_);
        capsule_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(capsule_vertices_, capsule_indices_);
        cylinder_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(cylinder_vertices_, cylinder_indices_);
        floor_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(floor_vertices_, floor_indices_);

        instances_.clear();
        instance_models_.clear();
        const int layer_count = 5;
        const int rows_per_layer = 8;
        const int cols_per_row = 12;
        const float col_spacing_x = 4.2f;
        const float row_spacing_z = 3.7f;
        const float layer_spacing_z = 16.0f;
        const float base_y = 1.1f;
        const float layer_y_step = 1.25f;
        std::mt19937 rng(1337u);
        std::uniform_real_distribution<float> jitter(-0.18f, 0.18f);
        std::uniform_real_distribution<float> hue(0.0f, 1.0f);
        std::uniform_real_distribution<float> scale_rand(0.54f, 1.18f);
        std::uniform_real_distribution<float> rot_rand(-0.28f, 0.28f);
        std::uniform_real_distribution<float> spin_rand(0.08f, 0.35f);
        for (int layer = 0; layer < layer_count; ++layer)
        {
            const float layer_z = (-0.5f * static_cast<float>(layer_count - 1) + static_cast<float>(layer)) * layer_spacing_z;
            for (int row = 0; row < rows_per_layer; ++row)
            {
                const float row_z = layer_z + (-0.5f * static_cast<float>(rows_per_layer - 1) + static_cast<float>(row)) * row_spacing_z;
                const float zig = (((row + layer) & 1) != 0) ? (0.45f * col_spacing_x) : 0.0f;
                for (int col = 0; col < cols_per_row; ++col)
                {
                    const uint32_t logical_idx =
                        static_cast<uint32_t>(layer * rows_per_layer * cols_per_row + row * cols_per_row + col);
                    Instance inst{};
                    switch (logical_idx % 5u)
                    {
                        case 1u:
                            inst.mesh_kind = Instance::MeshKind::Box;
                            break;
                        case 2u:
                            inst.mesh_kind = Instance::MeshKind::Cone;
                            break;
                        case 3u:
                            inst.mesh_kind = Instance::MeshKind::Capsule;
                            break;
                        case 4u:
                            inst.mesh_kind = Instance::MeshKind::Cylinder;
                            break;
                        case 0u:
                        default:
                            inst.mesh_kind = Instance::MeshKind::Sphere;
                            break;
                    }
                    inst.base_pos = glm::vec3(
                        (-0.5f * static_cast<float>(cols_per_row - 1) + static_cast<float>(col)) * col_spacing_x + zig + jitter(rng),
                        base_y + layer_y_step * static_cast<float>(layer) + 0.30f * static_cast<float>(col % 3),
                        row_z + jitter(rng));
                    const float h = hue(rng);
                    inst.base_color = glm::vec4(
                        0.45f + 0.55f * std::sin(6.28318f * (h + 0.00f)),
                        0.45f + 0.55f * std::sin(6.28318f * (h + 0.33f)),
                        0.45f + 0.55f * std::sin(6.28318f * (h + 0.66f)),
                        1.0f);
                    inst.scale = scale_rand(rng);
                    inst.phase = hue(rng) * 10.0f;
                    inst.base_rot = glm::vec3(rot_rand(rng), rot_rand(rng), rot_rand(rng));
                    inst.rot_speed = glm::vec3(spin_rand(rng), spin_rand(rng), spin_rand(rng));
                    inst.metallic = 0.04f + 0.22f * hue(rng);
                    inst.roughness = 0.24f + 0.42f * hue(rng);
                    inst.ao = 1.0f;
                    instances_.push_back(inst);
                }
            }
        }
        instance_models_.resize(instances_.size(), glm::mat4(1.0f));
        instance_visible_mask_.assign(instances_.size(), 1u);
        visible_instance_count_ = static_cast<uint32_t>(instances_.size());
        floor_visible_ = true;

        // Build a stable world-space caster bounds for sun shadow fitting.
        // This avoids per-frame shadow frustum jitter from animation/camera culling.
        shadow_scene_static_aabb_ = shs::transform_aabb(floor_local_aabb_, floor_model_);
        constexpr float kMaxBobAmplitude = 0.28f;
        for (const Instance& inst : instances_)
        {
            const float r = std::max(0.001f, local_bound_for_mesh(inst.mesh_kind).radius * inst.scale * 1.20f);
            const glm::vec3 minv = inst.base_pos + glm::vec3(-r, -r - kMaxBobAmplitude, -r);
            const glm::vec3 maxv = inst.base_pos + glm::vec3( r,  r + kMaxBobAmplitude,  r);
            shadow_scene_static_aabb_.expand(minv);
            shadow_scene_static_aabb_.expand(maxv);
        }
        shadow_scene_static_bounds_ready_ = true;

        light_anim_.clear();
        light_anim_.reserve(kMaxLights);
        gpu_lights_.resize(kMaxLights);
        shadow_lights_gpu_.assign(kMaxLights, ShadowLightGPU{});
        std::uniform_real_distribution<float> angle0(0.0f, 6.28318f);
        std::uniform_real_distribution<float> rad(8.0f, 34.0f);
        std::uniform_real_distribution<float> hgt(2.8f, 9.2f);
        std::uniform_real_distribution<float> spd(0.12f, 0.82f);
        std::uniform_real_distribution<float> radius(5.0f, 8.6f);
        std::uniform_real_distribution<float> inner_deg(12.0f, 20.0f);
        std::uniform_real_distribution<float> outer_extra_deg(6.0f, 14.0f);
        std::uniform_real_distribution<float> area_extent(0.8f, 2.4f);
        std::uniform_real_distribution<float> tube_half_len(0.7f, 2.2f);
        std::uniform_real_distribution<float> tube_rad(0.18f, 0.55f);
        std::uniform_real_distribution<float> axis_rand(-1.0f, 1.0f);
        std::uniform_real_distribution<float> att_pow(0.85f, 1.55f);
        std::uniform_real_distribution<float> att_bias(0.01f, 0.22f);
        std::uniform_real_distribution<float> right_rand(-1.0f, 1.0f);
        for (uint32_t i = 0; i < kMaxLights; ++i)
        {
            LightAnim l{};
            l.angle0 = angle0(rng);
            l.orbit_radius = rad(rng);
            l.height = hgt(rng);
            l.speed = spd(rng) * ((i & 1u) ? 1.0f : -1.0f);
            l.range = radius(rng);
            l.phase = hue(rng) * 10.0f;
            const float t = static_cast<float>(i) / static_cast<float>(kMaxLights);
            l.color = glm::vec3(
                0.35f + 0.65f * std::sin(6.28318f * (t + 0.00f)) * 0.5f + 0.5f,
                0.35f + 0.65f * std::sin(6.28318f * (t + 0.33f)) * 0.5f + 0.5f,
                0.35f + 0.65f * std::sin(6.28318f * (t + 0.66f)) * 0.5f + 0.5f);
            l.intensity = 6.0f + 8.0f * std::fmod(0.6180339f * static_cast<float>(i), 1.0f);
            l.attenuation_power = att_pow(rng);
            l.attenuation_bias = att_bias(rng);
            l.attenuation_cutoff = 0.0f;

            switch (i % 4u)
            {
                case 0u:
                    l.type = shs::LightType::Point;
                    l.attenuation_model = shs::LightAttenuationModel::InverseSquare;
                    l.intensity *= 0.95f;
                    l.color = glm::mix(l.color, glm::vec3(1.0f, 0.66f, 0.30f), 0.58f);
                    break;
                case 1u:
                {
                    l.type = shs::LightType::Spot;
                    l.attenuation_model = shs::LightAttenuationModel::InverseSquare;
                    const float inner = glm::radians(inner_deg(rng));
                    l.spot_inner_outer.x = inner;
                    l.spot_inner_outer.y = inner + glm::radians(outer_extra_deg(rng));
                    l.intensity *= 1.10f;
                    l.color = glm::mix(l.color, glm::vec3(0.34f, 0.84f, 1.0f), 0.63f);
                    break;
                }
                case 2u:
                    l.type = shs::LightType::RectArea;
                    l.attenuation_model = shs::LightAttenuationModel::Smooth;
                    l.shape_params = glm::vec4(area_extent(rng), area_extent(rng), 0.0f, 0.0f);
                    l.rect_right_ws = shs::normalize_or(glm::vec3(right_rand(rng), 0.0f, right_rand(rng)), glm::vec3(1.0f, 0.0f, 0.0f));
                    l.intensity *= 0.85f;
                    l.color = glm::mix(l.color, glm::vec3(0.98f, 0.44f, 0.80f), 0.64f);
                    break;
                case 3u:
                default:
                    l.type = shs::LightType::TubeArea;
                    l.attenuation_model = shs::LightAttenuationModel::Linear;
                    l.shape_params = glm::vec4(tube_half_len(rng), tube_rad(rng), 0.0f, 0.0f);
                    l.intensity *= 0.90f;
                    l.color = glm::mix(l.color, glm::vec3(0.36f, 1.0f, 0.58f), 0.60f);
                    break;
            }
            l.direction_ws = shs::normalize_or(glm::vec3(axis_rand(rng), -0.85f, axis_rand(rng)), glm::vec3(0.0f, -1.0f, 0.0f));
            light_anim_.push_back(l);
        }
        light_set_.points.reserve(kMaxLights);
        light_set_.spots.reserve(kMaxLights);
        light_set_.rect_areas.reserve(kMaxLights / 2u);
        light_set_.tube_areas.reserve(kMaxLights / 2u);

        shadow_settings_ = shs::make_default_shadow_composition_settings();
        shadow_settings_.quality.directional_resolution = kSunShadowMapSize;
        shadow_settings_.quality.local_resolution = kLocalShadowMapSize;
        shadow_settings_.quality.point_resolution = kLocalShadowMapSize;
        shadow_settings_.quality.filter = shs::ShadowFilter::PCF5x5;
        shadow_settings_.quality.pcf_step = 1.0f;
        shadow_settings_.budget.max_spot = std::min<uint32_t>(4u, kMaxSpotShadowMaps);
        shadow_settings_.budget.max_point = std::min<uint32_t>(2u, kMaxPointShadowLights);
        shadow_settings_.rect_area_proxy = false;
        shadow_settings_.tube_area_proxy = false;
        shadow_settings_.budget.max_rect_area = 0u;
        shadow_settings_.budget.max_tube_area = 0u;

        configure_render_path_defaults();
    }

    void configure_vulkan_culler_backend_from_env()
    {
        const char* env = std::getenv("SHS_VK_CULLER_BACKEND");
        if (!env || *env == '\0')
        {
            vulkan_culler_backend_ = VulkanCullerBackend::GpuCompute;
            return;
        }

        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (v == "off" || v == "0" || v == "disabled" || v == "none")
        {
            vulkan_culler_backend_ = VulkanCullerBackend::Disabled;
            return;
        }
        vulkan_culler_backend_ = VulkanCullerBackend::GpuCompute;
    }

    void init_gpu_resources()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) throw std::runtime_error("Vulkan device unavailable");

        create_worker_pools();
        create_descriptor_resources();
        create_geometry_buffers();
        create_dynamic_buffers();
        const VkExtent2D extent = vk_->swapchain_extent();
        ensure_render_targets(extent.width, extent.height);
        create_pipelines(true);
    }

    void create_worker_pools()
    {
        destroy_worker_pools();
        worker_pools_.resize(worker_count_);
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = vk_->graphics_queue_family_index();

        for (uint32_t i = 0; i < worker_count_; ++i)
        {
            for (uint32_t f = 0; f < kWorkerPoolRingSize; ++f)
            {
                if (vkCreateCommandPool(vk_->device(), &ci, nullptr, &worker_pools_[i].pools[f]) != VK_SUCCESS)
                {
                    throw std::runtime_error("vkCreateCommandPool failed for worker");
                }
            }
        }
    }

    void destroy_worker_pools()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        for (auto& w : worker_pools_)
        {
            for (auto& pool : w.pools)
            {
                if (pool == VK_NULL_HANDLE) continue;
                vkDestroyCommandPool(vk_->device(), pool, nullptr);
                pool = VK_NULL_HANDLE;
            }
        }
        worker_pools_.clear();
    }

    void create_buffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags mem_flags,
        GpuBuffer& out,
        bool map_memory)
    {
        destroy_buffer(out);
        if (!shs::vk_create_buffer(
                vk_->device(),
                vk_->physical_device(),
                size,
                usage,
                mem_flags,
                out.buffer,
                out.memory))
        {
            throw std::runtime_error("vk_create_buffer failed");
        }

        out.size = size;
        if (map_memory)
        {
            if (vkMapMemory(vk_->device(), out.memory, 0, size, 0, &out.mapped) != VK_SUCCESS)
            {
                shs::vk_destroy_buffer(vk_->device(), out.buffer, out.memory);
                throw std::runtime_error("vkMapMemory failed");
            }
        }
    }

    void destroy_buffer(GpuBuffer& b)
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        if (b.mapped)
        {
            vkUnmapMemory(vk_->device(), b.memory);
            b.mapped = nullptr;
        }
        shs::vk_destroy_buffer(vk_->device(), b.buffer, b.memory);
        b.size = 0;
    }

    void create_geometry_buffers()
    {
        const VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        create_buffer(
            static_cast<VkDeviceSize>(vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            vertex_buffer_,
            true);
        std::memcpy(vertex_buffer_.mapped, vertices_.data(), vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            index_buffer_,
            true);
        std::memcpy(index_buffer_.mapped, indices_.data(), indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(floor_vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            floor_vertex_buffer_,
            true);
        std::memcpy(floor_vertex_buffer_.mapped, floor_vertices_.data(), floor_vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(floor_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            floor_index_buffer_,
            true);
        std::memcpy(floor_index_buffer_.mapped, floor_indices_.data(), floor_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(cone_vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            cone_vertex_buffer_,
            true);
        std::memcpy(cone_vertex_buffer_.mapped, cone_vertices_.data(), cone_vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(cone_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            cone_index_buffer_,
            true);
        std::memcpy(cone_index_buffer_.mapped, cone_indices_.data(), cone_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(box_vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            box_vertex_buffer_,
            true);
        std::memcpy(box_vertex_buffer_.mapped, box_vertices_.data(), box_vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(box_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            box_index_buffer_,
            true);
        std::memcpy(box_index_buffer_.mapped, box_indices_.data(), box_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(sphere_line_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            sphere_line_index_buffer_,
            true);
        std::memcpy(
            sphere_line_index_buffer_.mapped,
            sphere_line_indices_.data(),
            sphere_line_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(cone_line_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            cone_line_index_buffer_,
            true);
        std::memcpy(
            cone_line_index_buffer_.mapped,
            cone_line_indices_.data(),
            cone_line_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(box_line_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            box_line_index_buffer_,
            true);
        std::memcpy(
            box_line_index_buffer_.mapped,
            box_line_indices_.data(),
            box_line_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(capsule_vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            capsule_vertex_buffer_,
            true);
        std::memcpy(capsule_vertex_buffer_.mapped, capsule_vertices_.data(), capsule_vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(capsule_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            capsule_index_buffer_,
            true);
        std::memcpy(capsule_index_buffer_.mapped, capsule_indices_.data(), capsule_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(cylinder_vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            cylinder_vertex_buffer_,
            true);
        std::memcpy(cylinder_vertex_buffer_.mapped, cylinder_vertices_.data(), cylinder_vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(cylinder_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            cylinder_index_buffer_,
            true);
        std::memcpy(cylinder_index_buffer_.mapped, cylinder_indices_.data(), cylinder_indices_.size() * sizeof(uint32_t));
    }

    void create_dynamic_buffers()
    {
        const VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        for (FrameResources& fr : frame_resources_)
        {
            create_buffer(
                sizeof(CameraUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                host_flags,
                fr.camera_buffer,
                true);

            create_buffer(
                static_cast<VkDeviceSize>(kMaxLights) * sizeof(shs::CullingLightGPU),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                host_flags,
                fr.light_buffer,
                true);

            create_buffer(
                static_cast<VkDeviceSize>(kMaxLights) * sizeof(ShadowLightGPU),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                host_flags,
                fr.shadow_light_buffer,
                true);

            std::memset(fr.shadow_light_buffer.mapped, 0, static_cast<size_t>(fr.shadow_light_buffer.size));
        }
    }

    VkFormat choose_depth_format() const
    {
        const std::array<VkFormat, 3> candidates{
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
        };

        for (VkFormat fmt : candidates)
        {
            VkFormatProperties props{};
            vkGetPhysicalDeviceFormatProperties(vk_->physical_device(), fmt, &props);
            const VkFormatFeatureFlags need =
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            if ((props.optimalTilingFeatures & need) == need)
            {
                return fmt;
            }
        }

        return VK_FORMAT_D32_SFLOAT;
    }

    void destroy_depth_target()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        VkDevice dev = vk_->device();
        if (depth_target_.framebuffer != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(dev, depth_target_.framebuffer, nullptr);
            depth_target_.framebuffer = VK_NULL_HANDLE;
        }
        if (depth_target_.render_pass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(dev, depth_target_.render_pass, nullptr);
            depth_target_.render_pass = VK_NULL_HANDLE;
        }
        if (depth_target_.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(dev, depth_target_.view, nullptr);
            depth_target_.view = VK_NULL_HANDLE;
        }
        if (depth_target_.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(dev, depth_target_.image, nullptr);
            depth_target_.image = VK_NULL_HANDLE;
        }
        if (depth_target_.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(dev, depth_target_.memory, nullptr);
            depth_target_.memory = VK_NULL_HANDLE;
        }
        depth_target_.w = 0;
        depth_target_.h = 0;
        depth_target_.format = VK_FORMAT_UNDEFINED;
    }

    void create_depth_target(uint32_t w, uint32_t h)
    {
        destroy_depth_target();
        depth_target_.w = w;
        depth_target_.h = h;
        depth_target_.format = choose_depth_format();

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.extent.width = w;
        ici.extent.height = h;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.format = depth_target_.format;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(vk_->device(), &ici, nullptr, &depth_target_.image) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImage failed for depth target");
        }

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(vk_->device(), depth_target_.image, &req);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = shs::vk_find_memory_type(
            vk_->physical_device(),
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX)
        {
            throw std::runtime_error("No compatible memory type for depth target");
        }
        if (vkAllocateMemory(vk_->device(), &mai, nullptr, &depth_target_.memory) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateMemory failed for depth target");
        }
        if (vkBindImageMemory(vk_->device(), depth_target_.image, depth_target_.memory, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBindImageMemory failed for depth target");
        }

        VkImageViewCreateInfo iv{};
        iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image = depth_target_.image;
        iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv.format = depth_target_.format;
        iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        iv.subresourceRange.baseMipLevel = 0;
        iv.subresourceRange.levelCount = 1;
        iv.subresourceRange.baseArrayLayer = 0;
        iv.subresourceRange.layerCount = 1;
        if (vkCreateImageView(vk_->device(), &iv, nullptr, &depth_target_.view) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImageView failed for depth target");
        }

        VkAttachmentDescription depth_att{};
        depth_att.format = depth_target_.format;
        depth_att.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_att.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference depth_ref{};
        depth_ref.attachment = 0;
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp.attachmentCount = 1;
        rp.pAttachments = &depth_att;
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        rp.dependencyCount = 2;
        rp.pDependencies = deps;
        if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &depth_target_.render_pass) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateRenderPass failed for depth prepass");
        }

        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = depth_target_.render_pass;
        fb.attachmentCount = 1;
        fb.pAttachments = &depth_target_.view;
        fb.width = w;
        fb.height = h;
        fb.layers = 1;
        if (vkCreateFramebuffer(vk_->device(), &fb, nullptr, &depth_target_.framebuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateFramebuffer failed for depth prepass");
        }
    }

    void destroy_layered_depth_target(LayeredDepthTarget& t)
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        VkDevice dev = vk_->device();
        for (VkFramebuffer fb : t.framebuffers)
        {
            if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(dev, fb, nullptr);
        }
        t.framebuffers.clear();
        for (VkImageView v : t.layer_views)
        {
            if (v != VK_NULL_HANDLE) vkDestroyImageView(dev, v, nullptr);
        }
        t.layer_views.clear();
        if (t.render_pass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(dev, t.render_pass, nullptr);
            t.render_pass = VK_NULL_HANDLE;
        }
        if (t.sampled_view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(dev, t.sampled_view, nullptr);
            t.sampled_view = VK_NULL_HANDLE;
        }
        if (t.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(dev, t.image, nullptr);
            t.image = VK_NULL_HANDLE;
        }
        if (t.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(dev, t.memory, nullptr);
            t.memory = VK_NULL_HANDLE;
        }
        t.w = 0;
        t.h = 0;
        t.layers = 0;
        t.format = VK_FORMAT_UNDEFINED;
    }

    void create_layered_depth_target(
        LayeredDepthTarget& out,
        uint32_t w,
        uint32_t h,
        uint32_t layers,
        VkImageViewType sampled_view_type)
    {
        destroy_layered_depth_target(out);
        out.w = w;
        out.h = h;
        out.layers = layers;
        out.format = choose_depth_format();

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.extent.width = w;
        ici.extent.height = h;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = layers;
        ici.format = out.format;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(vk_->device(), &ici, nullptr, &out.image) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImage failed for layered depth target");
        }

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(vk_->device(), out.image, &req);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = shs::vk_find_memory_type(
            vk_->physical_device(),
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX)
        {
            throw std::runtime_error("No compatible memory type for layered depth target");
        }
        if (vkAllocateMemory(vk_->device(), &mai, nullptr, &out.memory) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateMemory failed for layered depth target");
        }
        if (vkBindImageMemory(vk_->device(), out.image, out.memory, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBindImageMemory failed for layered depth target");
        }

        VkImageViewCreateInfo sv{};
        sv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        sv.image = out.image;
        sv.viewType = sampled_view_type;
        sv.format = out.format;
        sv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        sv.subresourceRange.baseMipLevel = 0;
        sv.subresourceRange.levelCount = 1;
        sv.subresourceRange.baseArrayLayer = 0;
        sv.subresourceRange.layerCount = layers;
        if (vkCreateImageView(vk_->device(), &sv, nullptr, &out.sampled_view) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImageView failed for layered depth sampled view");
        }

        VkAttachmentDescription depth_att{};
        depth_att.format = out.format;
        depth_att.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_att.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference depth_ref{};
        depth_ref.attachment = 0;
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp.attachmentCount = 1;
        rp.pAttachments = &depth_att;
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        rp.dependencyCount = 2;
        rp.pDependencies = deps;
        if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &out.render_pass) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateRenderPass failed for layered depth target");
        }

        out.layer_views.resize(layers, VK_NULL_HANDLE);
        out.framebuffers.resize(layers, VK_NULL_HANDLE);
        for (uint32_t i = 0; i < layers; ++i)
        {
            VkImageViewCreateInfo iv{};
            iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            iv.image = out.image;
            iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            iv.format = out.format;
            iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            iv.subresourceRange.baseMipLevel = 0;
            iv.subresourceRange.levelCount = 1;
            iv.subresourceRange.baseArrayLayer = i;
            iv.subresourceRange.layerCount = 1;
            if (vkCreateImageView(vk_->device(), &iv, nullptr, &out.layer_views[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateImageView failed for layered depth view");
            }

            VkFramebufferCreateInfo fb{};
            fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb.renderPass = out.render_pass;
            fb.attachmentCount = 1;
            fb.pAttachments = &out.layer_views[i];
            fb.width = w;
            fb.height = h;
            fb.layers = 1;
            if (vkCreateFramebuffer(vk_->device(), &fb, nullptr, &out.framebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateFramebuffer failed for layered depth target");
            }
        }
    }

    void ensure_shadow_targets()
    {
        const bool sun_ok =
            sun_shadow_target_.w == kSunShadowMapSize &&
            sun_shadow_target_.h == kSunShadowMapSize &&
            sun_shadow_target_.layers == 1u &&
            sun_shadow_target_.sampled_view != VK_NULL_HANDLE;
        const bool local_ok =
            local_shadow_target_.w == kLocalShadowMapSize &&
            local_shadow_target_.h == kLocalShadowMapSize &&
            local_shadow_target_.layers == kMaxLocalShadowLayers &&
            local_shadow_target_.sampled_view != VK_NULL_HANDLE;
        if (sun_ok && local_ok) return;

        create_layered_depth_target(
            sun_shadow_target_,
            kSunShadowMapSize,
            kSunShadowMapSize,
            1u,
            VK_IMAGE_VIEW_TYPE_2D);
        create_layered_depth_target(
            local_shadow_target_,
            kLocalShadowMapSize,
            kLocalShadowMapSize,
            kMaxLocalShadowLayers,
            VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    }

    void create_or_resize_tile_buffers(uint32_t tiles_x, uint32_t tiles_y)
    {
        const VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const VkDeviceSize tile_count = static_cast<VkDeviceSize>(tiles_x) * static_cast<VkDeviceSize>(tiles_y);
        const VkDeviceSize cluster_count = tile_count * static_cast<VkDeviceSize>(kClusterZSlices);
        const VkDeviceSize list_count = std::max(tile_count, cluster_count);
        const VkDeviceSize counts_size = list_count * sizeof(uint32_t);
        const VkDeviceSize indices_size = counts_size * kMaxLightsPerTile;
        const VkDeviceSize depth_ranges_size = tile_count * sizeof(glm::vec2);

        for (FrameResources& fr : frame_resources_)
        {
            create_buffer(counts_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, fr.tile_counts_buffer, true);
            create_buffer(indices_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, fr.tile_indices_buffer, true);
            create_buffer(depth_ranges_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, fr.tile_depth_ranges_buffer, true);

            std::memset(fr.tile_counts_buffer.mapped, 0, static_cast<size_t>(counts_size));
            std::memset(fr.tile_indices_buffer.mapped, 0, static_cast<size_t>(indices_size));
            std::memset(fr.tile_depth_ranges_buffer.mapped, 0, static_cast<size_t>(depth_ranges_size));
        }
    }

    void create_descriptor_resources()
    {
        if (depth_sampler_ == VK_NULL_HANDLE)
        {
            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = VK_FILTER_NEAREST;
            sci.minFilter = VK_FILTER_NEAREST;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.minLod = 0.0f;
            sci.maxLod = 0.0f;
            sci.maxAnisotropy = 1.0f;
            if (vkCreateSampler(vk_->device(), &sci, nullptr, &depth_sampler_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateSampler failed (depth)");
            }
        }

        if (global_set_layout_ == VK_NULL_HANDLE)
        {
            VkDescriptorSetLayoutBinding b[10]{};

            b[0].binding = 0;
            b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            b[0].descriptorCount = 1;
            b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

            for (uint32_t i = 1; i < 5; ++i)
            {
                b[i].binding = i;
                b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                b[i].descriptorCount = 1;
                b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
            }
            b[5].binding = 5;
            b[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[5].descriptorCount = 1;
            b[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            b[6].binding = 6;
            b[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[6].descriptorCount = 1;
            b[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            b[7].binding = 7;
            b[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[7].descriptorCount = 1;
            b[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            b[8].binding = 8;
            b[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[8].descriptorCount = 1;
            b[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            b[9].binding = 9;
            b[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[9].descriptorCount = 1;
            b[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = 10;
            ci.pBindings = b;
            if (vkCreateDescriptorSetLayout(vk_->device(), &ci, nullptr, &global_set_layout_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorSetLayout failed");
            }
        }

        if (descriptor_pool_ == VK_NULL_HANDLE)
        {
            VkDescriptorPoolSize sizes[3]{};
            sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            sizes[0].descriptorCount = kWorkerPoolRingSize;
            sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            sizes[1].descriptorCount = 5u * kWorkerPoolRingSize;
            sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sizes[2].descriptorCount = 4u * kWorkerPoolRingSize;

            VkDescriptorPoolCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            ci.maxSets = kWorkerPoolRingSize;
            ci.poolSizeCount = 3;
            ci.pPoolSizes = sizes;
            if (vkCreateDescriptorPool(vk_->device(), &ci, nullptr, &descriptor_pool_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorPool failed");
            }
        }

        if (frame_resources_.at_slot(0).global_set == VK_NULL_HANDLE)
        {
            std::array<VkDescriptorSet, kWorkerPoolRingSize> sets{};
            if (!shs::vk_allocate_descriptor_set_ring<kWorkerPoolRingSize>(
                    vk_->device(),
                    descriptor_pool_,
                    global_set_layout_,
                    sets))
            {
                throw std::runtime_error("vkAllocateDescriptorSets failed");
            }
            for (uint32_t i = 0; i < kWorkerPoolRingSize; ++i)
            {
                frame_resources_.at_slot(i).global_set = sets[i];
            }
        }
    }

    void update_global_descriptor_sets()
    {
        VkDescriptorImageInfo depth_info{};
        depth_info.sampler = depth_sampler_;
        depth_info.imageView = depth_target_.view;
        depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo sun_shadow_info{};
        sun_shadow_info.sampler = depth_sampler_;
        sun_shadow_info.imageView = sun_shadow_target_.sampled_view;
        sun_shadow_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo local_shadow_info{};
        local_shadow_info.sampler = depth_sampler_;
        local_shadow_info.imageView = local_shadow_target_.sampled_view;
        local_shadow_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo point_shadow_info{};
        point_shadow_info.sampler = depth_sampler_;
        point_shadow_info.imageView = local_shadow_target_.sampled_view;
        point_shadow_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        for (FrameResources& fr : frame_resources_)
        {
            if (fr.global_set == VK_NULL_HANDLE) continue;

            VkDescriptorBufferInfo camera_info{};
            camera_info.buffer = fr.camera_buffer.buffer;
            camera_info.offset = 0;
            camera_info.range = sizeof(CameraUBO);

            VkDescriptorBufferInfo light_info{};
            light_info.buffer = fr.light_buffer.buffer;
            light_info.offset = 0;
            light_info.range = static_cast<VkDeviceSize>(kMaxLights) * sizeof(shs::CullingLightGPU);

            VkDescriptorBufferInfo tile_counts_info{};
            tile_counts_info.buffer = fr.tile_counts_buffer.buffer;
            tile_counts_info.offset = 0;
            tile_counts_info.range = fr.tile_counts_buffer.size;

            VkDescriptorBufferInfo tile_indices_info{};
            tile_indices_info.buffer = fr.tile_indices_buffer.buffer;
            tile_indices_info.offset = 0;
            tile_indices_info.range = fr.tile_indices_buffer.size;

            VkDescriptorBufferInfo tile_depth_ranges_info{};
            tile_depth_ranges_info.buffer = fr.tile_depth_ranges_buffer.buffer;
            tile_depth_ranges_info.offset = 0;
            tile_depth_ranges_info.range = fr.tile_depth_ranges_buffer.size;

            VkDescriptorBufferInfo shadow_light_info{};
            shadow_light_info.buffer = fr.shadow_light_buffer.buffer;
            shadow_light_info.offset = 0;
            shadow_light_info.range = static_cast<VkDeviceSize>(kMaxLights) * sizeof(ShadowLightGPU);

            VkWriteDescriptorSet w[10]{};
            for (int i = 0; i < 10; ++i)
            {
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet = fr.global_set;
                w[i].dstBinding = static_cast<uint32_t>(i);
                w[i].descriptorCount = 1;
            }

            w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w[0].pBufferInfo = &camera_info;

            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[1].pBufferInfo = &light_info;

            w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[2].pBufferInfo = &tile_counts_info;

            w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[3].pBufferInfo = &tile_indices_info;

            w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[4].pBufferInfo = &tile_depth_ranges_info;

            w[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[5].pImageInfo = &depth_info;

            w[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[6].pImageInfo = &sun_shadow_info;

            w[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[7].pImageInfo = &local_shadow_info;

            w[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[8].pImageInfo = &point_shadow_info;

            w[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[9].pBufferInfo = &shadow_light_info;

            vkUpdateDescriptorSets(vk_->device(), 10, w, 0, nullptr);
        }
    }

    void destroy_pipelines()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;

        auto destroy_pipeline = [&](VkPipeline& p) {
            if (p != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(vk_->device(), p, nullptr);
                p = VK_NULL_HANDLE;
            }
        };
        auto destroy_layout = [&](VkPipelineLayout& l) {
            if (l != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(vk_->device(), l, nullptr);
                l = VK_NULL_HANDLE;
            }
        };

        destroy_pipeline(depth_pipeline_);
        destroy_layout(depth_pipeline_layout_);
        destroy_pipeline(shadow_pipeline_);
        destroy_layout(shadow_pipeline_layout_);

        destroy_pipeline(scene_pipeline_);
        destroy_pipeline(scene_wire_pipeline_);
        destroy_layout(scene_pipeline_layout_);

        destroy_pipeline(depth_reduce_pipeline_);
        destroy_pipeline(compute_pipeline_);
        destroy_layout(compute_pipeline_layout_);

        pipeline_gen_ = 0;
    }

    void create_pipelines(bool force)
    {
        if (!force && scene_pipeline_ != VK_NULL_HANDLE && pipeline_gen_ == vk_->swapchain_generation()) return;

        destroy_pipelines();

        const std::vector<char> shadow_vs_code = shs::vk_read_binary_file(SHS_VK_FP_SHADOW_VERT_SPV);
        const std::vector<char> scene_vs_code = shs::vk_read_binary_file(SHS_VK_FP_SCENE_VERT_SPV);
        const std::vector<char> scene_fs_code = shs::vk_read_binary_file(SHS_VK_FP_SCENE_FRAG_SPV);
        const std::vector<char> depth_reduce_cs_code = shs::vk_read_binary_file(SHS_VK_FP_DEPTH_REDUCE_COMP_SPV);
        const std::vector<char> cull_cs_code = shs::vk_read_binary_file(SHS_VK_FP_LIGHT_CULL_COMP_SPV);

        VkShaderModule shadow_vs = shs::vk_create_shader_module(vk_->device(), shadow_vs_code);
        VkShaderModule scene_vs = shs::vk_create_shader_module(vk_->device(), scene_vs_code);
        VkShaderModule scene_fs = shs::vk_create_shader_module(vk_->device(), scene_fs_code);
        VkShaderModule depth_reduce_cs = shs::vk_create_shader_module(vk_->device(), depth_reduce_cs_code);
        VkShaderModule cull_cs = shs::vk_create_shader_module(vk_->device(), cull_cs_code);

        const auto cleanup_modules = [&]() {
            if (shadow_vs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), shadow_vs, nullptr);
            if (scene_vs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), scene_vs, nullptr);
            if (scene_fs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), scene_fs, nullptr);
            if (depth_reduce_cs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), depth_reduce_cs, nullptr);
            if (cull_cs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), cull_cs, nullptr);
        };

        VkPushConstantRange shadow_pc{};
        shadow_pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        shadow_pc.offset = 0;
        shadow_pc.size = sizeof(ShadowPush);

        VkPipelineLayoutCreateInfo sh_pl{};
        sh_pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        sh_pl.pushConstantRangeCount = 1;
        sh_pl.pPushConstantRanges = &shadow_pc;
        if (vkCreatePipelineLayout(vk_->device(), &sh_pl, nullptr, &shadow_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (shadow)");
        }

        VkPushConstantRange draw_pc{};
        draw_pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        draw_pc.offset = 0;
        draw_pc.size = sizeof(DrawPush);

        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &global_set_layout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &draw_pc;
        if (vkCreatePipelineLayout(vk_->device(), &pli, nullptr, &depth_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (depth)");
        }
        if (vkCreatePipelineLayout(vk_->device(), &pli, nullptr, &scene_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (scene)");
        }

        VkPipelineLayoutCreateInfo cli{};
        cli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        cli.setLayoutCount = 1;
        cli.pSetLayouts = &global_set_layout_;
        if (vkCreatePipelineLayout(vk_->device(), &cli, nullptr, &compute_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (compute)");
        }

        VkPipelineShaderStageCreateInfo shadow_stage{};
        shadow_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shadow_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        shadow_stage.module = shadow_vs;
        shadow_stage.pName = "main";

        VkPipelineShaderStageCreateInfo depth_stage{};
        depth_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        depth_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        depth_stage.module = scene_vs;
        depth_stage.pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrs[2]{};
        attrs[0].location = 0;
        attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(Vertex, pos);
        attrs[1].location = 1;
        attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(Vertex, normal);

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 2;
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
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds_depth{};
        ds_depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds_depth.depthTestEnable = VK_TRUE;
        ds_depth.depthWriteEnable = VK_TRUE;
        ds_depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dyn_states;

        VkGraphicsPipelineCreateInfo gp_shadow{};
        gp_shadow.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp_shadow.stageCount = 1;
        gp_shadow.pStages = &shadow_stage;
        gp_shadow.pVertexInputState = &vi;
        gp_shadow.pInputAssemblyState = &ia;
        gp_shadow.pViewportState = &vp;
        gp_shadow.pRasterizationState = &rs;
        gp_shadow.pMultisampleState = &ms;
        gp_shadow.pDepthStencilState = &ds_depth;
        gp_shadow.pDynamicState = &dyn;
        gp_shadow.layout = shadow_pipeline_layout_;
        gp_shadow.renderPass = sun_shadow_target_.render_pass;
        gp_shadow.subpass = 0;
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_shadow, nullptr, &shadow_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (shadow)");
        }

        VkGraphicsPipelineCreateInfo gp_depth{};
        gp_depth.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp_depth.stageCount = 1;
        gp_depth.pStages = &depth_stage;
        gp_depth.pVertexInputState = &vi;
        gp_depth.pInputAssemblyState = &ia;
        gp_depth.pViewportState = &vp;
        gp_depth.pRasterizationState = &rs;
        gp_depth.pMultisampleState = &ms;
        gp_depth.pDepthStencilState = &ds_depth;
        gp_depth.pDynamicState = &dyn;
        gp_depth.layout = depth_pipeline_layout_;
        gp_depth.renderPass = depth_target_.render_pass;
        gp_depth.subpass = 0;

        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_depth, nullptr, &depth_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (depth)");
        }

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = scene_vs;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = scene_fs;
        stages[1].pName = "main";

        VkPipelineDepthStencilStateCreateInfo ds_scene{};
        ds_scene.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds_scene.depthTestEnable = vk_->has_depth_attachment() ? VK_TRUE : VK_FALSE;
        ds_scene.depthWriteEnable = vk_->has_depth_attachment() ? VK_TRUE : VK_FALSE;
        ds_scene.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        cba.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        VkGraphicsPipelineCreateInfo gp_scene{};
        gp_scene.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp_scene.stageCount = 2;
        gp_scene.pStages = stages;
        gp_scene.pVertexInputState = &vi;
        gp_scene.pInputAssemblyState = &ia;
        gp_scene.pViewportState = &vp;
        gp_scene.pRasterizationState = &rs;
        gp_scene.pMultisampleState = &ms;
        gp_scene.pDepthStencilState = &ds_scene;
        gp_scene.pColorBlendState = &cb;
        gp_scene.pDynamicState = &dyn;
        gp_scene.layout = scene_pipeline_layout_;
        gp_scene.renderPass = vk_->render_pass();
        gp_scene.subpass = 0;

        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_scene, nullptr, &scene_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (scene)");
        }

        VkPipelineInputAssemblyStateCreateInfo ia_lines = ia;
        ia_lines.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

        VkPipelineDepthStencilStateCreateInfo ds_wire = ds_scene;
        ds_wire.depthWriteEnable = VK_FALSE;

        VkGraphicsPipelineCreateInfo gp_scene_wire = gp_scene;
        gp_scene_wire.pInputAssemblyState = &ia_lines;
        gp_scene_wire.pDepthStencilState = &ds_wire;
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_scene_wire, nullptr, &scene_wire_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (scene wire)");
        }

        VkComputePipelineCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp.layout = compute_pipeline_layout_;
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = depth_reduce_cs;
        cp.stage.pName = "main";
        if (vkCreateComputePipelines(vk_->device(), VK_NULL_HANDLE, 1, &cp, nullptr, &depth_reduce_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateComputePipelines failed (depth reduce)");
        }

        cp.stage.module = cull_cs;
        cp.stage.pName = "main";
        if (vkCreateComputePipelines(vk_->device(), VK_NULL_HANDLE, 1, &cp, nullptr, &compute_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateComputePipelines failed");
        }

        cleanup_modules();
        pipeline_gen_ = vk_->swapchain_generation();
    }

    void ensure_render_targets(uint32_t w, uint32_t h)
    {
        if (w == 0 || h == 0) return;
        if (depth_target_.w == w && depth_target_.h == h && tile_w_ == ((w + kTileSize - 1) / kTileSize) && tile_h_ == ((h + kTileSize - 1) / kTileSize))
        {
            return;
        }

        create_depth_target(w, h);
        ensure_shadow_targets();
        tile_w_ = (w + kTileSize - 1) / kTileSize;
        tile_h_ = (h + kTileSize - 1) / kTileSize;
        create_or_resize_tile_buffers(tile_w_, tile_h_);
        update_global_descriptor_sets();
        create_pipelines(true);
    }

    void apply_technique_profile(shs::TechniqueMode mode, const shs::TechniqueProfile& profile)
    {
        active_technique_ = mode;
        const auto& modes = known_technique_modes();
        for (size_t i = 0; i < modes.size(); ++i)
        {
            if (modes[i] == mode)
            {
                technique_cycle_index_ = i;
                break;
            }
        }

        profile_depth_prepass_enabled_ = profile_has_pass(profile, "depth_prepass");
        enable_light_culling_ =
            profile_has_pass(profile, "light_culling") ||
            profile_has_pass(profile, "cluster_light_assign");

        shs::LightCullingMode mode_hint = default_culling_mode_for_technique(mode);
        if (!enable_light_culling_)
        {
            mode_hint = shs::LightCullingMode::None;
        }
        culling_mode_ = mode_hint;

        const bool has_forward_lighting =
            profile_has_pass(profile, "pbr_forward") ||
            profile_has_pass(profile, "pbr_forward_plus") ||
            profile_has_pass(profile, "pbr_forward_clustered");
        const bool has_deferred_lighting =
            profile_has_pass(profile, "deferred_lighting") ||
            profile_has_pass(profile, "deferred_lighting_tiled");
        enable_scene_pass_ = has_forward_lighting || has_deferred_lighting || profile_has_pass(profile, "gbuffer");
        if (!enable_scene_pass_) enable_scene_pass_ = true;

        refresh_depth_prepass_state();
        use_forward_plus_ = (culling_mode_ != shs::LightCullingMode::None);
        technique_switch_accum_sec_ = 0.0f;
    }

    void apply_technique_mode(shs::TechniqueMode mode)
    {
        const shs::TechniqueProfile profile = shs::make_default_technique_profile(mode);
        apply_technique_profile(mode, profile);
    }

    void init_render_path_registry()
    {
        render_path_registry_.clear();
        render_path_cycle_order_.clear();

        const auto& modes = known_technique_modes();
        render_path_cycle_order_.reserve(modes.size());
        for (const shs::TechniqueMode mode : modes)
        {
            shs::RenderPathRecipe recipe = make_default_stress_vk_recipe(mode);
            const std::string id = recipe.name;
            (void)render_path_registry_.register_recipe(std::move(recipe));
            render_path_cycle_order_.push_back(id);
        }
    }

    bool apply_render_path_recipe_by_index(size_t index)
    {
        if (render_path_cycle_order_.empty())
        {
            render_path_plan_valid_ = false;
            render_path_recipe_ = shs::RenderPathRecipe{};
            render_path_plan_ = shs::RenderPathExecutionPlan{};
            apply_technique_mode(shs::TechniqueMode::ForwardPlus);
            return false;
        }

        render_path_cycle_index_ = index % render_path_cycle_order_.size();
        const std::string& recipe_id = render_path_cycle_order_[render_path_cycle_index_];
        const shs::RenderPathRecipe* recipe = render_path_registry_.find_recipe(recipe_id);
        if (!recipe)
        {
            std::fprintf(stderr, "[render-path][stress][error] Missing recipe id '%s'.\n", recipe_id.c_str());
            render_path_plan_valid_ = false;
            render_path_recipe_ = shs::RenderPathRecipe{};
            render_path_plan_ = shs::RenderPathExecutionPlan{};
            apply_technique_mode(shs::TechniqueMode::ForwardPlus);
            return false;
        }

        render_path_recipe_ = *recipe;
        const shs::RenderPathCompiler compiler{};
        render_path_plan_ = compiler.compile(render_path_recipe_, ctx_, nullptr);

        for (const auto& w : render_path_plan_.warnings)
        {
            std::fprintf(stderr, "[render-path][stress][warn] %s\n", w.c_str());
        }
        for (const auto& e : render_path_plan_.errors)
        {
            std::fprintf(stderr, "[render-path][stress][error] %s\n", e.c_str());
        }

        render_path_plan_valid_ = render_path_plan_.valid;
        if (!render_path_plan_valid_)
        {
            std::fprintf(
                stderr,
                "[render-path][stress] Recipe '%s' invalid. Falling back to default technique profile.\n",
                render_path_recipe_.name.c_str());
            apply_technique_mode(render_path_recipe_.technique_mode);
            return false;
        }

        const shs::TechniqueProfile profile = shs::make_technique_profile(render_path_plan_);
        apply_technique_profile(render_path_plan_.technique_mode, profile);
        enable_scene_occlusion_ = render_path_plan_.runtime_state.view_occlusion_enabled;
        enable_light_occlusion_ = render_path_plan_.runtime_state.shadow_occlusion_enabled;
        shadow_settings_.enable = render_path_plan_.runtime_state.enable_shadows;

        std::fprintf(
            stderr,
            "[render-path][stress] Applied recipe '%s' (%s), passes:%zu.\n",
            render_path_plan_.recipe_name.c_str(),
            render_path_plan_valid_ ? "valid" : "invalid",
            render_path_plan_.pass_chain.size());
        return true;
    }

    void cycle_render_path_recipe()
    {
        if (render_path_cycle_order_.empty()) return;
        render_path_cycle_index_ = (render_path_cycle_index_ + 1u) % render_path_cycle_order_.size();
        (void)apply_render_path_recipe_by_index(render_path_cycle_index_);
    }

    void cycle_lighting_technique()
    {
        lighting_technique_ = next_lighting_technique(lighting_technique_);
    }

    void configure_render_path_defaults()
    {
        init_render_path_registry();
        const auto& modes = known_technique_modes();
        size_t preferred_index = 0u;
        for (size_t i = 0; i < modes.size(); ++i)
        {
            if (modes[i] == shs::TechniqueMode::ForwardPlus)
            {
                preferred_index = i;
                break;
            }
        }
        (void)apply_render_path_recipe_by_index(preferred_index);
    }

    void refresh_depth_prepass_state()
    {
        const bool needs_depth_for_culling =
            enable_light_culling_ &&
            culling_mode_ == shs::LightCullingMode::TiledDepthRange;
        enable_depth_prepass_ = profile_depth_prepass_enabled_ || needs_depth_for_culling;
    }

    void update_culling_debug_stats(uint32_t frame_slot)
    {
        if (!frame_resources_.valid_slot(frame_slot) || tile_w_ == 0 || tile_h_ == 0)
        {
            cull_debug_total_refs_ = 0;
            cull_debug_non_empty_lists_ = 0;
            cull_debug_list_count_ = 0;
            cull_debug_max_list_size_ = 0;
            return;
        }
        const GpuBuffer& tile_counts_buffer = frame_resources_.at_slot(frame_slot).tile_counts_buffer;
        if (!tile_counts_buffer.mapped || tile_counts_buffer.size < sizeof(uint32_t))
        {
            cull_debug_total_refs_ = 0;
            cull_debug_non_empty_lists_ = 0;
            cull_debug_list_count_ = 0;
            cull_debug_max_list_size_ = 0;
            return;
        }

        uint32_t list_count = tile_w_ * tile_h_;
        if (culling_mode_ == shs::LightCullingMode::Clustered)
        {
            list_count *= kClusterZSlices;
        }
        const uint32_t capacity = static_cast<uint32_t>(tile_counts_buffer.size / sizeof(uint32_t));
        list_count = std::min(list_count, capacity);

        const uint32_t* counts = reinterpret_cast<const uint32_t*>(tile_counts_buffer.mapped);
        uint64_t total_refs = 0;
        uint32_t non_empty = 0;
        uint32_t max_list = 0;
        for (uint32_t i = 0; i < list_count; ++i)
        {
            const uint32_t c = std::min(counts[i], kMaxLightsPerTile);
            total_refs += static_cast<uint64_t>(c);
            if (c > 0) ++non_empty;
            if (c > max_list) max_list = c;
        }

        cull_debug_total_refs_ = total_refs;
        cull_debug_non_empty_lists_ = non_empty;
        cull_debug_list_count_ = list_count;
        cull_debug_max_list_size_ = max_list;
    }

    void rebuild_instance_cull_shapes()
    {
        if (instance_cull_shapes_.size() != instances_.size())
        {
            instance_cull_shapes_.resize(instances_.size());
        }
        for (size_t i = 0; i < instances_.size(); ++i)
        {
            shs::SceneShape shape{};
            shape.shape = cull_shape_for_mesh(instances_[i].mesh_kind);
            shape.transform = shs::jolt::to_jph(instance_models_[i]);
            shape.stable_id = static_cast<uint32_t>(i);
            instance_cull_shapes_[i] = shape;
        }
    }

    void update_visibility_from_cell(const shs::CullingCell& cell)
    {
        if (instance_visible_mask_.size() != instances_.size())
        {
            instance_visible_mask_.assign(instances_.size(), 1u);
        }

        if (instance_cull_shapes_.size() != instances_.size())
        {
            rebuild_instance_cull_shapes();
        }

        const shs::CullResult instance_cull = shs::cull_vs_cell(std::span<const shs::SceneShape>{instance_cull_shapes_}, cell);
        frustum_visible_instance_indices_.clear();
        frustum_visible_instance_indices_.reserve(instances_.size());
        uint32_t visible_instances = 0;
        const size_t cull_count = std::min(instance_visible_mask_.size(), instance_cull.classes.size());
        for (size_t i = 0; i < cull_count; ++i)
        {
            const bool visible = shs::cull_class_is_visible(instance_cull.classes[i], true);
            instance_visible_mask_[i] = visible ? 1u : 0u;
            if (visible)
            {
                ++visible_instances;
                frustum_visible_instance_indices_.push_back(static_cast<uint32_t>(i));
            }
        }
        for (size_t i = cull_count; i < instance_visible_mask_.size(); ++i)
        {
            instance_visible_mask_[i] = 0u;
        }
        visible_instance_count_ = visible_instances;

        const shs::AABB floor_ws = shs::transform_aabb(floor_local_aabb_, floor_model_);
        const shs::CullClass floor_class = shs::classify_aabb_vs_cell(floor_ws, cell);
        floor_visible_ = shs::cull_class_is_visible(floor_class, true);
    }

    void apply_scene_software_occlusion()
    {
        if (!enable_scene_occlusion_)
        {
            return;
        }

        const size_t expected = static_cast<size_t>(kSceneOccW) * static_cast<size_t>(kSceneOccH);
        if (scene_occlusion_depth_.size() != expected)
        {
            scene_occlusion_depth_.assign(expected, 1.0f);
        }
        else
        {
            std::fill(scene_occlusion_depth_.begin(), scene_occlusion_depth_.end(), 1.0f);
        }

        std::vector<uint32_t> sorted = frustum_visible_instance_indices_;
        std::sort(
            sorted.begin(),
            sorted.end(),
            [&](uint32_t a, uint32_t b)
            {
                if (a >= instance_models_.size() || b >= instance_models_.size()) return a < b;
                const shs::AABB aa = shs::transform_aabb(local_aabb_for_mesh(instances_[a].mesh_kind), instance_models_[a]);
                const shs::AABB bb = shs::transform_aabb(local_aabb_for_mesh(instances_[b].mesh_kind), instance_models_[b]);
                const float da = shs::culling_sw::view_depth_of_aabb_center(aa, camera_ubo_.view);
                const float db = shs::culling_sw::view_depth_of_aabb_center(bb, camera_ubo_.view);
                return da < db;
            });

        uint32_t visible_instances = 0;
        for (const uint32_t idx : sorted)
        {
            if (idx >= instance_models_.size() || idx >= instance_visible_mask_.size()) continue;
            const shs::AABB world_box = shs::transform_aabb(local_aabb_for_mesh(instances_[idx].mesh_kind), instance_models_[idx]);
            const shs::culling_sw::ScreenRectDepth rect = shs::culling_sw::project_aabb_to_screen_rect(
                world_box,
                camera_ubo_.view_proj,
                kSceneOccW,
                kSceneOccH);
            const bool occluded = shs::culling_sw::is_rect_occluded(
                std::span<const float>(scene_occlusion_depth_.data(), scene_occlusion_depth_.size()),
                kSceneOccW,
                kSceneOccH,
                rect,
                1e-4f);

            if (occluded)
            {
                instance_visible_mask_[idx] = 0u;
                continue;
            }

            instance_visible_mask_[idx] = 1u;
            ++visible_instances;
            shs::culling_sw::rasterize_mesh_depth_transformed(
                std::span<float>(scene_occlusion_depth_.data(), scene_occlusion_depth_.size()),
                kSceneOccW,
                kSceneOccH,
                occluder_mesh_for_mesh(instances_[idx].mesh_kind),
                instance_models_[idx],
                camera_ubo_.view_proj);
        }
        visible_instance_count_ = visible_instances;
    }

    void build_light_occlusion_depth_from_scene()
    {
        if (!enable_light_occlusion_)
        {
            return;
        }

        const size_t expected = static_cast<size_t>(kLightOccW) * static_cast<size_t>(kLightOccH);
        if (light_occlusion_depth_.size() != expected)
        {
            light_occlusion_depth_.assign(expected, 1.0f);
        }
        else
        {
            std::fill(light_occlusion_depth_.begin(), light_occlusion_depth_.end(), 1.0f);
        }

        for (size_t idx = 0; idx < instance_visible_mask_.size() && idx < instance_models_.size(); ++idx)
        {
            if (instance_visible_mask_[idx] == 0u) continue;
            shs::culling_sw::rasterize_mesh_depth_transformed(
                std::span<float>(light_occlusion_depth_.data(), light_occlusion_depth_.size()),
                kLightOccW,
                kLightOccH,
                occluder_mesh_for_mesh(instances_[idx].mesh_kind),
                instance_models_[idx],
                camera_ubo_.view_proj);
        }

        if (floor_visible_)
        {
            shs::culling_sw::rasterize_mesh_depth_transformed(
                std::span<float>(light_occlusion_depth_.data(), light_occlusion_depth_.size()),
                kLightOccW,
                kLightOccH,
                floor_occluder_mesh_,
                floor_model_,
                camera_ubo_.view_proj);
        }
    }

    void refresh_visible_object_bounds_for_light_prefilter()
    {
        visible_object_aabbs_.clear();
        if (light_object_cull_mode_ == shs::LightObjectCullMode::None) return;
        visible_object_aabbs_.reserve(visible_instance_count_ + (floor_visible_ ? 1u : 0u));
        for (size_t i = 0; i < instance_visible_mask_.size() && i < instance_models_.size(); ++i)
        {
            if (instance_visible_mask_[i] == 0u) continue;
            visible_object_aabbs_.push_back(shs::transform_aabb(local_aabb_for_mesh(instances_[i].mesh_kind), instance_models_[i]));
        }
        if (floor_visible_)
        {
            visible_object_aabbs_.push_back(shs::transform_aabb(floor_local_aabb_, floor_model_));
        }
    }

    bool passes_light_object_prefilter(const shs::CullingLightGPU& packed) const
    {
        if (light_object_cull_mode_ == shs::LightObjectCullMode::None) return true;
        if (visible_object_aabbs_.empty()) return false;

        if (light_object_cull_mode_ == shs::LightObjectCullMode::SphereAabb)
        {
            shs::Sphere s{};
            s.center = glm::vec3(packed.cull_sphere);
            s.radius = std::max(packed.cull_sphere.w, 0.0f);
            for (const shs::AABB& obj : visible_object_aabbs_)
            {
                if (shs::intersect_sphere_aabb(s, obj)) return true;
            }
            return false;
        }

        shs::AABB light_box{};
        light_box.minv = glm::vec3(packed.cull_aabb_min);
        light_box.maxv = glm::vec3(packed.cull_aabb_max);
        for (const shs::AABB& obj : visible_object_aabbs_)
        {
            if (shs::intersect_aabb_aabb(light_box, obj)) return true;
        }
        return false;
    }

    void update_frame_data(float dt, float t, uint32_t w, uint32_t h, uint32_t frame_slot)
    {
        const float aspect = (h > 0) ? (static_cast<float>(w) / static_cast<float>(h)) : 1.0f;
        camera_.update(
            move_forward_,
            move_backward_,
            move_left_,
            move_right_,
            move_up_,
            move_down_,
            move_boost_,
            mouse_left_down_,
            mouse_right_down_,
            mouse_dx_accum_,
            mouse_dy_accum_,
            dt);
        mouse_dx_accum_ = 0.0f;
        mouse_dy_accum_ = 0.0f;

        const glm::vec3 cam_pos = camera_.pos;
        camera_ubo_.view = camera_.view_matrix();
        camera_ubo_.proj = shs::perspective_lh_no(glm::radians(62.0f), aspect, 0.1f, 260.0f);
        camera_ubo_.view_proj = camera_ubo_.proj * camera_ubo_.view;
        camera_ubo_.camera_pos_time = glm::vec4(cam_pos, t);
        camera_ubo_.sun_dir_intensity = glm::vec4(glm::normalize(glm::vec3(-0.35f, -1.0f, -0.18f)), 1.45f);
        camera_ubo_.screen_tile_lightcount = glm::uvec4(w, h, tile_w_, active_light_count_);
        camera_ubo_.params = glm::uvec4(tile_h_, kMaxLightsPerTile, kTileSize, static_cast<uint32_t>(culling_mode_));
        camera_ubo_.culling_params = glm::uvec4(
            kClusterZSlices,
            static_cast<uint32_t>(lighting_technique_),
            0u,
            0u);
        camera_ubo_.depth_params = glm::vec4(0.1f, 260.0f, 0.0f, 0.0f);
        camera_ubo_.exposure_gamma = glm::vec4(1.4f, 2.2f, 0.0f, 0.0f);
        // Keep directional shadow optional and subtle in this stress demo
        // so local-light behavior remains readable.
        const float dir_shadow_strength =
            (shadow_settings_.enable && enable_sun_shadow_)
                ? std::clamp(sun_shadow_strength_, 0.0f, 1.0f)
                : 0.0f;
        camera_ubo_.sun_shadow_params = glm::vec4(dir_shadow_strength, 0.0012f, 0.0030f, 2.0f);
        camera_ubo_.sun_shadow_filter = glm::vec4(
            shadow_settings_.quality.pcf_step,
            (shadow_settings_.enable && enable_sun_shadow_) ? 1.0f : 0.0f,
            0.0f,
            0.0f);

        for (size_t i = 0; i < instances_.size(); ++i)
        {
            const Instance& inst = instances_[i];
            const float bob = std::sin(t * 1.15f + inst.phase) * 0.24f;
            const glm::vec3 rot = inst.base_rot + inst.rot_speed * t;
            glm::mat4 m(1.0f);
            m = glm::translate(m, inst.base_pos + glm::vec3(0.0f, bob, 0.0f));
            m = glm::rotate(m, rot.x, glm::vec3(1.0f, 0.0f, 0.0f));
            m = glm::rotate(m, rot.y, glm::vec3(0.0f, 1.0f, 0.0f));
            m = glm::rotate(m, rot.z, glm::vec3(0.0f, 0.0f, 1.0f));
            m = glm::scale(m, glm::vec3(inst.scale));
            instance_models_[i] = m;
        }

        rebuild_instance_cull_shapes();
        const shs::CullingCell camera_cell = shs::extract_frustum_cell(
            camera_ubo_.view_proj,
            shs::CullingCellKind::CameraFrustumPerspective);
        update_visibility_from_cell(camera_cell);
        apply_scene_software_occlusion();
        build_light_occlusion_depth_from_scene();
        refresh_visible_object_bounds_for_light_prefilter();

        shs::AABB shadow_scene_aabb = shadow_scene_static_bounds_ready_
            ? shadow_scene_static_aabb_
            : shs::AABB{};
        if (!shadow_scene_static_bounds_ready_)
        {
            shadow_scene_aabb.expand(glm::vec3(-1.0f));
            shadow_scene_aabb.expand(glm::vec3(1.0f));
        }

        const glm::vec3 sun_dir = glm::normalize(glm::vec3(camera_ubo_.sun_dir_intensity));
        const shs::LightCamera sun_cam = shs::build_dir_light_camera_aabb(
            sun_dir,
            shadow_scene_aabb,
            14.0f,
            kSunShadowMapSize);
        sun_shadow_view_proj_ = sun_cam.viewproj;
        camera_ubo_.sun_shadow_view_proj = sun_shadow_view_proj_;

        if (shadow_lights_gpu_.size() != kMaxLights) shadow_lights_gpu_.assign(kMaxLights, ShadowLightGPU{});
        std::fill(shadow_lights_gpu_.begin(), shadow_lights_gpu_.end(), ShadowLightGPU{});
        local_shadow_casters_.clear();

        const auto build_local_shadow_vp = [&](const glm::vec3& pos_ws, const glm::vec3& dir_ws, float fov_rad, float range) -> glm::mat4 {
            const glm::vec3 dir = shs::normalize_or(dir_ws, glm::vec3(0.0f, -1.0f, 0.0f));
            glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
            if (std::abs(glm::dot(dir, up)) > 0.95f) up = glm::vec3(0.0f, 0.0f, 1.0f);
            const glm::mat4 v = glm::lookAtLH(pos_ws, pos_ws + dir, up);
            const glm::mat4 p = glm::perspectiveLH_NO(
                std::clamp(fov_rad, glm::radians(25.0f), glm::radians(150.0f)),
                1.0f,
                kShadowNearZ,
                std::max(range, kShadowNearZ + 0.2f));
            return p * v;
        };

        uint32_t used_spot_shadow = 0;
        uint32_t used_point_shadow = 0;
        uint32_t used_rect_shadow = 0;
        uint32_t used_tube_shadow = 0;

        const auto light_in_frustum = [&](const shs::Sphere& bounds) -> bool {
            shs::Sphere s = bounds;
            if (culling_mode_ == shs::LightCullingMode::TiledDepthRange)
            {
                // Keep tiled-depth conservative enough to avoid edge popping,
                // but still frustum-bound so light distribution matches other modes.
                s.radius = std::max(s.radius * 1.20f, s.radius + 0.75f);
            }
            else
            {
                // Slightly conservative light visibility to avoid edge flicker
                // when culling animated/orbiting lights against the camera frustum.
                s.radius = std::max(s.radius * 1.08f, s.radius + 0.25f);
            }
            shs::Sphere light_bounds = s; // Copy
            light_bounds.radius = std::max(light_bounds.radius, 0.0f); // Ensure valid
            const shs::CullClass light_class = shs::classify_sphere_vs_cell(light_bounds, camera_cell);
            return shs::cull_class_is_visible(light_class, true);
        };

        const auto light_in_occlusion = [&](const shs::Sphere& bounds) -> bool {
            if (!enable_light_occlusion_) return true;
            if (light_occlusion_depth_.empty()) return true;
            const shs::AABB light_box = shs::aabb_from_sphere(bounds);
            const shs::culling_sw::ScreenRectDepth rect = shs::culling_sw::project_aabb_to_screen_rect(
                light_box,
                camera_ubo_.view_proj,
                kLightOccW,
                kLightOccH);
            if (!rect.valid) return true;
            return !shs::culling_sw::is_rect_occluded(
                std::span<const float>(light_occlusion_depth_.data(), light_occlusion_depth_.size()),
                kLightOccW,
                kLightOccH,
                rect,
                1e-4f);
        };

        light_set_.clear_local_lights();
        const uint32_t lc = std::min<uint32_t>(active_light_count_, static_cast<uint32_t>(light_anim_.size()));
        uint32_t visible_light_count = 0;
        light_volume_debug_draws_.clear();
        light_volume_debug_draws_.reserve(lc);
        light_frustum_rejected_ = 0;
        light_occlusion_rejected_ = 0;
        light_prefilter_rejected_ = 0;
        for (uint32_t i = 0; i < lc; ++i)
        {
            const LightAnim& la = light_anim_[i];
            const float a = la.angle0 + la.speed * t;
            const float orbit_r = std::max(2.0f, la.orbit_radius * light_orbit_scale_);
            const float y = (la.height + light_height_bias_) + std::sin(a * 1.7f + la.phase) * 1.2f;
            const glm::vec3 p(std::cos(a) * orbit_r, y, std::sin(a) * orbit_r);
            float shape_range = la.range;
            switch (la.type)
            {
                case shs::LightType::RectArea:
                {
                    const float hx = std::max(0.10f, la.shape_params.x);
                    const float hy = std::max(0.10f, la.shape_params.y);
                    // Keep rect-area depth comparable to panel footprint.
                    shape_range = std::max(0.90f, std::max(hx, hy) * 2.25f);
                    break;
                }
                case shs::LightType::TubeArea:
                {
                    const float half_len = std::max(0.10f, la.shape_params.x);
                    const float radius = std::max(0.05f, la.shape_params.y);
                    // Capsule influence radius should stay tied to tube dimensions.
                    shape_range = std::max(0.90f, (half_len + radius) * 2.00f);
                    break;
                }
                case shs::LightType::Spot:
                {
                    // Keep cone depth in a practical range for scene readability.
                    shape_range = std::clamp(la.range, 2.20f, 7.50f);
                    break;
                }
                case shs::LightType::Point:
                default:
                {
                    shape_range = std::clamp(la.range, 1.20f, 6.80f);
                    break;
                }
            }
            const float tuned_range = std::max(0.60f, shape_range * light_range_scale_);
            const float tuned_intensity = std::max(0.0f, la.intensity * light_intensity_scale_);

            switch (la.type)
            {
                case shs::LightType::Spot:
                {
                    shs::SpotLight l{};
                    l.common.position_ws = p;
                    l.common.range = tuned_range;
                    l.common.color = la.color;
                    l.common.intensity = tuned_intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    l.direction_ws = la.direction_ws;
                    l.inner_angle_rad = la.spot_inner_outer.x;
                    l.outer_angle_rad = la.spot_inner_outer.y;
                    const shs::Sphere light_bounds = shs::spot_light_culling_sphere(l);
                    if (!light_in_frustum(light_bounds))
                    {
                        ++light_frustum_rejected_;
                        break;
                    }
                    if (!light_in_occlusion(light_bounds))
                    {
                        ++light_occlusion_rejected_;
                        break;
                    }
                    const shs::CullingLightGPU packed = shs::make_spot_culling_light(l);
                    if (!passes_light_object_prefilter(packed))
                    {
                        ++light_prefilter_rejected_;
                        break;
                    }
                    const uint32_t light_index = visible_light_count;
                    if (shadow_settings_.enable &&
                        shadow_settings_.spot &&
                        used_spot_shadow < std::min<uint32_t>(shadow_settings_.budget.max_spot, kMaxSpotShadowMaps))
                    {
                        const uint32_t layer = used_spot_shadow++;
                        l.common.flags |= shs::LightFlagAffectsShadows;
                        ShadowLightGPU sh{};
                        sh.light_view_proj = build_local_shadow_vp(l.common.position_ws, l.direction_ws, l.outer_angle_rad * 2.0f, l.common.range);
                        sh.position_range = glm::vec4(l.common.position_ws, l.common.range);
                        sh.shadow_params = glm::vec4(
                            0.72f,
                            camera_ubo_.sun_shadow_params.y,
                            camera_ubo_.sun_shadow_params.z,
                            camera_ubo_.sun_shadow_params.w);
                        sh.meta = glm::uvec4(
                            static_cast<uint32_t>(shs::ShadowTechnique::SpotMap2D),
                            layer,
                            0u,
                            1u);
                        shadow_lights_gpu_[light_index] = sh;

                        LocalShadowCaster caster{};
                        caster.light_index = light_index;
                        caster.technique = shs::ShadowTechnique::SpotMap2D;
                        caster.layer_base = layer;
                        caster.position_ws = l.common.position_ws;
                        caster.direction_ws = l.direction_ws;
                        caster.range = l.common.range;
                        caster.outer_angle_rad = l.outer_angle_rad;
                        caster.strength = sh.shadow_params.x;
                        local_shadow_casters_.push_back(caster);
                    }
                    light_set_.spots.push_back(l);
                    gpu_lights_[light_index] = shs::make_spot_culling_light(l);
                    {
                        LightVolumeDebugDraw d{};
                        d.mesh = DebugVolumeMeshKind::Cone;
                        d.model = make_spot_volume_debug_model(
                            l.common.position_ws,
                            l.direction_ws,
                            l.common.range,
                            l.outer_angle_rad);
                        const glm::vec3 c = glm::clamp(l.common.color * 1.08f, glm::vec3(0.05f), glm::vec3(1.0f));
                        d.color = glm::vec4(c, 1.0f);
                        light_volume_debug_draws_.push_back(d);
                    }
                    visible_light_count++;
                    break;
                }
                case shs::LightType::RectArea:
                {
                    shs::RectAreaLight l{};
                    l.common.position_ws = p;
                    l.common.range = tuned_range;
                    l.common.color = la.color;
                    l.common.intensity = tuned_intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    l.direction_ws = la.direction_ws;
                    l.right_ws = la.rect_right_ws;
                    l.half_extents = glm::vec2(la.shape_params.x, la.shape_params.y);
                    const shs::Sphere light_bounds = shs::rect_area_light_culling_sphere(l);
                    if (!light_in_frustum(light_bounds))
                    {
                        ++light_frustum_rejected_;
                        break;
                    }
                    if (!light_in_occlusion(light_bounds))
                    {
                        ++light_occlusion_rejected_;
                        break;
                    }
                    const shs::CullingLightGPU packed = shs::make_rect_area_culling_light(l);
                    if (!passes_light_object_prefilter(packed))
                    {
                        ++light_prefilter_rejected_;
                        break;
                    }
                    const uint32_t light_index = visible_light_count;
                    if (shadow_settings_.enable &&
                        shadow_settings_.rect_area_proxy &&
                        used_spot_shadow < kMaxSpotShadowMaps &&
                        used_rect_shadow < shadow_settings_.budget.max_rect_area)
                    {
                        ++used_rect_shadow;
                        const uint32_t layer = used_spot_shadow++;
                        l.common.flags |= shs::LightFlagAffectsShadows;
                        const float proxy_fov = glm::radians(76.0f);
                        ShadowLightGPU sh{};
                        sh.light_view_proj = build_local_shadow_vp(l.common.position_ws, l.direction_ws, proxy_fov, l.common.range);
                        sh.position_range = glm::vec4(l.common.position_ws, l.common.range);
                        sh.shadow_params = glm::vec4(
                            0.62f,
                            camera_ubo_.sun_shadow_params.y,
                            camera_ubo_.sun_shadow_params.z,
                            1.0f);
                        sh.meta = glm::uvec4(
                            static_cast<uint32_t>(shs::ShadowTechnique::AreaProxySpotMap2D),
                            layer,
                            0u,
                            1u);
                        shadow_lights_gpu_[light_index] = sh;

                        LocalShadowCaster caster{};
                        caster.light_index = light_index;
                        caster.technique = shs::ShadowTechnique::AreaProxySpotMap2D;
                        caster.layer_base = layer;
                        caster.position_ws = l.common.position_ws;
                        caster.direction_ws = l.direction_ws;
                        caster.range = l.common.range;
                        caster.outer_angle_rad = proxy_fov * 0.5f;
                        caster.strength = sh.shadow_params.x;
                        local_shadow_casters_.push_back(caster);
                    }
                    light_set_.rect_areas.push_back(l);
                    gpu_lights_[light_index] = shs::make_rect_area_culling_light(l);
                    {
                        LightVolumeDebugDraw d{};
                        d.mesh = DebugVolumeMeshKind::Box;
                        d.model = make_rect_volume_debug_model(
                            l.common.position_ws,
                            l.direction_ws,
                            l.right_ws,
                            l.half_extents.x,
                            l.half_extents.y,
                            l.common.range);
                        const glm::vec3 c = glm::clamp(l.common.color * 1.06f, glm::vec3(0.05f), glm::vec3(1.0f));
                        d.color = glm::vec4(c, 1.0f);
                        light_volume_debug_draws_.push_back(d);
                    }
                    visible_light_count++;
                    break;
                }
                case shs::LightType::TubeArea:
                {
                    shs::TubeAreaLight l{};
                    l.common.position_ws = p;
                    l.common.range = tuned_range;
                    l.common.color = la.color;
                    l.common.intensity = tuned_intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    l.axis_ws = la.direction_ws;
                    l.half_length = la.shape_params.x;
                    l.radius = la.shape_params.y;
                    const shs::Sphere light_bounds = shs::tube_area_light_culling_sphere(l);
                    if (!light_in_frustum(light_bounds))
                    {
                        ++light_frustum_rejected_;
                        break;
                    }
                    if (!light_in_occlusion(light_bounds))
                    {
                        ++light_occlusion_rejected_;
                        break;
                    }
                    const shs::CullingLightGPU packed = shs::make_tube_area_culling_light(l);
                    if (!passes_light_object_prefilter(packed))
                    {
                        ++light_prefilter_rejected_;
                        break;
                    }
                    const uint32_t light_index = visible_light_count;
                    if (shadow_settings_.enable &&
                        shadow_settings_.tube_area_proxy &&
                        used_spot_shadow < kMaxSpotShadowMaps &&
                        used_tube_shadow < shadow_settings_.budget.max_tube_area)
                    {
                        ++used_tube_shadow;
                        const uint32_t layer = used_spot_shadow++;
                        l.common.flags |= shs::LightFlagAffectsShadows;
                        const glm::vec3 dir = shs::normalize_or(l.axis_ws, glm::vec3(1.0f, 0.0f, 0.0f));
                        const float proxy_fov = glm::radians(70.0f);
                        ShadowLightGPU sh{};
                        sh.light_view_proj = build_local_shadow_vp(l.common.position_ws, dir, proxy_fov, l.common.range);
                        sh.position_range = glm::vec4(l.common.position_ws, l.common.range);
                        sh.shadow_params = glm::vec4(
                            0.58f,
                            camera_ubo_.sun_shadow_params.y,
                            camera_ubo_.sun_shadow_params.z,
                            1.0f);
                        sh.meta = glm::uvec4(
                            static_cast<uint32_t>(shs::ShadowTechnique::AreaProxySpotMap2D),
                            layer,
                            0u,
                            1u);
                        shadow_lights_gpu_[light_index] = sh;

                        LocalShadowCaster caster{};
                        caster.light_index = light_index;
                        caster.technique = shs::ShadowTechnique::AreaProxySpotMap2D;
                        caster.layer_base = layer;
                        caster.position_ws = l.common.position_ws;
                        caster.direction_ws = dir;
                        caster.range = l.common.range;
                        caster.outer_angle_rad = proxy_fov * 0.5f;
                        caster.strength = sh.shadow_params.x;
                        local_shadow_casters_.push_back(caster);
                    }
                    light_set_.tube_areas.push_back(l);
                    gpu_lights_[light_index] = shs::make_tube_area_culling_light(l);
                    {
                        LightVolumeDebugDraw d{};
                        d.mesh = DebugVolumeMeshKind::Box;
                        d.model = make_tube_volume_debug_model(
                            l.common.position_ws,
                            l.axis_ws,
                            l.half_length,
                            l.common.range);
                        const glm::vec3 c = glm::clamp(l.common.color * 1.05f, glm::vec3(0.05f), glm::vec3(1.0f));
                        d.color = glm::vec4(c, 1.0f);
                        light_volume_debug_draws_.push_back(d);
                    }
                    visible_light_count++;
                    break;
                }
                case shs::LightType::Point:
                default:
                {
                    shs::PointLight l{};
                    l.common.position_ws = p;
                    l.common.range = tuned_range;
                    l.common.color = la.color;
                    l.common.intensity = tuned_intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    const shs::Sphere light_bounds = shs::point_light_culling_sphere(l);
                    if (!light_in_frustum(light_bounds))
                    {
                        ++light_frustum_rejected_;
                        break;
                    }
                    if (!light_in_occlusion(light_bounds))
                    {
                        ++light_occlusion_rejected_;
                        break;
                    }
                    const shs::CullingLightGPU packed = shs::make_point_culling_light(l);
                    if (!passes_light_object_prefilter(packed))
                    {
                        ++light_prefilter_rejected_;
                        break;
                    }
                    const uint32_t light_index = visible_light_count;
                    if (shadow_settings_.enable &&
                        shadow_settings_.point &&
                        used_point_shadow < std::min<uint32_t>(shadow_settings_.budget.max_point, kMaxPointShadowLights))
                    {
                        const uint32_t layer_base = kMaxSpotShadowMaps + (used_point_shadow * kPointShadowFaceCount);
                        ++used_point_shadow;
                        l.common.flags |= shs::LightFlagAffectsShadows;
                        ShadowLightGPU sh{};
                        sh.position_range = glm::vec4(l.common.position_ws, l.common.range);
                        sh.shadow_params = glm::vec4(
                            0.68f,
                            camera_ubo_.sun_shadow_params.y,
                            camera_ubo_.sun_shadow_params.z,
                            camera_ubo_.sun_shadow_params.w);
                        sh.meta = glm::uvec4(
                            static_cast<uint32_t>(shs::ShadowTechnique::PointCube),
                            layer_base,
                            0u,
                            1u);
                        shadow_lights_gpu_[light_index] = sh;

                        LocalShadowCaster caster{};
                        caster.light_index = light_index;
                        caster.technique = shs::ShadowTechnique::PointCube;
                        caster.layer_base = layer_base;
                        caster.position_ws = l.common.position_ws;
                        caster.range = l.common.range;
                        caster.strength = sh.shadow_params.x;
                        local_shadow_casters_.push_back(caster);
                    }
                    light_set_.points.push_back(l);
                    gpu_lights_[light_index] = shs::make_point_culling_light(l);
                    {
                        LightVolumeDebugDraw d{};
                        d.mesh = DebugVolumeMeshKind::Sphere;
                        d.model = make_point_volume_debug_model(l.common.position_ws, l.common.range);
                        const glm::vec3 c = glm::clamp(l.common.color * 1.04f, glm::vec3(0.05f), glm::vec3(1.0f));
                        d.color = glm::vec4(c, 1.0f);
                        light_volume_debug_draws_.push_back(d);
                    }
                    visible_light_count++;
                    break;
                }
            }
        }
        visible_light_count_ = visible_light_count;
        camera_ubo_.screen_tile_lightcount.w = visible_light_count_;
        if (!frame_resources_.valid_slot(frame_slot))
        {
            throw std::runtime_error("Invalid frame slot for dynamic uploads");
        }
        FrameResources& fr = frame_resources_.at_slot(frame_slot);
        std::memcpy(fr.camera_buffer.mapped, &camera_ubo_, sizeof(CameraUBO));

        if (visible_light_count_ > 0u)
        {
            std::memcpy(fr.light_buffer.mapped, gpu_lights_.data(), static_cast<size_t>(visible_light_count_) * sizeof(shs::CullingLightGPU));
        }
        std::memcpy(fr.shadow_light_buffer.mapped, shadow_lights_gpu_.data(), static_cast<size_t>(kMaxLights) * sizeof(ShadowLightGPU));

        point_count_active_ = static_cast<uint32_t>(light_set_.points.size());
        spot_count_active_ = static_cast<uint32_t>(light_set_.spots.size());
        rect_count_active_ = static_cast<uint32_t>(light_set_.rect_areas.size());
        tube_count_active_ = static_cast<uint32_t>(light_set_.tube_areas.size());
        spot_shadow_count_ = used_spot_shadow;
        point_shadow_count_ = used_point_shadow;
    }

    void begin_render_pass_depth(VkCommandBuffer cmd)
    {
        VkClearValue clear{};
        clear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        bi.renderPass = depth_target_.render_pass;
        bi.framebuffer = depth_target_.framebuffer;
        bi.renderArea.offset = {0, 0};
        bi.renderArea.extent = {depth_target_.w, depth_target_.h};
        bi.clearValueCount = 1;
        bi.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    }

    void begin_render_pass_scene(VkCommandBuffer cmd, const shs::VulkanRenderBackend::FrameInfo& fi)
    {
        VkClearValue clear[2]{};
        clear[0].color = {{0.03f, 0.035f, 0.045f, 1.0f}};
        clear[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        bi.renderPass = fi.render_pass;
        bi.framebuffer = fi.framebuffer;
        bi.renderArea.offset = {0, 0};
        bi.renderArea.extent = fi.extent;
        bi.clearValueCount = vk_->has_depth_attachment() ? 2u : 1u;
        bi.pClearValues = clear;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    }

    void set_viewport_scissor(VkCommandBuffer cmd, uint32_t w, uint32_t h, bool flip_y)
    {
        shs::vk_cmd_set_viewport_scissor(cmd, w, h, flip_y);
    }

    void begin_render_pass_shadow(
        VkCommandBuffer cmd,
        const LayeredDepthTarget& target,
        uint32_t layer)
    {
        VkClearValue clear{};
        clear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        bi.renderPass = target.render_pass;
        bi.framebuffer = target.framebuffers[layer];
        bi.renderArea.offset = {0, 0};
        bi.renderArea.extent = {target.w, target.h};
        bi.clearValueCount = 1;
        bi.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
    }

    glm::mat4 make_point_shadow_face_view_proj(const glm::vec3& light_pos, float range, uint32_t face) const
    {
        static const glm::vec3 dirs[6] = {
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec3(-1.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 0.0f, -1.0f),
        };
        static const glm::vec3 ups[6] = {
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 0.0f, -1.0f),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec3(0.0f, -1.0f, 0.0f),
        };
        const uint32_t f = std::min<uint32_t>(face, 5u);
        const glm::mat4 v = glm::lookAtLH(light_pos, light_pos + dirs[f], ups[f]);
        const glm::mat4 p = glm::perspectiveLH_NO(glm::radians(90.0f), 1.0f, kShadowNearZ, std::max(range, kShadowNearZ + 0.2f));
        return p * v;
    }

    glm::mat4 make_local_shadow_view_proj(const LocalShadowCaster& caster) const
    {
        if (caster.technique == shs::ShadowTechnique::PointCube)
        {
            return glm::mat4(1.0f);
        }
        const glm::vec3 dir = shs::normalize_or(caster.direction_ws, glm::vec3(0.0f, -1.0f, 0.0f));
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        if (std::abs(glm::dot(dir, up)) > 0.95f) up = glm::vec3(0.0f, 0.0f, 1.0f);
        const glm::mat4 v = glm::lookAtLH(caster.position_ws, caster.position_ws + dir, up);
        const glm::mat4 p = glm::perspectiveLH_NO(
            std::clamp(caster.outer_angle_rad * 2.0f, glm::radians(25.0f), glm::radians(150.0f)),
            1.0f,
            kShadowNearZ,
            std::max(caster.range, kShadowNearZ + 0.2f));
        return p * v;
    }

    void draw_shadow_scene(VkCommandBuffer cmd, const glm::mat4& light_view_proj, shs::CullingCellKind cell_kind)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);
        const shs::CullingCell shadow_cell = shs::extract_frustum_cell(light_view_proj, cell_kind);
        if (instance_cull_shapes_.size() != instances_.size())
        {
            rebuild_instance_cull_shapes();
        }
        
        const VkDeviceSize vb_off = 0;
        const shs::AABB floor_ws = shs::transform_aabb(floor_local_aabb_, floor_model_);
        const bool floor_in_shadow_cell = shs::cull_class_is_visible(
            shs::classify_aabb_vs_cell(floor_ws, shadow_cell),
            true);

        if (floor_in_shadow_cell && !floor_indices_.empty() && floor_vertex_buffer_.buffer != VK_NULL_HANDLE)
        {
            vkCmdBindVertexBuffers(cmd, 0, 1, &floor_vertex_buffer_.buffer, &vb_off);
            vkCmdBindIndexBuffer(cmd, floor_index_buffer_.buffer, 0, VK_INDEX_TYPE_UINT32);
            ShadowPush pc{};
            pc.light_view_proj = light_view_proj;
            pc.model = floor_model_;
            vkCmdPushConstants(cmd, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &pc);
            vkCmdDrawIndexed(cmd, static_cast<uint32_t>(floor_indices_.size()), 1, 0, 0, 0);
        }

        const shs::CullResult shadow_cull = shs::cull_vs_cell(std::span<const shs::SceneShape>{instance_cull_shapes_}, shadow_cell);
        for (size_t idx : shadow_cull.visible_indices)
        {
            if (idx >= instance_models_.size()) continue;
            const uint32_t i = static_cast<uint32_t>(idx);
            const Instance::MeshKind mesh_kind = instances_[i].mesh_kind;
            const GpuBuffer& vb = vertex_buffer_for_mesh(mesh_kind);
            const GpuBuffer& ib = index_buffer_for_mesh(mesh_kind);
            const uint32_t index_count = index_count_for_mesh(mesh_kind);
            if (vb.buffer == VK_NULL_HANDLE || ib.buffer == VK_NULL_HANDLE || index_count == 0u) continue;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb.buffer, &vb_off);
            vkCmdBindIndexBuffer(cmd, ib.buffer, 0, VK_INDEX_TYPE_UINT32);
            ShadowPush pc{};
            pc.light_view_proj = light_view_proj;
            pc.model = instance_models_[i];
            vkCmdPushConstants(cmd, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &pc);
            vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
        }
    }

    void record_shadow_passes(VkCommandBuffer cmd)
    {
        if (!shadow_settings_.enable) return;
        if (shadow_pipeline_ == VK_NULL_HANDLE || shadow_pipeline_layout_ == VK_NULL_HANDLE) return;
        if (sun_shadow_target_.render_pass == VK_NULL_HANDLE || sun_shadow_target_.framebuffers.empty()) return;
        if (local_shadow_target_.render_pass == VK_NULL_HANDLE || local_shadow_target_.framebuffers.empty()) return;

        begin_render_pass_shadow(cmd, sun_shadow_target_, 0u);
        set_viewport_scissor(cmd, sun_shadow_target_.w, sun_shadow_target_.h, true);
        draw_shadow_scene(cmd, sun_shadow_view_proj_, shs::CullingCellKind::CascadeFrustum);
        vkCmdEndRenderPass(cmd);

        for (const LocalShadowCaster& caster : local_shadow_casters_)
        {
            if (caster.technique == shs::ShadowTechnique::PointCube)
            {
                for (uint32_t face = 0; face < kPointShadowFaceCount; ++face)
                {
                    const uint32_t layer = caster.layer_base + face;
                    if (layer >= local_shadow_target_.framebuffers.size()) continue;
                    const glm::mat4 vp = make_point_shadow_face_view_proj(caster.position_ws, caster.range, face);
                    begin_render_pass_shadow(cmd, local_shadow_target_, layer);
                    set_viewport_scissor(cmd, local_shadow_target_.w, local_shadow_target_.h, true);
                    draw_shadow_scene(cmd, vp, shs::CullingCellKind::PointShadowFaceFrustum);
                    vkCmdEndRenderPass(cmd);
                }
            }
            else
            {
                if (caster.layer_base >= local_shadow_target_.framebuffers.size()) continue;
                const glm::mat4 vp = make_local_shadow_view_proj(caster);
                begin_render_pass_shadow(cmd, local_shadow_target_, caster.layer_base);
                set_viewport_scissor(cmd, local_shadow_target_.w, local_shadow_target_.h, true);
                draw_shadow_scene(cmd, vp, shs::CullingCellKind::SpotShadowFrustum);
                vkCmdEndRenderPass(cmd);
            }
        }
    }

    glm::mat4 make_point_volume_debug_model(const glm::vec3& pos_ws, float range) const
    {
        const float r = std::max(range, 0.10f);
        // Source sphere mesh radius is 0.5, so multiply by 2*r for target radius r.
        return glm::translate(glm::mat4(1.0f), pos_ws) * glm::scale(glm::mat4(1.0f), glm::vec3(r * 2.0f));
    }

    glm::mat4 make_spot_volume_debug_model(
        const glm::vec3& pos_ws,
        const glm::vec3& dir_ws,
        float range,
        float outer_angle_rad) const
    {
        const glm::vec3 dir = shs::normalize_or(dir_ws, glm::vec3(0.0f, -1.0f, 0.0f));
        const float h = std::max(range, 0.25f);
        const float base_radius =
            std::tan(std::max(outer_angle_rad, glm::radians(3.0f))) * h;

        glm::vec3 bx{};
        glm::vec3 by{};
        glm::vec3 bz{};
        // Cone mesh tip is at +Y, so align +Y to -dir and offset center so tip sits at light position.
        basis_from_axis(-dir, bx, by, bz);
        const glm::vec3 center = pos_ws + dir * (h * 0.5f);
        return model_from_basis_and_scale(center, bx, by, bz, glm::vec3(base_radius, h, base_radius));
    }

    glm::mat4 make_rect_volume_debug_model(
        const glm::vec3& pos_ws,
        const glm::vec3& dir_ws,
        const glm::vec3& right_ws,
        float half_x,
        float half_y,
        float range) const
    {
        glm::vec3 fwd = shs::normalize_or(dir_ws, glm::vec3(0.0f, -1.0f, 0.0f));
        glm::vec3 right = right_ws - fwd * glm::dot(right_ws, fwd);
        right = shs::normalize_or(right, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 up = shs::normalize_or(glm::cross(fwd, right), glm::vec3(0.0f, 1.0f, 0.0f));
        right = shs::normalize_or(glm::cross(up, fwd), right);

        // Shader influence is a forward rounded-prism bound:
        // x/y expand by +range beyond panel half extents, z spans [0, range].
        // Source box mesh is centered and unit-sized, so scale by 2x half-extents.
        const float ex = std::max((half_x + range) * 2.0f, 0.10f);
        const float ey = std::max((half_y + range) * 2.0f, 0.10f);
        const float ez = std::max(range, 0.10f);
        const glm::vec3 center = pos_ws + fwd * (range * 0.5f);
        return model_from_basis_and_scale(center, right, up, fwd, glm::vec3(ex, ey, ez));
    }

    glm::mat4 make_tube_volume_debug_model(
        const glm::vec3& pos_ws,
        const glm::vec3& axis_ws,
        float half_length,
        float range) const
    {
        glm::vec3 axis = shs::normalize_or(axis_ws, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 up_hint = safe_perp_axis(axis);
        glm::vec3 up = shs::normalize_or(glm::cross(axis, up_hint), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::vec3 side = shs::normalize_or(glm::cross(up, axis), glm::vec3(0.0f, 0.0f, 1.0f));

        // Shader influence is a capsule around segment [ -half_length, +half_length ]
        // with capsule radius == range.
        const float ex = std::max((half_length + range) * 2.0f, 0.10f);
        const float ey = std::max(range * 2.0f, 0.10f);
        const float ez = std::max(range * 2.0f, 0.10f);
        return model_from_basis_and_scale(pos_ws, axis, up, side, glm::vec3(ex, ey, ez));
    }

    void draw_light_volumes_debug(VkCommandBuffer cmd, VkPipelineLayout layout, uint32_t frame_slot)
    {
        if (!show_light_volumes_debug_) return;
        if (light_volume_debug_draws_.empty()) return;
        if (!frame_resources_.valid_slot(frame_slot)) return;
        if (scene_wire_pipeline_ == VK_NULL_HANDLE) return;

        const VkDescriptorSet global_set = frame_resources_.at_slot(frame_slot).global_set;
        if (global_set == VK_NULL_HANDLE) return;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scene_wire_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set, 0, nullptr);

        const uint32_t draw_count = std::min<uint32_t>(static_cast<uint32_t>(light_volume_debug_draws_.size()), 512u);
        for (uint32_t i = 0; i < draw_count; ++i)
        {
            const LightVolumeDebugDraw& d = light_volume_debug_draws_[i];
            const GpuBuffer* vb = nullptr;
            const GpuBuffer* ib = nullptr;
            uint32_t index_count = 0u;

            switch (d.mesh)
            {
                case DebugVolumeMeshKind::Sphere:
                    vb = &vertex_buffer_;
                    ib = &sphere_line_index_buffer_;
                    index_count = static_cast<uint32_t>(sphere_line_indices_.size());
                    break;
                case DebugVolumeMeshKind::Cone:
                    vb = &cone_vertex_buffer_;
                    ib = &cone_line_index_buffer_;
                    index_count = static_cast<uint32_t>(cone_line_indices_.size());
                    break;
                case DebugVolumeMeshKind::Box:
                    vb = &box_vertex_buffer_;
                    ib = &box_line_index_buffer_;
                    index_count = static_cast<uint32_t>(box_line_indices_.size());
                    break;
            }

            if (!vb || !ib) continue;
            if (vb->buffer == VK_NULL_HANDLE || ib->buffer == VK_NULL_HANDLE || index_count == 0u) continue;

            const VkDeviceSize vb_off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb->buffer, &vb_off);
            vkCmdBindIndexBuffer(cmd, ib->buffer, 0, VK_INDEX_TYPE_UINT32);

            DrawPush pc{};
            pc.model = d.model;
            pc.base_color = d.color;
            // Unlit, colored wireframe overlay.
            pc.material_params = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
            vkCmdPushConstants(
                cmd,
                layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(DrawPush),
                &pc);
            vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
        }
    }

    void draw_floor(VkCommandBuffer cmd, VkPipelineLayout layout)
    {
        if (!floor_visible_) return;

        const VkDeviceSize vb_off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &floor_vertex_buffer_.buffer, &vb_off);
        vkCmdBindIndexBuffer(cmd, floor_index_buffer_.buffer, 0, VK_INDEX_TYPE_UINT32);

        DrawPush floor_pc{};
        floor_pc.model = floor_model_;
        floor_pc.base_color = floor_material_color_;
        floor_pc.material_params = floor_material_params_;
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DrawPush), &floor_pc);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(floor_indices_.size()), 1, 0, 0, 0);
    }

    void draw_sphere_range(VkCommandBuffer cmd, VkPipelineLayout layout, uint32_t start, uint32_t end)
    {
        const VkDeviceSize vb_off = 0;
        for (uint32_t i = start; i < end; ++i)
        {
            if (i >= instance_visible_mask_.size() || instance_visible_mask_[i] == 0u) continue;
            const Instance::MeshKind mesh_kind = instances_[i].mesh_kind;
            const GpuBuffer& vb = vertex_buffer_for_mesh(mesh_kind);
            const GpuBuffer& ib = index_buffer_for_mesh(mesh_kind);
            const uint32_t index_count = index_count_for_mesh(mesh_kind);
            if (vb.buffer == VK_NULL_HANDLE || ib.buffer == VK_NULL_HANDLE || index_count == 0u) continue;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb.buffer, &vb_off);
            vkCmdBindIndexBuffer(cmd, ib.buffer, 0, VK_INDEX_TYPE_UINT32);

            DrawPush pc{};
            pc.model = instance_models_[i];
            pc.base_color = instances_[i].base_color;
            pc.material_params = glm::vec4(
                instances_[i].metallic,
                instances_[i].roughness,
                instances_[i].ao,
                0.0f);
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DrawPush), &pc);
            vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
        }
    }

    bool record_secondary_batch(
        VkRenderPass rp,
        VkFramebuffer fb,
        VkPipeline pipeline,
        VkPipelineLayout layout,
        uint32_t w,
        uint32_t h,
        bool flip_y,
        uint32_t frame_slot,
        uint32_t worker_idx,
        uint32_t start,
        uint32_t end,
        bool draw_floor_here,
        VkCommandBuffer& out)
    {
        out = VK_NULL_HANDLE;
        if (start >= end && !draw_floor_here) return true;
        if (!frame_resources_.valid_slot(frame_slot)) return false;
        if (worker_idx >= worker_pools_.size()) return false;
        const VkDescriptorSet global_set = frame_resources_.at_slot(frame_slot).global_set;
        if (global_set == VK_NULL_HANDLE) return false;
        const VkCommandPool pool = worker_pools_[worker_idx].pools[frame_slot];
        if (pool == VK_NULL_HANDLE) return false;

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(vk_->device(), &ai, &out) != VK_SUCCESS)
        {
            return false;
        }

        VkCommandBufferInheritanceInfo inh{};
        inh.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        inh.renderPass = rp;
        inh.subpass = 0;
        inh.framebuffer = fb;

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        bi.pInheritanceInfo = &inh;
        if (vkBeginCommandBuffer(out, &bi) != VK_SUCCESS) return false;

        set_viewport_scissor(out, w, h, flip_y);
        vkCmdBindPipeline(out, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(out, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set, 0, nullptr);
        if (draw_floor_here) draw_floor(out, layout);
        if (start < end) draw_sphere_range(out, layout, start, end);

        return vkEndCommandBuffer(out) == VK_SUCCESS;
    }

    bool record_secondary_lists(
        VkRenderPass rp,
        VkFramebuffer fb,
        VkPipeline pipeline,
        VkPipelineLayout layout,
        uint32_t w,
        uint32_t h,
        bool flip_y,
        bool include_floor,
        uint32_t frame_slot,
        std::vector<VkCommandBuffer>& out)
    {
        out.clear();

        if (!use_multithread_recording_ || !jobs_ || worker_pools_.empty() || instances_.empty())
        {
            return true;
        }

        const uint32_t workers = std::min<uint32_t>(static_cast<uint32_t>(worker_pools_.size()), static_cast<uint32_t>(instances_.size()));
        if (workers <= 1) return true;
        if (frame_slot >= kWorkerPoolRingSize) return false;

        std::vector<VkCommandBuffer> tmp(workers, VK_NULL_HANDLE);
        std::atomic<bool> ok{true};
        shs::WaitGroup wg{};

        const uint32_t n = static_cast<uint32_t>(instances_.size());
        const uint32_t batch = (n + workers - 1) / workers;

        for (uint32_t wi = 0; wi < workers; ++wi)
        {
            const uint32_t start = wi * batch;
            const uint32_t end = std::min(n, start + batch);
            if (start >= end) continue;

            wg.add(1);
            jobs_->enqueue([&, wi, start, end]() {
                const bool draw_floor_here = include_floor && (wi == 0);
                if (!record_secondary_batch(rp, fb, pipeline, layout, w, h, flip_y, frame_slot, wi, start, end, draw_floor_here, tmp[wi]))
                {
                    ok.store(false, std::memory_order_release);
                }
                wg.done();
            });
        }

        wg.wait();
        if (!ok.load(std::memory_order_acquire)) return false;

        for (VkCommandBuffer cb : tmp)
        {
            if (cb != VK_NULL_HANDLE) out.push_back(cb);
        }
        return true;
    }

    bool reset_worker_pools_for_frame(uint32_t frame_slot)
    {
        if (!frame_resources_.valid_slot(frame_slot)) return false;
        if (!use_multithread_recording_ || !jobs_ || worker_pools_.empty() || instances_.empty()) return true;

        const uint32_t workers = std::min<uint32_t>(static_cast<uint32_t>(worker_pools_.size()), static_cast<uint32_t>(instances_.size()));
        if (workers <= 1) return true;

        for (uint32_t i = 0; i < workers; ++i)
        {
            const VkCommandPool pool = worker_pools_[i].pools[frame_slot];
            if (pool == VK_NULL_HANDLE) return false;
            vkResetCommandPool(vk_->device(), pool, 0);
        }
        return true;
    }

    void record_inline_scene(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout, uint32_t w, uint32_t h, uint32_t frame_slot)
    {
        if (!frame_resources_.valid_slot(frame_slot)) throw std::runtime_error("Invalid frame slot for scene recording");
        const VkDescriptorSet global_set = frame_resources_.at_slot(frame_slot).global_set;
        if (global_set == VK_NULL_HANDLE) throw std::runtime_error("Scene descriptor set unavailable");
        set_viewport_scissor(cmd, w, h, true);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set, 0, nullptr);
        draw_floor(cmd, layout);
        draw_sphere_range(cmd, layout, 0, static_cast<uint32_t>(instances_.size()));
    }

    void record_inline_depth(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout, uint32_t w, uint32_t h, uint32_t frame_slot)
    {
        if (!frame_resources_.valid_slot(frame_slot)) throw std::runtime_error("Invalid frame slot for depth recording");
        const VkDescriptorSet global_set = frame_resources_.at_slot(frame_slot).global_set;
        if (global_set == VK_NULL_HANDLE) throw std::runtime_error("Depth descriptor set unavailable");
        set_viewport_scissor(cmd, w, h, true);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set, 0, nullptr);
        draw_floor(cmd, layout);
        draw_sphere_range(cmd, layout, 0, static_cast<uint32_t>(instances_.size()));
    }

    VkPipelineStageFlags2 stage_flags_to_stage2(VkPipelineStageFlags stages) const
    {
        (void)stages;
        VkPipelineStageFlags2 out = 0;
#if defined(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)
        if ((stages & VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) != 0) out |= VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
#endif
#if defined(VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT)
        if ((stages & VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT) != 0) out |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
#endif
#if defined(VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
        if ((stages & VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT) != 0) out |= VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
#endif
#if defined(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)
        if ((stages & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT) != 0) out |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
#endif
#if defined(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
        if ((stages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) != 0) out |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
#endif
#if defined(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
        if (out == 0) out = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
#endif
        return out;
    }

    VkAccessFlags2 access_flags_to_access2(VkAccessFlags access) const
    {
        (void)access;
        VkAccessFlags2 out = 0;
#if defined(VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
        if ((access & VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) != 0) out |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
#endif
#if defined(VK_ACCESS_2_SHADER_READ_BIT)
        if ((access & VK_ACCESS_SHADER_READ_BIT) != 0) out |= VK_ACCESS_2_SHADER_READ_BIT;
#endif
#if defined(VK_ACCESS_2_SHADER_WRITE_BIT)
        if ((access & VK_ACCESS_SHADER_WRITE_BIT) != 0) out |= VK_ACCESS_2_SHADER_WRITE_BIT;
#endif
        return out;
    }

    void cmd_memory_barrier(
        VkCommandBuffer cmd,
        VkPipelineStageFlags src_stage,
        VkAccessFlags src_access,
        VkPipelineStageFlags dst_stage,
        VkAccessFlags dst_access)
    {
        if (cmd == VK_NULL_HANDLE) return;

#if defined(VK_STRUCTURE_TYPE_DEPENDENCY_INFO) && defined(VK_STRUCTURE_TYPE_MEMORY_BARRIER_2)
        if (vk_ && vk_->supports_synchronization2())
        {
            VkMemoryBarrier2 b2{};
            b2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            b2.srcStageMask = stage_flags_to_stage2(src_stage);
            b2.srcAccessMask = access_flags_to_access2(src_access);
            b2.dstStageMask = stage_flags_to_stage2(dst_stage);
            b2.dstAccessMask = access_flags_to_access2(dst_access);

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &b2;
            if (vk_->cmd_pipeline_barrier2(cmd, dep)) return;
        }
#endif

        VkMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = src_access;
        b.dstAccessMask = dst_access;
        vkCmdPipelineBarrier(
            cmd,
            src_stage,
            dst_stage,
            0,
            1,
            &b,
            0,
            nullptr,
            0,
            nullptr);
    }

    bool gpu_light_culler_enabled() const
    {
        return
            enable_light_culling_ &&
            vulkan_culler_backend_ == VulkanCullerBackend::GpuCompute &&
            compute_pipeline_layout_ != VK_NULL_HANDLE &&
            compute_pipeline_ != VK_NULL_HANDLE &&
            (culling_mode_ == shs::LightCullingMode::Tiled ||
             culling_mode_ == shs::LightCullingMode::TiledDepthRange ||
             culling_mode_ == shs::LightCullingMode::Clustered);
    }

    void clear_light_grid_cpu_buffers(uint32_t frame_slot)
    {
        if (!frame_resources_.valid_slot(frame_slot)) return;
        FrameResources& fr = frame_resources_.at_slot(frame_slot);
        if (fr.tile_counts_buffer.mapped && fr.tile_counts_buffer.size > 0)
        {
            std::memset(fr.tile_counts_buffer.mapped, 0, static_cast<size_t>(fr.tile_counts_buffer.size));
        }
        if (fr.tile_indices_buffer.mapped && fr.tile_indices_buffer.size > 0)
        {
            std::memset(fr.tile_indices_buffer.mapped, 0, static_cast<size_t>(fr.tile_indices_buffer.size));
        }
    }

    void draw_frame(float dt, float t)
    {
        int dw = 0;
        int dh = 0;
        SDL_Vulkan_GetDrawableSize(win_, &dw, &dh);
        if (dw <= 0 || dh <= 0)
        {
            SDL_Delay(16);
            return;
        }

        shs::RenderBackendFrameInfo frame{};
        frame.frame_index = ctx_.frame_index;
        frame.width = dw;
        frame.height = dh;

        shs::VulkanRenderBackend::FrameInfo fi{};
        if (!vk_->begin_frame(ctx_, frame, fi))
        {
            SDL_Delay(2);
            return;
        }
        const uint32_t frame_slot = shs::vk_frame_slot(frame.frame_index, kWorkerPoolRingSize);
        const VkDescriptorSet global_set = frame_resources_.at_slot(frame_slot).global_set;
        if (global_set == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Frame descriptor set unavailable");
        }

        ensure_render_targets(fi.extent.width, fi.extent.height);
        if (pipeline_gen_ != vk_->swapchain_generation())
        {
            create_pipelines(true);
        }
        update_culling_debug_stats(frame_slot);

        update_frame_data(dt, t, fi.extent.width, fi.extent.height, frame_slot);

        std::vector<VkCommandBuffer> depth_secondaries{};
        std::vector<VkCommandBuffer> scene_secondaries{};
        if (use_multithread_recording_)
        {
            if ((enable_depth_prepass_ || enable_scene_pass_) && !reset_worker_pools_for_frame(frame_slot))
            {
                throw std::runtime_error("Failed to reset worker command pools");
            }

            if (enable_depth_prepass_ &&
                !record_secondary_lists(
                    depth_target_.render_pass,
                    depth_target_.framebuffer,
                    depth_pipeline_,
                    depth_pipeline_layout_,
                    depth_target_.w,
                    depth_target_.h,
                    true,
                    true,
                    frame_slot,
                    depth_secondaries))
            {
                throw std::runtime_error("Failed to record depth secondary command buffers");
            }
            if (enable_scene_pass_ &&
                !record_secondary_lists(
                    fi.render_pass,
                    fi.framebuffer,
                    scene_pipeline_,
                    scene_pipeline_layout_,
                    fi.extent.width,
                    fi.extent.height,
                    true,
                    true,
                    frame_slot,
                    scene_secondaries))
            {
                throw std::runtime_error("Failed to record scene secondary command buffers");
            }
        }

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(fi.cmd, &bi) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBeginCommandBuffer failed");
        }

        record_shadow_passes(fi.cmd);

        cmd_memory_barrier(
            fi.cmd,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT);

        if (enable_depth_prepass_)
        {
            if (!depth_secondaries.empty())
            {
                begin_render_pass_depth(fi.cmd);
                vkCmdExecuteCommands(fi.cmd, static_cast<uint32_t>(depth_secondaries.size()), depth_secondaries.data());
                vkCmdEndRenderPass(fi.cmd);
            }
            else
            {
                VkClearValue clear{};
                clear.depthStencil = {1.0f, 0};
                VkRenderPassBeginInfo rp{};
                rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                rp.renderPass = depth_target_.render_pass;
                rp.framebuffer = depth_target_.framebuffer;
                rp.renderArea.offset = {0, 0};
                rp.renderArea.extent = {depth_target_.w, depth_target_.h};
                rp.clearValueCount = 1;
                rp.pClearValues = &clear;
                vkCmdBeginRenderPass(fi.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
                record_inline_depth(fi.cmd, depth_pipeline_, depth_pipeline_layout_, depth_target_.w, depth_target_.h, frame_slot);
                vkCmdEndRenderPass(fi.cmd);
            }
        }

        if (gpu_light_culler_enabled())
        {
            cmd_memory_barrier(
                fi.cmd,
                enable_depth_prepass_
                    ? (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)
                    : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                enable_depth_prepass_ ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : (VkAccessFlags)0,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            if (culling_mode_ == shs::LightCullingMode::TiledDepthRange)
            {
                vkCmdBindPipeline(fi.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, depth_reduce_pipeline_);
                vkCmdBindDescriptorSets(fi.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_layout_, 0, 1, &global_set, 0, nullptr);
                vkCmdDispatch(fi.cmd, tile_w_, tile_h_, 1);

                cmd_memory_barrier(
                    fi.cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
            }

            vkCmdBindPipeline(fi.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
            vkCmdBindDescriptorSets(fi.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_layout_, 0, 1, &global_set, 0, nullptr);
            const uint32_t dispatch_z = (culling_mode_ == shs::LightCullingMode::Clustered) ? kClusterZSlices : 1u;
            vkCmdDispatch(fi.cmd, tile_w_, tile_h_, dispatch_z);

            cmd_memory_barrier(
                fi.cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        }
        else if (enable_light_culling_)
        {
            clear_light_grid_cpu_buffers(frame_slot);
        }

        if (enable_scene_pass_)
        {
            if (!scene_secondaries.empty())
            {
                begin_render_pass_scene(fi.cmd, fi);
                vkCmdExecuteCommands(fi.cmd, static_cast<uint32_t>(scene_secondaries.size()), scene_secondaries.data());
                draw_light_volumes_debug(fi.cmd, scene_pipeline_layout_, frame_slot);
                vkCmdEndRenderPass(fi.cmd);
            }
            else
            {
                VkClearValue clear[2]{};
                clear[0].color = {{0.03f, 0.035f, 0.045f, 1.0f}};
                clear[1].depthStencil = {1.0f, 0};

                VkRenderPassBeginInfo rp{};
                rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                rp.renderPass = fi.render_pass;
                rp.framebuffer = fi.framebuffer;
                rp.renderArea.offset = {0, 0};
                rp.renderArea.extent = fi.extent;
                rp.clearValueCount = vk_->has_depth_attachment() ? 2u : 1u;
                rp.pClearValues = clear;

                vkCmdBeginRenderPass(fi.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
                record_inline_scene(fi.cmd, scene_pipeline_, scene_pipeline_layout_, fi.extent.width, fi.extent.height, frame_slot);
                draw_light_volumes_debug(fi.cmd, scene_pipeline_layout_, frame_slot);
                vkCmdEndRenderPass(fi.cmd);
            }
        }
        else
        {
            VkClearValue clear[2]{};
            clear[0].color = {{0.03f, 0.035f, 0.045f, 1.0f}};
            clear[1].depthStencil = {1.0f, 0};

            VkRenderPassBeginInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp.renderPass = fi.render_pass;
            rp.framebuffer = fi.framebuffer;
            rp.renderArea.offset = {0, 0};
            rp.renderArea.extent = fi.extent;
            rp.clearValueCount = vk_->has_depth_attachment() ? 2u : 1u;
            rp.pClearValues = clear;
            vkCmdBeginRenderPass(fi.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
            draw_light_volumes_debug(fi.cmd, scene_pipeline_layout_, frame_slot);
            vkCmdEndRenderPass(fi.cmd);
        }

        if (vkEndCommandBuffer(fi.cmd) != VK_SUCCESS)
        {
            throw std::runtime_error("vkEndCommandBuffer failed");
        }

        vk_->end_frame(fi);
        ctx_.frame_index++;
    }

    void update_window_title(float avg_ms)
    {
        const char* mode_name = shs::technique_mode_name(active_technique_);
        const char* light_tech_name = lighting_technique_name(lighting_technique_);
        const char* recipe_name = render_path_recipe_.name.empty() ? "n/a" : render_path_recipe_.name.c_str();
        const char* recipe_status = render_path_plan_valid_ ? "OK" : "Fallback";
        const char* cull_name = shs::light_culling_mode_name(culling_mode_);
        const char* culler_backend = vulkan_culler_backend_name(vulkan_culler_backend_);
        const char* rec_mode = use_multithread_recording_ ? "MT-secondary" : "inline";
        const float switch_in = auto_cycle_technique_ ? std::max(0.0f, kTechniqueSwitchPeriodSec - technique_switch_accum_sec_) : 0.0f;
        const double avg_refs = (cull_debug_list_count_ > 0)
            ? static_cast<double>(cull_debug_total_refs_) / static_cast<double>(cull_debug_list_count_)
            : 0.0;
        const uint32_t visible_draws = visible_instance_count_ + (floor_visible_ ? 1u : 0u);
        const uint32_t total_draws = static_cast<uint32_t>(instances_.size()) + 1u;
        const uint32_t culled_total = (active_light_count_ > visible_light_count_) ? (active_light_count_ - visible_light_count_) : 0u;

        char title[768];
        std::snprintf(
            title,
            sizeof(title),
            "%s | light:%s | rpath:%s(%s) mode:%s | cull:%s(%s) | rec:%s | lights:%u/%u[p:%u s:%u r:%u t:%u] | lvol:%s occ:%s/%s lobj:%s culled:%u[f:%u o:%u p:%u] | shad:sun:%s(%.2f) spot:%u point:%u | cfg:orb%.2f h%.1f r%.2f i%.2f | draws:%u/%u | tile:%ux%u | refs:%llu avg:%.1f max:%u nz:%u/%u | lightsw:%s %.1fs | %.2f ms",
            kAppName,
            light_tech_name,
            recipe_name,
            recipe_status,
            mode_name,
            cull_name,
            culler_backend,
            rec_mode,
            visible_light_count_,
            active_light_count_,
            point_count_active_,
            spot_count_active_,
            rect_count_active_,
            tube_count_active_,
            show_light_volumes_debug_ ? "on" : "off",
            enable_scene_occlusion_ ? "on" : "off",
            enable_light_occlusion_ ? "on" : "off",
            shs::light_object_cull_mode_name(light_object_cull_mode_),
            culled_total,
            light_frustum_rejected_,
            light_occlusion_rejected_,
            light_prefilter_rejected_,
            (shadow_settings_.enable && enable_sun_shadow_) ? "on" : "off",
            sun_shadow_strength_,
            spot_shadow_count_,
            point_shadow_count_,
            light_orbit_scale_,
            light_height_bias_,
            light_range_scale_,
            light_intensity_scale_,
            visible_draws,
            total_draws,
            tile_w_,
            tile_h_,
            static_cast<unsigned long long>(cull_debug_total_refs_),
            avg_refs,
            cull_debug_max_list_size_,
            cull_debug_non_empty_lists_,
            cull_debug_list_count_,
            auto_cycle_technique_ ? "auto" : "manual",
            switch_in,
            avg_ms);
        SDL_SetWindowTitle(win_, title);
    }

    void handle_event(const SDL_Event& e)
    {
        if (e.type == SDL_QUIT) running_ = false;

        if (e.type == SDL_KEYDOWN || e.type == SDL_KEYUP)
        {
            const bool down = (e.type == SDL_KEYDOWN);
            switch (e.key.keysym.sym)
            {
                case SDLK_w:
                    move_forward_ = down;
                    break;
                case SDLK_s:
                    move_backward_ = down;
                    break;
                case SDLK_a:
                    move_left_ = down;
                    break;
                case SDLK_d:
                    move_right_ = down;
                    break;
                case SDLK_q:
                    move_down_ = down;
                    break;
                case SDLK_e:
                    move_up_ = down;
                    break;
                case SDLK_LSHIFT:
                case SDLK_RSHIFT:
                    move_boost_ = down;
                    break;
                default:
                    break;
            }
        }

        if (e.type == SDL_MOUSEBUTTONDOWN || e.type == SDL_MOUSEBUTTONUP)
        {
            const bool down = (e.type == SDL_MOUSEBUTTONDOWN);
            if (e.button.button == SDL_BUTTON_LEFT) mouse_left_down_ = down;
            if (e.button.button == SDL_BUTTON_RIGHT) mouse_right_down_ = down;
        }

        if (e.type == SDL_MOUSEMOTION)
        {
            mouse_dx_accum_ += static_cast<float>(e.motion.xrel);
            mouse_dy_accum_ += static_cast<float>(e.motion.yrel);
        }

        if (e.type == SDL_KEYDOWN)
        {
            switch (e.key.keysym.sym)
            {
                case SDLK_ESCAPE:
                    running_ = false;
                    break;
                case SDLK_F1:
                    use_multithread_recording_ = !use_multithread_recording_;
                    break;
                case SDLK_F2:
                    if ((e.key.keysym.mod & KMOD_SHIFT) != 0)
                    {
                        cycle_lighting_technique();
                    }
                    else
                    {
                        cycle_render_path_recipe();
                    }
                    break;
                case SDLK_TAB:
                    cycle_render_path_recipe();
                    break;
                case SDLK_F6:
                    vulkan_culler_backend_ =
                        (vulkan_culler_backend_ == VulkanCullerBackend::GpuCompute)
                            ? VulkanCullerBackend::Disabled
                            : VulkanCullerBackend::GpuCompute;
                    break;
                case SDLK_F7:
                    show_light_volumes_debug_ = !show_light_volumes_debug_;
                    break;
                case SDLK_F11:
                    auto_cycle_technique_ = !auto_cycle_technique_;
                    technique_switch_accum_sec_ = 0.0f;
                    break;
                case SDLK_F12:
                    enable_sun_shadow_ = !enable_sun_shadow_;
                    break;
                case SDLK_1:
                    light_orbit_scale_ = std::clamp(light_orbit_scale_ - 0.10f, 0.35f, 2.50f);
                    break;
                case SDLK_2:
                    light_orbit_scale_ = std::clamp(light_orbit_scale_ + 0.10f, 0.35f, 2.50f);
                    break;
                case SDLK_3:
                    light_height_bias_ = std::clamp(light_height_bias_ - 0.50f, -8.0f, 12.0f);
                    break;
                case SDLK_4:
                    light_height_bias_ = std::clamp(light_height_bias_ + 0.50f, -8.0f, 12.0f);
                    break;
                case SDLK_5:
                    light_range_scale_ = std::clamp(light_range_scale_ - 0.10f, 0.35f, 2.50f);
                    break;
                case SDLK_6:
                    light_range_scale_ = std::clamp(light_range_scale_ + 0.10f, 0.35f, 2.50f);
                    break;
                case SDLK_7:
                    light_intensity_scale_ = std::clamp(light_intensity_scale_ - 0.10f, 0.15f, 3.00f);
                    break;
                case SDLK_8:
                    light_intensity_scale_ = std::clamp(light_intensity_scale_ + 0.10f, 0.15f, 3.00f);
                    break;
                case SDLK_9:
                    sun_shadow_strength_ = std::clamp(sun_shadow_strength_ - 0.05f, 0.0f, 1.0f);
                    break;
                case SDLK_0:
                    sun_shadow_strength_ = std::clamp(sun_shadow_strength_ + 0.05f, 0.0f, 1.0f);
                    break;
                case SDLK_r:
                    light_orbit_scale_ = 1.0f;
                    light_height_bias_ = 0.0f;
                    light_range_scale_ = 1.0f;
                    light_intensity_scale_ = 1.0f;
                    enable_sun_shadow_ = false;
                    sun_shadow_strength_ = 0.0f;
                    break;
                case SDLK_MINUS:
                case SDLK_KP_MINUS:
                    active_light_count_ = (active_light_count_ > 64u) ? (active_light_count_ - 64u) : 64u;
                    break;
                case SDLK_EQUALS:
                case SDLK_PLUS:
                case SDLK_KP_PLUS:
                    active_light_count_ = std::min<uint32_t>(kMaxLights, active_light_count_ + 64u);
                    break;
                default:
                    break;
            }
        }

        if (e.type == SDL_WINDOWEVENT &&
            (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED || e.window.event == SDL_WINDOWEVENT_RESIZED))
        {
            vk_->request_resize(e.window.data1, e.window.data2);
        }
    }

    void main_loop()
    {
        running_ = true;

        using clock = std::chrono::steady_clock;
        auto last = clock::now();
        auto title_t0 = last;
        float ema_ms = 16.0f;

        while (running_)
        {
            SDL_Event e{};
            while (SDL_PollEvent(&e))
            {
                handle_event(e);
            }

            auto now = clock::now();
            float dt = std::chrono::duration<float>(now - last).count();
            last = now;
            dt = std::clamp(dt, 1.0f / 240.0f, 1.0f / 15.0f);
            time_sec_ += dt;
            if (auto_cycle_technique_)
            {
                technique_switch_accum_sec_ += dt;
                if (technique_switch_accum_sec_ >= kTechniqueSwitchPeriodSec)
                {
                    cycle_lighting_technique();
                    technique_switch_accum_sec_ = 0.0f;
                }
            }

            auto cpu_t0 = clock::now();
            draw_frame(dt, time_sec_);
            auto cpu_t1 = clock::now();

            const float frame_ms = std::chrono::duration<float, std::milli>(cpu_t1 - cpu_t0).count();
            ema_ms = glm::mix(ema_ms, frame_ms, 0.08f);

            if (std::chrono::duration<float>(now - title_t0).count() >= 0.20f)
            {
                update_window_title(ema_ms);
                title_t0 = now;
            }
        }

        if (vk_ && vk_->device() != VK_NULL_HANDLE)
        {
            (void)vkDeviceWaitIdle(vk_->device());
        }
    }


    // cleanup consolidated at the top

    struct LightAnim
    {
        shs::LightType type = shs::LightType::Point;
        float angle0 = 0.0f;
        float orbit_radius = 10.0f;
        float height = 6.0f;
        float speed = 1.0f;
        float range = 6.0f;
        float phase = 0.0f;
        glm::vec3 color{1.0f};
        float intensity = 2.0f;
        shs::LightAttenuationModel attenuation_model = shs::LightAttenuationModel::Smooth;
        float attenuation_power = 1.0f;
        float attenuation_bias = 0.05f;
        float attenuation_cutoff = 0.0f;
        glm::vec3 direction_ws{0.0f, -1.0f, 0.0f};
        glm::vec3 rect_right_ws{1.0f, 0.0f, 0.0f};
        glm::vec2 spot_inner_outer{glm::radians(16.0f), glm::radians(26.0f)};
        glm::vec4 shape_params{0.0f};
    };

private:
    bool cleaned_up_ = false;
    bool running_ = false;
    bool sdl_ready_ = false;
    SDL_Window* win_ = nullptr;

    shs::Context ctx_{};
    std::vector<std::unique_ptr<shs::IRenderBackend>> keep_{};
    shs::VulkanRenderBackend* vk_ = nullptr;

    std::unique_ptr<shs::ThreadPoolJobSystem> jobs_{};
    uint32_t worker_count_ = 1;
    std::vector<WorkerPool> worker_pools_{};

    std::vector<Vertex> vertices_{};
    std::vector<uint32_t> indices_{};
    std::vector<Vertex> floor_vertices_{};
    std::vector<uint32_t> floor_indices_{};
    std::vector<Vertex> cone_vertices_{};
    std::vector<uint32_t> cone_indices_{};
    std::vector<Vertex> box_vertices_{};
    std::vector<uint32_t> box_indices_{};
    std::vector<uint32_t> sphere_line_indices_{};
    std::vector<uint32_t> cone_line_indices_{};
    std::vector<uint32_t> box_line_indices_{};
    std::vector<Vertex> capsule_vertices_{};
    std::vector<uint32_t> capsule_indices_{};
    std::vector<Vertex> cylinder_vertices_{};
    std::vector<uint32_t> cylinder_indices_{};
    std::vector<Instance> instances_{};
    std::vector<glm::mat4> instance_models_{};
    std::vector<uint8_t> instance_visible_mask_{};
    std::vector<uint32_t> frustum_visible_instance_indices_{};
    std::vector<shs::SceneShape> instance_cull_shapes_{};
    JPH::ShapeRefC sphere_shape_jolt_{};
    JPH::ShapeRefC box_shape_jolt_{};
    JPH::ShapeRefC cone_shape_jolt_{};
    JPH::ShapeRefC capsule_shape_jolt_{};
    JPH::ShapeRefC cylinder_shape_jolt_{};
    std::vector<LightAnim> light_anim_{};
    shs::LightSet light_set_{};
    std::vector<shs::CullingLightGPU> gpu_lights_{};
    std::vector<ShadowLightGPU> shadow_lights_gpu_{};
    std::vector<LocalShadowCaster> local_shadow_casters_{};
    std::vector<shs::AABB> visible_object_aabbs_{};
    shs::DebugMesh sphere_occluder_mesh_{};
    shs::DebugMesh cone_occluder_mesh_{};
    shs::DebugMesh box_occluder_mesh_{};
    shs::DebugMesh capsule_occluder_mesh_{};
    shs::DebugMesh cylinder_occluder_mesh_{};
    shs::DebugMesh floor_occluder_mesh_{};
    std::vector<float> scene_occlusion_depth_{};
    std::vector<float> light_occlusion_depth_{};
    glm::mat4 sun_shadow_view_proj_{1.0f};
    shs::AABB sphere_local_aabb_{};
    shs::AABB cone_local_aabb_{};
    shs::AABB box_local_aabb_{};
    shs::AABB capsule_local_aabb_{};
    shs::AABB cylinder_local_aabb_{};
    shs::Sphere sphere_local_bound_{};
    shs::Sphere cone_local_bound_{};
    shs::Sphere box_local_bound_{};
    shs::Sphere capsule_local_bound_{};
    shs::Sphere cylinder_local_bound_{};
    shs::AABB floor_local_aabb_{};
    shs::AABB shadow_scene_static_aabb_{};
    bool shadow_scene_static_bounds_ready_ = false;
    glm::mat4 floor_model_{1.0f};
    glm::vec4 floor_material_color_{1.0f};
    glm::vec4 floor_material_params_{0.0f, 0.72f, 1.0f, 0.0f};

    GpuBuffer vertex_buffer_{};
    GpuBuffer index_buffer_{};
    GpuBuffer floor_vertex_buffer_{};
    GpuBuffer floor_index_buffer_{};
    GpuBuffer cone_vertex_buffer_{};
    GpuBuffer cone_index_buffer_{};
    GpuBuffer box_vertex_buffer_{};
    GpuBuffer box_index_buffer_{};
    GpuBuffer sphere_line_index_buffer_{};
    GpuBuffer cone_line_index_buffer_{};
    GpuBuffer box_line_index_buffer_{};
    GpuBuffer capsule_vertex_buffer_{};
    GpuBuffer capsule_index_buffer_{};
    GpuBuffer cylinder_vertex_buffer_{};
    GpuBuffer cylinder_index_buffer_{};
    shs::VkFrameRing<FrameResources, kWorkerPoolRingSize> frame_resources_{};

    CameraUBO camera_ubo_{};
    DepthTarget depth_target_{};
    LayeredDepthTarget sun_shadow_target_{};
    LayeredDepthTarget local_shadow_target_{};

    VkDescriptorSetLayout global_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkSampler depth_sampler_ = VK_NULL_HANDLE;

    VkPipelineLayout shadow_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline shadow_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout depth_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline depth_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout scene_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline scene_pipeline_ = VK_NULL_HANDLE;
    VkPipeline scene_wire_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout compute_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline depth_reduce_pipeline_ = VK_NULL_HANDLE;
    VkPipeline compute_pipeline_ = VK_NULL_HANDLE;

    uint64_t pipeline_gen_ = 0;
    uint32_t tile_w_ = 0;
    uint32_t tile_h_ = 0;
    uint32_t active_light_count_ = kDefaultLightCount;
    uint32_t visible_light_count_ = 0;
    uint32_t visible_instance_count_ = 0;
    bool floor_visible_ = true;
    uint32_t point_count_active_ = 0;
    uint32_t spot_count_active_ = 0;
    uint32_t rect_count_active_ = 0;
    uint32_t tube_count_active_ = 0;
    uint32_t point_shadow_count_ = 0;
    uint32_t spot_shadow_count_ = 0;
    bool show_light_volumes_debug_ = false;
    std::vector<LightVolumeDebugDraw> light_volume_debug_draws_{};
    bool enable_scene_occlusion_ = false;
    bool enable_light_occlusion_ = false;
    shs::LightObjectCullMode light_object_cull_mode_ = shs::LightObjectCullMode::None;
    uint32_t light_frustum_rejected_ = 0;
    uint32_t light_occlusion_rejected_ = 0;
    uint32_t light_prefilter_rejected_ = 0;
    float light_orbit_scale_ = 1.0f;
    float light_height_bias_ = 0.0f;
    float light_range_scale_ = 1.0f;
    float light_intensity_scale_ = 1.0f;
    bool enable_sun_shadow_ = false;
    float sun_shadow_strength_ = 0.0f;
    bool use_forward_plus_ = true;
    shs::LightCullingMode culling_mode_ = shs::LightCullingMode::Tiled;
    shs::ShadowCompositionSettings shadow_settings_ = shs::make_default_shadow_composition_settings();
    VulkanCullerBackend vulkan_culler_backend_ = VulkanCullerBackend::GpuCompute;
    bool profile_depth_prepass_enabled_ = true;
    bool enable_depth_prepass_ = true;
    bool enable_light_culling_ = true;
    bool enable_scene_pass_ = true;
    uint64_t cull_debug_total_refs_ = 0;
    uint32_t cull_debug_non_empty_lists_ = 0;
    uint32_t cull_debug_list_count_ = 0;
    uint32_t cull_debug_max_list_size_ = 0;
    shs::RenderPathRegistry render_path_registry_{};
    std::vector<std::string> render_path_cycle_order_{};
    shs::RenderPathRecipe render_path_recipe_{};
    shs::RenderPathExecutionPlan render_path_plan_{};
    bool render_path_plan_valid_ = false;
    size_t render_path_cycle_index_ = 0;
    LightingTechnique lighting_technique_ = LightingTechnique::PBR;
    shs::TechniqueMode active_technique_ = shs::TechniqueMode::ForwardPlus;
    size_t technique_cycle_index_ = 1;
    float technique_switch_accum_sec_ = 0.0f;
    bool auto_cycle_technique_ = false;
    bool use_multithread_recording_ = false;
    FreeCamera camera_{};
    bool move_forward_ = false;
    bool move_backward_ = false;
    bool move_left_ = false;
    bool move_right_ = false;
    bool move_up_ = false;
    bool move_down_ = false;
    bool move_boost_ = false;
    bool mouse_left_down_ = false;
    bool mouse_right_down_ = false;
    float mouse_dx_accum_ = 0.0f;
    float mouse_dy_accum_ = 0.0f;
    float time_sec_ = 0.0f;
};
}

int main()
{
    try
    {
        HelloRenderingPathsApp app{};
        app.run();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
