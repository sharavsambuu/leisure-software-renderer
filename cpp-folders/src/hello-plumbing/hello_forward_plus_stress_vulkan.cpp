#define SDL_MAIN_HANDLED

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
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
#include <shs/camera/convention.hpp>
#include <shs/camera/light_camera.hpp>
#include <shs/frame/technique_mode.hpp>
#include <shs/geometry/frustum_culling.hpp>
#include <shs/job/thread_pool_job_system.hpp>
#include <shs/job/wait_group.hpp>
#include <shs/lighting/light_culling_mode.hpp>
#include <shs/lighting/light_set.hpp>
#include <shs/lighting/shadow_technique.hpp>
#include <shs/pipeline/technique_profile.hpp>
#include <shs/resources/loaders/primitive_import.hpp>
#include <shs/resources/resource_registry.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>

namespace
{
constexpr int kDefaultW = 1280;
constexpr int kDefaultH = 720;
constexpr uint32_t kTileSize = 16;
constexpr uint32_t kMaxLightsPerTile = 128;
constexpr uint32_t kMaxLights = 8192;
constexpr uint32_t kDefaultLightCount = 2048;
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
#if defined(SHS_FP_SHADOW_SHOWCASE)
constexpr const char* kAppName = "HelloVulkanShadowTechniques";
#else
constexpr const char* kAppName = "HelloForwardPlusStressVulkan";
#endif

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
    glm::uvec4 culling_params{0u};          // x: cluster_z_slices
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
    glm::vec3 base_pos{0.0f};
    glm::vec4 base_color{1.0f};
    float scale = 1.0f;
    float phase = 0.0f;
    float metallic = 0.08f;
    float roughness = 0.36f;
    float ao = 1.0f;
};

struct GpuBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
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

std::vector<char> read_file(const char* path)
{
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error(std::string("Failed to open shader file: ") + path);
    const size_t sz = static_cast<size_t>(f.tellg());
    if (sz == 0) throw std::runtime_error(std::string("Empty shader file: ") + path);
    std::vector<char> out(sz);
    f.seekg(0);
    f.read(out.data(), static_cast<std::streamsize>(sz));
    return out;
}

VkShaderModule create_shader_module(VkDevice dev, const std::vector<char>& code)
{
    if (dev == VK_NULL_HANDLE || code.empty() || (code.size() % 4) != 0)
    {
        throw std::runtime_error("Invalid SPIR-V blob");
    }
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &ci, nullptr, &sm) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateShaderModule failed");
    }
    return sm;
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

class HelloForwardPlusStressVulkanApp
{
public:
    ~HelloForwardPlusStressVulkanApp()
    {
        cleanup();
    }

    void run()
    {
        init_sdl();
        init_backend();
        init_jobs();
        init_scene_data();
        init_gpu_resources();
        main_loop();
    }

private:
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
        if (!vk_->init_sdl(init))
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

    void init_scene_data()
    {
        shs::ResourceRegistry resources{};
        const shs::MeshAssetHandle sphere_h = shs::import_sphere_primitive(resources, shs::SphereDesc{0.5f, 18, 12}, "fplus_sphere");
        const shs::MeshAssetHandle floor_h = shs::import_plane_primitive(resources, shs::PlaneDesc{300.0f, 300.0f, 64, 64}, "fplus_floor");

        const shs::MeshData* sphere_mesh = resources.get_mesh(sphere_h);
        if (!sphere_mesh || sphere_mesh->empty())
        {
            throw std::runtime_error("Failed to generate sphere primitive mesh");
        }
        const shs::MeshData* floor_mesh = resources.get_mesh(floor_h);
        if (!floor_mesh || floor_mesh->empty())
        {
            throw std::runtime_error("Failed to generate floor primitive mesh");
        }
        sphere_local_aabb_ = compute_local_aabb_from_positions(sphere_mesh->positions);
        floor_local_aabb_ = compute_local_aabb_from_positions(floor_mesh->positions);
        sphere_local_bound_ = shs::sphere_from_aabb(sphere_local_aabb_);

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

        floor_vertices_.clear();
        floor_vertices_.reserve(floor_mesh->positions.size());
        for (size_t i = 0; i < floor_mesh->positions.size(); ++i)
        {
            Vertex v{};
            v.pos = floor_mesh->positions[i];
            if (i < floor_mesh->normals.size()) v.normal = floor_mesh->normals[i];
            floor_vertices_.push_back(v);
        }
        floor_indices_ = floor_mesh->indices;
        floor_model_ = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.2f, 0.0f));
        floor_material_color_ = glm::vec4(120.0f / 255.0f, 122.0f / 255.0f, 128.0f / 255.0f, 1.0f);
        // PBR plastic floor material.
        floor_material_params_ = glm::vec4(0.0f, 0.62f, 1.0f, 0.0f);

        instances_.clear();
        instance_models_.clear();
        const int grid_x = 48;
        const int grid_z = 32;
        const float spacing = 2.4f;
        std::mt19937 rng(1337u);
        std::uniform_real_distribution<float> jitter(-0.18f, 0.18f);
        std::uniform_real_distribution<float> hue(0.0f, 1.0f);
        for (int z = 0; z < grid_z; ++z)
        {
            for (int x = 0; x < grid_x; ++x)
            {
                Instance inst{};
                inst.base_pos = glm::vec3(
                    (static_cast<float>(x) - static_cast<float>(grid_x - 1) * 0.5f) * spacing + jitter(rng),
                    0.0f,
                    (static_cast<float>(z) - static_cast<float>(grid_z - 1) * 0.5f) * spacing + jitter(rng));
                const float h = hue(rng);
                inst.base_color = glm::vec4(
                    0.45f + 0.55f * std::sin(6.28318f * (h + 0.00f)),
                    0.45f + 0.55f * std::sin(6.28318f * (h + 0.33f)),
                    0.45f + 0.55f * std::sin(6.28318f * (h + 0.66f)),
                    1.0f);
                inst.scale = 0.78f;
                inst.phase = hue(rng) * 10.0f;
                inst.metallic = 0.04f + 0.22f * hue(rng);
                inst.roughness = 0.22f + 0.45f * hue(rng);
                inst.ao = 1.0f;
                instances_.push_back(inst);
            }
        }
        instance_models_.resize(instances_.size(), glm::mat4(1.0f));
        instance_visible_mask_.assign(instances_.size(), 1u);
        visible_instance_count_ = static_cast<uint32_t>(instances_.size());
        floor_visible_ = true;

        light_anim_.clear();
        light_anim_.reserve(kMaxLights);
        gpu_lights_.resize(kMaxLights);
        shadow_lights_gpu_.assign(kMaxLights, ShadowLightGPU{});
        std::uniform_real_distribution<float> angle0(0.0f, 6.28318f);
        std::uniform_real_distribution<float> rad(8.0f, 82.0f);
        std::uniform_real_distribution<float> hgt(1.0f, 14.0f);
        std::uniform_real_distribution<float> spd(0.15f, 1.10f);
        std::uniform_real_distribution<float> radius(7.5f, 15.0f);
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

            const uint32_t bucket = i % 10u;
            if (bucket < 6u)
            {
                l.type = shs::LightType::Point;
                l.attenuation_model = shs::LightAttenuationModel::InverseSquare;
                // Warm-dominant palette for point lights.
                l.color = glm::mix(l.color, glm::vec3(1.0f, 0.62f, 0.28f), 0.58f);
            }
            else if (bucket < 9u)
            {
                l.type = shs::LightType::Spot;
                l.attenuation_model = shs::LightAttenuationModel::InverseSquare;
                const float inner = glm::radians(inner_deg(rng));
                l.spot_inner_outer.x = inner;
                l.spot_inner_outer.y = inner + glm::radians(outer_extra_deg(rng));
                // Cool palette for spot lights.
                l.color = glm::mix(l.color, glm::vec3(0.35f, 0.85f, 1.0f), 0.62f);
            }
            else if ((i & 1u) == 0u)
            {
                l.type = shs::LightType::RectArea;
                l.attenuation_model = shs::LightAttenuationModel::Smooth;
                l.shape_params = glm::vec4(area_extent(rng), area_extent(rng), 0.0f, 0.0f);
                l.rect_right_ws = shs::normalize_or(glm::vec3(right_rand(rng), 0.0f, right_rand(rng)), glm::vec3(1.0f, 0.0f, 0.0f));
                // Magenta-biased rect-area accents.
                l.color = glm::mix(l.color, glm::vec3(1.0f, 0.35f, 0.78f), 0.65f);
            }
            else
            {
                l.type = shs::LightType::TubeArea;
                l.attenuation_model = shs::LightAttenuationModel::Linear;
                l.shape_params = glm::vec4(tube_half_len(rng), tube_rad(rng), 0.0f, 0.0f);
                // Green tube-area accents.
                l.color = glm::mix(l.color, glm::vec3(0.32f, 1.0f, 0.55f), 0.62f);
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
        shadow_settings_.budget.max_rect_area = 2u;
        shadow_settings_.budget.max_tube_area = 2u;

        apply_technique_mode(shs::TechniqueMode::ForwardPlus);
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

    uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags required)
    {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(vk_->physical_device(), &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        {
            const bool type_ok = (type_bits & (1u << i)) != 0;
            const bool props_ok = (mp.memoryTypes[i].propertyFlags & required) == required;
            if (type_ok && props_ok) return i;
        }
        throw std::runtime_error("No compatible Vulkan memory type found");
    }

    void create_buffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags mem_flags,
        GpuBuffer& out,
        bool map_memory)
    {
        destroy_buffer(out);

        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(vk_->device(), &bci, nullptr, &out.buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateBuffer failed");
        }

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(vk_->device(), out.buffer, &req);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, mem_flags);
        if (vkAllocateMemory(vk_->device(), &mai, nullptr, &out.memory) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateMemory failed for buffer");
        }
        if (vkBindBufferMemory(vk_->device(), out.buffer, out.memory, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBindBufferMemory failed");
        }

        out.size = size;
        if (map_memory)
        {
            if (vkMapMemory(vk_->device(), out.memory, 0, size, 0, &out.mapped) != VK_SUCCESS)
            {
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
        if (b.buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(vk_->device(), b.buffer, nullptr);
            b.buffer = VK_NULL_HANDLE;
        }
        if (b.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(vk_->device(), b.memory, nullptr);
            b.memory = VK_NULL_HANDLE;
        }
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
    }

    void create_dynamic_buffers()
    {
        const VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        create_buffer(
            sizeof(CameraUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            host_flags,
            camera_buffer_,
            true);

        create_buffer(
            static_cast<VkDeviceSize>(kMaxLights) * sizeof(shs::CullingLightGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_flags,
            light_buffer_,
            true);

        create_buffer(
            static_cast<VkDeviceSize>(kMaxLights) * sizeof(ShadowLightGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_flags,
            shadow_light_buffer_,
            true);
        std::memset(shadow_light_buffer_.mapped, 0, static_cast<size_t>(shadow_light_buffer_.size));
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
            if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0)
            {
                return fmt;
            }
        }

        return VK_FORMAT_D32_SFLOAT;
    }

    bool has_stencil(VkFormat fmt) const
    {
        return fmt == VK_FORMAT_D24_UNORM_S8_UINT || fmt == VK_FORMAT_D32_SFLOAT_S8_UINT;
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
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
        if (has_stencil(depth_target_.format)) iv.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
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
        deps[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
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
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
        if (has_stencil(out.format)) sv.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
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
        deps[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
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
            if (has_stencil(out.format)) iv.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
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

        create_buffer(counts_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, tile_counts_buffer_, true);
        create_buffer(indices_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, tile_indices_buffer_, true);
        create_buffer(depth_ranges_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, tile_depth_ranges_buffer_, true);

        std::memset(tile_counts_buffer_.mapped, 0, static_cast<size_t>(counts_size));
        std::memset(tile_indices_buffer_.mapped, 0, static_cast<size_t>(indices_size));
        std::memset(tile_depth_ranges_buffer_.mapped, 0, static_cast<size_t>(depth_ranges_size));
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
            sizes[0].descriptorCount = 8;
            sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            sizes[1].descriptorCount = 96;
            sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sizes[2].descriptorCount = 32;

            VkDescriptorPoolCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            ci.maxSets = 4;
            ci.poolSizeCount = 3;
            ci.pPoolSizes = sizes;
            if (vkCreateDescriptorPool(vk_->device(), &ci, nullptr, &descriptor_pool_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorPool failed");
            }
        }

        if (global_set_ == VK_NULL_HANDLE)
        {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptor_pool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &global_set_layout_;
            if (vkAllocateDescriptorSets(vk_->device(), &ai, &global_set_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkAllocateDescriptorSets failed");
            }
        }
    }

    void update_global_descriptor_set()
    {
        VkDescriptorBufferInfo camera_info{};
        camera_info.buffer = camera_buffer_.buffer;
        camera_info.offset = 0;
        camera_info.range = sizeof(CameraUBO);

        VkDescriptorBufferInfo light_info{};
        light_info.buffer = light_buffer_.buffer;
        light_info.offset = 0;
        light_info.range = static_cast<VkDeviceSize>(kMaxLights) * sizeof(shs::CullingLightGPU);

        VkDescriptorBufferInfo tile_counts_info{};
        tile_counts_info.buffer = tile_counts_buffer_.buffer;
        tile_counts_info.offset = 0;
        tile_counts_info.range = tile_counts_buffer_.size;

        VkDescriptorBufferInfo tile_indices_info{};
        tile_indices_info.buffer = tile_indices_buffer_.buffer;
        tile_indices_info.offset = 0;
        tile_indices_info.range = tile_indices_buffer_.size;

        VkDescriptorBufferInfo tile_depth_ranges_info{};
        tile_depth_ranges_info.buffer = tile_depth_ranges_buffer_.buffer;
        tile_depth_ranges_info.offset = 0;
        tile_depth_ranges_info.range = tile_depth_ranges_buffer_.size;

        VkDescriptorBufferInfo shadow_light_info{};
        shadow_light_info.buffer = shadow_light_buffer_.buffer;
        shadow_light_info.offset = 0;
        shadow_light_info.range = static_cast<VkDeviceSize>(kMaxLights) * sizeof(ShadowLightGPU);

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

        VkWriteDescriptorSet w[10]{};
        for (int i = 0; i < 10; ++i)
        {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = global_set_;
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

        const std::vector<char> shadow_vs_code = read_file(SHS_VK_FP_SHADOW_VERT_SPV);
        const std::vector<char> scene_vs_code = read_file(SHS_VK_FP_SCENE_VERT_SPV);
        const std::vector<char> scene_fs_code = read_file(SHS_VK_FP_SCENE_FRAG_SPV);
        const std::vector<char> depth_reduce_cs_code = read_file(SHS_VK_FP_DEPTH_REDUCE_COMP_SPV);
        const std::vector<char> cull_cs_code = read_file(SHS_VK_FP_LIGHT_CULL_COMP_SPV);

        VkShaderModule shadow_vs = create_shader_module(vk_->device(), shadow_vs_code);
        VkShaderModule scene_vs = create_shader_module(vk_->device(), scene_vs_code);
        VkShaderModule scene_fs = create_shader_module(vk_->device(), scene_fs_code);
        VkShaderModule depth_reduce_cs = create_shader_module(vk_->device(), depth_reduce_cs_code);
        VkShaderModule cull_cs = create_shader_module(vk_->device(), cull_cs_code);

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
        update_global_descriptor_set();
        create_pipelines(true);
    }

    void apply_technique_mode(shs::TechniqueMode mode)
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
        const shs::TechniqueProfile profile = shs::make_default_technique_profile(mode);

        enable_depth_prepass_ = profile_has_pass(profile, "depth_prepass");
        enable_light_culling_ = profile_has_pass(profile, "light_culling");

        shs::LightCullingMode mode_hint = default_culling_mode_for_technique(mode);
        if (!enable_light_culling_)
        {
            mode_hint = shs::LightCullingMode::None;
        }
        else if (manual_culling_override_)
        {
            mode_hint = manual_culling_mode_;
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

        use_forward_plus_ = (culling_mode_ != shs::LightCullingMode::None);
        technique_switch_accum_sec_ = 0.0f;
    }

    void cycle_technique_mode()
    {
        const auto& modes = known_technique_modes();
        if (modes.empty()) return;
        technique_cycle_index_ = (technique_cycle_index_ + 1u) % modes.size();
        apply_technique_mode(modes[technique_cycle_index_]);
    }

    void cycle_culling_override_mode()
    {
        if (!manual_culling_override_)
        {
            manual_culling_override_ = true;
            manual_culling_mode_ = culling_mode_;
        }

        switch (manual_culling_mode_)
        {
            case shs::LightCullingMode::None:
                manual_culling_mode_ = shs::LightCullingMode::Tiled;
                break;
            case shs::LightCullingMode::Tiled:
                manual_culling_mode_ = shs::LightCullingMode::TiledDepthRange;
                break;
            case shs::LightCullingMode::TiledDepthRange:
                manual_culling_mode_ = shs::LightCullingMode::Clustered;
                break;
            case shs::LightCullingMode::Clustered:
            default:
                manual_culling_mode_ = shs::LightCullingMode::None;
                break;
        }

        culling_mode_ = enable_light_culling_ ? manual_culling_mode_ : shs::LightCullingMode::None;
    }

    void clear_culling_override_mode()
    {
        manual_culling_override_ = false;
        culling_mode_ = enable_light_culling_ ?
            default_culling_mode_for_technique(active_technique_) :
            shs::LightCullingMode::None;
    }

    void update_culling_debug_stats()
    {
        if (!tile_counts_buffer_.mapped || tile_counts_buffer_.size < sizeof(uint32_t) || tile_w_ == 0 || tile_h_ == 0)
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
        const uint32_t capacity = static_cast<uint32_t>(tile_counts_buffer_.size / sizeof(uint32_t));
        list_count = std::min(list_count, capacity);

        const uint32_t* counts = reinterpret_cast<const uint32_t*>(tile_counts_buffer_.mapped);
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

    void update_visibility_from_frustum(const shs::Frustum& frustum)
    {
        if (instance_visible_mask_.size() != instances_.size())
        {
            instance_visible_mask_.assign(instances_.size(), 1u);
        }

        uint32_t visible_instances = 0;
        for (size_t i = 0; i < instances_.size(); ++i)
        {
            const shs::Sphere ws = shs::transform_sphere(sphere_local_bound_, instance_models_[i]);
            const bool visible = shs::intersects_frustum_sphere(frustum, ws);
            instance_visible_mask_[i] = visible ? 1u : 0u;
            if (visible) ++visible_instances;
        }
        visible_instance_count_ = visible_instances;

        const shs::AABB floor_ws = shs::transform_aabb(floor_local_aabb_, floor_model_);
        floor_visible_ = shs::intersects_frustum_aabb(frustum, floor_ws);
    }

    void update_frame_data(float dt, float t, uint32_t w, uint32_t h)
    {
        (void)dt;

        const float aspect = (h > 0) ? (static_cast<float>(w) / static_cast<float>(h)) : 1.0f;
        const float orbit_r = 68.0f;
        const glm::vec3 cam_pos = glm::vec3(std::sin(t * 0.22f) * orbit_r, 26.0f + std::sin(t * 0.35f) * 5.0f, std::cos(t * 0.22f) * orbit_r);
        const glm::vec3 cam_target = glm::vec3(0.0f, 2.0f, 0.0f);

        camera_ubo_.view = shs::look_at_lh(cam_pos, cam_target, glm::vec3(0.0f, 1.0f, 0.0f));
        camera_ubo_.proj = shs::perspective_lh_no(glm::radians(62.0f), aspect, 0.1f, 260.0f);
        camera_ubo_.view_proj = camera_ubo_.proj * camera_ubo_.view;
        camera_ubo_.camera_pos_time = glm::vec4(cam_pos, t);
        camera_ubo_.sun_dir_intensity = glm::vec4(glm::normalize(glm::vec3(-0.35f, -1.0f, -0.18f)), 1.65f);
        camera_ubo_.screen_tile_lightcount = glm::uvec4(w, h, tile_w_, active_light_count_);
        camera_ubo_.params = glm::uvec4(tile_h_, kMaxLightsPerTile, kTileSize, static_cast<uint32_t>(culling_mode_));
        camera_ubo_.culling_params = glm::uvec4(kClusterZSlices, 0u, 0u, 0u);
        camera_ubo_.depth_params = glm::vec4(0.1f, 260.0f, 0.0f, 0.0f);
        camera_ubo_.exposure_gamma = glm::vec4(1.4f, 2.2f, 0.0f, 0.0f);
        camera_ubo_.sun_shadow_params = glm::vec4(0.88f, 0.0008f, 0.0018f, 2.0f);
        camera_ubo_.sun_shadow_filter = glm::vec4(shadow_settings_.quality.pcf_step, shadow_settings_.enable ? 1.0f : 0.0f, 0.0f, 0.0f);

        for (size_t i = 0; i < instances_.size(); ++i)
        {
            const Instance& inst = instances_[i];
            const float bob = std::sin(t * 1.2f + inst.phase) * 0.28f;
            const float rot = t * (0.2f + 0.03f * std::sin(inst.phase));
            glm::mat4 m(1.0f);
            m = glm::translate(m, inst.base_pos + glm::vec3(0.0f, bob, 0.0f));
            m = glm::rotate(m, rot, glm::vec3(0.0f, 1.0f, 0.0f));
            m = glm::scale(m, glm::vec3(inst.scale));
            instance_models_[i] = m;
        }

        const shs::Frustum camera_frustum = shs::extract_frustum_planes(camera_ubo_.view_proj);
        update_visibility_from_frustum(camera_frustum);

        shs::AABB shadow_scene_aabb{};
        bool has_shadow_bounds = false;
        if (floor_visible_)
        {
            const shs::AABB floor_ws = shs::transform_aabb(floor_local_aabb_, floor_model_);
            shadow_scene_aabb.expand(floor_ws.minv);
            shadow_scene_aabb.expand(floor_ws.maxv);
            has_shadow_bounds = true;
        }
        for (size_t i = 0; i < instance_models_.size(); ++i)
        {
            if (i >= instance_visible_mask_.size() || instance_visible_mask_[i] == 0u) continue;
            const shs::Sphere ws_sphere = shs::transform_sphere(sphere_local_bound_, instance_models_[i]);
            shadow_scene_aabb.expand(ws_sphere.center - glm::vec3(ws_sphere.radius));
            shadow_scene_aabb.expand(ws_sphere.center + glm::vec3(ws_sphere.radius));
            has_shadow_bounds = true;
        }
        if (!has_shadow_bounds)
        {
            shadow_scene_aabb.expand(glm::vec3(-1.0f));
            shadow_scene_aabb.expand(glm::vec3(1.0f));
        }

        const glm::vec3 sun_dir = glm::normalize(glm::vec3(camera_ubo_.sun_dir_intensity));
        const shs::LightCamera sun_cam = shs::build_dir_light_camera_aabb(sun_dir, shadow_scene_aabb, 14.0f);
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

        light_set_.clear_local_lights();
        const uint32_t lc = std::min<uint32_t>(active_light_count_, static_cast<uint32_t>(light_anim_.size()));
        uint32_t visible_light_count = 0;
        for (uint32_t i = 0; i < lc; ++i)
        {
            const LightAnim& la = light_anim_[i];
            const float a = la.angle0 + la.speed * t;
            const float y = la.height + std::sin(a * 1.7f + la.phase) * 2.6f;
            const glm::vec3 p(std::cos(a) * la.orbit_radius, y, std::sin(a) * la.orbit_radius);
            bool visible_light = false;

            switch (la.type)
            {
                case shs::LightType::Spot:
                {
                    shs::SpotLight l{};
                    l.common.position_ws = p;
                    l.common.range = la.range;
                    l.common.color = la.color;
                    l.common.intensity = la.intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    l.direction_ws = la.direction_ws;
                    l.inner_angle_rad = la.spot_inner_outer.x;
                    l.outer_angle_rad = la.spot_inner_outer.y;
                    const shs::Sphere light_bounds = shs::spot_light_culling_sphere(l);
                    visible_light = shs::intersects_frustum_sphere(camera_frustum, light_bounds);
                    if (!visible_light) break;
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
                            0.92f,
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
                    visible_light_count++;
                    break;
                }
                case shs::LightType::RectArea:
                {
                    shs::RectAreaLight l{};
                    l.common.position_ws = p;
                    l.common.range = la.range;
                    l.common.color = la.color;
                    l.common.intensity = la.intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    l.direction_ws = la.direction_ws;
                    l.right_ws = la.rect_right_ws;
                    l.half_extents = glm::vec2(la.shape_params.x, la.shape_params.y);
                    const shs::Sphere light_bounds = shs::rect_area_light_culling_sphere(l);
                    visible_light = shs::intersects_frustum_sphere(camera_frustum, light_bounds);
                    if (!visible_light) break;
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
                            0.78f,
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
                    visible_light_count++;
                    break;
                }
                case shs::LightType::TubeArea:
                {
                    shs::TubeAreaLight l{};
                    l.common.position_ws = p;
                    l.common.range = la.range;
                    l.common.color = la.color;
                    l.common.intensity = la.intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    l.axis_ws = la.direction_ws;
                    l.half_length = la.shape_params.x;
                    l.radius = la.shape_params.y;
                    const shs::Sphere light_bounds = shs::tube_area_light_culling_sphere(l);
                    visible_light = shs::intersects_frustum_sphere(camera_frustum, light_bounds);
                    if (!visible_light) break;
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
                            0.72f,
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
                    visible_light_count++;
                    break;
                }
                case shs::LightType::Point:
                default:
                {
                    shs::PointLight l{};
                    l.common.position_ws = p;
                    l.common.range = la.range;
                    l.common.color = la.color;
                    l.common.intensity = la.intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    const shs::Sphere light_bounds = shs::point_light_culling_sphere(l);
                    visible_light = shs::intersects_frustum_sphere(camera_frustum, light_bounds);
                    if (!visible_light) break;
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
                            0.86f,
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
                    visible_light_count++;
                    break;
                }
            }
        }
        visible_light_count_ = visible_light_count;
        camera_ubo_.screen_tile_lightcount.w = visible_light_count_;
        std::memcpy(camera_buffer_.mapped, &camera_ubo_, sizeof(CameraUBO));

        if (visible_light_count_ > 0u)
        {
            std::memcpy(light_buffer_.mapped, gpu_lights_.data(), static_cast<size_t>(visible_light_count_) * sizeof(shs::CullingLightGPU));
        }
        std::memcpy(shadow_light_buffer_.mapped, shadow_lights_gpu_.data(), static_cast<size_t>(kMaxLights) * sizeof(ShadowLightGPU));

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
        VkViewport vp{};
        vp.x = 0.0f;
        vp.y = flip_y ? static_cast<float>(h) : 0.0f;
        vp.width = static_cast<float>(w);
        vp.height = flip_y ? -static_cast<float>(h) : static_cast<float>(h);
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        VkRect2D sc{};
        sc.offset = {0, 0};
        sc.extent = {w, h};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
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

    void draw_shadow_scene(VkCommandBuffer cmd, const glm::mat4& light_view_proj)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);

        const VkDeviceSize vb_off = 0;
        if (floor_visible_)
        {
            vkCmdBindVertexBuffers(cmd, 0, 1, &floor_vertex_buffer_.buffer, &vb_off);
            vkCmdBindIndexBuffer(cmd, floor_index_buffer_.buffer, 0, VK_INDEX_TYPE_UINT32);
            ShadowPush pc{};
            pc.light_view_proj = light_view_proj;
            pc.model = floor_model_;
            vkCmdPushConstants(cmd, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &pc);
            vkCmdDrawIndexed(cmd, static_cast<uint32_t>(floor_indices_.size()), 1, 0, 0, 0);
        }

        vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer_.buffer, &vb_off);
        vkCmdBindIndexBuffer(cmd, index_buffer_.buffer, 0, VK_INDEX_TYPE_UINT32);
        for (uint32_t i = 0; i < static_cast<uint32_t>(instances_.size()); ++i)
        {
            if (i >= instance_visible_mask_.size() || instance_visible_mask_[i] == 0u) continue;
            ShadowPush pc{};
            pc.light_view_proj = light_view_proj;
            pc.model = instance_models_[i];
            vkCmdPushConstants(cmd, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &pc);
            vkCmdDrawIndexed(cmd, static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);
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
        draw_shadow_scene(cmd, sun_shadow_view_proj_);
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
                    draw_shadow_scene(cmd, vp);
                    vkCmdEndRenderPass(cmd);
                }
            }
            else
            {
                if (caster.layer_base >= local_shadow_target_.framebuffers.size()) continue;
                const glm::mat4 vp = make_local_shadow_view_proj(caster);
                begin_render_pass_shadow(cmd, local_shadow_target_, caster.layer_base);
                set_viewport_scissor(cmd, local_shadow_target_.w, local_shadow_target_.h, true);
                draw_shadow_scene(cmd, vp);
                vkCmdEndRenderPass(cmd);
            }
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
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer_.buffer, &vb_off);
        vkCmdBindIndexBuffer(cmd, index_buffer_.buffer, 0, VK_INDEX_TYPE_UINT32);

        for (uint32_t i = start; i < end; ++i)
        {
            if (i >= instance_visible_mask_.size() || instance_visible_mask_[i] == 0u) continue;

            DrawPush pc{};
            pc.model = instance_models_[i];
            pc.base_color = instances_[i].base_color;
            pc.material_params = glm::vec4(
                instances_[i].metallic,
                instances_[i].roughness,
                instances_[i].ao,
                0.0f);
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DrawPush), &pc);
            vkCmdDrawIndexed(cmd, static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);
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
        if (frame_slot >= kWorkerPoolRingSize) return false;
        if (worker_idx >= worker_pools_.size()) return false;
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
        vkCmdBindDescriptorSets(out, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set_, 0, nullptr);
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
        if (frame_slot >= kWorkerPoolRingSize) return false;
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

    void record_inline_scene(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout, uint32_t w, uint32_t h)
    {
        set_viewport_scissor(cmd, w, h, true);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set_, 0, nullptr);
        draw_floor(cmd, layout);
        draw_sphere_range(cmd, layout, 0, static_cast<uint32_t>(instances_.size()));
    }

    void record_inline_depth(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout, uint32_t w, uint32_t h)
    {
        set_viewport_scissor(cmd, w, h, true);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set_, 0, nullptr);
        draw_floor(cmd, layout);
        draw_sphere_range(cmd, layout, 0, static_cast<uint32_t>(instances_.size()));
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
        const uint32_t frame_slot = static_cast<uint32_t>(frame.frame_index % kWorkerPoolRingSize);

        ensure_render_targets(fi.extent.width, fi.extent.height);
        if (pipeline_gen_ != vk_->swapchain_generation())
        {
            create_pipelines(true);
        }
        update_culling_debug_stats();

        update_frame_data(dt, t, fi.extent.width, fi.extent.height);

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

        VkMemoryBarrier shadow_to_sample{};
        shadow_to_sample.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        shadow_to_sample.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        shadow_to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            fi.cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            1,
            &shadow_to_sample,
            0,
            nullptr,
            0,
            nullptr);

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
                record_inline_depth(fi.cmd, depth_pipeline_, depth_pipeline_layout_, depth_target_.w, depth_target_.h);
                vkCmdEndRenderPass(fi.cmd);
            }
        }

        if (enable_light_culling_)
        {
            VkMemoryBarrier pre{};
            pre.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            pre.srcAccessMask = enable_depth_prepass_ ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : 0u;
            pre.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(
                fi.cmd,
                enable_depth_prepass_ ? VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1,
                &pre,
                0,
                nullptr,
                0,
                nullptr);

            if (culling_mode_ == shs::LightCullingMode::TiledDepthRange)
            {
                vkCmdBindPipeline(fi.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, depth_reduce_pipeline_);
                vkCmdBindDescriptorSets(fi.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_layout_, 0, 1, &global_set_, 0, nullptr);
                vkCmdDispatch(fi.cmd, tile_w_, tile_h_, 1);

                VkMemoryBarrier reduce_to_cull{};
                reduce_to_cull.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                reduce_to_cull.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                reduce_to_cull.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(
                    fi.cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &reduce_to_cull,
                    0,
                    nullptr,
                    0,
                    nullptr);
            }

            vkCmdBindPipeline(fi.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
            vkCmdBindDescriptorSets(fi.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_layout_, 0, 1, &global_set_, 0, nullptr);
            const uint32_t dispatch_z = (culling_mode_ == shs::LightCullingMode::Clustered) ? kClusterZSlices : 1u;
            vkCmdDispatch(fi.cmd, tile_w_, tile_h_, dispatch_z);

            VkMemoryBarrier post{};
            post.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(
                fi.cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                1,
                &post,
                0,
                nullptr,
                0,
                nullptr);
        }

        if (enable_scene_pass_)
        {
            if (!scene_secondaries.empty())
            {
                begin_render_pass_scene(fi.cmd, fi);
                vkCmdExecuteCommands(fi.cmd, static_cast<uint32_t>(scene_secondaries.size()), scene_secondaries.data());
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
                record_inline_scene(fi.cmd, scene_pipeline_, scene_pipeline_layout_, fi.extent.width, fi.extent.height);
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
        const char* cull_name = shs::light_culling_mode_name(culling_mode_);
        const char* cull_src = manual_culling_override_ ? "manual" : "tech";
        const char* rec_mode = use_multithread_recording_ ? "MT-secondary" : "inline";
        const float switch_in = std::max(0.0f, kTechniqueSwitchPeriodSec - technique_switch_accum_sec_);
        const double avg_refs = (cull_debug_list_count_ > 0)
            ? static_cast<double>(cull_debug_total_refs_) / static_cast<double>(cull_debug_list_count_)
            : 0.0;
        const uint32_t visible_draws = visible_instance_count_ + (floor_visible_ ? 1u : 0u);
        const uint32_t total_draws = static_cast<uint32_t>(instances_.size()) + 1u;

        char title[512];
        std::snprintf(
            title,
            sizeof(title),
            "%s | mode:%s | cull:%s(%s) | rec:%s | lights:%u/%u[p:%u s:%u r:%u t:%u] | shad:sun:%s spot:%u point:%u | draws:%u/%u | tile:%ux%u | refs:%llu avg:%.1f max:%u nz:%u/%u | switch:%.1fs | %.2f ms",
            kAppName,
            mode_name,
            cull_name,
            cull_src,
            rec_mode,
            visible_light_count_,
            active_light_count_,
            point_count_active_,
            spot_count_active_,
            rect_count_active_,
            tube_count_active_,
            shadow_settings_.enable ? "on" : "off",
            spot_shadow_count_,
            point_shadow_count_,
            visible_draws,
            total_draws,
            tile_w_,
            tile_h_,
            static_cast<unsigned long long>(cull_debug_total_refs_),
            avg_refs,
            cull_debug_max_list_size_,
            cull_debug_non_empty_lists_,
            cull_debug_list_count_,
            switch_in,
            avg_ms);
        SDL_SetWindowTitle(win_, title);
    }

    void handle_event(const SDL_Event& e)
    {
        if (e.type == SDL_QUIT) running_ = false;
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
                    cycle_technique_mode();
                    break;
                case SDLK_F3:
                    cycle_culling_override_mode();
                    break;
                case SDLK_F4:
                    clear_culling_override_mode();
                    break;
                case SDLK_F5:
                    shadow_settings_.enable = !shadow_settings_.enable;
                    break;
                case SDLK_MINUS:
                case SDLK_KP_MINUS:
                    if (active_light_count_ > 256) active_light_count_ -= 256;
                    break;
                case SDLK_EQUALS:
                case SDLK_PLUS:
                case SDLK_KP_PLUS:
                    active_light_count_ = std::min<uint32_t>(kMaxLights, active_light_count_ + 256);
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
            technique_switch_accum_sec_ += dt;
            if (technique_switch_accum_sec_ >= kTechniqueSwitchPeriodSec)
            {
                cycle_technique_mode();
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

    void cleanup()
    {
        if (cleaned_up_) return;
        cleaned_up_ = true;

        if (vk_ && vk_->device() != VK_NULL_HANDLE)
        {
            (void)vkDeviceWaitIdle(vk_->device());
        }

        destroy_pipelines();
        destroy_depth_target();
        destroy_layered_depth_target(sun_shadow_target_);
        destroy_layered_depth_target(local_shadow_target_);

        destroy_buffer(tile_depth_ranges_buffer_);
        destroy_buffer(tile_indices_buffer_);
        destroy_buffer(tile_counts_buffer_);
        destroy_buffer(shadow_light_buffer_);
        destroy_buffer(light_buffer_);
        destroy_buffer(camera_buffer_);
        destroy_buffer(floor_index_buffer_);
        destroy_buffer(floor_vertex_buffer_);
        destroy_buffer(index_buffer_);
        destroy_buffer(vertex_buffer_);

        destroy_worker_pools();
        jobs_.reset();

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
    }

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
    std::vector<Instance> instances_{};
    std::vector<glm::mat4> instance_models_{};
    std::vector<uint8_t> instance_visible_mask_{};
    std::vector<LightAnim> light_anim_{};
    shs::LightSet light_set_{};
    std::vector<shs::CullingLightGPU> gpu_lights_{};
    std::vector<ShadowLightGPU> shadow_lights_gpu_{};
    std::vector<LocalShadowCaster> local_shadow_casters_{};
    glm::mat4 sun_shadow_view_proj_{1.0f};
    shs::AABB sphere_local_aabb_{};
    shs::Sphere sphere_local_bound_{};
    shs::AABB floor_local_aabb_{};
    glm::mat4 floor_model_{1.0f};
    glm::vec4 floor_material_color_{1.0f};
    glm::vec4 floor_material_params_{0.0f, 0.72f, 1.0f, 0.0f};

    GpuBuffer vertex_buffer_{};
    GpuBuffer index_buffer_{};
    GpuBuffer floor_vertex_buffer_{};
    GpuBuffer floor_index_buffer_{};
    GpuBuffer camera_buffer_{};
    GpuBuffer light_buffer_{};
    GpuBuffer shadow_light_buffer_{};
    GpuBuffer tile_counts_buffer_{};
    GpuBuffer tile_indices_buffer_{};
    GpuBuffer tile_depth_ranges_buffer_{};

    CameraUBO camera_ubo_{};
    DepthTarget depth_target_{};
    LayeredDepthTarget sun_shadow_target_{};
    LayeredDepthTarget local_shadow_target_{};

    VkDescriptorSetLayout global_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet global_set_ = VK_NULL_HANDLE;
    VkSampler depth_sampler_ = VK_NULL_HANDLE;

    VkPipelineLayout shadow_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline shadow_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout depth_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline depth_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout scene_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline scene_pipeline_ = VK_NULL_HANDLE;
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
    bool use_forward_plus_ = true;
    shs::LightCullingMode culling_mode_ = shs::LightCullingMode::Tiled;
    shs::ShadowCompositionSettings shadow_settings_ = shs::make_default_shadow_composition_settings();
    bool manual_culling_override_ = false;
    shs::LightCullingMode manual_culling_mode_ = shs::LightCullingMode::Tiled;
    bool enable_depth_prepass_ = true;
    bool enable_light_culling_ = true;
    bool enable_scene_pass_ = true;
    uint64_t cull_debug_total_refs_ = 0;
    uint32_t cull_debug_non_empty_lists_ = 0;
    uint32_t cull_debug_list_count_ = 0;
    uint32_t cull_debug_max_list_size_ = 0;
    shs::TechniqueMode active_technique_ = shs::TechniqueMode::ForwardPlus;
    size_t technique_cycle_index_ = 1;
    float technique_switch_accum_sec_ = 0.0f;
    bool use_multithread_recording_ = true;
    float time_sec_ = 0.0f;
};
}

int main()
{
    try
    {
        HelloForwardPlusStressVulkanApp app{};
        app.run();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
