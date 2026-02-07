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
#include <shs/frame/technique_mode.hpp>
#include <shs/job/thread_pool_job_system.hpp>
#include <shs/job/wait_group.hpp>
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

enum class GpuCullingMode : uint32_t
{
    None = 0,
    Tiled = 1,
    TiledDepthRange = 2,
    Clustered = 3
};

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
};

struct PointLightGPU
{
    glm::vec4 pos_radius{0.0f};
    glm::vec4 color_intensity{1.0f};
};

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

struct WorkerPool
{
    VkCommandPool pool = VK_NULL_HANDLE;
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

const char* culling_mode_name(GpuCullingMode mode)
{
    switch (mode)
    {
        case GpuCullingMode::None: return "none";
        case GpuCullingMode::Tiled: return "tiled";
        case GpuCullingMode::TiledDepthRange: return "tiled-depth";
        case GpuCullingMode::Clustered: return "clustered";
    }
    return "unknown";
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
            "HelloForwardPlusStressVulkan",
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
        init.app_name = "HelloForwardPlusStressVulkan";
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

        light_anim_.clear();
        light_anim_.reserve(kMaxLights);
        lights_cpu_.resize(kMaxLights);
        std::uniform_real_distribution<float> angle0(0.0f, 6.28318f);
        std::uniform_real_distribution<float> rad(8.0f, 82.0f);
        std::uniform_real_distribution<float> hgt(1.0f, 14.0f);
        std::uniform_real_distribution<float> spd(0.15f, 1.10f);
        std::uniform_real_distribution<float> radius(7.5f, 15.0f);
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
            light_anim_.push_back(l);
        }

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
            if (vkCreateCommandPool(vk_->device(), &ci, nullptr, &worker_pools_[i].pool) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateCommandPool failed for worker");
            }
        }
    }

    void destroy_worker_pools()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        for (auto& w : worker_pools_)
        {
            if (w.pool != VK_NULL_HANDLE)
            {
                vkDestroyCommandPool(vk_->device(), w.pool, nullptr);
                w.pool = VK_NULL_HANDLE;
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
            static_cast<VkDeviceSize>(kMaxLights) * sizeof(glm::vec4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_flags,
            light_pos_radius_buffer_,
            true);

        create_buffer(
            static_cast<VkDeviceSize>(kMaxLights) * sizeof(glm::vec4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_flags,
            light_color_intensity_buffer_,
            true);
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
            VkDescriptorSetLayoutBinding b[7]{};

            b[0].binding = 0;
            b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            b[0].descriptorCount = 1;
            b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

            for (uint32_t i = 1; i < 6; ++i)
            {
                b[i].binding = i;
                b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                b[i].descriptorCount = 1;
                b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
            }
            b[6].binding = 6;
            b[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[6].descriptorCount = 1;
            b[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = 7;
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
            sizes[0].descriptorCount = 4;
            sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            sizes[1].descriptorCount = 64;
            sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sizes[2].descriptorCount = 4;

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

        VkDescriptorBufferInfo light_pos_info{};
        light_pos_info.buffer = light_pos_radius_buffer_.buffer;
        light_pos_info.offset = 0;
        light_pos_info.range = static_cast<VkDeviceSize>(kMaxLights) * sizeof(glm::vec4);

        VkDescriptorBufferInfo light_color_info{};
        light_color_info.buffer = light_color_intensity_buffer_.buffer;
        light_color_info.offset = 0;
        light_color_info.range = static_cast<VkDeviceSize>(kMaxLights) * sizeof(glm::vec4);

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

        VkDescriptorImageInfo depth_info{};
        depth_info.sampler = depth_sampler_;
        depth_info.imageView = depth_target_.view;
        depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet w[7]{};
        for (int i = 0; i < 7; ++i)
        {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = global_set_;
            w[i].dstBinding = static_cast<uint32_t>(i);
            w[i].descriptorCount = 1;
        }

        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[0].pBufferInfo = &camera_info;

        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[1].pBufferInfo = &light_pos_info;

        w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[2].pBufferInfo = &light_color_info;

        w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[3].pBufferInfo = &tile_counts_info;

        w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[4].pBufferInfo = &tile_indices_info;

        w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[5].pBufferInfo = &tile_depth_ranges_info;

        w[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[6].pImageInfo = &depth_info;

        vkUpdateDescriptorSets(vk_->device(), 7, w, 0, nullptr);
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

        const std::vector<char> scene_vs_code = read_file(SHS_VK_FP_SCENE_VERT_SPV);
        const std::vector<char> scene_fs_code = read_file(SHS_VK_FP_SCENE_FRAG_SPV);
        const std::vector<char> depth_reduce_cs_code = read_file(SHS_VK_FP_DEPTH_REDUCE_COMP_SPV);
        const std::vector<char> cull_cs_code = read_file(SHS_VK_FP_LIGHT_CULL_COMP_SPV);

        VkShaderModule scene_vs = create_shader_module(vk_->device(), scene_vs_code);
        VkShaderModule scene_fs = create_shader_module(vk_->device(), scene_fs_code);
        VkShaderModule depth_reduce_cs = create_shader_module(vk_->device(), depth_reduce_cs_code);
        VkShaderModule cull_cs = create_shader_module(vk_->device(), cull_cs_code);

        const auto cleanup_modules = [&]() {
            if (scene_vs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), scene_vs, nullptr);
            if (scene_fs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), scene_fs, nullptr);
            if (depth_reduce_cs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), depth_reduce_cs, nullptr);
            if (cull_cs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), cull_cs, nullptr);
        };

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

        GpuCullingMode mode_hint = GpuCullingMode::None;
        switch (mode)
        {
            case shs::TechniqueMode::ForwardPlus:
                mode_hint = GpuCullingMode::Tiled;
                break;
            case shs::TechniqueMode::TiledDeferred:
                mode_hint = GpuCullingMode::TiledDepthRange;
                break;
            case shs::TechniqueMode::ClusteredForward:
                mode_hint = GpuCullingMode::Clustered;
                break;
            case shs::TechniqueMode::Forward:
            case shs::TechniqueMode::Deferred:
            default:
                mode_hint = GpuCullingMode::None;
                break;
        }
        if (!enable_light_culling_) mode_hint = GpuCullingMode::None;
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

        use_forward_plus_ = (culling_mode_ != GpuCullingMode::None);
        technique_switch_accum_sec_ = 0.0f;
    }

    void cycle_technique_mode()
    {
        const auto& modes = known_technique_modes();
        if (modes.empty()) return;
        technique_cycle_index_ = (technique_cycle_index_ + 1u) % modes.size();
        apply_technique_mode(modes[technique_cycle_index_]);
    }

    void update_frame_data(float dt, float t, uint32_t w, uint32_t h)
    {
        (void)dt;

        const float aspect = (h > 0) ? (static_cast<float>(w) / static_cast<float>(h)) : 1.0f;
        const float orbit_r = 68.0f;
        const glm::vec3 cam_pos = glm::vec3(std::sin(t * 0.22f) * orbit_r, 26.0f + std::sin(t * 0.35f) * 5.0f, std::cos(t * 0.22f) * orbit_r);
        const glm::vec3 cam_target = glm::vec3(0.0f, 2.0f, 0.0f);

        camera_ubo_.view = glm::lookAt(cam_pos, cam_target, glm::vec3(0.0f, 1.0f, 0.0f));
        camera_ubo_.proj = glm::perspective(glm::radians(62.0f), aspect, 0.1f, 260.0f);
        camera_ubo_.view_proj = camera_ubo_.proj * camera_ubo_.view;
        camera_ubo_.camera_pos_time = glm::vec4(cam_pos, t);
        camera_ubo_.sun_dir_intensity = glm::vec4(glm::normalize(glm::vec3(-0.35f, -1.0f, -0.18f)), 1.65f);
        camera_ubo_.screen_tile_lightcount = glm::uvec4(w, h, tile_w_, active_light_count_);
        camera_ubo_.params = glm::uvec4(tile_h_, kMaxLightsPerTile, kTileSize, static_cast<uint32_t>(culling_mode_));
        camera_ubo_.culling_params = glm::uvec4(kClusterZSlices, 0u, 0u, 0u);
        camera_ubo_.depth_params = glm::vec4(0.1f, 260.0f, 0.0f, 0.0f);
        camera_ubo_.exposure_gamma = glm::vec4(1.4f, 2.2f, 0.0f, 0.0f);

        std::memcpy(camera_buffer_.mapped, &camera_ubo_, sizeof(CameraUBO));

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

        glm::vec4* pos_out = reinterpret_cast<glm::vec4*>(light_pos_radius_buffer_.mapped);
        glm::vec4* color_out = reinterpret_cast<glm::vec4*>(light_color_intensity_buffer_.mapped);

        const uint32_t lc = std::min<uint32_t>(active_light_count_, static_cast<uint32_t>(light_anim_.size()));
        for (uint32_t i = 0; i < lc; ++i)
        {
            const LightAnim& la = light_anim_[i];
            const float a = la.angle0 + la.speed * t;
            const float y = la.height + std::sin(a * 1.7f + la.phase) * 2.6f;
            const glm::vec3 p(std::cos(a) * la.orbit_radius, y, std::sin(a) * la.orbit_radius);
            pos_out[i] = glm::vec4(p, la.range);
            color_out[i] = glm::vec4(la.color, la.intensity);
        }
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

    void draw_floor(VkCommandBuffer cmd, VkPipelineLayout layout)
    {
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
        uint32_t worker_idx,
        uint32_t start,
        uint32_t end,
        bool draw_floor_here,
        VkCommandBuffer& out)
    {
        out = VK_NULL_HANDLE;
        if (start >= end && !draw_floor_here) return true;

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = worker_pools_[worker_idx].pool;
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
        std::vector<VkCommandBuffer>& out)
    {
        out.clear();

        if (!use_multithread_recording_ || !jobs_ || worker_pools_.empty() || instances_.empty())
        {
            return true;
        }

        const uint32_t workers = std::min<uint32_t>(static_cast<uint32_t>(worker_pools_.size()), static_cast<uint32_t>(instances_.size()));
        if (workers <= 1) return true;

        for (uint32_t i = 0; i < workers; ++i)
        {
            vkResetCommandPool(vk_->device(), worker_pools_[i].pool, 0);
        }

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
                if (!record_secondary_batch(rp, fb, pipeline, layout, w, h, flip_y, wi, start, end, draw_floor_here, tmp[wi]))
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

        ensure_render_targets(fi.extent.width, fi.extent.height);
        if (pipeline_gen_ != vk_->swapchain_generation())
        {
            create_pipelines(true);
        }

        update_frame_data(dt, t, fi.extent.width, fi.extent.height);

        std::vector<VkCommandBuffer> depth_secondaries{};
        std::vector<VkCommandBuffer> scene_secondaries{};
        if (use_multithread_recording_)
        {
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

            if (culling_mode_ == GpuCullingMode::TiledDepthRange)
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
            const uint32_t dispatch_z = (culling_mode_ == GpuCullingMode::Clustered) ? kClusterZSlices : 1u;
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
        const char* cull_name = culling_mode_name(culling_mode_);
        const char* rec_mode = use_multithread_recording_ ? "MT-secondary" : "inline";
        const float switch_in = std::max(0.0f, kTechniqueSwitchPeriodSec - technique_switch_accum_sec_);

        char title[512];
        std::snprintf(
            title,
            sizeof(title),
            "HelloForwardPlusStressVulkan | mode:%s | cull:%s | rec:%s | lights:%u | draws:%zu | tile:%ux%u | switch:%.1fs | %.2f ms",
            mode_name,
            cull_name,
            rec_mode,
            active_light_count_,
            instances_.size() + 1u,
            tile_w_,
            tile_h_,
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

        destroy_buffer(tile_depth_ranges_buffer_);
        destroy_buffer(tile_indices_buffer_);
        destroy_buffer(tile_counts_buffer_);
        destroy_buffer(light_color_intensity_buffer_);
        destroy_buffer(light_pos_radius_buffer_);
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
        float angle0 = 0.0f;
        float orbit_radius = 10.0f;
        float height = 6.0f;
        float speed = 1.0f;
        float range = 6.0f;
        float phase = 0.0f;
        glm::vec3 color{1.0f};
        float intensity = 2.0f;
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
    std::vector<LightAnim> light_anim_{};
    std::vector<PointLightGPU> lights_cpu_{};
    glm::mat4 floor_model_{1.0f};
    glm::vec4 floor_material_color_{1.0f};
    glm::vec4 floor_material_params_{0.0f, 0.72f, 1.0f, 0.0f};

    GpuBuffer vertex_buffer_{};
    GpuBuffer index_buffer_{};
    GpuBuffer floor_vertex_buffer_{};
    GpuBuffer floor_index_buffer_{};
    GpuBuffer camera_buffer_{};
    GpuBuffer light_pos_radius_buffer_{};
    GpuBuffer light_color_intensity_buffer_{};
    GpuBuffer tile_counts_buffer_{};
    GpuBuffer tile_indices_buffer_{};
    GpuBuffer tile_depth_ranges_buffer_{};

    CameraUBO camera_ubo_{};
    DepthTarget depth_target_{};

    VkDescriptorSetLayout global_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet global_set_ = VK_NULL_HANDLE;
    VkSampler depth_sampler_ = VK_NULL_HANDLE;

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
    bool use_forward_plus_ = true;
    GpuCullingMode culling_mode_ = GpuCullingMode::Tiled;
    bool enable_depth_prepass_ = true;
    bool enable_light_culling_ = true;
    bool enable_scene_pass_ = true;
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
