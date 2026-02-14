#define SDL_MAIN_HANDLED

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <SDL2/SDL.h>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/camera/camera_math.hpp>
#include <shs/camera/convention.hpp>
#include <shs/camera/light_camera.hpp>
#include <shs/core/context.hpp>
#include <shs/geometry/culling_runtime.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/job/thread_pool_job_system.hpp>
#include <shs/job/wait_group.hpp>
#include <shs/platform/platform_input.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>
#include <shs/rhi/drivers/vulkan/vk_cmd_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_memory_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_shader_utils.hpp>
#include <shs/scene/scene_culling.hpp>
#include <shs/scene/scene_elements.hpp>

using namespace shs;

#ifndef SHS_VK_CULLING_VERT_SPV
#error "SHS_VK_CULLING_VERT_SPV is not defined"
#endif
#ifndef SHS_VK_CULLING_FRAG_SPV
#error "SHS_VK_CULLING_FRAG_SPV is not defined"
#endif
#ifndef SHS_VK_SOFT_SHADOW_CULLING_VERT_SPV
#error "SHS_VK_SOFT_SHADOW_CULLING_VERT_SPV is not defined"
#endif
#ifndef SHS_VK_SOFT_SHADOW_CULLING_FRAG_SPV
#error "SHS_VK_SOFT_SHADOW_CULLING_FRAG_SPV is not defined"
#endif
#ifndef SHS_VK_PB_SHADOW_VERT_SPV
#error "SHS_VK_PB_SHADOW_VERT_SPV is not defined"
#endif

namespace
{
constexpr int kWindowW = 1200;
constexpr int kWindowH = 900;
// Vulkan backend currently runs with max_frames_in_flight = 1, so keep ring resources in lockstep.
constexpr uint32_t kFrameRing = 1u;
constexpr uint32_t kShadowMapSize = 2048u;
constexpr float kSunHeightLift = 6.0f;
constexpr float kShadowStrength = 0.75f;
constexpr float kShadowBiasConst = 0.0010f;
constexpr float kShadowBiasSlope = 0.0020f;
constexpr float kShadowPcfStep = 1.0f;
constexpr int kShadowPcfRadius = 2;
constexpr float kShadowRangeScale = 50.0f;
const glm::vec3 kFloorBaseColor(0.30f, 0.30f, 0.35f);
constexpr uint8_t kOcclusionHideConfirmFrames = 3u;
constexpr uint8_t kOcclusionShowConfirmFrames = 2u;
constexpr uint64_t kOcclusionMinVisibleSamples = 1u;
constexpr uint32_t kOcclusionWarmupFramesAfterCameraMove = 2u;
constexpr uint32_t kMaxRecordingWorkers = 8u;

struct Vertex
{
    glm::vec3 pos{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
};

struct alignas(16) CameraUBO
{
    glm::mat4 view_proj{1.0f};
    glm::vec4 camera_pos{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 light_dir_ws{0.0f, -1.0f, 0.0f, 0.0f};
    glm::mat4 light_view_proj{1.0f};
    glm::vec4 shadow_params{kShadowStrength, kShadowBiasConst, kShadowBiasSlope, kShadowPcfStep};
    glm::vec4 shadow_misc{static_cast<float>(kShadowPcfRadius), 0.0f, 0.0f, 0.0f};
};

struct alignas(16) DrawPush
{
    glm::mat4 model{1.0f};
    glm::vec4 base_color{1.0f};
    glm::uvec4 mode_pad{0u, 0u, 0u, 0u};
};

struct alignas(16) ShadowPush
{
    glm::mat4 light_mvp{1.0f};
};

struct GpuBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

struct MeshGPU
{
    GpuBuffer vertex{};
    GpuBuffer tri_indices{};
    GpuBuffer line_indices{};
    uint32_t tri_index_count = 0;
    uint32_t line_index_count = 0;
};

struct WorkerPool
{
    std::array<VkCommandPool, kFrameRing> pools{};
};

struct DepthTarget
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

struct ShapeInstance
{
    SceneShape shape;
    uint32_t mesh_index = 0;
    glm::vec3 color{1.0f};
    glm::vec3 base_pos{0.0f};
    glm::vec3 base_rot{0.0f};
    glm::vec3 angular_vel{0.0f};
    glm::mat4 model{1.0f};
    bool visible = true;
    bool frustum_visible = true;
    bool occluded = false;
    bool animated = true;
    bool casts_shadow = true;
};

struct FreeCamera
{
    glm::vec3 pos{0.0f, 14.0f, -28.0f};
    float yaw = glm::half_pi<float>();
    float pitch = -0.25f;
    float move_speed = 20.0f;
    float look_speed = 0.003f;

    void update(const PlatformInputState& input, float dt)
    {
        if (input.right_mouse_down)
        {
            yaw -= input.mouse_dx * look_speed;
            pitch -= input.mouse_dy * look_speed;
            pitch = std::clamp(pitch, -glm::half_pi<float>() + 0.01f, glm::half_pi<float>() - 0.01f);
        }

        const glm::vec3 fwd = forward_from_yaw_pitch(yaw, pitch);
        const glm::vec3 right = right_from_forward(fwd);
        const glm::vec3 up(0.0f, 1.0f, 0.0f);

        const float speed = move_speed * (input.boost ? 2.0f : 1.0f);
        if (input.forward) pos += fwd * speed * dt;
        if (input.backward) pos -= fwd * speed * dt;
        if (input.left) pos += right * speed * dt;
        if (input.right) pos -= right * speed * dt;
        if (input.ascend) pos += up * speed * dt;
        if (input.descend) pos -= up * speed * dt;
    }

    glm::mat4 view_matrix() const
    {
        return look_at_lh(pos, pos + forward_from_yaw_pitch(yaw, pitch), glm::vec3(0.0f, 1.0f, 0.0f));
    }
};

inline glm::mat4 compose_model(const glm::vec3& pos, const glm::vec3& rot_euler)
{
    glm::mat4 model(1.0f);
    model = glm::translate(model, pos);
    model = glm::rotate(model, rot_euler.x, glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, rot_euler.y, glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, rot_euler.z, glm::vec3(0.0f, 0.0f, 1.0f));
    return model;
}

inline std::vector<uint32_t> make_line_indices_from_triangles(const std::vector<uint32_t>& tri_indices)
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

inline std::vector<Vertex> make_vertices_with_normals(const DebugMesh& mesh)
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
        // Mesh winding follows LH + clockwise front faces, so flip RH cross order.
        glm::vec3 n = glm::cross(p2 - p0, p1 - p0);
        const float n2 = glm::dot(n, n);
        if (n2 <= 1e-12f) n = glm::vec3(0.0f, 1.0f, 0.0f);
        else n *= 1.0f / std::sqrt(n2);

        verts[i0].normal += n;
        verts[i1].normal += n;
        verts[i2].normal += n;
    }

    for (auto& v : verts)
    {
        const float n2 = glm::dot(v.normal, v.normal);
        if (n2 <= 1e-12f) v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        else v.normal *= 1.0f / std::sqrt(n2);
    }

    return verts;
}

AABB compute_shadow_caster_bounds(const std::vector<ShapeInstance>& instances)
{
    AABB out{};
    bool any = false;
    for (const auto& inst : instances)
    {
        if (!inst.casts_shadow) continue;
        const AABB box = inst.shape.world_aabb();
        if (!any)
        {
            out.minv = box.minv;
            out.maxv = box.maxv;
            any = true;
            continue;
        }
        out.expand(box.minv);
        out.expand(box.maxv);
    }
    if (!any)
    {
        out.minv = glm::vec3(-1.0f);
        out.maxv = glm::vec3(1.0f);
    }
    return out;
}

AABB scale_aabb_about_center(const AABB& src, float scale)
{
    const float s = std::max(scale, 1.0f);
    const glm::vec3 c = src.center();
    const glm::vec3 e = src.extent() * s;
    AABB out{};
    out.minv = c - e;
    out.maxv = c + e;
    return out;
}

enum class DemoShapeKind : uint8_t
{
    Sphere = 0,
    Box = 1,
    Capsule = 2,
    Cylinder = 3,
    TaperedCapsule = 4,
    ConvexHull = 5,
    Mesh = 6,
    ConvexFromMesh = 7,
    PointLightVolume = 8,
    SpotLightVolume = 9,
    RectLightVolume = 10,
    TubeLightVolume = 11
};

float pseudo_random01(uint32_t seed)
{
    uint32_t x = seed;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (float)(x & 0x00ffffffu) / (float)0x01000000u;
}

std::vector<glm::vec3> scaled_custom_hull(float s)
{
    return {
        {-0.8f * s, -0.7f * s, -0.4f * s},
        { 0.9f * s, -0.6f * s, -0.5f * s},
        { 1.0f * s,  0.4f * s, -0.1f * s},
        {-0.7f * s,  0.6f * s, -0.2f * s},
        {-0.3f * s, -0.4f * s,  0.9f * s},
        { 0.4f * s,  0.7f * s,  0.8f * s},
    };
}

MeshData scaled_wedge_mesh(float s)
{
    MeshData wedge_mesh{};
    wedge_mesh.positions = {
        {-0.9f * s, -0.6f * s, -0.6f * s},
        { 0.9f * s, -0.6f * s, -0.6f * s},
        { 0.0f * s,  0.8f * s, -0.6f * s},
        {-0.9f * s, -0.6f * s,  0.6f * s},
        { 0.9f * s, -0.6f * s,  0.6f * s},
        { 0.0f * s,  0.8f * s,  0.6f * s},
    };
    wedge_mesh.indices = {
        0, 1, 2,
        5, 4, 3,
        0, 3, 4, 0, 4, 1,
        1, 4, 5, 1, 5, 2,
        2, 5, 3, 2, 3, 0
    };
    return wedge_mesh;
}

glm::vec3 color_for_demo_shape_kind(DemoShapeKind kind)
{
    switch (kind)
    {
        case DemoShapeKind::Sphere: return {0.95f, 0.35f, 0.35f};
        case DemoShapeKind::Box: return {0.35f, 0.90f, 0.45f};
        case DemoShapeKind::Capsule: return {0.35f, 0.55f, 0.95f};
        case DemoShapeKind::Cylinder: return {0.95f, 0.80f, 0.30f};
        case DemoShapeKind::TaperedCapsule: return {0.80f, 0.40f, 0.95f};
        case DemoShapeKind::ConvexHull: return {0.30f, 0.85f, 0.90f};
        case DemoShapeKind::Mesh: return {0.92f, 0.55f, 0.25f};
        case DemoShapeKind::ConvexFromMesh: return {0.55f, 0.95f, 0.55f};
        case DemoShapeKind::PointLightVolume: return {0.95f, 0.45f, 0.65f};
        case DemoShapeKind::SpotLightVolume: return {0.95f, 0.70f, 0.35f};
        case DemoShapeKind::RectLightVolume: return {0.35f, 0.95f, 0.80f};
        case DemoShapeKind::TubeLightVolume: return {0.70f, 0.65f, 0.95f};
    }
    return {0.9f, 0.9f, 0.9f};
}

JPH::ShapeRefC make_scaled_demo_shape(DemoShapeKind kind, float s)
{
    const float ss = std::max(s, 0.25f);
    switch (kind)
    {
        case DemoShapeKind::Sphere:
            return jolt::make_sphere(1.0f * ss);
        case DemoShapeKind::Box:
            return jolt::make_box(glm::vec3(0.9f, 0.7f, 0.6f) * ss);
        case DemoShapeKind::Capsule:
            return jolt::make_capsule(0.9f * ss, 0.45f * ss);
        case DemoShapeKind::Cylinder:
            return jolt::make_cylinder(0.9f * ss, 0.5f * ss);
        case DemoShapeKind::TaperedCapsule:
            return jolt::make_tapered_capsule(0.9f * ss, 0.25f * ss, 0.65f * ss);
        case DemoShapeKind::ConvexHull:
            return jolt::make_convex_hull(scaled_custom_hull(ss));
        case DemoShapeKind::Mesh:
            return jolt::make_mesh_shape(scaled_wedge_mesh(ss));
        case DemoShapeKind::ConvexFromMesh:
            return jolt::make_convex_hull_from_mesh(scaled_wedge_mesh(ss));
        case DemoShapeKind::PointLightVolume:
            return jolt::make_point_light_volume(1.0f * ss);
        case DemoShapeKind::SpotLightVolume:
            return jolt::make_spot_light_volume(1.8f * ss, glm::radians(28.0f), 20);
        case DemoShapeKind::RectLightVolume:
            return jolt::make_rect_area_light_volume(glm::vec2(0.8f, 0.5f) * ss, 2.0f * ss);
        case DemoShapeKind::TubeLightVolume:
            return jolt::make_tube_area_light_volume(0.9f * ss, 0.35f * ss);
    }
    return jolt::make_sphere(1.0f * ss);
}

DebugMesh make_tessellated_floor_mesh(float half_extent, int subdivisions)
{
    DebugMesh mesh{};
    const int div = std::max(1, subdivisions);
    const int verts_per_row = div + 1;
    const float full = std::max(half_extent, 1.0f) * 2.0f;
    const float step = full / (float)div;

    mesh.vertices.reserve((size_t)verts_per_row * (size_t)verts_per_row);
    mesh.indices.reserve((size_t)div * (size_t)div * 6u);

    for (int z = 0; z <= div; ++z)
    {
        for (int x = 0; x <= div; ++x)
        {
            const float px = -half_extent + (float)x * step;
            const float pz = -half_extent + (float)z * step;
            mesh.vertices.push_back(glm::vec3(px, 0.0f, pz));
        }
    }

    const auto idx_of = [verts_per_row](int x, int z) -> uint32_t {
        return (uint32_t)(z * verts_per_row + x);
    };

    for (int z = 0; z < div; ++z)
    {
        for (int x = 0; x < div; ++x)
        {
            const uint32_t i00 = idx_of(x + 0, z + 0);
            const uint32_t i10 = idx_of(x + 1, z + 0);
            const uint32_t i01 = idx_of(x + 0, z + 1);
            const uint32_t i11 = idx_of(x + 1, z + 1);

            mesh.indices.push_back(i00);
            mesh.indices.push_back(i10);
            mesh.indices.push_back(i11);

            mesh.indices.push_back(i00);
            mesh.indices.push_back(i11);
            mesh.indices.push_back(i01);
        }
    }

    return mesh;
}

class HelloSoftShadowCullingVkApp
{
public:
    ~HelloSoftShadowCullingVkApp()
    {
        cleanup();
    }

    void run()
    {
        jolt::init_jolt();
        init_sdl();
        init_backend();
        configure_recording_workers();
        create_worker_pools();
        create_upload_resources();
        create_descriptor_resources();
        create_scene();
        create_occlusion_query_resources();
        create_pipelines();
        main_loop();
        jolt::shutdown_jolt();
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
            "Soft Shadow Culling Demo (Vulkan)",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            kWindowW,
            kWindowH,
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        if (!win_)
        {
            throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
        }
    }

    void init_backend()
    {
        RenderBackendCreateResult created = create_render_backend(RenderBackendType::Vulkan);
        if (!created.note.empty()) std::fprintf(stderr, "[shs] %s\n", created.note.c_str());
        if (!created.backend) throw std::runtime_error("Backend factory did not return backend");

        keep_.push_back(std::move(created.backend));
        for (auto& aux : created.auxiliary_backends)
        {
            if (aux) keep_.push_back(std::move(aux));
        }
        for (auto& b : keep_)
        {
            ctx_.register_backend(b.get());
        }

        vk_ = dynamic_cast<VulkanRenderBackend*>(ctx_.backend(RenderBackendType::Vulkan));
        if (!vk_) throw std::runtime_error("Vulkan backend unavailable");

        int dw = 0;
        int dh = 0;
        SDL_Vulkan_GetDrawableSize(win_, &dw, &dh);
        if (dw <= 0 || dh <= 0)
        {
            dw = kWindowW;
            dh = kWindowH;
        }

        VulkanRenderBackend::InitDesc init{};
        init.window = win_;
        init.width = dw;
        init.height = dh;
        init.enable_validation = false;
        init.app_name = "hello_soft_shadow_culling_vk";
        if (!vk_->init(init)) throw std::runtime_error("Vulkan init failed");

        ctx_.set_primary_backend(vk_);
    }

    void configure_recording_workers()
    {
        uint32_t hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 2u;
        worker_count_ = std::clamp<uint32_t>(hc > 1u ? hc - 1u : 1u, 1u, kMaxRecordingWorkers);
        use_multithread_recording_ = worker_count_ > 1u;
        if (use_multithread_recording_)
        {
            jobs_ = std::make_unique<ThreadPoolJobSystem>(worker_count_);
        }
        else
        {
            jobs_.reset();
        }
    }

    void create_worker_pools()
    {
        destroy_worker_pools();
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        if (!use_multithread_recording_) return;

        worker_pools_.resize(worker_count_);
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = vk_->graphics_queue_family_index();

        for (uint32_t i = 0; i < worker_count_; ++i)
        {
            for (uint32_t ring = 0; ring < kFrameRing; ++ring)
            {
                if (vkCreateCommandPool(vk_->device(), &ci, nullptr, &worker_pools_[i].pools[ring]) != VK_SUCCESS)
                {
                    throw std::runtime_error("vkCreateCommandPool(worker) failed");
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

    void create_upload_resources()
    {
        destroy_upload_resources();
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;

        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = vk_->graphics_queue_family_index();
        if (vkCreateCommandPool(vk_->device(), &ci, nullptr, &upload_pool_) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateCommandPool(upload) failed");
        }

        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(vk_->device(), &fi, nullptr, &upload_fence_) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateFence(upload) failed");
        }
    }

    void destroy_upload_resources()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        if (upload_fence_ != VK_NULL_HANDLE)
        {
            vkDestroyFence(vk_->device(), upload_fence_, nullptr);
            upload_fence_ = VK_NULL_HANDLE;
        }
        if (upload_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(vk_->device(), upload_pool_, nullptr);
            upload_pool_ = VK_NULL_HANDLE;
        }
    }

    void create_buffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags mem_props,
        GpuBuffer& out,
        bool map_memory)
    {
        destroy_buffer(out);
        if (!vk_create_buffer(
                vk_->device(),
                vk_->physical_device(),
                size,
                usage,
                mem_props,
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
                vk_destroy_buffer(vk_->device(), out.buffer, out.memory);
                out.size = 0;
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
        vk_destroy_buffer(vk_->device(), b.buffer, b.memory);
        b.size = 0;
    }

    void copy_buffer_once(VkBuffer src, VkBuffer dst, VkDeviceSize size)
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        if (upload_pool_ == VK_NULL_HANDLE || upload_fence_ == VK_NULL_HANDLE)
        {
            throw std::runtime_error("upload resources are not initialized");
        }

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = upload_pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(vk_->device(), &ai, &cmd) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateCommandBuffers(upload) failed");
        }

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
        {
            vkFreeCommandBuffers(vk_->device(), upload_pool_, 1, &cmd);
            throw std::runtime_error("vkBeginCommandBuffer(upload) failed");
        }

        VkBufferCopy copy{};
        copy.srcOffset = 0;
        copy.dstOffset = 0;
        copy.size = size;
        vkCmdCopyBuffer(cmd, src, dst, 1, &copy);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        {
            vkFreeCommandBuffers(vk_->device(), upload_pool_, 1, &cmd);
            throw std::runtime_error("vkEndCommandBuffer(upload) failed");
        }

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        if (vkQueueSubmit(vk_->graphics_queue(), 1, &si, upload_fence_) != VK_SUCCESS)
        {
            vkFreeCommandBuffers(vk_->device(), upload_pool_, 1, &cmd);
            throw std::runtime_error("vkQueueSubmit(upload) failed");
        }
        if (vkWaitForFences(vk_->device(), 1, &upload_fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        {
            vkFreeCommandBuffers(vk_->device(), upload_pool_, 1, &cmd);
            throw std::runtime_error("vkWaitForFences(upload) failed");
        }
        (void)vkResetFences(vk_->device(), 1, &upload_fence_);
        vkFreeCommandBuffers(vk_->device(), upload_pool_, 1, &cmd);
    }

    void upload_static_device_buffer(
        const void* src_data,
        VkDeviceSize src_size,
        VkBufferUsageFlags dst_usage,
        GpuBuffer& out)
    {
        if (src_data == nullptr || src_size == 0)
        {
            throw std::runtime_error("upload_static_device_buffer: empty source");
        }

        GpuBuffer staging{};
        const VkMemoryPropertyFlags host_mem =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        create_buffer(
            src_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            host_mem,
            staging,
            true);
        std::memcpy(staging.mapped, src_data, static_cast<size_t>(src_size));

        create_buffer(
            src_size,
            dst_usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            out,
            false);

        copy_buffer_once(staging.buffer, out.buffer, src_size);
        destroy_buffer(staging);
    }

    uint32_t upload_debug_mesh(const DebugMesh& mesh)
    {
        if (mesh.vertices.empty() || mesh.indices.empty())
        {
            throw std::runtime_error("upload_debug_mesh: mesh is empty");
        }

        MeshGPU gpu{};
        const auto vertices = make_vertices_with_normals(mesh);
        const auto line_indices = make_line_indices_from_triangles(mesh.indices);
        upload_static_device_buffer(
            vertices.data(),
            static_cast<VkDeviceSize>(vertices.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            gpu.vertex);
        upload_static_device_buffer(
            mesh.indices.data(),
            static_cast<VkDeviceSize>(mesh.indices.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            gpu.tri_indices);
        upload_static_device_buffer(
            line_indices.data(),
            static_cast<VkDeviceSize>(line_indices.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            gpu.line_indices);

        gpu.tri_index_count = static_cast<uint32_t>(mesh.indices.size());
        gpu.line_index_count = static_cast<uint32_t>(line_indices.size());

        meshes_.push_back(gpu);
        return static_cast<uint32_t>(meshes_.size() - 1u);
    }

    void create_scene()
    {
        instances_.clear();
        view_cull_scene_.clear();
        shadow_cull_scene_.clear();

        // Large floor
        {
            ShapeInstance floor{};
            floor.shape.shape = jolt::make_box(glm::vec3(120.0f, 0.1f, 120.0f));
            floor.base_pos = glm::vec3(0.0f, -0.2f, 0.0f);
            floor.base_rot = glm::vec3(0.0f);
            floor.model = compose_model(floor.base_pos, floor.base_rot);
            floor.shape.transform = jolt::to_jph(floor.model);
            floor.shape.stable_id = 9000;
            floor.color = kFloorBaseColor;
            floor.animated = false;
            floor.casts_shadow = true;
            floor.mesh_index = upload_debug_mesh(make_tessellated_floor_mesh(120.0f, 96));
            instances_.push_back(floor);
        }

        const std::array<DemoShapeKind, 12> shape_kinds = {
            DemoShapeKind::Sphere,
            DemoShapeKind::Box,
            DemoShapeKind::Capsule,
            DemoShapeKind::Cylinder,
            DemoShapeKind::TaperedCapsule,
            DemoShapeKind::ConvexHull,
            DemoShapeKind::Mesh,
            DemoShapeKind::ConvexFromMesh,
            DemoShapeKind::PointLightVolume,
            DemoShapeKind::SpotLightVolume,
            DemoShapeKind::RectLightVolume,
            DemoShapeKind::TubeLightVolume
        };

        uint32_t next_id = 1u;
        const int layer_count = 3;
        const int rows_per_layer = 8;
        const int cols_per_row = 10;
        const float col_spacing_x = 5.2f;
        const float row_spacing_z = 4.6f;
        const float layer_spacing_z = 24.0f;
        const float base_y = 1.3f;
        const float layer_y_step = 0.9f;

        for (int layer = 0; layer < layer_count; ++layer)
        {
            const float layer_z = (-0.5f * (float)(layer_count - 1) + (float)layer) * layer_spacing_z;
            for (int row = 0; row < rows_per_layer; ++row)
            {
                const float row_z = layer_z + (-0.5f * (float)(rows_per_layer - 1) + (float)row) * row_spacing_z;
                const float zig = (((row + layer) & 1) != 0) ? (0.42f * col_spacing_x) : 0.0f;
                for (int col = 0; col < cols_per_row; ++col)
                {
                    const uint32_t logical_idx =
                        (uint32_t)layer * (uint32_t)(rows_per_layer * cols_per_row) +
                        (uint32_t)row * (uint32_t)cols_per_row +
                        (uint32_t)col;
                    const DemoShapeKind kind = shape_kinds[(logical_idx * 7u + 3u) % shape_kinds.size()];
                    const float scale = 0.58f + 1.02f * pseudo_random01(logical_idx * 1664525u + 1013904223u);

                    ShapeInstance inst{};
                    inst.shape.shape = make_scaled_demo_shape(kind, scale);
                    inst.mesh_index = upload_debug_mesh(debug_mesh_from_shape(*inst.shape.shape, JPH::Mat44::sIdentity()));
                    inst.base_pos = glm::vec3(
                        (-0.5f * (float)(cols_per_row - 1) + (float)col) * col_spacing_x + zig,
                        base_y + layer_y_step * (float)layer + 0.22f * (float)(col % 3),
                        row_z);
                    inst.base_rot = glm::vec3(
                        0.21f * pseudo_random01(logical_idx * 279470273u + 1u),
                        0.35f * pseudo_random01(logical_idx * 2246822519u + 7u),
                        0.19f * pseudo_random01(logical_idx * 3266489917u + 11u));
                    inst.angular_vel = glm::vec3(
                        0.20f + 0.26f * pseudo_random01(logical_idx * 747796405u + 13u),
                        0.18f + 0.24f * pseudo_random01(logical_idx * 2891336453u + 17u),
                        0.16f + 0.21f * pseudo_random01(logical_idx * 1181783497u + 19u));
                    inst.model = compose_model(inst.base_pos, inst.base_rot);
                    inst.shape.transform = jolt::to_jph(inst.model);
                    inst.shape.stable_id = next_id++;
                    inst.color = color_for_demo_shape_kind(kind);
                    inst.animated = true;
                    inst.casts_shadow = true;
                    instances_.push_back(std::move(inst));
                }
            }
        }

        // Unit cube for AABB wire overlay.
        {
            AABB unit{};
            unit.minv = glm::vec3(-0.5f);
            unit.maxv = glm::vec3(0.5f);
            aabb_mesh_index_ = upload_debug_mesh(debug_mesh_from_aabb(unit));
        }

        view_cull_scene_.reserve(instances_.size());
        shadow_cull_scene_.reserve(instances_.size());
        for (size_t i = 0; i < instances_.size(); ++i)
        {
            SceneElement view_elem{};
            view_elem.geometry = instances_[i].shape;
            view_elem.user_index = static_cast<uint32_t>(i);
            view_elem.visible = instances_[i].visible;
            view_elem.frustum_visible = instances_[i].frustum_visible;
            view_elem.occluded = instances_[i].occluded;
            view_elem.casts_shadow = instances_[i].casts_shadow;
            view_cull_scene_.add(std::move(view_elem));

            SceneElement shadow_elem{};
            shadow_elem.geometry = instances_[i].shape;
            shadow_elem.user_index = static_cast<uint32_t>(i);
            shadow_elem.visible = true;
            shadow_elem.frustum_visible = true;
            shadow_elem.occluded = false;
            shadow_elem.casts_shadow = instances_[i].casts_shadow;
            shadow_elem.enabled = instances_[i].casts_shadow;
            shadow_cull_scene_.add(std::move(shadow_elem));
        }
    }

    void create_occlusion_query_resources()
    {
        destroy_occlusion_query_resources();
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;

        max_view_query_count_ = std::max<uint32_t>(1u, static_cast<uint32_t>(view_cull_scene_.size()));
        max_shadow_query_count_ = std::max<uint32_t>(1u, static_cast<uint32_t>(shadow_cull_scene_.size()));
        for (uint32_t i = 0; i < kFrameRing; ++i)
        {
            VkQueryPoolCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            ci.queryType = VK_QUERY_TYPE_OCCLUSION;
            ci.queryCount = max_view_query_count_;
            if (vkCreateQueryPool(vk_->device(), &ci, nullptr, &view_query_pools_[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateQueryPool(view) failed");
            }
            ci.queryCount = max_shadow_query_count_;
            if (vkCreateQueryPool(vk_->device(), &ci, nullptr, &shadow_query_pools_[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateQueryPool(shadow) failed");
            }
            view_query_counts_[i] = 0;
            shadow_query_counts_[i] = 0;
            view_query_scene_indices_[i].clear();
            shadow_query_scene_indices_[i].clear();
        }
    }

    void destroy_occlusion_query_resources()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        for (uint32_t i = 0; i < kFrameRing; ++i)
        {
            if (view_query_pools_[i] != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(vk_->device(), view_query_pools_[i], nullptr);
                view_query_pools_[i] = VK_NULL_HANDLE;
            }
            if (shadow_query_pools_[i] != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(vk_->device(), shadow_query_pools_[i], nullptr);
                shadow_query_pools_[i] = VK_NULL_HANDLE;
            }
            view_query_counts_[i] = 0;
            shadow_query_counts_[i] = 0;
            view_query_scene_indices_[i].clear();
            shadow_query_scene_indices_[i].clear();
        }
        max_view_query_count_ = 0;
        max_shadow_query_count_ = 0;
    }

    void create_descriptor_resources()
    {
        if (camera_set_layout_ == VK_NULL_HANDLE)
        {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = 1;
            ci.pBindings = &binding;
            if (vkCreateDescriptorSetLayout(vk_->device(), &ci, nullptr, &camera_set_layout_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorSetLayout(camera) failed");
            }
        }

        if (shadow_set_layout_ == VK_NULL_HANDLE)
        {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = 1;
            ci.pBindings = &binding;
            if (vkCreateDescriptorSetLayout(vk_->device(), &ci, nullptr, &shadow_set_layout_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorSetLayout(shadow) failed");
            }
        }

        if (descriptor_pool_ == VK_NULL_HANDLE)
        {
            VkDescriptorPoolSize sizes[2]{};
            sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            sizes[0].descriptorCount = kFrameRing;
            sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sizes[1].descriptorCount = 1;

            VkDescriptorPoolCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            ci.maxSets = kFrameRing + 1;
            ci.poolSizeCount = 2;
            ci.pPoolSizes = sizes;
            if (vkCreateDescriptorPool(vk_->device(), &ci, nullptr, &descriptor_pool_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorPool failed");
            }
        }

        if (shadow_sampler_ == VK_NULL_HANDLE)
        {
            VkSamplerCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            ci.magFilter = VK_FILTER_LINEAR;
            ci.minFilter = VK_FILTER_LINEAR;
            ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci.minLod = 0.0f;
            ci.maxLod = 0.0f;
            ci.maxAnisotropy = 1.0f;
            if (vkCreateSampler(vk_->device(), &ci, nullptr, &shadow_sampler_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateSampler(shadow) failed");
            }
        }

        const VkMemoryPropertyFlags host_mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        std::array<VkDescriptorSetLayout, kFrameRing> layouts{};
        layouts.fill(camera_set_layout_);

        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptor_pool_;
        ai.descriptorSetCount = kFrameRing;
        ai.pSetLayouts = layouts.data();

        VkDescriptorSet sets[kFrameRing]{};
        if (vkAllocateDescriptorSets(vk_->device(), &ai, sets) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateDescriptorSets failed");
        }

        for (uint32_t i = 0; i < kFrameRing; ++i)
        {
            create_buffer(
                sizeof(CameraUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                host_mem,
                camera_ubos_[i],
                true);
            camera_sets_[i] = sets[i];

            VkDescriptorBufferInfo bi{};
            bi.buffer = camera_ubos_[i].buffer;
            bi.offset = 0;
            bi.range = sizeof(CameraUBO);

            VkWriteDescriptorSet wr{};
            wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr.dstSet = camera_sets_[i];
            wr.dstBinding = 0;
            wr.descriptorCount = 1;
            wr.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            wr.pBufferInfo = &bi;

            vkUpdateDescriptorSets(vk_->device(), 1, &wr, 0, nullptr);
        }

        if (shadow_set_ == VK_NULL_HANDLE)
        {
            VkDescriptorSetAllocateInfo sai{};
            sai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            sai.descriptorPool = descriptor_pool_;
            sai.descriptorSetCount = 1;
            sai.pSetLayouts = &shadow_set_layout_;
            if (vkAllocateDescriptorSets(vk_->device(), &sai, &shadow_set_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkAllocateDescriptorSets(shadow) failed");
            }
        }
    }

    void destroy_depth_target(DepthTarget& t)
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        VkDevice dev = vk_->device();
        if (t.view != VK_NULL_HANDLE) vkDestroyImageView(dev, t.view, nullptr);
        if (t.image != VK_NULL_HANDLE) vkDestroyImage(dev, t.image, nullptr);
        if (t.memory != VK_NULL_HANDLE) vkFreeMemory(dev, t.memory, nullptr);
        t = DepthTarget{};
    }

    bool create_depth_target(
        uint32_t w,
        uint32_t h,
        VkFormat format,
        VkImageUsageFlags usage,
        DepthTarget& out)
    {
        out = DepthTarget{};
        out.format = format;

        VkImageCreateInfo ii{};
        ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.extent = {w, h, 1u};
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
        const uint32_t mt = vk_find_memory_type(
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
        iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
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

    void update_shadow_descriptor_set()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        if (shadow_set_ == VK_NULL_HANDLE) return;
        if (shadow_sampler_ == VK_NULL_HANDLE) return;
        if (shadow_depth_target_.view == VK_NULL_HANDLE) return;

        VkDescriptorImageInfo ii{};
        ii.sampler = shadow_sampler_;
        ii.imageView = shadow_depth_target_.view;
        ii.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet wr{};
        wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet = shadow_set_;
        wr.dstBinding = 0;
        wr.descriptorCount = 1;
        wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wr.pImageInfo = &ii;
        vkUpdateDescriptorSets(vk_->device(), 1, &wr, 0, nullptr);
    }

    void destroy_shadow_resources()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        if (shadow_fb_ != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(vk_->device(), shadow_fb_, nullptr);
            shadow_fb_ = VK_NULL_HANDLE;
        }
        if (shadow_render_pass_ != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(vk_->device(), shadow_render_pass_, nullptr);
            shadow_render_pass_ = VK_NULL_HANDLE;
        }
        destroy_depth_target(shadow_depth_target_);
    }

    void create_shadow_resources()
    {
        destroy_shadow_resources();
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;

        const VkFormat depth_fmt =
            (vk_->depth_format() != VK_FORMAT_UNDEFINED)
                ? vk_->depth_format()
                : VK_FORMAT_D32_SFLOAT;

        if (!create_depth_target(
                kShadowMapSize,
                kShadowMapSize,
                depth_fmt,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                shadow_depth_target_))
        {
            throw std::runtime_error("create_depth_target(shadow) failed");
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
        if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &shadow_render_pass_) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateRenderPass(shadow) failed");
        }

        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = shadow_render_pass_;
        fb.attachmentCount = 1;
        fb.pAttachments = &shadow_depth_target_.view;
        fb.width = kShadowMapSize;
        fb.height = kShadowMapSize;
        fb.layers = 1;
        if (vkCreateFramebuffer(vk_->device(), &fb, nullptr, &shadow_fb_) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateFramebuffer(shadow) failed");
        }

        update_shadow_descriptor_set();
    }

    void destroy_pipelines()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        VkDevice dev = vk_->device();
        if (pipeline_tri_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(dev, pipeline_tri_, nullptr);
            pipeline_tri_ = VK_NULL_HANDLE;
        }
        if (pipeline_line_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(dev, pipeline_line_, nullptr);
            pipeline_line_ = VK_NULL_HANDLE;
        }
        if (pipeline_depth_prepass_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(dev, pipeline_depth_prepass_, nullptr);
            pipeline_depth_prepass_ = VK_NULL_HANDLE;
        }
        if (pipeline_occ_query_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(dev, pipeline_occ_query_, nullptr);
            pipeline_occ_query_ = VK_NULL_HANDLE;
        }
        if (shadow_pipeline_depth_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(dev, shadow_pipeline_depth_, nullptr);
            shadow_pipeline_depth_ = VK_NULL_HANDLE;
        }
        if (shadow_pipeline_occ_query_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(dev, shadow_pipeline_occ_query_, nullptr);
            shadow_pipeline_occ_query_ = VK_NULL_HANDLE;
        }
        if (pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(dev, pipeline_layout_, nullptr);
            pipeline_layout_ = VK_NULL_HANDLE;
        }
        if (shadow_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(dev, shadow_pipeline_layout_, nullptr);
            shadow_pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    VkPipeline create_main_pipeline(
        const char* vert_spv_path,
        const char* frag_spv_path,
        VkPrimitiveTopology topology,
        VkPolygonMode polygon_mode,
        VkCullModeFlags cull_mode,
        bool depth_test,
        bool depth_write,
        bool color_write)
    {
        const VkDevice dev = vk_->device();

        const std::vector<char> vs_code = vk_read_binary_file(vert_spv_path);
        const std::vector<char> fs_code = vk_read_binary_file(frag_spv_path);
        VkShaderModule vs = vk_create_shader_module(dev, vs_code);
        VkShaderModule fs = vk_create_shader_module(dev, fs_code);

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

        VkVertexInputAttributeDescription attrs[2]{};
        attrs[0].location = 0;
        attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = static_cast<uint32_t>(offsetof(Vertex, pos));
        attrs[1].location = 1;
        attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = static_cast<uint32_t>(offsetof(Vertex, normal));

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = topology;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = polygon_mode;
        rs.cullMode = cull_mode;
        // We render with flipped-Y Vulkan viewport; with LH/clockwise mesh winding this maps to CCW front faces.
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = depth_test ? VK_TRUE : VK_FALSE;
        ds.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = color_write
            ? (VK_COLOR_COMPONENT_R_BIT |
               VK_COLOR_COMPONENT_G_BIT |
               VK_COLOR_COMPONENT_B_BIT |
               VK_COLOR_COMPONENT_A_BIT)
            : 0u;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        const VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
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
        gp.layout = pipeline_layout_;
        gp.renderPass = vk_->render_pass();
        gp.subpass = 0;

        VkPipeline out = VK_NULL_HANDLE;
        const VkResult res = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &out);
        vkDestroyShaderModule(dev, vs, nullptr);
        vkDestroyShaderModule(dev, fs, nullptr);
        if (res != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateGraphicsPipelines failed");
        }
        return out;
    }

    VkPipeline create_shadow_pipeline(bool depth_write, VkCullModeFlags cull_mode)
    {
        const VkDevice dev = vk_->device();
        if (shadow_render_pass_ == VK_NULL_HANDLE)
        {
            throw std::runtime_error("shadow_render_pass not initialized");
        }

        const std::vector<char> vs_code = vk_read_binary_file(SHS_VK_PB_SHADOW_VERT_SPV);
        VkShaderModule vs = vk_create_shader_module(dev, vs_code);

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
        attr.offset = static_cast<uint32_t>(offsetof(Vertex, pos));

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
        rs.cullMode = cull_mode;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 0;
        cb.pAttachments = nullptr;

        const VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
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
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dyn;
        gp.layout = shadow_pipeline_layout_;
        gp.renderPass = shadow_render_pass_;
        gp.subpass = 0;

        VkPipeline out = VK_NULL_HANDLE;
        const VkResult res = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &out);
        vkDestroyShaderModule(dev, vs, nullptr);
        if (res != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateGraphicsPipelines(shadow) failed");
        }
        return out;
    }

    void create_pipelines()
    {
        destroy_pipelines();
        if (shadow_render_pass_ == VK_NULL_HANDLE || shadow_fb_ == VK_NULL_HANDLE)
        {
            create_shadow_resources();
        }

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(DrawPush);

        VkDescriptorSetLayout set_layouts[2] = {camera_set_layout_, shadow_set_layout_};
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 2;
        pl.pSetLayouts = set_layouts;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(vk_->device(), &pl, nullptr, &pipeline_layout_) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreatePipelineLayout failed");
        }

        VkPushConstantRange sp{};
        sp.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        sp.offset = 0;
        sp.size = sizeof(ShadowPush);

        VkPipelineLayoutCreateInfo spl{};
        spl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        spl.setLayoutCount = 0;
        spl.pSetLayouts = nullptr;
        spl.pushConstantRangeCount = 1;
        spl.pPushConstantRanges = &sp;
        if (vkCreatePipelineLayout(vk_->device(), &spl, nullptr, &shadow_pipeline_layout_) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreatePipelineLayout(shadow) failed");
        }

        pipeline_tri_ = create_main_pipeline(
            SHS_VK_SOFT_SHADOW_CULLING_VERT_SPV,
            SHS_VK_SOFT_SHADOW_CULLING_FRAG_SPV,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_BACK_BIT,
            true,
            true,
            true);
        // Match software debug behavior: lines are overlay (no depth test/write).
        pipeline_line_ = create_main_pipeline(
            SHS_VK_CULLING_VERT_SPV,
            SHS_VK_CULLING_FRAG_SPV,
            VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            false,
            false,
            true);
        pipeline_depth_prepass_ = create_main_pipeline(
            SHS_VK_CULLING_VERT_SPV,
            SHS_VK_CULLING_FRAG_SPV,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_BACK_BIT,
            true,
            true,
            false);
        // Occlusion queries use proxy AABBs; avoid winding sensitivity by disabling face culling.
        pipeline_occ_query_ = create_main_pipeline(
            SHS_VK_CULLING_VERT_SPV,
            SHS_VK_CULLING_FRAG_SPV,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            true,
            false,
            false);
        shadow_pipeline_depth_ = create_shadow_pipeline(true, VK_CULL_MODE_BACK_BIT);
        shadow_pipeline_occ_query_ = create_shadow_pipeline(false, VK_CULL_MODE_NONE);
        pipeline_gen_ = vk_->swapchain_generation();
    }

    bool pump_input(PlatformInputState& out)
    {
        out = PlatformInputState{};

        SDL_Event e{};
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT) out.quit = true;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) out.quit = true;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_l) out.toggle_light_shafts = true;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_b) out.toggle_bot = true;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F1) out.cycle_debug_view = true;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F2) out.cycle_cull_mode = true;

            if (e.type == SDL_MOUSEMOTION)
            {
                out.mouse_dx += static_cast<float>(e.motion.xrel);
                out.mouse_dy += static_cast<float>(e.motion.yrel);
            }

            if (e.type == SDL_WINDOWEVENT &&
                (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED || e.window.event == SDL_WINDOWEVENT_RESIZED))
            {
                if (vk_) vk_->request_resize(e.window.data1, e.window.data2);
            }
        }

        uint32_t ms = SDL_GetMouseState(nullptr, nullptr);
        out.right_mouse_down = (ms & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
        out.left_mouse_down = (ms & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;

        const uint8_t* ks = SDL_GetKeyboardState(nullptr);
        out.forward = ks[SDL_SCANCODE_W] != 0;
        out.backward = ks[SDL_SCANCODE_S] != 0;
        out.left = ks[SDL_SCANCODE_A] != 0;
        out.right = ks[SDL_SCANCODE_D] != 0;
        out.descend = ks[SDL_SCANCODE_Q] != 0;
        out.ascend = ks[SDL_SCANCODE_E] != 0;
        out.boost = ks[SDL_SCANCODE_LSHIFT] != 0;

        SDL_SetRelativeMouseMode(out.right_mouse_down ? SDL_TRUE : SDL_FALSE);
        return !out.quit;
    }

    void update_scene_and_culling(float time_s)
    {
        for (auto& inst : instances_)
        {
            if (inst.animated)
            {
                const glm::vec3 rot = inst.base_rot + inst.angular_vel * time_s;
                inst.model = compose_model(inst.base_pos, rot);
            }
            inst.shape.transform = jolt::to_jph(inst.model);
            inst.visible = true;
            inst.frustum_visible = true;
            inst.occluded = false;
        }

        auto view_elems = view_cull_scene_.elements();
        auto shadow_elems = shadow_cull_scene_.elements();
        for (size_t i = 0; i < instances_.size(); ++i)
        {
            view_elems[i].geometry = instances_[i].shape;
            view_elems[i].visible = true;
            view_elems[i].frustum_visible = true;
            view_elems[i].occluded = false;
            view_elems[i].enabled = true;

            shadow_elems[i].geometry = instances_[i].shape;
            shadow_elems[i].visible = true;
            shadow_elems[i].frustum_visible = true;
            shadow_elems[i].occluded = false;
            shadow_elems[i].enabled = instances_[i].casts_shadow;
        }

        view_mtx_ = camera_.view_matrix();
        proj_mtx_ = perspective_lh_no(glm::radians(60.0f), aspect_, 0.1f, 1000.0f);
        vp_mtx_ = proj_mtx_ * view_mtx_;
        frustum_ = extract_frustum_planes(vp_mtx_);

        shadow_caster_bounds_ = compute_shadow_caster_bounds(instances_);
        const AABB shadow_bounds = scale_aabb_about_center(shadow_caster_bounds_, kShadowRangeScale);
        const glm::vec3 scene_center = shadow_caster_bounds_.center();
        const float scene_radius = std::max(42.0f, glm::length(shadow_caster_bounds_.extent()) * 1.8f);
        const float orbit_angle = 0.17f * time_s;
        const glm::vec3 sun_pos_ws = scene_center + glm::vec3(
            std::cos(orbit_angle) * scene_radius,
            std::max(26.0f, shadow_caster_bounds_.maxv.y + 22.0f) + kSunHeightLift,
            std::sin(orbit_angle) * scene_radius);
        sun_dir_ws_ = glm::normalize(scene_center - sun_pos_ws);

        light_cam_ = build_dir_light_camera_aabb(
            sun_dir_ws_,
            shadow_bounds,
            8.0f,
            kShadowMapSize);
        // Culling frustum uses the canonical LH NO matrix conventions in library space.
        light_frustum_ = extract_frustum_planes(light_cam_.viewproj);
        // Shadow map sampling/rendering in Vulkan expects depth in [0, 1].
        glm::mat4 clip(1.0f);
        clip[2][2] = 0.5f;
        clip[3][2] = 0.5f;
        light_view_proj_mtx_ = clip * light_cam_.viewproj;

        view_cull_ctx_.run_frustum(view_cull_scene_, frustum_);
        shadow_cull_ctx_.run_frustum(shadow_cull_scene_, light_frustum_);
    }

    void consume_query_results(
        VkQueryPool query_pool,
        uint32_t query_count,
        const std::vector<uint32_t>& scene_indices,
        SceneCullingContext& cull_ctx,
        SceneElementSet& cull_scene)
    {
        if (query_pool == VK_NULL_HANDLE) return;
        if (query_count == 0u) return;
        if (scene_indices.empty()) return;
        const uint32_t n = std::min<uint32_t>(query_count, static_cast<uint32_t>(scene_indices.size()));
        if (n == 0u) return;

        std::vector<uint64_t> query_data(static_cast<size_t>(n), 0u);
        const VkResult qr = vkGetQueryPoolResults(
            vk_->device(),
            query_pool,
            0,
            n,
            static_cast<VkDeviceSize>(query_data.size() * sizeof(uint64_t)),
            query_data.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (qr != VK_SUCCESS) return;

        cull_ctx.apply_occlusion_query_samples(
            cull_scene,
            std::span<const uint32_t>(scene_indices.data(), n),
            std::span<const uint64_t>(query_data.data(), n),
            kOcclusionMinVisibleSamples);
    }

    void consume_occlusion_results(uint32_t ring)
    {
        if (!enable_occlusion_) return;
        if (ring >= kFrameRing) return;
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;

        if (vk_->has_depth_attachment())
        {
            consume_query_results(
                view_query_pools_[ring],
                view_query_counts_[ring],
                view_query_scene_indices_[ring],
                view_cull_ctx_,
                view_cull_scene_);
        }
        consume_query_results(
            shadow_query_pools_[ring],
            shadow_query_counts_[ring],
            shadow_query_scene_indices_[ring],
            shadow_cull_ctx_,
            shadow_cull_scene_);
    }

    void finalize_visibility_lists(uint32_t ring)
    {
        view_cull_ctx_.finalize_visibility(view_cull_scene_, apply_occlusion_this_frame_);
        shadow_cull_ctx_.finalize_visibility(shadow_cull_scene_, apply_occlusion_this_frame_);

        (void)view_cull_ctx_.apply_frustum_fallback_if_needed(
            view_cull_scene_,
            enable_occlusion_,
            vk_ && vk_->has_depth_attachment(),
            (ring < kFrameRing) ? view_query_counts_[ring] : 0u);
        (void)shadow_cull_ctx_.apply_frustum_fallback_if_needed(
            shadow_cull_scene_,
            enable_occlusion_,
            true,
            (ring < kFrameRing) ? shadow_query_counts_[ring] : 0u);

        render_view_scene_indices_ = view_cull_ctx_.visible_indices();
        render_shadow_scene_indices_ = shadow_cull_ctx_.visible_indices();
        scene_stats_ = view_cull_ctx_.stats();
        shadow_stats_ = shadow_cull_ctx_.stats();

        // Keep floor render-stable when it is in frustum.
        if (!view_cull_scene_.empty())
        {
            const uint32_t floor_scene_idx = 0u;
            const auto elems = view_cull_scene_.elements();
            if (floor_scene_idx < elems.size() && elems[floor_scene_idx].frustum_visible)
            {
                if (std::find(
                        render_view_scene_indices_.begin(),
                        render_view_scene_indices_.end(),
                        floor_scene_idx) == render_view_scene_indices_.end())
                {
                    render_view_scene_indices_.push_back(floor_scene_idx);
                    scene_stats_.visible_count += 1u;
                    if (scene_stats_.occluded_count > 0u) scene_stats_.occluded_count -= 1u;
                    normalize_culling_stats(scene_stats_);
                }
            }
        }
    }

    bool get_view_scene_instance(uint32_t scene_idx, const ShapeInstance*& out_inst, const MeshGPU*& out_mesh) const
    {
        out_inst = nullptr;
        out_mesh = nullptr;
        if (scene_idx >= view_cull_scene_.size()) return false;
        const uint32_t idx = view_cull_scene_[scene_idx].user_index;
        if (idx >= instances_.size()) return false;
        const ShapeInstance& inst = instances_[idx];
        if (inst.mesh_index >= meshes_.size()) return false;
        const MeshGPU& mesh = meshes_[inst.mesh_index];
        out_inst = &inst;
        out_mesh = &mesh;
        return true;
    }

    void prepare_view_occlusion_query_work(uint32_t ring)
    {
        if (ring >= kFrameRing) return;
        view_query_scene_indices_[ring].clear();
        view_query_counts_[ring] = 0;
        if (!enable_occlusion_ || !vk_ || !vk_->has_depth_attachment()) return;
        if (view_query_pools_[ring] == VK_NULL_HANDLE || max_view_query_count_ == 0u) return;

        view_query_scene_indices_[ring].reserve(view_cull_ctx_.frustum_visible_indices().size());
        for (const uint32_t scene_idx : view_cull_ctx_.frustum_visible_indices())
        {
            if (view_query_scene_indices_[ring].size() >= max_view_query_count_) break;

            const ShapeInstance* inst = nullptr;
            const MeshGPU* mesh = nullptr;
            if (!get_view_scene_instance(scene_idx, inst, mesh)) continue;
            if (mesh->tri_indices.buffer == VK_NULL_HANDLE || mesh->tri_index_count == 0) continue;
            view_query_scene_indices_[ring].push_back(scene_idx);
        }
        view_query_counts_[ring] = static_cast<uint32_t>(view_query_scene_indices_[ring].size());
    }

    void record_depth_prepass_range(
        VkCommandBuffer cmd,
        VkDescriptorSet camera_set,
        uint32_t begin,
        uint32_t end)
    {
        if (pipeline_depth_prepass_ == VK_NULL_HANDLE) return;
        const uint32_t draw_n = static_cast<uint32_t>(render_view_scene_indices_.size());
        begin = std::min(begin, draw_n);
        end = std::min(end, draw_n);
        if (begin >= end) return;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_depth_prepass_);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout_,
            0,
            1,
            &camera_set,
            0,
            nullptr);

        for (uint32_t i = begin; i < end; ++i)
        {
            const uint32_t scene_idx = render_view_scene_indices_[i];
            const ShapeInstance* inst = nullptr;
            const MeshGPU* mesh = nullptr;
            if (!get_view_scene_instance(scene_idx, inst, mesh)) continue;
            if (mesh->tri_indices.buffer == VK_NULL_HANDLE || mesh->tri_index_count == 0) continue;

            const VkBuffer vb = mesh->vertex.buffer;
            const VkDeviceSize vb_off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vb_off);
            vkCmdBindIndexBuffer(cmd, mesh->tri_indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            DrawPush push{};
            push.model = inst->model;
            push.base_color = glm::vec4(inst->color, 1.0f);
            push.mode_pad.x = 1u;
            vkCmdPushConstants(
                cmd,
                pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(DrawPush),
                &push);
            vkCmdDrawIndexed(cmd, mesh->tri_index_count, 1, 0, 0, 0);
        }
    }

    void record_main_draws_range(
        VkCommandBuffer cmd,
        VkDescriptorSet camera_set,
        uint32_t begin,
        uint32_t end)
    {
        const uint32_t draw_n = static_cast<uint32_t>(render_view_scene_indices_.size());
        begin = std::min(begin, draw_n);
        end = std::min(end, draw_n);
        if (begin >= end) return;

        if (render_lit_surfaces_)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_tri_);
        }
        else
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_line_);
        }
        if (shadow_set_ != VK_NULL_HANDLE)
        {
            const VkDescriptorSet sets[2] = {camera_set, shadow_set_};
            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_layout_,
                0,
                2,
                sets,
                0,
                nullptr);
        }
        else
        {
            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_layout_,
                0,
                1,
                &camera_set,
                0,
                nullptr);
        }

        for (uint32_t i = begin; i < end; ++i)
        {
            const uint32_t scene_idx = render_view_scene_indices_[i];
            const ShapeInstance* inst = nullptr;
            const MeshGPU* mesh = nullptr;
            if (!get_view_scene_instance(scene_idx, inst, mesh)) continue;

            uint32_t index_count = 0;
            VkBuffer ib = VK_NULL_HANDLE;
            if (render_lit_surfaces_)
            {
                ib = mesh->tri_indices.buffer;
                index_count = mesh->tri_index_count;
            }
            else
            {
                ib = mesh->line_indices.buffer;
                index_count = mesh->line_index_count;
            }
            if (ib == VK_NULL_HANDLE || index_count == 0) continue;

            const VkBuffer vb = mesh->vertex.buffer;
            const VkDeviceSize vb_off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vb_off);
            vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);

            DrawPush push{};
            push.model = inst->model;
            push.base_color = glm::vec4(inst->color, 1.0f);
            push.mode_pad.x = render_lit_surfaces_ ? 1u : 0u;
            vkCmdPushConstants(
                cmd,
                pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(DrawPush),
                &push);
            vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
        }
    }

    void record_aabb_overlay_range(
        VkCommandBuffer cmd,
        VkDescriptorSet camera_set,
        uint32_t begin,
        uint32_t end)
    {
        if (!show_aabb_debug_) return;
        if (aabb_mesh_index_ >= meshes_.size()) return;

        const uint32_t draw_n = static_cast<uint32_t>(render_view_scene_indices_.size());
        begin = std::min(begin, draw_n);
        end = std::min(end, draw_n);
        if (begin >= end) return;

        const glm::vec4 aabb_color(1.0f, 0.94f, 0.31f, 1.0f);
        const MeshGPU& aabb_mesh = meshes_[aabb_mesh_index_];
        if (aabb_mesh.line_indices.buffer == VK_NULL_HANDLE || aabb_mesh.line_index_count == 0) return;

        const VkBuffer vb = aabb_mesh.vertex.buffer;
        const VkDeviceSize vb_off = 0;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_line_);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout_,
            0,
            1,
            &camera_set,
            0,
            nullptr);
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vb_off);
        vkCmdBindIndexBuffer(cmd, aabb_mesh.line_indices.buffer, 0, VK_INDEX_TYPE_UINT32);

        for (uint32_t i = begin; i < end; ++i)
        {
            const uint32_t scene_idx = render_view_scene_indices_[i];
            const ShapeInstance* inst = nullptr;
            const MeshGPU* mesh = nullptr;
            if (!get_view_scene_instance(scene_idx, inst, mesh)) continue;
            (void)mesh;

            const AABB box = inst->shape.world_aabb();
            const glm::vec3 center = (box.minv + box.maxv) * 0.5f;
            const glm::vec3 size = glm::max(box.maxv - box.minv, glm::vec3(1e-4f));

            DrawPush push{};
            push.model = glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), size);
            push.base_color = aabb_color;
            push.mode_pad.x = 0u;
            vkCmdPushConstants(
                cmd,
                pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(DrawPush),
                &push);
            vkCmdDrawIndexed(cmd, aabb_mesh.line_index_count, 1, 0, 0, 0);
        }
    }

    void record_view_occlusion_queries_range(
        VkCommandBuffer cmd,
        VkDescriptorSet camera_set,
        uint32_t ring,
        uint32_t begin,
        uint32_t end)
    {
        if (!enable_occlusion_) return;
        if (!vk_ || !vk_->has_depth_attachment()) return;
        if (ring >= kFrameRing) return;
        if (view_query_pools_[ring] == VK_NULL_HANDLE) return;
        if (pipeline_occ_query_ == VK_NULL_HANDLE) return;

        const uint32_t query_n = view_query_counts_[ring];
        begin = std::min(begin, query_n);
        end = std::min(end, query_n);
        if (begin >= end) return;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_occ_query_);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout_,
            0,
            1,
            &camera_set,
            0,
            nullptr);

        for (uint32_t query_idx = begin; query_idx < end; ++query_idx)
        {
            const uint32_t scene_idx = view_query_scene_indices_[ring][query_idx];
            const ShapeInstance* inst = nullptr;
            const MeshGPU* mesh = nullptr;
            if (!get_view_scene_instance(scene_idx, inst, mesh)) continue;
            if (mesh->tri_indices.buffer == VK_NULL_HANDLE || mesh->tri_index_count == 0) continue;

            const VkBuffer vb = mesh->vertex.buffer;
            const VkDeviceSize vb_off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vb_off);
            vkCmdBindIndexBuffer(cmd, mesh->tri_indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            DrawPush push{};
            push.model = inst->model;
            push.base_color = glm::vec4(inst->color, 1.0f);
            push.mode_pad.x = 1u;
            vkCmdPushConstants(
                cmd,
                pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(DrawPush),
                &push);

            vkCmdBeginQuery(cmd, view_query_pools_[ring], query_idx, 0);
            vkCmdDrawIndexed(cmd, mesh->tri_index_count, 1, 0, 0, 0);
            vkCmdEndQuery(cmd, view_query_pools_[ring], query_idx);
        }
    }

    void record_depth_prepass(VkCommandBuffer cmd, VkDescriptorSet camera_set)
    {
        record_depth_prepass_range(cmd, camera_set, 0u, static_cast<uint32_t>(render_view_scene_indices_.size()));
    }

    void record_main_draws(VkCommandBuffer cmd, VkDescriptorSet camera_set)
    {
        record_main_draws_range(cmd, camera_set, 0u, static_cast<uint32_t>(render_view_scene_indices_.size()));
        record_aabb_overlay_range(cmd, camera_set, 0u, static_cast<uint32_t>(render_view_scene_indices_.size()));
    }

    void record_view_occlusion_queries(VkCommandBuffer cmd, VkDescriptorSet camera_set, uint32_t ring)
    {
        record_view_occlusion_queries_range(cmd, camera_set, ring, 0u, view_query_counts_[ring]);
    }

    bool reset_worker_pools_for_frame(uint32_t ring)
    {
        if (ring >= kFrameRing) return false;
        if (!use_multithread_recording_ || worker_pools_.empty()) return true;
        for (uint32_t i = 0; i < worker_count_ && i < worker_pools_.size(); ++i)
        {
            VkCommandPool pool = worker_pools_[i].pools[ring];
            if (pool == VK_NULL_HANDLE) return false;
            (void)vkResetCommandPool(vk_->device(), pool, 0);
        }
        return true;
    }

    bool record_main_secondary_batch(
        VkRenderPass render_pass,
        VkFramebuffer framebuffer,
        VkExtent2D extent,
        VkDescriptorSet camera_set,
        uint32_t ring,
        uint32_t worker_idx,
        uint32_t draw_begin,
        uint32_t draw_end,
        uint32_t query_begin,
        uint32_t query_end,
        bool record_depth,
        bool record_queries,
        bool record_main,
        VkCommandBuffer& out_cmd)
    {
        out_cmd = VK_NULL_HANDLE;
        if (worker_idx >= worker_pools_.size()) return false;
        if (ring >= kFrameRing) return false;
        if (draw_begin >= draw_end && query_begin >= query_end) return true;

        VkCommandPool pool = worker_pools_[worker_idx].pools[ring];
        if (pool == VK_NULL_HANDLE) return false;

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(vk_->device(), &ai, &out_cmd) != VK_SUCCESS)
        {
            return false;
        }

        VkCommandBufferInheritanceInfo inh{};
        inh.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        inh.renderPass = render_pass;
        inh.subpass = 0;
        inh.framebuffer = framebuffer;
        inh.occlusionQueryEnable = VK_TRUE;
        inh.queryFlags = 0;
        inh.pipelineStatistics = 0;

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        bi.pInheritanceInfo = &inh;
        if (vkBeginCommandBuffer(out_cmd, &bi) != VK_SUCCESS)
        {
            return false;
        }

        vk_cmd_set_viewport_scissor(out_cmd, extent.width, extent.height, true);
        if (record_depth && draw_begin < draw_end)
        {
            record_depth_prepass_range(out_cmd, camera_set, draw_begin, draw_end);
        }
        if (record_queries && query_begin < query_end)
        {
            record_view_occlusion_queries_range(out_cmd, camera_set, ring, query_begin, query_end);
        }
        if (record_main && draw_begin < draw_end)
        {
            record_main_draws_range(out_cmd, camera_set, draw_begin, draw_end);
            record_aabb_overlay_range(out_cmd, camera_set, draw_begin, draw_end);
        }

        return vkEndCommandBuffer(out_cmd) == VK_SUCCESS;
    }

    bool record_main_secondary_lists(
        VkRenderPass render_pass,
        VkFramebuffer framebuffer,
        VkExtent2D extent,
        VkDescriptorSet camera_set,
        uint32_t ring,
        bool record_depth,
        bool record_queries,
        bool record_main,
        std::vector<VkCommandBuffer>& out)
    {
        out.clear();
        if (!use_multithread_recording_ || worker_pools_.empty() || !jobs_) return false;
        if (ring >= kFrameRing) return false;

        const uint32_t draw_n = static_cast<uint32_t>(render_view_scene_indices_.size());
        const uint32_t query_n = view_query_counts_[ring];
        const bool use_query_ranges = record_queries && !record_depth && !record_main;
        const uint32_t total = use_query_ranges ? query_n : draw_n;
        if (total == 0u) return false;

        const uint32_t workers =
            std::min<uint32_t>(std::min<uint32_t>(worker_count_, static_cast<uint32_t>(worker_pools_.size())), total);
        if (workers <= 1u) return false;

        const uint32_t batch = (total + workers - 1u) / workers;
        std::vector<VkCommandBuffer> tmp(workers, VK_NULL_HANDLE);
        std::atomic<bool> ok{true};
        WaitGroup wg{};

        for (uint32_t wi = 0; wi < workers; ++wi)
        {
            const uint32_t begin = std::min(total, wi * batch);
            const uint32_t end = std::min(total, begin + batch);
            const uint32_t draw_begin = use_query_ranges ? 0u : begin;
            const uint32_t draw_end = use_query_ranges ? 0u : end;
            const uint32_t query_begin = use_query_ranges ? begin : 0u;
            const uint32_t query_end = use_query_ranges ? end : 0u;
            if (draw_begin >= draw_end && query_begin >= query_end) continue;

            wg.add(1);
            jobs_->enqueue([&, wi, draw_begin, draw_end, query_begin, query_end]() {
                if (!record_main_secondary_batch(
                        render_pass,
                        framebuffer,
                        extent,
                        camera_set,
                        ring,
                        wi,
                        draw_begin,
                        draw_end,
                        query_begin,
                        query_end,
                        record_depth,
                        record_queries,
                        record_main,
                        tmp[wi]))
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
        return !out.empty();
    }

    void draw_shadow_scene_element(VkCommandBuffer cmd, uint32_t shadow_scene_idx)
    {
        if (shadow_scene_idx >= shadow_cull_scene_.size()) return;
        const uint32_t idx = shadow_cull_scene_[shadow_scene_idx].user_index;
        if (idx >= instances_.size()) return;
        const ShapeInstance& inst = instances_[idx];
        if (!inst.casts_shadow) return;
        if (inst.mesh_index >= meshes_.size()) return;
        const MeshGPU& mesh = meshes_[inst.mesh_index];
        if (mesh.tri_indices.buffer == VK_NULL_HANDLE || mesh.tri_index_count == 0) return;

        const VkBuffer vb = mesh.vertex.buffer;
        const VkDeviceSize vb_off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vb_off);
        vkCmdBindIndexBuffer(cmd, mesh.tri_indices.buffer, 0, VK_INDEX_TYPE_UINT32);

        ShadowPush sp{};
        sp.light_mvp = light_view_proj_mtx_ * inst.model;
        vkCmdPushConstants(cmd, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &sp);
        vkCmdDrawIndexed(cmd, mesh.tri_index_count, 1, 0, 0, 0);
    }

    void record_shadow_occlusion_queries(VkCommandBuffer cmd, uint32_t ring)
    {
        if (!enable_occlusion_) return;
        if (ring >= kFrameRing) return;
        if (shadow_query_pools_[ring] == VK_NULL_HANDLE) return;
        if (shadow_pipeline_occ_query_ == VK_NULL_HANDLE) return;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_occ_query_);
        shadow_query_scene_indices_[ring].clear();
        shadow_query_scene_indices_[ring].reserve(shadow_cull_ctx_.frustum_visible_indices().size());
        shadow_query_counts_[ring] = 0;

        for (const uint32_t shadow_scene_idx : shadow_cull_ctx_.frustum_visible_indices())
        {
            if (shadow_query_counts_[ring] >= max_shadow_query_count_) break;
            if (shadow_scene_idx >= shadow_cull_scene_.size()) continue;
            const uint32_t idx = shadow_cull_scene_[shadow_scene_idx].user_index;
            if (idx >= instances_.size()) continue;
            const ShapeInstance& inst = instances_[idx];
            if (!inst.casts_shadow) continue;
            if (inst.mesh_index >= meshes_.size()) continue;
            const MeshGPU& mesh = meshes_[inst.mesh_index];
            if (mesh.tri_indices.buffer == VK_NULL_HANDLE || mesh.tri_index_count == 0) continue;

            const VkBuffer vb = mesh.vertex.buffer;
            const VkDeviceSize vb_off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vb_off);
            vkCmdBindIndexBuffer(cmd, mesh.tri_indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            ShadowPush sp{};
            sp.light_mvp = light_view_proj_mtx_ * inst.model;
            vkCmdPushConstants(cmd, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &sp);

            const uint32_t query_idx = shadow_query_counts_[ring];
            shadow_query_scene_indices_[ring].push_back(shadow_scene_idx);
            shadow_query_counts_[ring]++;

            vkCmdBeginQuery(cmd, shadow_query_pools_[ring], query_idx, 0);
            vkCmdDrawIndexed(cmd, mesh.tri_index_count, 1, 0, 0, 0);
            vkCmdEndQuery(cmd, shadow_query_pools_[ring], query_idx);
        }
    }

    void record_shadow_pass(VkCommandBuffer cmd, uint32_t ring)
    {
        if (shadow_render_pass_ == VK_NULL_HANDLE || shadow_fb_ == VK_NULL_HANDLE) return;
        if (shadow_pipeline_depth_ == VK_NULL_HANDLE) return;

        VkClearValue clear{};
        clear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = shadow_render_pass_;
        rp.framebuffer = shadow_fb_;
        rp.renderArea.offset = {0, 0};
        rp.renderArea.extent = {kShadowMapSize, kShadowMapSize};
        rp.clearValueCount = 1;
        rp.pClearValues = &clear;

        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vk_cmd_set_viewport_scissor(cmd, kShadowMapSize, kShadowMapSize, true);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_depth_);
        for (const uint32_t shadow_scene_idx : render_shadow_scene_indices_)
        {
            draw_shadow_scene_element(cmd, shadow_scene_idx);
        }

        record_shadow_occlusion_queries(cmd, ring);
        vkCmdEndRenderPass(cmd);
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

    void draw_frame()
    {
        int dw = 0;
        int dh = 0;
        SDL_Vulkan_GetDrawableSize(win_, &dw, &dh);
        if (dw <= 0 || dh <= 0)
        {
            SDL_Delay(8);
            return;
        }
        aspect_ = static_cast<float>(dw) / static_cast<float>(std::max(1, dh));

        RenderBackendFrameInfo frame{};
        frame.frame_index = ctx_.frame_index;
        frame.width = dw;
        frame.height = dh;

        VulkanRenderBackend::FrameInfo fi{};
        if (!vk_->begin_frame(ctx_, frame, fi))
        {
            SDL_Delay(1);
            return;
        }

        if (pipeline_tri_ == VK_NULL_HANDLE || pipeline_gen_ != vk_->swapchain_generation())
        {
            create_pipelines();
        }

        const uint32_t ring = static_cast<uint32_t>(ctx_.frame_index % kFrameRing);
        apply_occlusion_this_frame_ =
            enable_occlusion_ &&
            (occlusion_warmup_frames_ == 0u);

        if (!apply_occlusion_this_frame_)
        {
            auto view_elems = view_cull_scene_.elements();
            for (auto& elem : view_elems) elem.occluded = false;
            auto shadow_elems = shadow_cull_scene_.elements();
            for (auto& elem : shadow_elems) elem.occluded = false;
        }

        // Consume occlusion results only after begin_frame() fence wait.
        // Reading before that can race GPU completion and produce flicker.
        if (apply_occlusion_this_frame_) consume_occlusion_results(ring);
        finalize_visibility_lists(ring);

        auto view_elems = view_cull_scene_.elements();
        for (size_t i = 0; i < instances_.size() && i < view_elems.size(); ++i)
        {
            instances_[i].visible = view_elems[i].visible;
            instances_[i].frustum_visible = view_elems[i].frustum_visible;
            instances_[i].occluded = view_elems[i].occluded;
        }

        CameraUBO cam{};
        cam.view_proj = vp_mtx_;
        cam.camera_pos = glm::vec4(camera_.pos, 1.0f);
        cam.light_dir_ws = glm::vec4(sun_dir_ws_, 0.0f);
        cam.light_view_proj = light_view_proj_mtx_;
        cam.shadow_params = glm::vec4(kShadowStrength, kShadowBiasConst, kShadowBiasSlope, kShadowPcfStep);
        cam.shadow_misc = glm::vec4((float)kShadowPcfRadius, 0.0f, 0.0f, 0.0f);
        std::memcpy(camera_ubos_[ring].mapped, &cam, sizeof(CameraUBO));

        prepare_view_occlusion_query_work(ring);

        const bool has_draw_work = !render_view_scene_indices_.empty();
        const bool has_query_work = enable_occlusion_ && vk_->has_depth_attachment() && (view_query_counts_[ring] > 0u);
        std::vector<VkCommandBuffer> depth_secondaries{};
        std::vector<VkCommandBuffer> query_secondaries{};
        std::vector<VkCommandBuffer> main_secondaries{};
        bool use_main_secondaries = false;
        if (use_multithread_recording_ && has_draw_work && reset_worker_pools_for_frame(ring))
        {
            const bool depth_ok =
                record_main_secondary_lists(
                    fi.render_pass,
                    fi.framebuffer,
                    fi.extent,
                    camera_sets_[ring],
                    ring,
                    true,
                    false,
                    false,
                    depth_secondaries);
            const bool query_ok =
                !has_query_work ||
                record_main_secondary_lists(
                    fi.render_pass,
                    fi.framebuffer,
                    fi.extent,
                    camera_sets_[ring],
                    ring,
                    false,
                    true,
                    false,
                    query_secondaries);
            const bool draw_ok =
                record_main_secondary_lists(
                    fi.render_pass,
                    fi.framebuffer,
                    fi.extent,
                    camera_sets_[ring],
                    ring,
                    false,
                    false,
                    true,
                    main_secondaries);
            use_main_secondaries = depth_ok && query_ok && draw_ok;
        }
        used_secondary_this_frame_ = use_main_secondaries;

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(fi.cmd, &bi) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBeginCommandBuffer failed");
        }

        if (enable_occlusion_ &&
            vk_->has_depth_attachment() &&
            view_query_pools_[ring] != VK_NULL_HANDLE &&
            max_view_query_count_ > 0)
        {
            vkCmdResetQueryPool(fi.cmd, view_query_pools_[ring], 0, max_view_query_count_);
        }
        else
        {
            view_query_counts_[ring] = 0;
            view_query_scene_indices_[ring].clear();
        }
        if (enable_occlusion_ &&
            shadow_query_pools_[ring] != VK_NULL_HANDLE &&
            max_shadow_query_count_ > 0)
        {
            vkCmdResetQueryPool(fi.cmd, shadow_query_pools_[ring], 0, max_shadow_query_count_);
        }
        else
        {
            shadow_query_counts_[ring] = 0;
            shadow_query_scene_indices_[ring].clear();
        }

        record_shadow_pass(fi.cmd, ring);
        cmd_memory_barrier(
            fi.cmd,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT);

        VkClearValue clear_values[2]{};
        clear_values[0].color = {{0.047f, 0.051f, 0.070f, 1.0f}};
        clear_values[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = fi.render_pass;
        rp.framebuffer = fi.framebuffer;
        rp.renderArea.offset = {0, 0};
        rp.renderArea.extent = fi.extent;
        rp.clearValueCount = vk_->has_depth_attachment() ? 2u : 1u;
        rp.pClearValues = clear_values;

        vkCmdBeginRenderPass(
            fi.cmd,
            &rp,
            use_main_secondaries ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
        if (use_main_secondaries && !main_secondaries.empty() && !depth_secondaries.empty())
        {
            vkCmdExecuteCommands(fi.cmd, static_cast<uint32_t>(depth_secondaries.size()), depth_secondaries.data());
            if (!query_secondaries.empty())
            {
                vkCmdExecuteCommands(fi.cmd, static_cast<uint32_t>(query_secondaries.size()), query_secondaries.data());
            }
            vkCmdExecuteCommands(fi.cmd, static_cast<uint32_t>(main_secondaries.size()), main_secondaries.data());
        }
        else
        {
            vk_cmd_set_viewport_scissor(fi.cmd, fi.extent.width, fi.extent.height, true);
            record_depth_prepass(fi.cmd, camera_sets_[ring]);
            record_view_occlusion_queries(fi.cmd, camera_sets_[ring], ring);
            record_main_draws(fi.cmd, camera_sets_[ring]);
        }
        vkCmdEndRenderPass(fi.cmd);

        if (vkEndCommandBuffer(fi.cmd) != VK_SUCCESS)
        {
            throw std::runtime_error("vkEndCommandBuffer failed");
        }

        vk_->end_frame(fi);
        ctx_.frame_index++;
        if (occlusion_warmup_frames_ > 0u) --occlusion_warmup_frames_;
    }

    void update_title(float avg_ms)
    {
        char title[384];
        std::snprintf(
            title,
            sizeof(title),
            "Soft Shadow Culling Demo (VK) | Scene:%u Frustum:%u Occ:%u Vis:%u | Shadow F:%u O:%u V:%u | Occ:%s | Mode:%s | AABB:%s | Rec:%s | %.2f ms",
            scene_stats_.scene_count,
            scene_stats_.frustum_visible_count,
            scene_stats_.occluded_count,
            scene_stats_.visible_count,
            shadow_stats_.frustum_visible_count,
            shadow_stats_.occluded_count,
            shadow_stats_.visible_count,
            (enable_occlusion_ && vk_ && vk_->has_depth_attachment()) ? "ON" : "OFF",
            render_lit_surfaces_ ? "Lit" : "Debug",
            show_aabb_debug_ ? "ON" : "OFF",
            used_secondary_this_frame_ ? "MT-secondary" : "Inline",
            avg_ms);
        SDL_SetWindowTitle(win_, title);
    }

    void main_loop()
    {
        std::printf("Controls: RMB look, WASD+QE move, Shift boost, B toggle AABB, L toggle debug/lit, F1 toggle MT-secondary recording, F2 toggle occlusion\n");

        bool running = true;
        auto t0 = std::chrono::steady_clock::now();
        auto prev = t0;
        auto title_tick = t0;
        float ema_ms = 16.0f;

        while (running)
        {
            const auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - prev).count();
            prev = now;
            dt = std::clamp(dt, 1.0f / 240.0f, 1.0f / 12.0f);
            const float time_s = std::chrono::duration<float>(now - t0).count();

            PlatformInputState input{};
            if (!pump_input(input)) break;
            if (input.quit) break;
            if (input.toggle_bot) show_aabb_debug_ = !show_aabb_debug_;
            if (input.toggle_light_shafts) render_lit_surfaces_ = !render_lit_surfaces_;
            if (input.cycle_debug_view)
            {
                if (worker_count_ <= 1u)
                {
                    use_multithread_recording_ = false;
                    destroy_worker_pools();
                }
                else
                {
                    use_multithread_recording_ = !use_multithread_recording_;
                    if (use_multithread_recording_)
                    {
                        if (!jobs_) jobs_ = std::make_unique<ThreadPoolJobSystem>(worker_count_);
                        if (worker_pools_.empty()) create_worker_pools();
                    }
                    else
                    {
                        destroy_worker_pools();
                    }
                }
            }
            if (input.cycle_cull_mode)
            {
                enable_occlusion_ = !enable_occlusion_;
                view_cull_ctx_.clear();
                shadow_cull_ctx_.clear();
                auto view_elems = view_cull_scene_.elements();
                for (auto& elem : view_elems) elem.occluded = false;
                auto shadow_elems = shadow_cull_scene_.elements();
                for (auto& elem : shadow_elems) elem.occluded = false;
                occlusion_warmup_frames_ = kOcclusionWarmupFramesAfterCameraMove;
            }

            camera_.update(input, dt);
            if (camera_prev_valid_)
            {
                const float pos_delta = glm::length(camera_.pos - camera_prev_pos_);
                const float yaw_delta = std::abs(camera_.yaw - camera_prev_yaw_);
                const float pitch_delta = std::abs(camera_.pitch - camera_prev_pitch_);
                if (pos_delta > 0.03f || yaw_delta > 0.0025f || pitch_delta > 0.0025f)
                {
                    occlusion_warmup_frames_ = kOcclusionWarmupFramesAfterCameraMove;
                }
            }
            camera_prev_valid_ = true;
            camera_prev_pos_ = camera_.pos;
            camera_prev_yaw_ = camera_.yaw;
            camera_prev_pitch_ = camera_.pitch;
            update_scene_and_culling(time_s);

            const auto cpu0 = std::chrono::steady_clock::now();
            draw_frame();
            const auto cpu1 = std::chrono::steady_clock::now();
            const float frame_ms = std::chrono::duration<float, std::milli>(cpu1 - cpu0).count();
            ema_ms = glm::mix(ema_ms, frame_ms, 0.08f);

            if (std::chrono::duration<float>(now - title_tick).count() >= 0.15f)
            {
                update_title(ema_ms);
                title_tick = now;
            }

            running = true;
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

        if (vk_ && vk_->device() != VK_NULL_HANDLE)
        {
            for (auto& mesh : meshes_)
            {
                destroy_buffer(mesh.vertex);
                destroy_buffer(mesh.tri_indices);
                destroy_buffer(mesh.line_indices);
            }
            meshes_.clear();

            for (auto& b : camera_ubos_)
            {
                destroy_buffer(b);
            }

            destroy_occlusion_query_resources();
            destroy_pipelines();
            destroy_shadow_resources();
            destroy_upload_resources();
            destroy_worker_pools();

            if (shadow_sampler_ != VK_NULL_HANDLE)
            {
                vkDestroySampler(vk_->device(), shadow_sampler_, nullptr);
                shadow_sampler_ = VK_NULL_HANDLE;
            }

            if (descriptor_pool_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(vk_->device(), descriptor_pool_, nullptr);
                descriptor_pool_ = VK_NULL_HANDLE;
            }
            if (camera_set_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(vk_->device(), camera_set_layout_, nullptr);
                camera_set_layout_ = VK_NULL_HANDLE;
            }
            if (shadow_set_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(vk_->device(), shadow_set_layout_, nullptr);
                shadow_set_layout_ = VK_NULL_HANDLE;
            }
        }

        jobs_.reset();
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

private:
    bool cleaned_up_ = false;
    bool sdl_ready_ = false;
    SDL_Window* win_ = nullptr;

    Context ctx_{};
    std::vector<std::unique_ptr<IRenderBackend>> keep_{};
    VulkanRenderBackend* vk_ = nullptr;
    VkCommandPool upload_pool_ = VK_NULL_HANDLE;
    VkFence upload_fence_ = VK_NULL_HANDLE;
    std::unique_ptr<ThreadPoolJobSystem> jobs_{};
    uint32_t worker_count_ = 1u;
    std::vector<WorkerPool> worker_pools_{};

    VkDescriptorSetLayout camera_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout shadow_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    std::array<GpuBuffer, kFrameRing> camera_ubos_{};
    std::array<VkDescriptorSet, kFrameRing> camera_sets_{};
    VkDescriptorSet shadow_set_ = VK_NULL_HANDLE;
    VkSampler shadow_sampler_ = VK_NULL_HANDLE;

    DepthTarget shadow_depth_target_{};
    VkRenderPass shadow_render_pass_ = VK_NULL_HANDLE;
    VkFramebuffer shadow_fb_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout shadow_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_tri_ = VK_NULL_HANDLE;
    VkPipeline pipeline_line_ = VK_NULL_HANDLE;
    VkPipeline pipeline_depth_prepass_ = VK_NULL_HANDLE;
    VkPipeline pipeline_occ_query_ = VK_NULL_HANDLE;
    VkPipeline shadow_pipeline_depth_ = VK_NULL_HANDLE;
    VkPipeline shadow_pipeline_occ_query_ = VK_NULL_HANDLE;
    uint64_t pipeline_gen_ = 0;

    std::array<VkQueryPool, kFrameRing> view_query_pools_{};
    std::array<VkQueryPool, kFrameRing> shadow_query_pools_{};
    std::array<uint32_t, kFrameRing> view_query_counts_{};
    std::array<uint32_t, kFrameRing> shadow_query_counts_{};
    std::array<std::vector<uint32_t>, kFrameRing> view_query_scene_indices_{};
    std::array<std::vector<uint32_t>, kFrameRing> shadow_query_scene_indices_{};
    uint32_t max_view_query_count_ = 0;
    uint32_t max_shadow_query_count_ = 0;

    std::vector<MeshGPU> meshes_{};
    std::vector<ShapeInstance> instances_{};
    SceneElementSet view_cull_scene_{};
    SceneElementSet shadow_cull_scene_{};
    SceneCullingContext view_cull_ctx_{
        VisibilityHistoryPolicy{
            kOcclusionHideConfirmFrames,
            kOcclusionShowConfirmFrames}};
    SceneCullingContext shadow_cull_ctx_{
        VisibilityHistoryPolicy{
            kOcclusionHideConfirmFrames,
            kOcclusionShowConfirmFrames}};
    std::vector<uint32_t> render_view_scene_indices_{};
    std::vector<uint32_t> render_shadow_scene_indices_{};
    uint32_t aabb_mesh_index_ = 0;

    FreeCamera camera_{};
    float aspect_ = static_cast<float>(kWindowW) / static_cast<float>(kWindowH);
    glm::mat4 view_mtx_{1.0f};
    glm::mat4 proj_mtx_{1.0f};
    glm::mat4 vp_mtx_{1.0f};
    glm::mat4 light_view_proj_mtx_{1.0f};
    glm::vec3 sun_dir_ws_{0.0f, -1.0f, 0.0f};
    LightCamera light_cam_{};
    AABB shadow_caster_bounds_{};
    Frustum frustum_{};
    Frustum light_frustum_{};

    bool show_aabb_debug_ = false;
    bool render_lit_surfaces_ = true;
    bool use_multithread_recording_ = false;
    bool used_secondary_this_frame_ = false;
    bool enable_occlusion_ = true;
    bool apply_occlusion_this_frame_ = false;
    uint32_t occlusion_warmup_frames_ = 0;
    bool camera_prev_valid_ = false;
    glm::vec3 camera_prev_pos_{0.0f};
    float camera_prev_yaw_ = 0.0f;
    float camera_prev_pitch_ = 0.0f;
    CullingStats scene_stats_{};
    CullingStats shadow_stats_{};
};

} // namespace

int main()
{
    try
    {
        HelloSoftShadowCullingVkApp app{};
        app.run();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
