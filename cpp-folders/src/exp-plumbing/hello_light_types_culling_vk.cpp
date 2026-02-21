#define SDL_MAIN_HANDLED

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <SDL2/SDL.h>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/camera/camera_math.hpp>
#include <shs/camera/convention.hpp>
#include <shs/core/context.hpp>
#include <shs/core/units.hpp>
#include <shs/geometry/culling_runtime.hpp>
#include <shs/geometry/culling_software.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/geometry/volumes.hpp>
#include <shs/lighting/light_culling_runtime.hpp>
#include <shs/lighting/light_runtime.hpp>
#include <shs/platform/platform_input.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>
#include <shs/rhi/drivers/vulkan/vk_cmd_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_memory_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_shader_utils.hpp>
#include <shs/scene/scene_culling.hpp>
#include <shs/scene/scene_elements.hpp>

using namespace shs;

#ifndef SHS_VK_LIGHT_TYPES_CULLING_VERT_SPV
#error "SHS_VK_LIGHT_TYPES_CULLING_VERT_SPV is not defined"
#endif
#ifndef SHS_VK_LIGHT_TYPES_CULLING_FRAG_SPV
#error "SHS_VK_LIGHT_TYPES_CULLING_FRAG_SPV is not defined"
#endif

namespace
{
constexpr int kWindowW = 1200;
constexpr int kWindowH = 900;
constexpr uint32_t kFrameRing = 1u;
constexpr int kOccW = 320;
constexpr int kOccH = 240;
constexpr int kLightOccW = 240;
constexpr int kLightOccH = 180;
constexpr uint32_t kMaxLightsPerObject = kLightSelectionCapacity;
constexpr uint32_t kGpuMaxLights = 64u;
constexpr uint32_t kLightBinTileSize = 32u;
constexpr uint32_t kLightClusterDepthSlices = 16u;
constexpr float kCameraNear = 0.05f;
constexpr float kCameraFar = 300.0f;
constexpr bool kLightOcclusionDefault = false;
constexpr float kDemoFloorHalfExtentM = 24.0f * units::meter;
constexpr float kDemoFloorVisualSizeM = 48.0f * units::meter;

struct Vertex
{
    glm::vec3 pos{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
};

struct alignas(16) CameraUBO
{
    glm::mat4 view_proj{1.0f};
    glm::vec4 camera_pos{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 sun_dir_to_scene_ws{glm::normalize(glm::vec3(0.20f, -1.0f, 0.16f)), 0.0f};
};

struct alignas(16) DrawPush
{
    glm::mat4 model{1.0f};
    glm::vec4 base_color{1.0f};
    glm::uvec4 mode_pad{0u, 0u, 0u, 0u}; // x: lit mode, y: light count
    glm::uvec4 light_indices_01{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    glm::uvec4 light_indices_23{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
};
static_assert(sizeof(DrawPush) <= 128u, "DrawPush must fit minimum Vulkan push constant size");

struct alignas(16) GpuLight
{
    glm::vec4 position_range{0.0f};
    glm::vec4 color_intensity{0.0f};
    glm::vec4 direction_inner{0.0f};
    glm::vec4 axis_outer{0.0f};
    glm::vec4 up_shape_x{0.0f};
    glm::vec4 shape_attenuation{0.0f};
    glm::uvec4 type_shape_flags{0u};
};
static_assert(sizeof(GpuLight) % 16u == 0u, "GpuLight must stay 16-byte aligned");

struct alignas(16) LightUBO
{
    glm::uvec4 counts{0u}; // x: valid light count
    std::array<GpuLight, kGpuMaxLights> lights{};
};
static_assert(sizeof(LightUBO) % 16u == 0u, "LightUBO must stay 16-byte aligned");

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
};

struct FreeCamera
{
    glm::vec3 pos{0.0f, 4.2f, -15.5f};
    float yaw = glm::half_pi<float>();
    float pitch = -0.18f;
    float move_speed = 7.0f;
    float look_speed = 0.003f;
    static constexpr float kMouseSpikeThreshold = 240.0f;
    static constexpr float kMouseDeltaClamp = 90.0f;

    void update(const PlatformInputState& input, float dt)
    {
        if (input.right_mouse_down || input.left_mouse_down)
        {
            float mdx = input.mouse_dx;
            float mdy = input.mouse_dy;
            // WSL2 relative-mode can produce one-frame spikes.
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

inline glm::mat4 compose_model(const glm::vec3& pos, const glm::vec3& rot_euler)
{
    glm::mat4 model(1.0f);
    model = glm::translate(model, pos);
    model = glm::rotate(model, rot_euler.x, glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, rot_euler.y, glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, rot_euler.z, glm::vec3(0.0f, 0.0f, 1.0f));
    return model;
}

uint8_t to_u8(float v)
{
    return static_cast<uint8_t>(std::clamp(v * 255.0f, 0.0f, 255.0f));
}

float pseudo_random01(uint32_t seed)
{
    uint32_t x = seed;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

std::vector<uint32_t> make_line_indices_from_triangles(const std::vector<uint32_t>& tri_indices)
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

std::vector<Vertex> make_vertices_with_normals(const DebugMesh& mesh)
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

AABB compute_local_aabb_from_debug_mesh(const DebugMesh& mesh)
{
    AABB out{};
    if (mesh.vertices.empty())
    {
        out.minv = glm::vec3(-0.5f);
        out.maxv = glm::vec3(0.5f);
        return out;
    }
    out.minv = mesh.vertices[0];
    out.maxv = mesh.vertices[0];
    for (const glm::vec3& p : mesh.vertices)
    {
        out.expand(p);
    }
    return out;
}

AABB compute_scene_bounds(
    const std::vector<ShapeInstance>& instances,
    const std::vector<AABB>& mesh_local_aabbs,
    bool animated_only)
{
    AABB out{};
    bool any = false;
    for (const ShapeInstance& inst : instances)
    {
        if (animated_only && !inst.animated) continue;
        if (inst.mesh_index >= mesh_local_aabbs.size()) continue;
        const AABB box = transform_aabb(mesh_local_aabbs[inst.mesh_index], inst.model);
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
        out.minv = glm::vec3(-10.0f);
        out.maxv = glm::vec3(10.0f);
    }

    return out;
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
            return jolt::make_spot_light_volume(1.2f * ss, glm::radians(28.0f), 20);
        case DemoShapeKind::RectLightVolume:
            // For general visualization scaling, use a very small attenuation bound
            // so the shape draws reasonably as a panel rather than a giant cube.
            // Jolt BoxShape asserts if extents < 0.05f, so clamp minimum thickness.
            return jolt::make_rect_area_light_volume(glm::vec2(0.8f, 0.5f) * ss, std::max(0.1f * ss, 0.055f));
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
    const float step = full / static_cast<float>(div);

    mesh.vertices.reserve(static_cast<size_t>(verts_per_row) * static_cast<size_t>(verts_per_row));
    mesh.indices.reserve(static_cast<size_t>(div) * static_cast<size_t>(div) * 6u);

    for (int z = 0; z <= div; ++z)
    {
        for (int x = 0; x <= div; ++x)
        {
            const float px = -half_extent + static_cast<float>(x) * step;
            const float pz = -half_extent + static_cast<float>(z) * step;
            mesh.vertices.push_back(glm::vec3(px, 0.0f, pz));
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

void sync_instances_to_scene(SceneElementSet& scene, const std::vector<ShapeInstance>& instances)
{
    auto elems = scene.elements();
    for (size_t i = 0; i < instances.size() && i < elems.size(); ++i)
    {
        elems[i].geometry = instances[i].shape;
        elems[i].visible = true;
        elems[i].frustum_visible = true;
        elems[i].occluded = false;
        elems[i].enabled = true;
    }
}

void sync_lights_to_scene(SceneElementSet& scene, const std::vector<LightInstance>& lights)
{
    auto elems = scene.elements();
    for (size_t i = 0; i < lights.size() && i < elems.size(); ++i)
    {
        elems[i].geometry = lights[i].volume;
        elems[i].visible = true;
        elems[i].frustum_visible = true;
        elems[i].occluded = false;
        elems[i].enabled = true;
    }
}

GpuLight make_gpu_light(const LightInstance& light)
{
    GpuLight out{};
    out.position_range = light.packed.position_range;
    out.color_intensity = light.packed.color_intensity;
    out.direction_inner = light.packed.direction_spot;
    out.axis_outer = light.packed.axis_spot_outer;
    out.up_shape_x = light.packed.up_shape_x;
    out.shape_attenuation = light.packed.shape_attenuation;
    out.type_shape_flags = light.packed.type_shape_flags;
    return out;
}

class HelloLightTypesCullingVkApp
{
public:
    ~HelloLightTypesCullingVkApp()
    {
        cleanup();
    }

    void run()
    {
        jolt::init_jolt();
        init_sdl();
        init_backend();
        create_descriptor_resources();
        create_scene();
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
            "Light Types + Culling Demo (Vulkan)",
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
        init.app_name = "hello_light_types_culling_vk";
        if (!vk_->init(init)) throw std::runtime_error("Vulkan init failed");

        ctx_.set_primary_backend(vk_);
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

    uint32_t upload_debug_mesh(const DebugMesh& mesh)
    {
        if (mesh.vertices.empty() || mesh.indices.empty())
        {
            throw std::runtime_error("upload_debug_mesh: mesh is empty");
        }

        MeshGPU gpu{};
        const auto vertices = make_vertices_with_normals(mesh);
        const auto line_indices = make_line_indices_from_triangles(mesh.indices);

        const VkMemoryPropertyFlags host_mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        create_buffer(
            static_cast<VkDeviceSize>(vertices.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_mem,
            gpu.vertex,
            true);
        std::memcpy(gpu.vertex.mapped, vertices.data(), static_cast<size_t>(gpu.vertex.size));

        create_buffer(
            static_cast<VkDeviceSize>(mesh.indices.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_mem,
            gpu.tri_indices,
            true);
        std::memcpy(gpu.tri_indices.mapped, mesh.indices.data(), static_cast<size_t>(gpu.tri_indices.size));

        create_buffer(
            static_cast<VkDeviceSize>(line_indices.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_mem,
            gpu.line_indices,
            true);
        std::memcpy(gpu.line_indices.mapped, line_indices.data(), static_cast<size_t>(gpu.line_indices.size));

        gpu.tri_index_count = static_cast<uint32_t>(mesh.indices.size());
        gpu.line_index_count = static_cast<uint32_t>(line_indices.size());

        meshes_.push_back(gpu);
        mesh_cpu_.push_back(mesh);
        mesh_local_aabbs_.push_back(compute_local_aabb_from_debug_mesh(mesh));
        return static_cast<uint32_t>(meshes_.size() - 1u);
    }

    void create_scene()
    {
        instances_.clear();
        lights_.clear();
        meshes_.clear();
        mesh_cpu_.clear();
        mesh_local_aabbs_.clear();

        // Floor.
        {
            ShapeInstance floor{};
            floor.shape.shape = jolt::make_box(glm::vec3(kDemoFloorHalfExtentM, 0.12f * units::meter, kDemoFloorHalfExtentM));
            floor.base_pos = glm::vec3(0.0f, -0.12f * units::meter, 0.0f);
            floor.base_rot = glm::vec3(0.0f);
            floor.model = compose_model(floor.base_pos, floor.base_rot);
            floor.shape.transform = jolt::to_jph(floor.model);
            floor.shape.stable_id = 9000;
            floor.color = glm::vec3(0.44f, 0.44f, 0.46f);
            floor.animated = false;

            floor.mesh_index = upload_debug_mesh(make_tessellated_floor_mesh(kDemoFloorVisualSizeM, 64));
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

        uint32_t next_shape_id = 1;
        const int layer_count = 2;
        const int rows_per_layer = 6;
        const int cols_per_row = 8;
        const float col_spacing_x = 2.6f * units::meter;
        const float row_spacing_z = 2.4f * units::meter;
        const float layer_spacing_z = 9.0f * units::meter;
        const float base_y = 0.9f * units::meter;
        const float layer_y_step = 0.55f * units::meter;

        for (int layer = 0; layer < layer_count; ++layer)
        {
            const float layer_z = (-0.5f * static_cast<float>(layer_count - 1) + static_cast<float>(layer)) * layer_spacing_z;
            for (int row = 0; row < rows_per_layer; ++row)
            {
                const float row_z = layer_z + (-0.5f * static_cast<float>(rows_per_layer - 1) + static_cast<float>(row)) * row_spacing_z;
                const float zig = (((row + layer) & 1) != 0) ? (0.44f * col_spacing_x) : 0.0f;
                for (int col = 0; col < cols_per_row; ++col)
                {
                    const uint32_t logical_idx =
                        static_cast<uint32_t>(layer * rows_per_layer * cols_per_row + row * cols_per_row + col);
                    const DemoShapeKind kind = shape_kinds[(logical_idx * 7u + 3u) % shape_kinds.size()];
                    const float scale = 0.42f + 0.52f * pseudo_random01(logical_idx * 1664525u + 1013904223u);

                    ShapeInstance inst{};
                    inst.shape.shape = make_scaled_demo_shape(kind, scale);
                    inst.mesh_index = upload_debug_mesh(debug_mesh_from_shape(*inst.shape.shape, JPH::Mat44::sIdentity()));

                    inst.base_pos = glm::vec3(
                        (-0.5f * static_cast<float>(cols_per_row - 1) + static_cast<float>(col)) * col_spacing_x + zig,
                        base_y + layer_y_step * static_cast<float>(layer) + 0.18f * units::meter * static_cast<float>(col % 3),
                        row_z);
                    inst.base_rot = glm::vec3(
                        0.21f * pseudo_random01(logical_idx * 279470273u + 1u),
                        0.35f * pseudo_random01(logical_idx * 2246822519u + 7u),
                        0.19f * pseudo_random01(logical_idx * 3266489917u + 11u));
                    inst.angular_vel = glm::vec3(
                        0.10f + 0.14f * pseudo_random01(logical_idx * 747796405u + 13u),
                        0.09f + 0.16f * pseudo_random01(logical_idx * 2891336453u + 17u),
                        0.08f + 0.12f * pseudo_random01(logical_idx * 1181783497u + 19u));
                    inst.model = compose_model(inst.base_pos, inst.base_rot);
                    inst.shape.transform = jolt::to_jph(inst.model);
                    inst.shape.stable_id = next_shape_id++;
                    inst.color = color_for_demo_shape_kind(kind);
                    inst.animated = true;
                    instances_.push_back(std::move(inst));
                }
            }
        }

        unit_aabb_mesh_index_ = upload_debug_mesh(debug_mesh_from_aabb(AABB{glm::vec3(-0.5f), glm::vec3(0.5f)}));

        const AABB dynamic_scene_bounds = compute_scene_bounds(instances_, mesh_local_aabbs_, true);
        const glm::vec3 dynamic_center = dynamic_scene_bounds.center();
        const glm::vec3 dynamic_extent = glm::max(dynamic_scene_bounds.extent(), glm::vec3(6.0f * units::meter));

        const std::array<const ILightModel*, 4> light_models = {
            &point_model_,
            &spot_model_,
            &rect_model_,
            &tube_model_
        };

        const std::array<glm::vec3, 10> light_palette = {
            glm::vec3(0.98f, 0.45f, 0.50f),
            glm::vec3(0.45f, 0.82f, 1.00f),
            glm::vec3(0.55f, 1.00f, 0.60f),
            glm::vec3(1.00f, 0.85f, 0.48f),
            glm::vec3(0.92f, 0.52f, 1.00f),
            glm::vec3(1.00f, 0.62f, 0.40f),
            glm::vec3(0.62f, 0.78f, 1.00f),
            glm::vec3(0.90f, 1.00f, 0.60f),
            glm::vec3(1.00f, 0.58f, 0.78f),
            glm::vec3(0.60f, 0.98f, 0.96f)
        };

        uint32_t next_light_id = 50000;
        const uint32_t lights_per_type = 5;
        for (uint32_t type_i = 0; type_i < light_models.size(); ++type_i)
        {
            for (uint32_t li = 0; li < lights_per_type; ++li)
            {
                const uint32_t light_index = type_i * lights_per_type + li;
                const float r0 = pseudo_random01(light_index * 747796405u + 13u);
                const float r1 = pseudo_random01(light_index * 2891336453u + 17u);
                const float r2 = pseudo_random01(light_index * 1181783497u + 19u);
                const float r3 = pseudo_random01(light_index * 2246822519u + 23u);
                const float r4 = pseudo_random01(light_index * 3266489917u + 29u);
                const float r5 = pseudo_random01(light_index * 668265263u + 31u);

                LightInstance light{};
                light.model = light_models[type_i];
                light.props.color = light_palette[(light_index * 3u + type_i) % light_palette.size()] * (0.82f + 0.30f * r0);
                light.props.flags = LightFlagsDefault;

                switch (light.model->type())
                {
                    case LightType::Point:
                        light.props.range = 3.5f * units::meter + (2.0f * units::meter) * r1;
                        light.props.intensity = 2.0f + 1.0f * r2;
                        light.props.attenuation_model = LightAttenuationModel::Smooth;
                        light.props.attenuation_power = 1.25f;
                        break;
                    case LightType::Spot:
                        light.props.range = 5.0f * units::meter + (3.0f * units::meter) * r1;
                        light.props.intensity = 2.6f + 1.2f * r2;
                        light.props.inner_angle_rad = glm::radians(12.0f + 8.0f * r3);
                        light.props.outer_angle_rad = light.props.inner_angle_rad + glm::radians(8.0f + 12.0f * r4);
                        light.props.attenuation_model = LightAttenuationModel::Smooth;
                        light.props.attenuation_power = 1.30f;
                        break;
                    case LightType::RectArea:
                        light.props.range = 4.5f * units::meter + (2.5f * units::meter) * r1;
                        light.props.intensity = 1.9f + 0.8f * r2;
                        light.props.rect_half_extents = glm::vec2(
                            0.45f * units::meter + (0.50f * units::meter) * r3,
                            0.25f * units::meter + (0.30f * units::meter) * r4);
                        light.props.attenuation_model = LightAttenuationModel::InverseSquare;
                        light.props.attenuation_bias = 0.16f;
                        light.props.attenuation_power = 1.0f;
                        break;
                    case LightType::TubeArea:
                        light.props.range = 4.0f * units::meter + (2.8f * units::meter) * r1;
                        light.props.intensity = 2.0f + 0.9f * r2;
                        light.props.tube_half_length = 0.55f * units::meter + (0.60f * units::meter) * r3;
                        light.props.tube_radius = 0.10f * units::meter + (0.18f * units::meter) * r4;
                        light.props.attenuation_model = LightAttenuationModel::InverseSquare;
                        light.props.attenuation_bias = 0.14f;
                        light.props.attenuation_power = 1.0f;
                        break;
                    default:
                        break;
                }

                light.motion.orbit_center = dynamic_center + glm::vec3(
                    (r0 - 0.5f) * dynamic_extent.x * 0.50f,
                    1.5f * units::meter + (1.8f * units::meter) * r1,
                    (r2 - 0.5f) * dynamic_extent.z * 0.50f);
                light.motion.aim_center = dynamic_center + glm::vec3(
                    (r3 - 0.5f) * dynamic_extent.x * 0.25f,
                    0.9f * units::meter + (0.7f * units::meter) * r4,
                    (r5 - 0.5f) * dynamic_extent.z * 0.25f);
                light.motion.orbit_axis = normalize_or(glm::vec3(r2 - 0.5f, 1.0f, r3 - 0.5f), glm::vec3(0.0f, 1.0f, 0.0f));
                light.motion.radial_axis = normalize_or(glm::vec3(r4 - 0.5f, 0.2f * (r0 - 0.5f), r5 - 0.5f), glm::vec3(1.0f, 0.0f, 0.0f));
                light.motion.orbit_radius = 1.4f * units::meter + (3.5f * units::meter) * r4;
                light.motion.orbit_speed = 0.25f + 0.65f * r5;
                light.motion.orbit_phase = glm::two_pi<float>() * r3;
                light.motion.vertical_amplitude = 0.15f * units::meter + (0.55f * units::meter) * r2;
                light.motion.vertical_speed = 0.7f + 1.1f * r1;
                light.motion.direction_lead = 0.12f + 0.28f * r0;
                light.motion.vertical_aim_bias = -0.04f * units::meter - (0.10f * units::meter) * r5;

                update_light_motion(light, 0.0f);
                light.volume_model = light.model->volume_model_matrix(light.props);
                light.volume.shape = light.model->create_volume_shape(light.props);
                light.volume.transform = jolt::to_jph(light.volume_model);
                light.volume.stable_id = next_light_id++;
                light.packed = light.model->pack_for_culling(light.props);
                light.mesh_index = upload_debug_mesh(debug_mesh_from_shape(*light.volume.shape, JPH::Mat44::sIdentity()));
                lights_.push_back(std::move(light));
            }
        }

        view_cull_scene_.clear();
        view_cull_scene_.reserve(instances_.size());
        for (size_t i = 0; i < instances_.size(); ++i)
        {
            SceneElement elem{};
            elem.geometry = instances_[i].shape;
            elem.user_index = static_cast<uint32_t>(i);
            elem.visible = true;
            elem.frustum_visible = true;
            elem.occluded = false;
            elem.enabled = true;
            view_cull_scene_.add(std::move(elem));
        }

        light_cull_scene_.clear();
        light_cull_scene_.reserve(lights_.size());
        for (size_t i = 0; i < lights_.size(); ++i)
        {
            SceneElement elem{};
            elem.geometry = lights_[i].volume;
            elem.user_index = static_cast<uint32_t>(i);
            elem.visible = true;
            elem.frustum_visible = true;
            elem.occluded = false;
            elem.enabled = true;
            light_cull_scene_.add(std::move(elem));
        }
    }

    void create_descriptor_resources()
    {
        if (set_layout_ == VK_NULL_HANDLE)
        {
            std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[1].binding = 1;
            bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = static_cast<uint32_t>(bindings.size());
            ci.pBindings = bindings.data();
            if (vkCreateDescriptorSetLayout(vk_->device(), &ci, nullptr, &set_layout_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorSetLayout failed");
            }
        }

        if (descriptor_pool_ == VK_NULL_HANDLE)
        {
            VkDescriptorPoolSize ps{};
            ps.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ps.descriptorCount = 2u * kFrameRing;

            VkDescriptorPoolCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            ci.maxSets = kFrameRing;
            ci.poolSizeCount = 1;
            ci.pPoolSizes = &ps;
            if (vkCreateDescriptorPool(vk_->device(), &ci, nullptr, &descriptor_pool_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorPool failed");
            }
        }

        const VkMemoryPropertyFlags host_mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        std::array<VkDescriptorSetLayout, kFrameRing> layouts{};
        layouts.fill(set_layout_);

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
            create_buffer(
                sizeof(LightUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                host_mem,
                light_ubos_[i],
                true);
            camera_sets_[i] = sets[i];

            VkDescriptorBufferInfo cam_bi{};
            cam_bi.buffer = camera_ubos_[i].buffer;
            cam_bi.offset = 0;
            cam_bi.range = sizeof(CameraUBO);

            VkDescriptorBufferInfo light_bi{};
            light_bi.buffer = light_ubos_[i].buffer;
            light_bi.offset = 0;
            light_bi.range = sizeof(LightUBO);

            std::array<VkWriteDescriptorSet, 2> writes{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = camera_sets_[i];
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &cam_bi;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = camera_sets_[i];
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[1].pBufferInfo = &light_bi;

            vkUpdateDescriptorSets(vk_->device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
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
        if (pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(dev, pipeline_layout_, nullptr);
            pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    VkPipeline create_pipeline(VkPrimitiveTopology topology, VkPolygonMode polygon_mode)
    {
        const VkDevice dev = vk_->device();

        const std::vector<char> vs_code = vk_read_binary_file(SHS_VK_LIGHT_TYPES_CULLING_VERT_SPV);
        const std::vector<char> fs_code = vk_read_binary_file(SHS_VK_LIGHT_TYPES_CULLING_FRAG_SPV);
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
        rs.cullMode = (topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
        // We render with flipped-Y Vulkan viewport; with LH/clockwise mesh winding this maps to CCW front faces.
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;

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

    void create_pipelines()
    {
        destroy_pipelines();

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(DrawPush);

        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &set_layout_;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(vk_->device(), &pl, nullptr, &pipeline_layout_) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreatePipelineLayout failed");
        }

        pipeline_tri_ = create_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_POLYGON_MODE_FILL);
        pipeline_line_ = create_pipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_POLYGON_MODE_FILL);
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
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F3) out.toggle_front_face = true;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F4) out.toggle_shading_model = true;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F5) out.toggle_sky_mode = true;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F6) out.toggle_follow_camera = true;

            if (e.type == SDL_MOUSEMOTION)
            {
                if (!ignore_next_mouse_dt_)
                {
                    out.mouse_dx += static_cast<float>(e.motion.xrel);
                    out.mouse_dy += static_cast<float>(e.motion.yrel);
                }
            }
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_RIGHT) mouse_right_held_ = true;
            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_RIGHT) mouse_right_held_ = false;
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) mouse_left_held_ = true;
            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) mouse_left_held_ = false;

            if (e.type == SDL_WINDOWEVENT &&
                (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED || e.window.event == SDL_WINDOWEVENT_RESIZED))
            {
                if (vk_) vk_->request_resize(e.window.data1, e.window.data2);
            }
            if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_FOCUS_LOST)
            {
                mouse_right_held_ = false;
                mouse_left_held_ = false;
            }
        }

        const uint32_t ms = SDL_GetMouseState(nullptr, nullptr);
        if ((ms & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0) mouse_right_held_ = true;
        if ((ms & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0) mouse_left_held_ = true;
        if (!relative_mouse_mode_)
        {
            if ((ms & SDL_BUTTON(SDL_BUTTON_RIGHT)) == 0) mouse_right_held_ = false;
            if ((ms & SDL_BUTTON(SDL_BUTTON_LEFT)) == 0) mouse_left_held_ = false;
        }
        out.right_mouse_down = mouse_right_held_;
        out.left_mouse_down = mouse_left_held_;

        const uint8_t* ks = SDL_GetKeyboardState(nullptr);
        out.forward = ks[SDL_SCANCODE_W] != 0;
        out.backward = ks[SDL_SCANCODE_S] != 0;
        out.left = ks[SDL_SCANCODE_A] != 0;
        out.right = ks[SDL_SCANCODE_D] != 0;
        out.descend = ks[SDL_SCANCODE_Q] != 0;
        out.ascend = ks[SDL_SCANCODE_E] != 0;
        out.boost = ks[SDL_SCANCODE_LSHIFT] != 0;

        if (ignore_next_mouse_dt_) ignore_next_mouse_dt_ = false;

        const bool look_drag = out.right_mouse_down || out.left_mouse_down;
        if (look_drag != relative_mouse_mode_)
        {
            relative_mouse_mode_ = look_drag;
            SDL_SetRelativeMouseMode(relative_mouse_mode_ ? SDL_TRUE : SDL_FALSE);
            if (relative_mouse_mode_) ignore_next_mouse_dt_ = true;
            out.mouse_dx = 0.0f;
            out.mouse_dy = 0.0f;
        }

        return !out.quit;
    }

    void update_aspect_from_drawable()
    {
        int dw = 0;
        int dh = 0;
        SDL_Vulkan_GetDrawableSize(win_, &dw, &dh);
        if (dw > 0 && dh > 0)
        {
            viewport_w_ = static_cast<uint32_t>(dw);
            viewport_h_ = static_cast<uint32_t>(dh);
            aspect_ = static_cast<float>(dw) / static_cast<float>(dh);
        }
    }

    void update_scene_and_culling(float time_s)
    {
        for (ShapeInstance& inst : instances_)
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

        if (!freeze_lights_)
        {
            for (LightInstance& light : lights_)
            {
                update_light_motion(light, time_s);
            }
        }

        for (LightInstance& light : lights_)
        {
            light.volume_model = light.model->volume_model_matrix(light.props);
            light.volume.transform = jolt::to_jph(light.volume_model);
            light.packed = light.model->pack_for_culling(light.props);
            light.visible = true;
            light.frustum_visible = true;
            light.occluded = false;
        }

        sync_instances_to_scene(view_cull_scene_, instances_);
        sync_lights_to_scene(light_cull_scene_, lights_);

        view_matrix_ = camera_.view_matrix();
        proj_matrix_ = perspective_lh_no(glm::radians(60.0f), aspect_, kCameraNear, kCameraFar);
        view_proj_matrix_ = proj_matrix_ * view_matrix_;
        frustum_ = extract_frustum_planes(view_proj_matrix_);

        view_cull_ctx_.run_frustum(view_cull_scene_, frustum_);
        view_cull_ctx_.run_software_occlusion(
            view_cull_scene_,
            enable_scene_occlusion_,
            std::span<float>(occlusion_depth_.data(), occlusion_depth_.size()),
            kOccW,
            kOccH,
            view_matrix_,
            view_proj_matrix_,
            [&](const SceneElement& elem, uint32_t, std::span<float> depth_span) {
                if (elem.user_index >= instances_.size()) return;
                const ShapeInstance& inst = instances_[elem.user_index];
                if (inst.mesh_index >= mesh_cpu_.size()) return;
                culling_sw::rasterize_mesh_depth_transformed(
                    depth_span,
                    kOccW,
                    kOccH,
                    mesh_cpu_[inst.mesh_index],
                    inst.model,
                    view_proj_matrix_);
            });
        (void)view_cull_ctx_.apply_frustum_fallback_if_needed(
            view_cull_scene_,
            enable_scene_occlusion_,
            true,
            0u);

        light_cull_ctx_.run_frustum(light_cull_scene_, frustum_);
        light_cull_ctx_.run_software_occlusion(
            light_cull_scene_,
            enable_light_occlusion_,
            std::span<float>(light_occlusion_depth_.data(), light_occlusion_depth_.size()),
            kLightOccW,
            kLightOccH,
            view_matrix_,
            view_proj_matrix_,
            [&](const SceneElement& elem, uint32_t, std::span<float> depth_span) {
                if (elem.user_index >= lights_.size()) return;
                const LightInstance& light = lights_[elem.user_index];
                if (light.mesh_index >= mesh_cpu_.size()) return;
                culling_sw::rasterize_mesh_depth_transformed(
                    depth_span,
                    kLightOccW,
                    kLightOccH,
                    mesh_cpu_[light.mesh_index],
                    light.volume_model,
                    view_proj_matrix_);
            });
        (void)light_cull_ctx_.apply_frustum_fallback_if_needed(
            light_cull_scene_,
            enable_light_occlusion_,
            true,
            0u);

        const auto view_elems = view_cull_scene_.elements();
        for (size_t i = 0; i < instances_.size() && i < view_elems.size(); ++i)
        {
            instances_[i].visible = view_elems[i].visible;
            instances_[i].frustum_visible = view_elems[i].frustum_visible;
            instances_[i].occluded = view_elems[i].occluded;
        }

        const auto light_elems = light_cull_scene_.elements();
        for (size_t i = 0; i < lights_.size() && i < light_elems.size(); ++i)
        {
            lights_[i].visible = light_elems[i].visible;
            lights_[i].frustum_visible = light_elems[i].frustum_visible;
            lights_[i].occluded = light_elems[i].occluded;
        }

        object_stats_ = view_cull_ctx_.stats();
        light_stats_ = light_cull_ctx_.stats();

        draw_scene_indices_ = view_cull_ctx_.visible_indices();
        draw_stats_ = object_stats_;
        if (!view_cull_scene_.empty())
        {
            const uint32_t floor_scene_idx = 0u;
            if (floor_scene_idx < view_elems.size() && view_elems[floor_scene_idx].frustum_visible)
            {
                if (std::find(draw_scene_indices_.begin(), draw_scene_indices_.end(), floor_scene_idx) == draw_scene_indices_.end())
                {
                    draw_scene_indices_.push_back(floor_scene_idx);
                    draw_stats_.visible_count += 1u;
                    if (draw_stats_.occluded_count > 0u) draw_stats_.occluded_count -= 1u;
                    normalize_culling_stats(draw_stats_);
                }
            }
        }

        visible_light_scene_indices_ = light_cull_ctx_.visible_indices();

        light_bin_cfg_.mode = light_culling_mode_;
        light_bin_cfg_.tile_size = kLightBinTileSize;
        light_bin_cfg_.cluster_depth_slices = kLightClusterDepthSlices;
        light_bin_cfg_.z_near = kCameraNear;
        light_bin_cfg_.z_far = kCameraFar;

        TileViewDepthRange tile_depth_range{};
        std::span<const float> tile_min_depth{};
        std::span<const float> tile_max_depth{};
        if (light_culling_mode_ == LightCullingMode::TiledDepthRange)
        {
            tile_depth_range = build_tile_view_depth_range_from_scene(
                std::span<const uint32_t>(draw_scene_indices_.data(), draw_scene_indices_.size()),
                view_cull_scene_,
                view_matrix_,
                view_proj_matrix_,
                viewport_w_,
                viewport_h_,
                kLightBinTileSize,
                kCameraNear,
                kCameraFar);

            if (tile_depth_range.valid())
            {
                tile_min_depth = std::span<const float>(tile_depth_range.min_view_depth.data(), tile_depth_range.min_view_depth.size());
                tile_max_depth = std::span<const float>(tile_depth_range.max_view_depth.data(), tile_depth_range.max_view_depth.size());
            }
        }

        light_bin_data_ = build_light_bin_culling(
            std::span<const uint32_t>(visible_light_scene_indices_.data(), visible_light_scene_indices_.size()),
            light_cull_scene_,
            view_proj_matrix_,
            viewport_w_,
            viewport_h_,
            light_bin_cfg_,
            tile_min_depth,
            tile_max_depth);
    }

    void bind_and_draw_mesh(
        VkCommandBuffer cmd,
        VkDescriptorSet camera_set,
        const MeshGPU& mesh,
        const glm::mat4& model,
        const glm::vec3& color,
        const LightSelection* selection,
        bool triangle_fill,
        bool lit_mode)
    {
        const VkBuffer vb = mesh.vertex.buffer;
        const VkDeviceSize vb_off = 0;
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

        VkBuffer ib = VK_NULL_HANDLE;
        uint32_t index_count = 0;
        if (triangle_fill)
        {
            ib = mesh.tri_indices.buffer;
            index_count = mesh.tri_index_count;
        }
        else
        {
            ib = mesh.line_indices.buffer;
            index_count = mesh.line_index_count;
        }

        if (ib == VK_NULL_HANDLE || index_count == 0) return;
        vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);

        DrawPush push{};
        push.model = model;
        push.base_color = glm::vec4(glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f)), 1.0f);
        push.mode_pad.x = lit_mode ? 1u : 0u;
        if (selection != nullptr)
        {
            push.mode_pad.y = std::min(selection->count, kMaxLightsPerObject);
            push.light_indices_01 = glm::uvec4(
                (push.mode_pad.y > 0u) ? selection->indices[0] : 0xffffffffu,
                (push.mode_pad.y > 1u) ? selection->indices[1] : 0xffffffffu,
                (push.mode_pad.y > 2u) ? selection->indices[2] : 0xffffffffu,
                (push.mode_pad.y > 3u) ? selection->indices[3] : 0xffffffffu);
            push.light_indices_23 = glm::uvec4(
                (push.mode_pad.y > 4u) ? selection->indices[4] : 0xffffffffu,
                (push.mode_pad.y > 5u) ? selection->indices[5] : 0xffffffffu,
                (push.mode_pad.y > 6u) ? selection->indices[6] : 0xffffffffu,
                (push.mode_pad.y > 7u) ? selection->indices[7] : 0xffffffffu);
        }
        vkCmdPushConstants(
            cmd,
            pipeline_layout_,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(DrawPush),
            &push);
        vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
    }

    void record_draws(VkCommandBuffer cmd, VkDescriptorSet camera_set)
    {
        last_light_links_total_ = 0;
        last_max_lights_per_object_ = 0;
        last_light_candidates_total_ = 0;
        last_max_light_candidates_ = 0;

        if (render_lit_surfaces_)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_tri_);
        }
        else
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_line_);
        }

        for (const uint32_t scene_idx : draw_scene_indices_)
        {
            if (scene_idx >= view_cull_scene_.size()) continue;
            const uint32_t obj_idx = view_cull_scene_[scene_idx].user_index;
            if (obj_idx >= instances_.size()) continue;

            const ShapeInstance& inst = instances_[obj_idx];
            if (inst.mesh_index >= meshes_.size()) continue;

            LightSelection draw_selection{};
            const LightSelection* draw_selection_ptr = nullptr;
            if (render_lit_surfaces_)
            {
                const AABB world_box = inst.shape.world_aabb();
                const std::span<const uint32_t> candidate_light_scene_indices =
                    gather_light_scene_candidates_for_aabb(
                        light_bin_data_,
                        world_box,
                        view_matrix_,
                        view_proj_matrix_,
                        light_candidate_scene_scratch_);

                last_light_candidates_total_ += candidate_light_scene_indices.size();
                last_max_light_candidates_ = std::max(last_max_light_candidates_, static_cast<uint32_t>(candidate_light_scene_indices.size()));

                const LightSelection selection = collect_object_lights(
                    world_box,
                    candidate_light_scene_indices,
                    light_cull_scene_,
                    lights_,
                    light_object_cull_mode_);

                for (uint32_t i = 0; i < selection.count; ++i)
                {
                    const uint32_t idx = selection.indices[i];
                    if (idx >= kGpuMaxLights) continue;
                    draw_selection.indices[draw_selection.count] = idx;
                    draw_selection.dist2[draw_selection.count] = selection.dist2[i];
                    ++draw_selection.count;
                }
                draw_selection_ptr = &draw_selection;

                last_light_links_total_ += draw_selection.count;
                last_max_lights_per_object_ = std::max(last_max_lights_per_object_, draw_selection.count);
            }

            bind_and_draw_mesh(
                cmd,
                camera_set,
                meshes_[inst.mesh_index],
                inst.model,
                inst.color,
                draw_selection_ptr,
                render_lit_surfaces_,
                render_lit_surfaces_);
        }

        if (show_aabb_debug_ && unit_aabb_mesh_index_ < meshes_.size())
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_line_);
            for (const uint32_t scene_idx : draw_scene_indices_)
            {
                if (scene_idx >= view_cull_scene_.size()) continue;
                const uint32_t obj_idx = view_cull_scene_[scene_idx].user_index;
                if (obj_idx >= instances_.size()) continue;

                const ShapeInstance& inst = instances_[obj_idx];
                const AABB box = inst.shape.world_aabb();
                const glm::vec3 center = (box.minv + box.maxv) * 0.5f;
                const glm::vec3 size = glm::max(box.maxv - box.minv, glm::vec3(1e-4f));
                const glm::mat4 aabb_model =
                    glm::translate(glm::mat4(1.0f), center) *
                    glm::scale(glm::mat4(1.0f), size);

                bind_and_draw_mesh(
                    cmd,
                    camera_set,
                    meshes_[unit_aabb_mesh_index_],
                    aabb_model,
                    glm::vec3(1.0f, 0.94f, 0.31f),
                    nullptr,
                    false,
                    false);
            }
        }

        if (draw_light_volumes_ && !render_lit_surfaces_)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_line_);
            for (const uint32_t light_scene_idx : visible_light_scene_indices_)
            {
                if (light_scene_idx >= light_cull_scene_.size()) continue;
                const uint32_t light_idx = light_cull_scene_[light_scene_idx].user_index;
                if (light_idx >= lights_.size()) continue;
                const LightInstance& light = lights_[light_idx];
                if (light.mesh_index >= meshes_.size()) continue;

                const glm::vec3 lc = glm::clamp(light.props.color * 1.05f, glm::vec3(0.0f), glm::vec3(1.0f));
                bind_and_draw_mesh(
                    cmd,
                    camera_set,
                    meshes_[light.mesh_index],
                    light.volume_model,
                    lc,
                    nullptr,
                    false,
                    false);
            }
        }

        if (!draw_scene_indices_.empty())
        {
            last_avg_lights_per_object_ = static_cast<float>(last_light_links_total_) / static_cast<float>(draw_scene_indices_.size());
            last_avg_light_candidates_per_object_ = static_cast<float>(last_light_candidates_total_) / static_cast<float>(draw_scene_indices_.size());
        }
        else
        {
            last_avg_lights_per_object_ = 0.0f;
            last_avg_light_candidates_per_object_ = 0.0f;
        }
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
        CameraUBO cam{};
        cam.view_proj = view_proj_matrix_;
        cam.camera_pos = glm::vec4(camera_.pos, 1.0f);
        cam.sun_dir_to_scene_ws = glm::vec4(glm::normalize(glm::vec3(0.20f, -1.0f, 0.16f)), 0.0f);
        std::memcpy(camera_ubos_[ring].mapped, &cam, sizeof(CameraUBO));

        LightUBO light_ubo{};
        const uint32_t light_count = std::min(static_cast<uint32_t>(lights_.size()), kGpuMaxLights);
        light_ubo.counts = glm::uvec4(light_count, 0u, 0u, 0u);
        for (uint32_t i = 0; i < light_count; ++i)
        {
            light_ubo.lights[i] = make_gpu_light(lights_[i]);
        }
        std::memcpy(light_ubos_[ring].mapped, &light_ubo, sizeof(LightUBO));

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(fi.cmd, &bi) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBeginCommandBuffer failed");
        }

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

        vkCmdBeginRenderPass(fi.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vk_cmd_set_viewport_scissor(fi.cmd, fi.extent.width, fi.extent.height, true);
        record_draws(fi.cmd, camera_sets_[ring]);
        vkCmdEndRenderPass(fi.cmd);

        if (vkEndCommandBuffer(fi.cmd) != VK_SUCCESS)
        {
            throw std::runtime_error("vkEndCommandBuffer failed");
        }

        vk_->end_frame(fi);
        ctx_.frame_index++;
    }

    void update_title(float avg_ms)
    {
        char title[512];
        std::snprintf(
            title,
            sizeof(title),
            "Light Types Culling (VK) | Obj F:%u O:%u V:%u | Light F:%u O:%u V:%u | Cand %.2f (max %u) | L/Obj %.2f (max %u) | LMode:%s | LCull:%s | Occ:%s/%s | Vol:%s | %s | %.2f ms",
            draw_stats_.frustum_visible_count,
            draw_stats_.occluded_count,
            draw_stats_.visible_count,
            light_stats_.frustum_visible_count,
            light_stats_.occluded_count,
            light_stats_.visible_count,
            last_avg_light_candidates_per_object_,
            last_max_light_candidates_,
            last_avg_lights_per_object_,
            last_max_lights_per_object_,
            light_culling_mode_name(light_culling_mode_),
            light_object_cull_mode_name(light_object_cull_mode_),
            enable_scene_occlusion_ ? "ON" : "OFF",
            enable_light_occlusion_ ? "ON" : "OFF",
            draw_light_volumes_ ? "ON" : "OFF",
            render_lit_surfaces_ ? "Lit" : "Debug",
            avg_ms);
        SDL_SetWindowTitle(win_, title);
    }

    void main_loop()
    {
        std::printf(
            "Controls: LMB/RMB drag look, WASD+QE move, Shift boost | "
            "L lit/debug, B AABB, F1 light volumes, F2 scene occlusion, F3 light occlusion, F4 light/object culling, F5 freeze lights, F6 light bin mode\n");

        auto t0 = std::chrono::steady_clock::now();
        auto prev = t0;
        auto title_tick = t0;
        float ema_ms = 16.0f;

        while (true)
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
            if (input.cycle_debug_view) draw_light_volumes_ = !draw_light_volumes_;
            if (input.cycle_cull_mode) enable_scene_occlusion_ = !enable_scene_occlusion_;
            if (input.toggle_front_face) enable_light_occlusion_ = !enable_light_occlusion_;
            if (input.toggle_shading_model) light_object_cull_mode_ = next_light_object_cull_mode(light_object_cull_mode_);
            if (input.toggle_sky_mode) freeze_lights_ = !freeze_lights_;
            if (input.toggle_follow_camera) light_culling_mode_ = next_light_culling_mode(light_culling_mode_);

            update_aspect_from_drawable();
            camera_.update(input, dt);
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

            std::printf(
                "Obj F:%u O:%u V:%u | Light F:%u O:%u V:%u | Cand:%4.2f max:%u | L/Obj:%4.2f max:%u | LMode:%s | LCull:%s | Occ:%s/%s | Vol:%s | Mode:%s\r",
                draw_stats_.frustum_visible_count,
                draw_stats_.occluded_count,
                draw_stats_.visible_count,
                light_stats_.frustum_visible_count,
                light_stats_.occluded_count,
                light_stats_.visible_count,
                last_avg_light_candidates_per_object_,
                last_max_light_candidates_,
                last_avg_lights_per_object_,
                last_max_lights_per_object_,
                light_culling_mode_name(light_culling_mode_),
                light_object_cull_mode_name(light_object_cull_mode_),
                enable_scene_occlusion_ ? "ON " : "OFF",
                enable_light_occlusion_ ? "ON " : "OFF",
                draw_light_volumes_ ? "ON " : "OFF",
                render_lit_surfaces_ ? "Lit  " : "Debug");
            std::fflush(stdout);
        }

        std::printf("\n");

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
            for (auto& b : light_ubos_)
            {
                destroy_buffer(b);
            }

            destroy_pipelines();

            if (descriptor_pool_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(vk_->device(), descriptor_pool_, nullptr);
                descriptor_pool_ = VK_NULL_HANDLE;
            }
            if (set_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(vk_->device(), set_layout_, nullptr);
                set_layout_ = VK_NULL_HANDLE;
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

private:
    bool cleaned_up_ = false;
    bool sdl_ready_ = false;
    SDL_Window* win_ = nullptr;

    Context ctx_{};
    std::vector<std::unique_ptr<IRenderBackend>> keep_{};
    VulkanRenderBackend* vk_ = nullptr;

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    std::array<GpuBuffer, kFrameRing> camera_ubos_{};
    std::array<GpuBuffer, kFrameRing> light_ubos_{};
    std::array<VkDescriptorSet, kFrameRing> camera_sets_{};

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_tri_ = VK_NULL_HANDLE;
    VkPipeline pipeline_line_ = VK_NULL_HANDLE;
    uint64_t pipeline_gen_ = 0;

    std::vector<MeshGPU> meshes_{};
    std::vector<DebugMesh> mesh_cpu_{};
    std::vector<AABB> mesh_local_aabbs_{};

    std::vector<ShapeInstance> instances_{};
    std::vector<LightInstance> lights_{};

    uint32_t unit_aabb_mesh_index_ = 0;

    FreeCamera camera_{};
    float aspect_ = static_cast<float>(kWindowW) / static_cast<float>(kWindowH);
    uint32_t viewport_w_ = static_cast<uint32_t>(kWindowW);
    uint32_t viewport_h_ = static_cast<uint32_t>(kWindowH);
    glm::mat4 view_matrix_{1.0f};
    glm::mat4 proj_matrix_{1.0f};
    glm::mat4 view_proj_matrix_{1.0f};
    Frustum frustum_{};

    SceneElementSet view_cull_scene_{};
    SceneElementSet light_cull_scene_{};
    SceneCullingContext view_cull_ctx_{};
    SceneCullingContext light_cull_ctx_{};

    std::vector<float> occlusion_depth_{static_cast<size_t>(kOccW) * static_cast<size_t>(kOccH), 1.0f};
    std::vector<float> light_occlusion_depth_{static_cast<size_t>(kLightOccW) * static_cast<size_t>(kLightOccH), 1.0f};

    std::vector<uint32_t> draw_scene_indices_{};
    std::vector<uint32_t> visible_light_scene_indices_{};

    CullingStats object_stats_{};
    CullingStats light_stats_{};
    CullingStats draw_stats_{};

    uint64_t last_light_links_total_ = 0;
    uint32_t last_max_lights_per_object_ = 0;
    float last_avg_lights_per_object_ = 0.0f;
    uint64_t last_light_candidates_total_ = 0;
    uint32_t last_max_light_candidates_ = 0;
    float last_avg_light_candidates_per_object_ = 0.0f;

    bool show_aabb_debug_ = false;
    bool render_lit_surfaces_ = true;
    bool draw_light_volumes_ = true;
    bool enable_scene_occlusion_ = true;
    bool enable_light_occlusion_ = kLightOcclusionDefault;
    bool freeze_lights_ = false;
    LightCullingMode light_culling_mode_ = LightCullingMode::Clustered;
    LightObjectCullMode light_object_cull_mode_ = LightObjectCullMode::VolumeAabb;
    LightBinCullingConfig light_bin_cfg_{};
    LightBinCullingData light_bin_data_{};
    std::vector<uint32_t> light_candidate_scene_scratch_{};

    bool relative_mouse_mode_ = false;
    bool ignore_next_mouse_dt_ = false;
    bool mouse_right_held_ = false;
    bool mouse_left_held_ = false;

    PointLightModel point_model_{};
    SpotLightModel spot_model_{};
    RectAreaLightModel rect_model_{};
    TubeAreaLightModel tube_model_{};
};

} // namespace

int main()
{
    try
    {
        HelloLightTypesCullingVkApp app{};
        app.run();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
