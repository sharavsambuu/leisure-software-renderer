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
#include <shs/geometry/culling_runtime.hpp>
#include <shs/geometry/culling_visibility.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/scene/scene_instance.hpp>
#include <shs/platform/platform_input.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>
#include <shs/rhi/drivers/vulkan/vk_cmd_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_memory_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_shader_utils.hpp>

using namespace shs;

#ifndef SHS_VK_CULLING_VERT_SPV
#error "SHS_VK_CULLING_VERT_SPV is not defined"
#endif
#ifndef SHS_VK_CULLING_FRAG_SPV
#error "SHS_VK_CULLING_FRAG_SPV is not defined"
#endif

namespace
{
constexpr int kWindowW = 1200;
constexpr int kWindowH = 900;
// Vulkan backend currently runs with max_frames_in_flight = 1, so keep ring resources in lockstep.
constexpr uint32_t kFrameRing = 1u;
const glm::vec3 kSunLightDirWs = glm::normalize(glm::vec3(0.20f, -1.0f, 0.16f));
constexpr uint8_t kOcclusionHideConfirmFrames = 2u;
constexpr uint8_t kOcclusionShowConfirmFrames = 1u;
constexpr uint64_t kOcclusionMinVisibleSamples = 1u;
constexpr uint32_t kOcclusionWarmupFramesAfterCameraMove = 0u;

struct Vertex
{
    glm::vec3 pos{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
};

struct alignas(16) CameraUBO
{
    glm::mat4 view_proj{1.0f};
    glm::vec4 camera_pos{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 light_dir_ws{kSunLightDirWs, 0.0f};
};

struct alignas(16) DrawPush
{
    glm::mat4 model{1.0f};
    glm::vec4 base_color{1.0f};
    glm::uvec4 mode_pad{0u, 0u, 0u, 0u};
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

struct FreeCamera
{
    glm::vec3 pos{0.0f, 14.0f, -28.0f};
    float yaw = glm::half_pi<float>();
    float pitch = -0.25f;
    float move_speed = 20.0f;
    float look_speed = 0.003f;
    static constexpr float kMouseSpikeThreshold = 240.0f;
    static constexpr float kMouseDeltaClamp = 90.0f;

    void update(const PlatformInputState& input, float dt)
    {
        if (input.right_mouse_down || input.left_mouse_down)
        {
            float mdx = input.mouse_dx;
            float mdy = input.mouse_dy;
            // WSL2 relative-mode occasionally reports large one-frame spikes.
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

class HelloOcclusionCullingVkApp
{
public:
    ~HelloOcclusionCullingVkApp()
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
            "Occlusion + Frustum Culling Demo (Vulkan)",
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
        init.app_name = "hello_occlusion_culling_vk";
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
        return static_cast<uint32_t>(meshes_.size() - 1u);
    }

    void create_scene()
    {
        instances_.clear();

        // Floor
        {
            SceneInstance floor{};
            floor.geometry.shape = jolt::make_box(glm::vec3(50.0f, 0.1f, 50.0f));
            floor.anim.base_pos = glm::vec3(0.0f, -0.2f, 0.0f);
            floor.anim.base_rot = glm::vec3(0.0f);
            floor.geometry.transform = jolt::to_jph(compose_model(floor.anim.base_pos, floor.anim.base_rot));
            floor.geometry.stable_id = 9000;
            floor.tint_color = glm::vec3(0.18f, 0.18f, 0.22f);
            floor.anim.animated = false;

            const DebugMesh floor_mesh = debug_mesh_from_shape(*floor.geometry.shape, JPH::Mat44::sIdentity());
            floor.user_index = upload_debug_mesh(floor_mesh);
            instances_.push_back(floor);
        }

        const std::vector<glm::vec3> custom_hull_verts = {
            {-0.8f, -0.7f, -0.4f},
            { 0.9f, -0.6f, -0.5f},
            { 1.0f,  0.4f, -0.1f},
            {-0.7f,  0.6f, -0.2f},
            {-0.3f, -0.4f,  0.9f},
            { 0.4f,  0.7f,  0.8f},
        };

        MeshData wedge_mesh{};
        wedge_mesh.positions = {
            {-0.9f, -0.6f, -0.6f},
            { 0.9f, -0.6f, -0.6f},
            { 0.0f,  0.8f, -0.6f},
            {-0.9f, -0.6f,  0.6f},
            { 0.9f, -0.6f,  0.6f},
            { 0.0f,  0.8f,  0.6f},
        };
        wedge_mesh.indices = {
            0, 1, 2,
            5, 4, 3,
            0, 3, 4, 0, 4, 1,
            1, 4, 5, 1, 5, 2,
            2, 5, 3, 2, 3, 0
        };

        const JPH::ShapeRefC sphere_shape = jolt::make_sphere(1.0f);
        const JPH::ShapeRefC box_shape = jolt::make_box(glm::vec3(0.9f, 0.7f, 0.6f));
        const JPH::ShapeRefC capsule_shape = jolt::make_capsule(0.9f, 0.45f);
        const JPH::ShapeRefC cylinder_shape = jolt::make_cylinder(0.9f, 0.5f);
        const JPH::ShapeRefC tapered_capsule_shape = jolt::make_tapered_capsule(0.9f, 0.25f, 0.65f);
        const JPH::ShapeRefC convex_hull_shape = jolt::make_convex_hull(custom_hull_verts);
        const JPH::ShapeRefC mesh_shape = jolt::make_mesh_shape(wedge_mesh);
        const JPH::ShapeRefC convex_from_mesh_shape = jolt::make_convex_hull_from_mesh(wedge_mesh);
        const JPH::ShapeRefC point_light_volume_shape = jolt::make_point_light_volume(1.0f);
        const JPH::ShapeRefC spot_light_volume_shape = jolt::make_spot_light_volume(1.2f, glm::radians(28.0f), 20);
        const JPH::ShapeRefC rect_light_volume_shape = jolt::make_rect_area_light_volume(glm::vec2(0.8f, 0.5f), 0.1f);
        const JPH::ShapeRefC tube_light_volume_shape = jolt::make_tube_area_light_volume(0.9f, 0.35f);

        struct ShapeTypeDef
        {
            JPH::ShapeRefC shape;
            glm::vec3 color;
            uint32_t mesh_index = 0;
        };

        std::vector<ShapeTypeDef> shape_types = {
            {sphere_shape,             {0.95f, 0.35f, 0.35f}, 0u},
            {box_shape,                {0.35f, 0.90f, 0.45f}, 0u},
            {capsule_shape,            {0.35f, 0.55f, 0.95f}, 0u},
            {cylinder_shape,           {0.95f, 0.80f, 0.30f}, 0u},
            {tapered_capsule_shape,    {0.80f, 0.40f, 0.95f}, 0u},
            {convex_hull_shape,        {0.30f, 0.85f, 0.90f}, 0u},
            {mesh_shape,               {0.92f, 0.55f, 0.25f}, 0u},
            {convex_from_mesh_shape,   {0.55f, 0.95f, 0.55f}, 0u},
            {point_light_volume_shape, {0.95f, 0.45f, 0.65f}, 0u},
            {spot_light_volume_shape,  {0.95f, 0.70f, 0.35f}, 0u},
            {rect_light_volume_shape,  {0.35f, 0.95f, 0.80f}, 0u},
            {tube_light_volume_shape,  {0.70f, 0.65f, 0.95f}, 0u},
        };

        for (auto& type : shape_types)
        {
            const DebugMesh mesh = debug_mesh_from_shape(*type.shape, JPH::Mat44::sIdentity());
            type.mesh_index = upload_debug_mesh(mesh);
        }

        uint32_t next_id = 0;
        const int copies_per_type = 6;
        const float spacing_x = 5.6f;
        const float spacing_z = 4.8f;
        for (size_t t = 0; t < shape_types.size(); ++t)
        {
            for (int c = 0; c < copies_per_type; ++c)
            {
                SceneInstance inst{};
                inst.geometry.shape = shape_types[t].shape;
                inst.user_index = shape_types[t].mesh_index;
                inst.anim.base_pos = glm::vec3(
                    (-0.5f * (copies_per_type - 1) + static_cast<float>(c)) * spacing_x,
                    1.25f + 0.25f * static_cast<float>(c % 3),
                    (-0.5f * static_cast<float>(shape_types.size() - 1) + static_cast<float>(t)) * spacing_z);
                inst.anim.base_rot = glm::vec3(
                    0.17f * static_cast<float>(c),
                    0.23f * static_cast<float>(t),
                    0.11f * static_cast<float>(c + static_cast<int>(t)));
                inst.anim.angular_vel = glm::vec3(
                    0.30f + 0.07f * static_cast<float>((c + static_cast<int>(t)) % 5),
                    0.42f + 0.06f * static_cast<float>(c % 4),
                    0.36f + 0.05f * static_cast<float>(static_cast<int>(t) % 6));
                inst.geometry.transform = jolt::to_jph(compose_model(inst.anim.base_pos, inst.anim.base_rot));
                inst.geometry.stable_id = next_id++;
                inst.tint_color = shape_types[t].color;
                inst.anim.animated = true;
                instances_.push_back(std::move(inst));
            }
        }

        // Unit cube for AABB wire overlay (scale/translate in model matrix).
        {
            AABB unit{};
            unit.minv = glm::vec3(-0.5f);
            unit.maxv = glm::vec3(0.5f);
            const DebugMesh unit_mesh = debug_mesh_from_aabb(unit);
            aabb_mesh_index_ = upload_debug_mesh(unit_mesh);
        }
    }

    void create_occlusion_query_resources()
    {
        destroy_occlusion_query_resources();
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;

        max_query_count_ = std::max<uint32_t>(1u, static_cast<uint32_t>(instances_.size()));
        for (uint32_t i = 0; i < kFrameRing; ++i)
        {
            VkQueryPoolCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            ci.queryType = VK_QUERY_TYPE_OCCLUSION;
            ci.queryCount = max_query_count_;
            if (vkCreateQueryPool(vk_->device(), &ci, nullptr, &occlusion_query_pools_[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateQueryPool failed");
            }
            occlusion_query_counts_[i] = 0;
            occlusion_query_instances_[i].clear();
        }
    }

    void destroy_occlusion_query_resources()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        for (uint32_t i = 0; i < kFrameRing; ++i)
        {
            if (occlusion_query_pools_[i] != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(vk_->device(), occlusion_query_pools_[i], nullptr);
                occlusion_query_pools_[i] = VK_NULL_HANDLE;
            }
            occlusion_query_counts_[i] = 0;
            occlusion_query_instances_[i].clear();
        }
        max_query_count_ = 0;
    }

    void create_descriptor_resources()
    {
        if (set_layout_ == VK_NULL_HANDLE)
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
            if (vkCreateDescriptorSetLayout(vk_->device(), &ci, nullptr, &set_layout_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorSetLayout failed");
            }
        }

        if (descriptor_pool_ == VK_NULL_HANDLE)
        {
            VkDescriptorPoolSize ps{};
            ps.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ps.descriptorCount = kFrameRing;

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
        if (pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(dev, pipeline_layout_, nullptr);
            pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    VkPipeline create_pipeline(
        VkPrimitiveTopology topology,
        VkPolygonMode polygon_mode,
        VkCullModeFlags cull_mode,
        bool depth_test,
        bool depth_write,
        bool color_write)
    {
        const VkDevice dev = vk_->device();

        const std::vector<char> vs_code = vk_read_binary_file(SHS_VK_CULLING_VERT_SPV);
        const std::vector<char> fs_code = vk_read_binary_file(SHS_VK_CULLING_FRAG_SPV);
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

        pipeline_tri_ = create_pipeline(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_BACK_BIT,
            true,
            true,
            true);
        // Match software debug behavior: lines are overlay (no depth test/write).
        pipeline_line_ = create_pipeline(
            VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            false,
            false,
            true);
        pipeline_depth_prepass_ = create_pipeline(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_BACK_BIT,
            true,
            true,
            false);
        // Occlusion queries use proxy AABBs; avoid winding sensitivity by disabling face culling.
        pipeline_occ_query_ = create_pipeline(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            true,
            false,
            false);
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
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F2) out.cycle_cull_mode = true;

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

    void update_scene_and_culling(float time_s)
    {
        for (auto& inst : instances_)
        {
            if (inst.anim.animated)
            {
                const glm::vec3 rot = inst.anim.base_rot + inst.anim.angular_vel * time_s;
                inst.geometry.transform = jolt::to_jph(compose_model(inst.anim.base_pos, rot));
            }
        }

        const glm::mat4 view = camera_.view_matrix();
        const glm::mat4 proj = perspective_lh_no(glm::radians(60.0f), aspect_, 0.1f, 1000.0f);
        const glm::mat4 vp = proj * view;
        frustum_ = extract_frustum_planes(vp);

        const CullingResultEx frustum_result = run_frustum_culling(
            std::span<const SceneInstance>(instances_.data(), instances_.size()),
            frustum_,
            [](const SceneInstance& inst) -> const SceneShape& { return inst.geometry; });

        frustum_visible_indices_ = frustum_result.frustum_visible_indices;
        for (size_t i = 0; i < instances_.size(); ++i)
        {
            auto& inst = instances_[i];
            const bool frustum_visible =
                (i < frustum_result.frustum_classes.size()) &&
                cull_class_is_visible(
                    frustum_result.frustum_classes[i],
                    frustum_result.request.include_intersecting);
            inst.frustum_visible = frustum_visible;
            inst.visible = false;
            if (!frustum_visible)
            {
                inst.occluded = false;
                visibility_history_.reset(inst.geometry.stable_id);
            }
        }
    }

    void consume_occlusion_results(uint32_t ring)
    {
        if (!enable_occlusion_) return;
        if (!vk_->has_depth_attachment()) return;
        if (ring >= kFrameRing) return;
        if (occlusion_query_pools_[ring] == VK_NULL_HANDLE) return;

        const uint32_t query_count = occlusion_query_counts_[ring];
        if (query_count == 0) return;

        std::vector<uint64_t> query_data(static_cast<size_t>(query_count), 0u);
        const VkResult qr = vkGetQueryPoolResults(
            vk_->device(),
            occlusion_query_pools_[ring],
            0,
            query_count,
            static_cast<VkDeviceSize>(query_data.size() * sizeof(uint64_t)),
            query_data.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        if (qr != VK_SUCCESS) return;

        const auto& inst_map = occlusion_query_instances_[ring];
        apply_query_visibility_samples(
            std::span<SceneInstance>(instances_.data(), instances_.size()),
            std::span<const uint32_t>(inst_map.data(), inst_map.size()),
            std::span<const uint64_t>(query_data.data(), query_data.size()),
            kOcclusionMinVisibleSamples,
            visibility_history_,
            [](const SceneInstance& inst) -> uint32_t {
                return inst.geometry.stable_id;
            },
            [](SceneInstance& inst, bool occluded) {
                inst.occluded = occluded;
            });
    }

    void record_depth_prepass(VkCommandBuffer cmd, VkDescriptorSet camera_set)
    {
        if (pipeline_depth_prepass_ == VK_NULL_HANDLE) return;

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

        for (const uint32_t idx : render_visible_indices_)
        {
            if (idx >= instances_.size()) continue;
            const auto& inst = instances_[idx];
            if (inst.user_index >= meshes_.size()) continue;
            const MeshGPU& mesh = meshes_[inst.user_index];
            if (mesh.tri_indices.buffer == VK_NULL_HANDLE || mesh.tri_index_count == 0) continue;

            const VkBuffer vb = mesh.vertex.buffer;
            const VkDeviceSize vb_off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vb_off);
            vkCmdBindIndexBuffer(cmd, mesh.tri_indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            DrawPush push{};
            push.model = jolt::to_glm(inst.geometry.transform);
            push.base_color = glm::vec4(inst.tint_color, 1.0f);
            push.mode_pad.x = 1u;
            vkCmdPushConstants(
                cmd,
                pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(DrawPush),
                &push);
            vkCmdDrawIndexed(cmd, mesh.tri_index_count, 1, 0, 0, 0);
        }
    }

    void build_visible_lists(uint32_t ring)
    {
        stats_ = build_visibility_from_frustum(
            std::span<SceneInstance>(instances_.data(), instances_.size()),
            std::span<const uint32_t>(frustum_visible_indices_.data(), frustum_visible_indices_.size()),
            apply_occlusion_this_frame_,
            [](const SceneInstance& inst) -> bool {
                return inst.occluded;
            },
            [](SceneInstance& inst, bool visible) {
                inst.visible = visible;
            },
            render_visible_indices_);

        // Safety: never allow occlusion logic to blank the full frustum-visible scene.
        if (should_use_frustum_visibility_fallback(
                enable_occlusion_,
                vk_ && vk_->has_depth_attachment(),
                (ring < kFrameRing) ? occlusion_query_counts_[ring] : 0u,
                stats_))
        {
            render_visible_indices_ = frustum_visible_indices_;
            stats_ = make_culling_stats(
                static_cast<uint32_t>(instances_.size()),
                static_cast<uint32_t>(frustum_visible_indices_.size()),
                static_cast<uint32_t>(render_visible_indices_.size()));
        }
    }

    void record_main_draws(VkCommandBuffer cmd, VkDescriptorSet camera_set)
    {
        const glm::vec4 aabb_color(1.0f, 0.94f, 0.31f, 1.0f);
        // Occlusion demo should render only frustum+occlusion-visible objects in both modes.
        const std::vector<uint32_t>& draw_indices = render_visible_indices_;

        if (render_lit_surfaces_)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_tri_);
        }
        else
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_line_);
        }
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout_,
            0,
            1,
            &camera_set,
            0,
            nullptr);

        for (const uint32_t idx : draw_indices)
        {
            if (idx >= instances_.size()) continue;
            const auto& inst = instances_[idx];
            if (inst.user_index >= meshes_.size()) continue;
            const MeshGPU& mesh = meshes_[inst.user_index];

            const VkBuffer vb = mesh.vertex.buffer;
            const VkDeviceSize vb_off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vb_off);

            uint32_t index_count = 0;
            VkBuffer ib = VK_NULL_HANDLE;
            if (render_lit_surfaces_)
            {
                ib = mesh.tri_indices.buffer;
                index_count = mesh.tri_index_count;
            }
            else
            {
                ib = mesh.line_indices.buffer;
                index_count = mesh.line_index_count;
            }
            if (ib == VK_NULL_HANDLE || index_count == 0) continue;

            vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);

            DrawPush push{};
            push.model = jolt::to_glm(inst.geometry.transform);
            push.base_color = glm::vec4(inst.tint_color, 1.0f);
            push.mode_pad.x = render_lit_surfaces_ ? 1u : 0u;
            vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DrawPush), &push);
            vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
        }

        if (!show_aabb_debug_) return;
        if (aabb_mesh_index_ >= meshes_.size()) return;

        const MeshGPU& aabb_mesh = meshes_[aabb_mesh_index_];
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

        for (const uint32_t idx : draw_indices)
        {
            if (idx >= instances_.size()) continue;
            const auto& inst = instances_[idx];
            const AABB box = inst.geometry.world_aabb();
            const glm::vec3 center = (box.minv + box.maxv) * 0.5f;
            const glm::vec3 size = glm::max(box.maxv - box.minv, glm::vec3(1e-4f));

            DrawPush push{};
            push.model = glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), size);
            push.base_color = aabb_color;
            push.mode_pad.x = 0u;
            vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DrawPush), &push);
            vkCmdDrawIndexed(cmd, aabb_mesh.line_index_count, 1, 0, 0, 0);
        }
    }

    void record_occlusion_queries(VkCommandBuffer cmd, VkDescriptorSet camera_set, uint32_t ring)
    {
        if (!enable_occlusion_) return;
        if (!vk_->has_depth_attachment()) return;
        if (ring >= kFrameRing) return;
        if (occlusion_query_pools_[ring] == VK_NULL_HANDLE) return;
        if (pipeline_occ_query_ == VK_NULL_HANDLE) return;

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

        occlusion_query_instances_[ring].clear();
        occlusion_query_instances_[ring].reserve(frustum_visible_indices_.size());
        occlusion_query_counts_[ring] = 0;

        for (const uint32_t idx : frustum_visible_indices_)
        {
            if (idx >= instances_.size()) continue;
            const auto& inst = instances_[idx];
            if (inst.user_index >= meshes_.size()) continue;
            if (occlusion_query_counts_[ring] >= max_query_count_) break;

            const MeshGPU& mesh = meshes_[inst.user_index];
            if (mesh.tri_indices.buffer == VK_NULL_HANDLE || mesh.tri_index_count == 0) continue;

            const uint32_t query_idx = occlusion_query_counts_[ring];
            occlusion_query_instances_[ring].push_back(idx);
            occlusion_query_counts_[ring]++;

            const VkBuffer vb = mesh.vertex.buffer;
            const VkDeviceSize vb_off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vb_off);
            vkCmdBindIndexBuffer(cmd, mesh.tri_indices.buffer, 0, VK_INDEX_TYPE_UINT32);

            DrawPush push{};
            push.model = jolt::to_glm(inst.geometry.transform);
            push.base_color = glm::vec4(inst.tint_color, 1.0f);
            push.mode_pad.x = 1u;
            vkCmdPushConstants(
                cmd,
                pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(DrawPush),
                &push);

            vkCmdBeginQuery(cmd, occlusion_query_pools_[ring], query_idx, 0);
            vkCmdDrawIndexed(cmd, mesh.tri_index_count, 1, 0, 0, 0);
            vkCmdEndQuery(cmd, occlusion_query_pools_[ring], query_idx);
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
            vk_->has_depth_attachment() &&
            (occlusion_warmup_frames_ == 0u);

        if (!apply_occlusion_this_frame_)
        {
            for (auto& inst : instances_) inst.occluded = false;
            if (!enable_occlusion_) visibility_history_.clear();
        }

        // Consume occlusion results only after begin_frame() fence wait.
        // Reading before that can race GPU completion and produce flicker.
        if (apply_occlusion_this_frame_) consume_occlusion_results(ring);
        build_visible_lists(ring);

        CameraUBO cam{};
        const glm::mat4 view = camera_.view_matrix();
        const glm::mat4 proj = perspective_lh_no(glm::radians(60.0f), aspect_, 0.1f, 1000.0f);
        cam.view_proj = proj * view;
        cam.camera_pos = glm::vec4(camera_.pos, 1.0f);
        cam.light_dir_ws = glm::vec4(kSunLightDirWs, 0.0f);
        std::memcpy(camera_ubos_[ring].mapped, &cam, sizeof(CameraUBO));

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(fi.cmd, &bi) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBeginCommandBuffer failed");
        }

        if (enable_occlusion_ &&
            vk_->has_depth_attachment() &&
            occlusion_query_pools_[ring] != VK_NULL_HANDLE &&
            max_query_count_ > 0)
        {
            vkCmdResetQueryPool(fi.cmd, occlusion_query_pools_[ring], 0, max_query_count_);
        }
        else
        {
            occlusion_query_counts_[ring] = 0;
            occlusion_query_instances_[ring].clear();
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
        record_depth_prepass(fi.cmd, camera_sets_[ring]);
        record_occlusion_queries(fi.cmd, camera_sets_[ring], ring);
        record_main_draws(fi.cmd, camera_sets_[ring]);
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
        char title[320];
        std::snprintf(
            title,
            sizeof(title),
            "Occlusion Culling Demo (VK) | Scene:%u Frustum:%u Occluded:%u Visible:%u | Occ:%s | Mode:%s | AABB:%s | %.2f ms",
            stats_.scene_count,
            stats_.frustum_visible_count,
            stats_.occluded_count,
            stats_.visible_count,
            (enable_occlusion_ && vk_ && vk_->has_depth_attachment()) ? "ON" : "OFF",
            render_lit_surfaces_ ? "Lit" : "Debug",
            show_aabb_debug_ ? "ON" : "OFF",
            avg_ms);
        SDL_SetWindowTitle(win_, title);
    }

    void main_loop()
    {
        std::printf("Controls: LMB/RMB drag look, WASD+QE move, Shift boost, B toggle AABB, L toggle debug/lit, F2 toggle occlusion\n");

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
            if (input.cycle_cull_mode)
            {
                enable_occlusion_ = !enable_occlusion_;
                visibility_history_.clear();
                for (auto& inst : instances_)
                {
                    inst.occluded = false;
                }
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
    std::array<VkDescriptorSet, kFrameRing> camera_sets_{};

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_tri_ = VK_NULL_HANDLE;
    VkPipeline pipeline_line_ = VK_NULL_HANDLE;
    VkPipeline pipeline_depth_prepass_ = VK_NULL_HANDLE;
    VkPipeline pipeline_occ_query_ = VK_NULL_HANDLE;
    uint64_t pipeline_gen_ = 0;

    std::array<VkQueryPool, kFrameRing> occlusion_query_pools_{};
    std::array<uint32_t, kFrameRing> occlusion_query_counts_{};
    std::array<std::vector<uint32_t>, kFrameRing> occlusion_query_instances_{};
    uint32_t max_query_count_ = 0;

    std::vector<MeshGPU> meshes_{};
    std::vector<SceneInstance> instances_{};
    std::vector<uint32_t> frustum_visible_indices_{};
    std::vector<uint32_t> render_visible_indices_{};
    uint32_t aabb_mesh_index_ = 0;

    FreeCamera camera_{};
    float aspect_ = static_cast<float>(kWindowW) / static_cast<float>(kWindowH);
    Frustum frustum_{};

    bool show_aabb_debug_ = false;
    bool render_lit_surfaces_ = false;
    bool enable_occlusion_ = true;
    bool relative_mouse_mode_ = false;
    bool ignore_next_mouse_dt_ = false;
    bool mouse_right_held_ = false;
    bool mouse_left_held_ = false;
    bool apply_occlusion_this_frame_ = false;
    uint32_t occlusion_warmup_frames_ = 0;
    bool camera_prev_valid_ = false;
    glm::vec3 camera_prev_pos_{0.0f};
    float camera_prev_yaw_ = 0.0f;
    float camera_prev_pitch_ = 0.0f;
    VisibilityHistory visibility_history_{
        VisibilityHistoryPolicy{
            kOcclusionHideConfirmFrames,
            kOcclusionShowConfirmFrames}};
    CullingStats stats_{};
};

} // namespace

int main()
{
    try
    {
        HelloOcclusionCullingVkApp app{};
        app.run();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
