#define SDL_MAIN_HANDLED

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/scene_shape.hpp>
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
constexpr uint32_t kFrameRing = 2u;

struct Vertex
{
    glm::vec3 pos{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
};

struct alignas(16) CameraUBO
{
    glm::mat4 view_proj{1.0f};
    glm::vec4 camera_pos{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 light_dir_ws{0.45f, -1.0f, 0.35f, 0.0f};
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
    bool animated = true;
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
        glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
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

class HelloCullingVkApp
{
public:
    ~HelloCullingVkApp()
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
            "Culling & Debug Draw Demo (Vulkan)",
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
        init.app_name = "hello_culling_vk";
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
            ShapeInstance floor{};
            floor.shape.shape = jolt::make_box(glm::vec3(50.0f, 0.1f, 50.0f));
            floor.base_pos = glm::vec3(0.0f, -0.2f, 0.0f);
            floor.base_rot = glm::vec3(0.0f);
            floor.model = compose_model(floor.base_pos, floor.base_rot);
            floor.shape.transform = jolt::to_jph(floor.model);
            floor.shape.stable_id = 9000;
            floor.color = glm::vec3(0.18f, 0.18f, 0.22f);
            floor.animated = false;

            const DebugMesh floor_mesh = debug_mesh_from_shape(*floor.shape.shape, JPH::Mat44::sIdentity());
            floor.mesh_index = upload_debug_mesh(floor_mesh);
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
        const JPH::ShapeRefC spot_light_volume_shape = jolt::make_spot_light_volume(1.8f, glm::radians(28.0f), 20);
        const JPH::ShapeRefC rect_light_volume_shape = jolt::make_rect_area_light_volume(glm::vec2(0.8f, 0.5f), 2.0f);
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
                ShapeInstance inst{};
                inst.shape.shape = shape_types[t].shape;
                inst.mesh_index = shape_types[t].mesh_index;
                inst.base_pos = glm::vec3(
                    (-0.5f * (copies_per_type - 1) + static_cast<float>(c)) * spacing_x,
                    1.25f + 0.25f * static_cast<float>(c % 3),
                    (-0.5f * static_cast<float>(shape_types.size() - 1) + static_cast<float>(t)) * spacing_z);
                inst.base_rot = glm::vec3(
                    0.17f * static_cast<float>(c),
                    0.23f * static_cast<float>(t),
                    0.11f * static_cast<float>(c + static_cast<int>(t)));
                inst.angular_vel = glm::vec3(
                    0.30f + 0.07f * static_cast<float>((c + static_cast<int>(t)) % 5),
                    0.42f + 0.06f * static_cast<float>(c % 4),
                    0.36f + 0.05f * static_cast<float>(static_cast<int>(t) % 6));
                inst.model = compose_model(inst.base_pos, inst.base_rot);
                inst.shape.transform = jolt::to_jph(inst.model);
                inst.shape.stable_id = next_id++;
                inst.color = shape_types[t].color;
                inst.animated = true;
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
        if (pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(dev, pipeline_layout_, nullptr);
            pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    VkPipeline create_pipeline(VkPrimitiveTopology topology, VkPolygonMode polygon_mode)
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
        rs.cullMode = (topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
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
        }

        const glm::mat4 view = camera_.view_matrix();
        const glm::mat4 proj = perspective_lh_no(glm::radians(60.0f), aspect_, 0.1f, 1000.0f);
        const glm::mat4 vp = proj * view;
        frustum_ = extract_frustum_planes(vp);

        visible_count_ = 0;
        for (auto& inst : instances_)
        {
            const CullClass cc = classify_aabb_vs_frustum(inst.shape.world_aabb(), frustum_);
            inst.visible = (cc != CullClass::Outside);
            if (inst.visible) ++visible_count_;
        }
        scene_count_ = static_cast<uint32_t>(instances_.size());
        culled_count_ = scene_count_ - visible_count_;
    }

    void record_draws(VkCommandBuffer cmd, VkDescriptorSet camera_set)
    {
        const glm::vec4 aabb_color(1.0f, 0.94f, 0.31f, 1.0f);

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

        for (const auto& inst : instances_)
        {
            if (!inst.visible) continue;
            if (inst.mesh_index >= meshes_.size()) continue;
            const MeshGPU& mesh = meshes_[inst.mesh_index];

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
            push.model = inst.model;
            push.base_color = glm::vec4(inst.color, 1.0f);
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

        for (const auto& inst : instances_)
        {
            if (!inst.visible) continue;
            const AABB box = inst.shape.world_aabb();
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
        CameraUBO cam{};
        const glm::mat4 view = camera_.view_matrix();
        const glm::mat4 proj = perspective_lh_no(glm::radians(60.0f), aspect_, 0.1f, 1000.0f);
        cam.view_proj = proj * view;
        cam.camera_pos = glm::vec4(camera_.pos, 1.0f);
        cam.light_dir_ws = glm::vec4(0.45f, -1.0f, 0.35f, 0.0f);
        std::memcpy(camera_ubos_[ring].mapped, &cam, sizeof(CameraUBO));

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
        char title[320];
        std::snprintf(
            title,
            sizeof(title),
            "Culling Demo (VK) | Scene:%u Visible:%u Culled:%u | Mode:%s | AABB:%s | %.2f ms",
            scene_count_,
            visible_count_,
            culled_count_,
            render_lit_surfaces_ ? "Lit" : "Debug",
            show_aabb_debug_ ? "ON" : "OFF",
            avg_ms);
        SDL_SetWindowTitle(win_, title);
    }

    void main_loop()
    {
        std::printf("Controls: RMB look, WASD+QE move, Shift boost, B toggle AABB, L toggle debug/lit\n");

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
    uint64_t pipeline_gen_ = 0;

    std::vector<MeshGPU> meshes_{};
    std::vector<ShapeInstance> instances_{};
    uint32_t aabb_mesh_index_ = 0;

    FreeCamera camera_{};
    float aspect_ = static_cast<float>(kWindowW) / static_cast<float>(kWindowH);
    Frustum frustum_{};

    bool show_aabb_debug_ = false;
    bool render_lit_surfaces_ = false;

    uint32_t scene_count_ = 0;
    uint32_t visible_count_ = 0;
    uint32_t culled_count_ = 0;
};

} // namespace

int main()
{
    try
    {
        HelloCullingVkApp app{};
        app.run();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
