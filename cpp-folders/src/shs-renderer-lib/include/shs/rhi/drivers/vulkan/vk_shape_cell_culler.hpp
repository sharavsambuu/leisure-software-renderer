#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_shape_cell_culler.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Generic ShapeVolume vs ConvexCell Vulkan compute culler helpers.
*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

#include <glm/glm.hpp>

#include "shs/geometry/convex_cell.hpp"
#include "shs/geometry/shape_volume.hpp"

#ifdef SHS_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace shs
{
#ifdef SHS_HAS_VULKAN
    inline constexpr uint32_t k_vk_shape_cell_culler_group_size_x = 64u;
    inline constexpr uint32_t k_vk_shape_cell_culler_set_index = 0u;
    inline constexpr uint32_t k_vk_shape_cell_culler_binding_shapes = 0u;
    inline constexpr uint32_t k_vk_shape_cell_culler_binding_cells = 1u;
    inline constexpr uint32_t k_vk_shape_cell_culler_binding_jobs = 2u;
    inline constexpr uint32_t k_vk_shape_cell_culler_binding_results = 3u;
    inline constexpr uint32_t k_vk_shape_cell_culler_binding_aux_vertices = 4u;
    inline constexpr uint32_t k_vk_shape_cell_payload_flag_has_aux_vertices = 1u << 0;
    inline constexpr uint32_t k_vk_shape_cell_payload_flag_broad_fallback = 1u << 1;

    struct alignas(16) VkShapeVolumeGPU
    {
        // Broad phase sphere.
        glm::vec4 center_radius{0.0f};

        // Shape-specific payload slots (kind dependent). Use vk_pack_shape_volume_gpu.
        glm::vec4 p0{0.0f};
        glm::vec4 p1{0.0f};
        glm::vec4 p2{0.0f};
        glm::vec4 p3{0.0f};
        glm::vec4 p4{0.0f};
        glm::vec4 p5{0.0f};

        // x: ShapeVolumeKind
        // y: aux vertex offset
        // z: aux vertex count
        // w: flags
        glm::uvec4 meta{0u, 0u, 0u, 0u};
    };
    static_assert(sizeof(VkShapeVolumeGPU) % 16 == 0, "VkShapeVolumeGPU must be std430 aligned");

    struct alignas(16) VkConvexCellGPU
    {
        // x: plane_count, y: ConvexCellKind, z/w: user.
        glm::uvec4 meta{0u, 0u, 0u, 0u};
        std::array<glm::vec4, k_convex_cell_max_planes> planes{};
    };
    static_assert(sizeof(VkConvexCellGPU) % 16 == 0, "VkConvexCellGPU must be std430 aligned");

    struct alignas(16) VkCullJobGPU
    {
        uint32_t shape_index = 0;
        uint32_t cell_index = 0;
        uint32_t out_index = 0;
        uint32_t flags = 0;
    };
    static_assert(sizeof(VkCullJobGPU) == 16, "VkCullJobGPU must be 16 bytes");

    struct alignas(16) VkShapeCellCullerPushConstants
    {
        // x: job_count, y: shape_count, z: cell_count, w: flags
        glm::uvec4 counts{0u, 0u, 0u, 0u};
        // x: outside_epsilon, y: inside_epsilon
        glm::vec4 eps{1e-5f, 1e-5f, 0.0f, 0.0f};
    };
    static_assert(sizeof(VkShapeCellCullerPushConstants) == 32, "VkShapeCellCullerPushConstants must stay 32 bytes");

    struct VkShapeCellCullerPipeline
    {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

    inline uint32_t vk_shape_cell_culler_dispatch_groups(uint32_t job_count)
    {
        return (job_count + k_vk_shape_cell_culler_group_size_x - 1u) / k_vk_shape_cell_culler_group_size_x;
    }

    inline bool vk_append_aux_vertices(
        const std::vector<glm::vec3>& vertices,
        std::vector<glm::vec4>* aux_vertices,
        VkShapeVolumeGPU& out)
    {
        if (vertices.empty()) return false;
        if (!aux_vertices) return false;

        out.meta.y = static_cast<uint32_t>(aux_vertices->size());
        out.meta.z = static_cast<uint32_t>(vertices.size());
        out.meta.w |= k_vk_shape_cell_payload_flag_has_aux_vertices;
        out.meta.w &= ~k_vk_shape_cell_payload_flag_broad_fallback;
        aux_vertices->reserve(aux_vertices->size() + vertices.size());
        for (const glm::vec3& v : vertices)
        {
            aux_vertices->push_back(glm::vec4(v, 1.0f));
        }
        return true;
    }

    inline bool vk_shape_cell_volume_has_aux_vertices(const VkShapeVolumeGPU& packed)
    {
        return (packed.meta.w & k_vk_shape_cell_payload_flag_has_aux_vertices) != 0u;
    }

    inline bool vk_shape_cell_volume_uses_broad_fallback(const VkShapeVolumeGPU& packed)
    {
        return (packed.meta.w & k_vk_shape_cell_payload_flag_broad_fallback) != 0u;
    }

    inline bool vk_pack_shape_volume_gpu(
        const ShapeVolume& shape,
        VkShapeVolumeGPU& out,
        std::vector<glm::vec4>* aux_vertices = nullptr)
    {
        out = VkShapeVolumeGPU{};
        out.meta.x = static_cast<uint32_t>(shape.kind());

        const Sphere broad = conservative_bounds_sphere(shape);
        out.center_radius = glm::vec4(broad.center, std::max(broad.radius, 0.0f));

        return std::visit([&](const auto& s) -> bool {
            const auto pack_aux_vertices_or_fallback = [&](const std::vector<glm::vec3>& verts) -> bool {
                if (vk_append_aux_vertices(verts, aux_vertices, out)) return true;
                out.meta.y = 0u;
                out.meta.z = 0u;
                out.meta.w &= ~k_vk_shape_cell_payload_flag_has_aux_vertices;
                out.meta.w |= k_vk_shape_cell_payload_flag_broad_fallback;
                return false;
            };

            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, Sphere>)
            {
                out.center_radius = glm::vec4(s.center, std::max(s.radius, 0.0f));
                return true;
            }
            else if constexpr (std::is_same_v<T, AABB>)
            {
                out.p0 = glm::vec4(s.minv, 0.0f);
                out.p1 = glm::vec4(s.maxv, 0.0f);
                return true;
            }
            else if constexpr (std::is_same_v<T, OBB>)
            {
                out.p0 = glm::vec4(s.axis_x, std::max(s.half_extents.x, 0.0f));
                out.p1 = glm::vec4(s.axis_y, std::max(s.half_extents.y, 0.0f));
                out.p2 = glm::vec4(s.axis_z, std::max(s.half_extents.z, 0.0f));
                return true;
            }
            else if constexpr (std::is_same_v<T, Capsule>)
            {
                out.p0 = glm::vec4(s.a, 0.0f);
                out.p1 = glm::vec4(s.b, 0.0f);
                out.p2 = glm::vec4(std::max(s.radius, 0.0f), 0.0f, 0.0f, 0.0f);
                return true;
            }
            else if constexpr (std::is_same_v<T, Cone>)
            {
                const glm::vec3 axis = normalize_or(s.axis, glm::vec3(0.0f, -1.0f, 0.0f));
                out.p0 = glm::vec4(s.apex, 0.0f);
                out.p1 = glm::vec4(axis, std::max(s.height, 0.0f));
                out.p2 = glm::vec4(std::max(s.radius, 0.0f), 0.0f, 0.0f, 0.0f);
                return true;
            }
            else if constexpr (std::is_same_v<T, ConeFrustum>)
            {
                const glm::vec3 axis = normalize_or(s.axis, glm::vec3(0.0f, -1.0f, 0.0f));
                out.p0 = glm::vec4(s.apex, 0.0f);
                out.p1 = glm::vec4(axis, std::max(s.near_distance, 0.0f));
                out.p2 = glm::vec4(
                    std::max(s.far_distance, std::max(s.near_distance, 0.0f)),
                    std::max(s.near_radius, 0.0f),
                    std::max(s.far_radius, 0.0f),
                    0.0f);
                return true;
            }
            else if constexpr (std::is_same_v<T, Cylinder>)
            {
                const glm::vec3 axis = normalize_or(s.axis, glm::vec3(0.0f, 1.0f, 0.0f));
                out.p0 = glm::vec4(s.center, 0.0f);
                out.p1 = glm::vec4(axis, std::max(s.half_height, 0.0f));
                out.p2 = glm::vec4(std::max(s.radius, 0.0f), 0.0f, 0.0f, 0.0f);
                return true;
            }
            else if constexpr (std::is_same_v<T, SweptCapsule>)
            {
                out.p0 = glm::vec4(s.at_t0.a, 0.0f);
                out.p1 = glm::vec4(s.at_t0.b, 0.0f);
                out.p2 = glm::vec4(s.at_t1.a, 0.0f);
                out.p3 = glm::vec4(s.at_t1.b, 0.0f);
                out.p4 = glm::vec4(std::max(s.at_t0.radius, 0.0f), std::max(s.at_t1.radius, 0.0f), 0.0f, 0.0f);
                return true;
            }
            else if constexpr (std::is_same_v<T, SweptOBB>)
            {
                const std::vector<glm::vec3> verts = swept_obb_vertices(s);
                return pack_aux_vertices_or_fallback(verts);
            }
            else if constexpr (std::is_same_v<T, ConvexPolyhedron>)
            {
                return pack_aux_vertices_or_fallback(convex_polyhedron_vertices(s));
            }
            else if constexpr (std::is_same_v<T, KDOP18>)
            {
                return pack_aux_vertices_or_fallback(kdop18_vertices(s));
            }
            else if constexpr (std::is_same_v<T, KDOP26>)
            {
                return pack_aux_vertices_or_fallback(kdop26_vertices(s));
            }
            else if constexpr (std::is_same_v<T, MeshletHull>)
            {
                return pack_aux_vertices_or_fallback(convex_polyhedron_vertices(s.hull));
            }
            else if constexpr (std::is_same_v<T, ClusterHull>)
            {
                return pack_aux_vertices_or_fallback(convex_polyhedron_vertices(s.hull));
            }
            else
            {
                return true;
            }
        }, shape.value);
    }

    inline size_t vk_pack_shape_volumes_gpu(
        const std::vector<ShapeVolume>& shapes,
        std::vector<VkShapeVolumeGPU>& out_shapes,
        std::vector<glm::vec4>& out_aux_vertices)
    {
        out_shapes.resize(shapes.size());
        out_aux_vertices.clear();

        size_t packed = 0;
        for (size_t i = 0; i < shapes.size(); ++i)
        {
            if (vk_pack_shape_volume_gpu(shapes[i], out_shapes[i], &out_aux_vertices))
            {
                ++packed;
            }
            else
            {
                // Aux vertex packing failed: keep broad-phase sphere fallback.
                out_shapes[i].meta.y = 0u;
                out_shapes[i].meta.z = 0u;
                out_shapes[i].meta.w &= ~k_vk_shape_cell_payload_flag_has_aux_vertices;
                out_shapes[i].meta.w |= k_vk_shape_cell_payload_flag_broad_fallback;
            }
        }
        return packed;
    }

    inline bool vk_pack_convex_cell_gpu(const ConvexCell& cell, VkConvexCellGPU& out)
    {
        out = VkConvexCellGPU{};
        out.meta.x = std::min(cell.plane_count, k_convex_cell_max_planes);
        out.meta.y = static_cast<uint32_t>(cell.kind);
        out.meta.z = cell.user_data.z;
        out.meta.w = cell.user_data.w;

        for (uint32_t i = 0; i < out.meta.x; ++i)
        {
            const Plane& p = cell.planes[i];
            out.planes[i] = glm::vec4(p.normal, p.d);
        }
        return out.meta.x > 0u;
    }

    inline size_t vk_pack_convex_cells_gpu(
        const std::vector<ConvexCell>& cells,
        std::vector<VkConvexCellGPU>& out)
    {
        out.resize(cells.size());
        size_t packed = 0;
        for (size_t i = 0; i < cells.size(); ++i)
        {
            if (vk_pack_convex_cell_gpu(cells[i], out[i])) ++packed;
        }
        return packed;
    }

    inline VkShapeCellCullerPushConstants vk_make_shape_cell_culler_push_constants(
        uint32_t job_count,
        uint32_t shape_count,
        uint32_t cell_count,
        float outside_eps = 1e-5f,
        float inside_eps = 1e-5f,
        uint32_t flags = 0u)
    {
        VkShapeCellCullerPushConstants out{};
        out.counts = glm::uvec4(job_count, shape_count, cell_count, flags);
        out.eps = glm::vec4(std::max(outside_eps, 0.0f), std::max(inside_eps, 0.0f), 0.0f, 0.0f);
        return out;
    }

    inline void vk_destroy_shape_cell_culler_pipeline(VkDevice device, VkShapeCellCullerPipeline& pipeline)
    {
        if (device == VK_NULL_HANDLE) return;
        if (pipeline.pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, pipeline.pipeline, nullptr);
            pipeline.pipeline = VK_NULL_HANDLE;
        }
        if (pipeline.pipeline_layout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device, pipeline.pipeline_layout, nullptr);
            pipeline.pipeline_layout = VK_NULL_HANDLE;
        }
        if (pipeline.set_layout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device, pipeline.set_layout, nullptr);
            pipeline.set_layout = VK_NULL_HANDLE;
        }
    }

    inline bool vk_create_shape_cell_culler_pipeline(
        VkDevice device,
        VkShaderModule compute_shader_module,
        VkShapeCellCullerPipeline& out_pipeline)
    {
        out_pipeline = VkShapeCellCullerPipeline{};
        if (device == VK_NULL_HANDLE || compute_shader_module == VK_NULL_HANDLE) return false;

        const VkDescriptorSetLayoutBinding bindings[] = {
            {k_vk_shape_cell_culler_binding_shapes, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {k_vk_shape_cell_culler_binding_cells, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {k_vk_shape_cell_culler_binding_jobs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {k_vk_shape_cell_culler_binding_results, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {k_vk_shape_cell_culler_binding_aux_vertices, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };

        VkDescriptorSetLayoutCreateInfo set_ci{};
        set_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_ci.bindingCount = static_cast<uint32_t>(std::size(bindings));
        set_ci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &set_ci, nullptr, &out_pipeline.set_layout) != VK_SUCCESS)
        {
            return false;
        }

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(VkShapeCellCullerPushConstants);

        VkPipelineLayoutCreateInfo layout_ci{};
        layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_ci.setLayoutCount = 1;
        layout_ci.pSetLayouts = &out_pipeline.set_layout;
        layout_ci.pushConstantRangeCount = 1;
        layout_ci.pPushConstantRanges = &push_range;
        if (vkCreatePipelineLayout(device, &layout_ci, nullptr, &out_pipeline.pipeline_layout) != VK_SUCCESS)
        {
            vk_destroy_shape_cell_culler_pipeline(device, out_pipeline);
            return false;
        }

        VkPipelineShaderStageCreateInfo stage_ci{};
        stage_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_ci.module = compute_shader_module;
        stage_ci.pName = "main";

        VkComputePipelineCreateInfo pipe_ci{};
        pipe_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipe_ci.layout = out_pipeline.pipeline_layout;
        pipe_ci.stage = stage_ci;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipe_ci, nullptr, &out_pipeline.pipeline) != VK_SUCCESS)
        {
            vk_destroy_shape_cell_culler_pipeline(device, out_pipeline);
            return false;
        }
        return true;
    }

    inline void vk_cmd_dispatch_shape_cell_culler(
        VkCommandBuffer cmd,
        const VkShapeCellCullerPipeline& pipeline,
        VkDescriptorSet descriptor_set,
        const VkShapeCellCullerPushConstants& push)
    {
        if (cmd == VK_NULL_HANDLE || pipeline.pipeline == VK_NULL_HANDLE || pipeline.pipeline_layout == VK_NULL_HANDLE) return;
        if (push.counts.x == 0) return;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout,
            k_vk_shape_cell_culler_set_index,
            1,
            &descriptor_set,
            0,
            nullptr);
        vkCmdPushConstants(
            cmd,
            pipeline.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(VkShapeCellCullerPushConstants),
            &push);

        const uint32_t groups = vk_shape_cell_culler_dispatch_groups(push.counts.x);
        vkCmdDispatch(cmd, std::max(groups, 1u), 1u, 1u);
    }
#else
    struct VkShapeVolumeGPU {};
    struct VkConvexCellGPU {};
    struct VkCullJobGPU {};
    struct VkShapeCellCullerPushConstants {};
    struct VkShapeCellCullerPipeline {};
#endif
}
