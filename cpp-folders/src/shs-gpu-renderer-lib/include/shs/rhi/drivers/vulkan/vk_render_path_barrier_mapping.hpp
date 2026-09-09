#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_render_path_barrier_mapping.hpp
    MODULE: rhi/vulkan
    PURPOSE: Convert graph-owned render-path barrier edges into Vulkan stage/access templates.
*/


#include <vulkan/vulkan.h>

#include "shs/pipeline/pass_contract.hpp"
#include "shs/pipeline/pass_id.hpp"
#include "shs/pipeline/render_path_barrier_plan.hpp"

namespace shs
{
    struct VkRenderPathBarrierTemplate
    {
        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkAccessFlags src_access = 0u;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        VkAccessFlags dst_access = 0u;
        bool valid = false;
    };

    inline bool vk_render_path_is_compute_pass(PassId pass_id)
    {
        switch (pass_id)
        {
            case PassId::LightCulling:
            case PassId::ClusterBuild:
            case PassId::ClusterLightAssign:
                return true;
            default:
                return false;
        }
    }

    inline bool vk_render_path_is_depth_write_pass(PassId pass_id)
    {
        switch (pass_id)
        {
            case PassId::DepthPrepass:
            case PassId::ShadowMap:
                return true;
            default:
                return false;
        }
    }

    inline VkPipelineStageFlags vk_render_path_stage_for_pass_side(
        PassId pass_id,
        PassSemantic semantic,
        ContractAccess access)
    {
        const bool reads = contract_access_has_read(access);
        const bool writes = contract_access_has_write(access);
        if (vk_render_path_is_compute_pass(pass_id))
        {
            return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }

        VkPipelineStageFlags out = 0u;
        if (reads)
        {
            out |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        if (writes)
        {
            const bool depth_semantic =
                semantic == PassSemantic::Depth || semantic == PassSemantic::ShadowMap;
            if (depth_semantic || vk_render_path_is_depth_write_pass(pass_id))
            {
                out |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            }
            else
            {
                out |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            }
        }
        if (out == 0u)
        {
            out = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
        return out;
    }

    inline VkAccessFlags vk_render_path_access_for_pass_side(
        PassId pass_id,
        PassSemantic semantic,
        ContractAccess access)
    {
        const bool reads = contract_access_has_read(access);
        const bool writes = contract_access_has_write(access);
        VkAccessFlags out = 0u;

        if (reads)
        {
            out |= VK_ACCESS_SHADER_READ_BIT;
        }
        if (writes)
        {
            if (vk_render_path_is_compute_pass(pass_id))
            {
                out |= VK_ACCESS_SHADER_WRITE_BIT;
            }
            else if (semantic == PassSemantic::Depth || semantic == PassSemantic::ShadowMap)
            {
                out |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            }
            else
            {
                out |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            }
        }
        return out;
    }

    inline VkRenderPathBarrierTemplate vk_make_render_path_barrier_template(
        const RenderPathBarrierEdge& edge)
    {
        VkRenderPathBarrierTemplate out{};
        out.src_stage = vk_render_path_stage_for_pass_side(edge.from_pass_kind, edge.semantic, edge.from_access);
        out.src_access = vk_render_path_access_for_pass_side(edge.from_pass_kind, edge.semantic, edge.from_access);
        out.dst_stage = vk_render_path_stage_for_pass_side(edge.to_pass_kind, edge.semantic, edge.to_access);
        out.dst_access = vk_render_path_access_for_pass_side(edge.to_pass_kind, edge.semantic, edge.to_access);

        // Keep barriers conservative for unknown/custom pass kinds.
        if (edge.from_pass_kind == PassId::Unknown)
        {
            out.src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            if (out.src_access == 0u) out.src_access = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        }
        if (edge.to_pass_kind == PassId::Unknown)
        {
            out.dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            if (out.dst_access == 0u) out.dst_access = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        }

        out.valid = (out.src_stage != 0u) && (out.dst_stage != 0u);
        return out;
    }
}
