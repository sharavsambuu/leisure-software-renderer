#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_standard_pass_execution.hpp
    MODULE: pipeline
    PURPOSE: Shared Vulkan standard pass execution flow helpers.
*/


#include <utility>

#include <vulkan/vulkan.h>

#include "shs/pipeline/render_path_compiler.hpp"

namespace shs
{
    template <typename TContext, typename TRecordShadowPasses, typename TMemoryBarrier>
    inline bool vk_execute_shadow_map_pass(
        TContext& ctx,
        const RenderPathCompiledPass& pass,
        TRecordShadowPasses&& record_shadow_passes,
        TMemoryBarrier&& memory_barrier)
    {
        (void)pass;
        if (!ctx.fi) return false;

        record_shadow_passes(ctx.fi->cmd);
        memory_barrier(
            ctx.fi->cmd,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT);
        return true;
    }

    template <typename TContext, typename TBeginDepthSecondaryPass, typename TRecordInlineDepth>
    inline bool vk_execute_depth_prepass_pass(
        TContext& ctx,
        const RenderPathCompiledPass& pass,
        VkRenderPass depth_render_pass,
        VkFramebuffer depth_framebuffer,
        uint32_t depth_w,
        uint32_t depth_h,
        TBeginDepthSecondaryPass&& begin_depth_secondary_pass,
        TRecordInlineDepth&& record_inline_depth)
    {
        (void)pass;
        if (!ctx.fi) return false;
        if (ctx.depth_pass_executed) return true;
        if (!ctx.depth_prepass_enabled) return true;

        if (ctx.depth_secondaries && !ctx.depth_secondaries->empty())
        {
            begin_depth_secondary_pass(ctx.fi->cmd);
            vkCmdExecuteCommands(
                ctx.fi->cmd,
                static_cast<uint32_t>(ctx.depth_secondaries->size()),
                ctx.depth_secondaries->data());
            vkCmdEndRenderPass(ctx.fi->cmd);
        }
        else
        {
            VkClearValue clear{};
            clear.depthStencil = {1.0f, 0};

            VkRenderPassBeginInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp.renderPass = depth_render_pass;
            rp.framebuffer = depth_framebuffer;
            rp.renderArea.offset = {0, 0};
            rp.renderArea.extent = {depth_w, depth_h};
            rp.clearValueCount = 1;
            rp.pClearValues = &clear;
            vkCmdBeginRenderPass(ctx.fi->cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

            record_inline_depth(ctx.fi->cmd, ctx.frame_slot);
            vkCmdEndRenderPass(ctx.fi->cmd);
        }

        ctx.depth_pass_executed = true;
        return true;
    }

    template <
        typename TContext,
        typename TClearLightGrid,
        typename TMemoryBarrier,
        typename TDispatchDepthReduce,
        typename TDispatchLightCull>
    inline bool vk_execute_light_culling_pass(
        TContext& ctx,
        const RenderPathCompiledPass& pass,
        bool use_depth_range_reduction,
        uint32_t dispatch_z,
        TClearLightGrid&& clear_light_grid,
        TMemoryBarrier&& memory_barrier,
        TDispatchDepthReduce&& dispatch_depth_reduce,
        TDispatchLightCull&& dispatch_light_cull)
    {
        (void)pass;
        if (!ctx.fi) return false;
        if (ctx.light_culling_executed) return true;
        if (!ctx.light_culling_enabled) return true;

        if (ctx.gpu_light_culler_enabled)
        {
            const VkPipelineStageFlags depth_stage =
                ctx.depth_pass_executed
                    ? (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)
                    : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            const VkAccessFlags depth_access =
                ctx.depth_pass_executed
                    ? static_cast<VkAccessFlags>(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
                    : static_cast<VkAccessFlags>(0);

            memory_barrier(
                ctx.fi->cmd,
                depth_stage,
                depth_access,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            if (use_depth_range_reduction)
            {
                dispatch_depth_reduce(ctx.fi->cmd, ctx.global_set);
                memory_barrier(
                    ctx.fi->cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
            }

            dispatch_light_cull(ctx.fi->cmd, ctx.global_set, dispatch_z);
            memory_barrier(
                ctx.fi->cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        }
        else if (!ctx.light_grid_cleared)
        {
            clear_light_grid(ctx.frame_slot);
            ctx.light_grid_cleared = true;
        }

        ctx.light_culling_executed = true;
        return true;
    }

    template <
        typename TContext,
        typename THasDepthAttachment,
        typename TBeginSceneSecondaryPass,
        typename TRecordInlineScene,
        typename TDrawSceneOverlay>
    inline bool vk_execute_scene_pass(
        TContext& ctx,
        const RenderPathCompiledPass& pass,
        THasDepthAttachment&& has_depth_attachment,
        TBeginSceneSecondaryPass&& begin_scene_secondary_pass,
        TRecordInlineScene&& record_inline_scene,
        TDrawSceneOverlay&& draw_scene_overlay)
    {
        (void)pass;
        if (!ctx.fi) return false;
        if (ctx.scene_pass_executed) return true;
        if (!ctx.scene_enabled) return true;

        if (ctx.scene_secondaries && !ctx.scene_secondaries->empty())
        {
            begin_scene_secondary_pass(ctx.fi->cmd, *ctx.fi);
            vkCmdExecuteCommands(
                ctx.fi->cmd,
                static_cast<uint32_t>(ctx.scene_secondaries->size()),
                ctx.scene_secondaries->data());
            draw_scene_overlay(ctx.fi->cmd, ctx.frame_slot);
            vkCmdEndRenderPass(ctx.fi->cmd);
        }
        else
        {
            VkClearValue clear[2]{};
            clear[0].color = {{0.03f, 0.035f, 0.045f, 1.0f}};
            clear[1].depthStencil = {1.0f, 0};

            VkRenderPassBeginInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp.renderPass = ctx.fi->render_pass;
            rp.framebuffer = ctx.fi->framebuffer;
            rp.renderArea.offset = {0, 0};
            rp.renderArea.extent = ctx.fi->extent;
            rp.clearValueCount = has_depth_attachment() ? 2u : 1u;
            rp.pClearValues = clear;
            vkCmdBeginRenderPass(ctx.fi->cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

            record_inline_scene(
                ctx.fi->cmd,
                ctx.frame_slot,
                ctx.fi->extent.width,
                ctx.fi->extent.height);
            draw_scene_overlay(ctx.fi->cmd, ctx.frame_slot);
            vkCmdEndRenderPass(ctx.fi->cmd);
        }

        ctx.scene_pass_executed = true;
        return true;
    }

    template <
        typename TContext,
        typename TBeginGBufferPass,
        typename TRecordInlineGBuffer,
        typename TMemoryBarrier>
    inline bool vk_execute_gbuffer_pass(
        TContext& ctx,
        const RenderPathCompiledPass& pass,
        bool gbuffer_resources_ready,
        TBeginGBufferPass&& begin_gbuffer_pass,
        TRecordInlineGBuffer&& record_inline_gbuffer,
        TMemoryBarrier&& memory_barrier)
    {
        (void)pass;
        if (!ctx.fi) return false;
        if (ctx.gbuffer_pass_executed) return true;
        if (!gbuffer_resources_ready) return false;

        begin_gbuffer_pass(ctx.fi->cmd);
        record_inline_gbuffer(ctx.fi->cmd, ctx.frame_slot);
        vkCmdEndRenderPass(ctx.fi->cmd);

        memory_barrier(
            ctx.fi->cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT);

        ctx.depth_pass_executed = true;
        ctx.gbuffer_pass_executed = true;
        return true;
    }

    template <typename TContext, typename THasDepthAttachment, typename TRecordDeferredInline>
    inline bool vk_execute_deferred_lighting_pass(
        TContext& ctx,
        const RenderPathCompiledPass& pass,
        bool deferred_resources_ready,
        THasDepthAttachment&& has_depth_attachment,
        TRecordDeferredInline&& record_deferred_inline)
    {
        (void)pass;
        if (!ctx.fi) return false;
        if (ctx.deferred_lighting_pass_executed) return true;
        if (!deferred_resources_ready) return false;

        VkClearValue clear[2]{};
        clear[0].color = {{0.03f, 0.035f, 0.045f, 1.0f}};
        clear[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = ctx.fi->render_pass;
        rp.framebuffer = ctx.fi->framebuffer;
        rp.renderArea.offset = {0, 0};
        rp.renderArea.extent = ctx.fi->extent;
        rp.clearValueCount = has_depth_attachment() ? 2u : 1u;
        rp.pClearValues = clear;
        vkCmdBeginRenderPass(ctx.fi->cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        record_deferred_inline(ctx.fi->cmd, ctx);
        vkCmdEndRenderPass(ctx.fi->cmd);

        ctx.deferred_lighting_pass_executed = true;
        ctx.scene_pass_executed = true;
        return true;
    }
}
