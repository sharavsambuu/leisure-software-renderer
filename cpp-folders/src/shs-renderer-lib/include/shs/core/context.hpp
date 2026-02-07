#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: context.hpp
    МОДУЛЬ: core
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн core модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>
#include <array>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

#include "shs/job/job_system.hpp"
#include "shs/gfx/rt_shadow.hpp"
#include "shs/rhi/core/backend.hpp"
#include "shs/rhi/sync/vk_runtime.hpp"

namespace shs
{
    struct RenderDebugStats
    {
        uint64_t tri_input = 0;
        uint64_t tri_after_clip = 0;
        uint64_t tri_raster = 0;
        float ms_shadow = 0.0f;
        float ms_pbr = 0.0f;
        float ms_tonemap = 0.0f;
        float ms_shafts = 0.0f;
        float ms_motion_blur = 0.0f;
        uint64_t vk_like_submissions = 0;
        uint64_t vk_like_tasks = 0;
        uint64_t vk_like_stalls = 0;

        void reset()
        {
            tri_input = 0;
            tri_after_clip = 0;
            tri_raster = 0;
            ms_shadow = 0.0f;
            ms_pbr = 0.0f;
            ms_tonemap = 0.0f;
            ms_shafts = 0.0f;
            ms_motion_blur = 0.0f;
            vk_like_submissions = 0;
            vk_like_tasks = 0;
            vk_like_stalls = 0;
        }
    };

    struct ShadowRuntimeState
    {
        const RT_ShadowDepth* map = nullptr;
        glm::mat4 light_viewproj{1.0f};
        bool valid = false;

        void reset()
        {
            map = nullptr;
            light_viewproj = glm::mat4(1.0f);
            valid = false;
        }
    };

    struct RenderHistoryState
    {
        std::unordered_map<uint64_t, glm::mat4> prev_model_by_object{};
        bool has_prev_frame = false;

        void reset()
        {
            prev_model_by_object.clear();
            has_prev_frame = false;
        }
    };

    struct ForwardPlusRuntimeState
    {
        bool depth_prepass_valid = false;
        bool light_culling_valid = false;
        uint32_t tile_size = 16;
        uint32_t tile_count_x = 0;
        uint32_t tile_count_y = 0;
        uint32_t max_lights_per_tile = 128;
        uint32_t visible_light_count = 0;
        // CPU fallback-ын хувьд tile бүрийн гэрлийн тооны мэдээлэл.
        std::vector<uint32_t> tile_light_counts{};

        void reset()
        {
            depth_prepass_valid = false;
            light_culling_valid = false;
            tile_size = 16;
            tile_count_x = 0;
            tile_count_y = 0;
            max_lights_per_tile = 128;
            visible_light_count = 0;
            tile_light_counts.clear();
        }
    };

    struct Context
    {
        IJobSystem* job_system = nullptr;
        // Backward-compat: анхны single-backend pointer.
        IRenderBackend* render_backend = nullptr;
        uint64_t frame_index = 0;
        RenderDebugStats debug{};
        ShadowRuntimeState shadow{};
        RenderHistoryState history{};
        ForwardPlusRuntimeState forward_plus{};
        std::array<IRenderBackend*, 3> backends{nullptr, nullptr, nullptr};
        RenderBackendType primary_backend = RenderBackendType::Software;
        VulkanLikeRuntime vk_like{};

        static constexpr size_t backend_index(RenderBackendType t)
        {
            return (size_t)t;
        }

        void register_backend(IRenderBackend* backend)
        {
            if (!backend) return;
            backends[backend_index(backend->type())] = backend;
            if (!render_backend)
            {
                render_backend = backend;
                primary_backend = backend->type();
            }
        }

        void set_primary_backend(IRenderBackend* backend)
        {
            if (!backend) return;
            register_backend(backend);
            render_backend = backend;
            primary_backend = backend->type();
        }

        void set_primary_backend(RenderBackendType type)
        {
            primary_backend = type;
            if (auto* b = this->backend(type)) render_backend = b;
        }

        IRenderBackend* backend(RenderBackendType type) const
        {
            return backends[backend_index(type)];
        }

        bool has_backend(RenderBackendType type) const
        {
            return backend(type) != nullptr;
        }

        IRenderBackend* active_backend() const
        {
            if (auto* b = backend(primary_backend)) return b;
            if (render_backend) return render_backend;
            return backend(RenderBackendType::Software);
        }

        RenderBackendType active_backend_type() const
        {
            const auto* b = active_backend();
            return b ? b->type() : RenderBackendType::Software;
        }

        const char* active_backend_name() const
        {
            const auto* b = active_backend();
            return b ? b->name() : render_backend_type_name(RenderBackendType::Software);
        }
    };
}
