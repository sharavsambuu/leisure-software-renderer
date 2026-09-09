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
#include <utility>
#include <vector>
#include <glm/glm.hpp>

#include "shs/job/job_system.hpp"
#include "shs/gfx/rt_shadow.hpp"
#include "shs/gfx/rt_types.hpp"
#include "shs/rhi/core/backend.hpp"

namespace shs
{
    // Рендерлэлтийн үеийн гүйцэтгэл болон дебаг мэдээллийг хадгалах бүтэц.
    // Хэдэн гурвалжин зурагдсан, ямар функц хэр удаан ажилласан зэргийг хянана.
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

    // Сүүдрийн зураглал (Shadow Map) болон сүүдэр тооцоолох үеийн кэш санах ойн төлөв.
    // Объектуудын хязгаарын хайрцаг (Bounding Box)-ийг дахин дахин тооцоолохгүйн тулд кэш ашигладаг.
    struct ShadowRuntimeState
    {
        using MeshBoundsPair = std::pair<glm::vec3, glm::vec3>;
        const RT_ShadowDepth* map = nullptr;
        glm::mat4 light_viewproj{1.0f};
        bool valid = false;
        std::unordered_map<const void*, MeshBoundsPair> mesh_bounds_cache{};

        void reset()
        {
            map = nullptr;
            light_viewproj = glm::mat4(1.0f);
            valid = false;
        }

        void reset_caches()
        {
            mesh_bounds_cache.clear();
        }
    };

    // Өмнөх фрэймийн рендер төлөвийг хадгалах бүтэц. 
    // Хөдөлгөөний бүдэгрүүлэлт (Motion Blur) зэрэг өмнөх мэдээлэл шаарддаг эффектүүдэд ашиглана.
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

    // TAA (Temporal Anti-Aliasing) буюу цагийн зурвасын ирмэг толигоржуулалтын төлөв.
    // Өмнөх фрэймийн өнгийг одоогийн өнгөтэй хольж ирмэгийн арзгарыг дарна.
    struct TemporalAARuntimeState
    {
        std::vector<Color> history{};
        int history_w = 0;
        int history_h = 0;
        bool history_valid = false;

        void reset()
        {
            history.clear();
            history_w = 0;
            history_h = 0;
            history_valid = false;
        }
    };

    // Системийн ерөнхий контекст. Хамгийн чухал сангуудын холбоосыг 
    // болон ажиллах үеийн төлөвүүдийг хадгална.
    struct Context
    {
        IJobSystem* job_system = nullptr;
        uint64_t frame_index = 0;
        RenderDebugStats debug{};
        ShadowRuntimeState shadow{};
        RenderHistoryState history{};
        TemporalAARuntimeState temporal_aa{};
        std::array<IRenderBackend*, 3> backends{nullptr, nullptr, nullptr};
        RenderBackendType primary_backend = RenderBackendType::Software;

        static constexpr size_t backend_index(RenderBackendType t)
        {
            return (size_t)t;
        }

        void register_backend(IRenderBackend* backend)
        {
            if (!backend) return;
            const bool had_any_backend = has_backend(RenderBackendType::Software)
                || has_backend(RenderBackendType::OpenGL)
                || has_backend(RenderBackendType::Vulkan);
            backends[backend_index(backend->type())] = backend;
            if (!had_any_backend)
            {
                primary_backend = backend->type();
            }
        }

        void set_primary_backend(IRenderBackend* backend)
        {
            if (!backend) return;
            register_backend(backend);
            primary_backend = backend->type();
        }

        void set_primary_backend(RenderBackendType type)
        {
            primary_backend = type;
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
            if (auto* b = backend(RenderBackendType::Software)) return b;
            if (auto* b = backend(RenderBackendType::OpenGL)) return b;
            if (auto* b = backend(RenderBackendType::Vulkan)) return b;
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
