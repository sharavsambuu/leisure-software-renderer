/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_context.hpp
    МОДУЛЬ: passes
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн passes модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#pragma once

#include <cstdint>
#include "shs/core/context.hpp"
#include "shs/gfx/rt_types.hpp"

namespace shs
{
    struct Scene;
    class ResourceRegistry;
    struct RendererResources;

    enum class PassContextSceneBinding : uint8_t
    {
        Unknown = 0,
        ModernScene = 1
    };

    enum class PassContextResourceBinding : uint8_t
    {
        Unknown = 0,
        ResourceRegistry = 1,
        RendererResources = 2
    };

    struct PassContext
    {
        // --- App / Engine context ---
        Context* ctx = nullptr;          // Цонх, input, timing, device гэх мэт (demo-гийн үндсэн context)

        // Нэг кадрын үндсэн render target.
        DefaultRT* rt = nullptr;

        // --- Frame timing ---
        uint64_t frame_index = 0;        // Кадрын дугаар
        float    dt          = 0.0f;     // delta time (сек)

        // --- Camera ---
        // Камерын матриц/вектор pointer-ууд.
        const void* view          = nullptr; // (const glm::mat4*)
        const void* proj          = nullptr; // (const glm::mat4*)
        const void* viewproj      = nullptr; // (const glm::mat4*)
        const void* prev_viewproj = nullptr; // (const glm::mat4*)

        const void* cam_pos_ws    = nullptr; // (const glm::vec3*)
        const void* sun_dir_ws    = nullptr; // (const glm::vec3*)

        // --- Common post params ---
        float exposure = 1.0f;           // Tonemap exposure
        float gamma    = 2.2f;           // Display gamma

        // --- Shared resource hubs ---
        // Raw pointer slots kept for compatibility with older resource plumbing.
        // Use bind_* / *_scene / *_resources helpers below instead of raw casts.
        void* resources = nullptr;       // ResourceRegistry* or RendererResources*
        void* scene     = nullptr;       // Scene*
        PassContextResourceBinding resource_binding = PassContextResourceBinding::Unknown;
        PassContextSceneBinding scene_binding = PassContextSceneBinding::Unknown;

        inline void bind_scene(Scene* s)
        {
            scene = static_cast<void*>(s);
            scene_binding = s ? PassContextSceneBinding::ModernScene : PassContextSceneBinding::Unknown;
        }

        inline Scene* modern_scene()
        {
            if (scene_binding != PassContextSceneBinding::ModernScene) return nullptr;
            return static_cast<Scene*>(scene);
        }

        inline const Scene* modern_scene() const
        {
            if (scene_binding != PassContextSceneBinding::ModernScene) return nullptr;
            return static_cast<const Scene*>(scene);
        }

        inline void bind_resource_registry(ResourceRegistry* r)
        {
            resources = static_cast<void*>(r);
            resource_binding = r ? PassContextResourceBinding::ResourceRegistry : PassContextResourceBinding::Unknown;
        }

        inline void bind_renderer_resources(RendererResources* r)
        {
            resources = static_cast<void*>(r);
            resource_binding = r ? PassContextResourceBinding::RendererResources : PassContextResourceBinding::Unknown;
        }

        inline ResourceRegistry* resource_registry()
        {
            if (resource_binding != PassContextResourceBinding::ResourceRegistry) return nullptr;
            return static_cast<ResourceRegistry*>(resources);
        }

        inline const ResourceRegistry* resource_registry() const
        {
            if (resource_binding != PassContextResourceBinding::ResourceRegistry) return nullptr;
            return static_cast<const ResourceRegistry*>(resources);
        }

        inline RendererResources* renderer_resources()
        {
            if (resource_binding != PassContextResourceBinding::RendererResources) return nullptr;
            return static_cast<RendererResources*>(resources);
        }

        inline const RendererResources* renderer_resources() const
        {
            if (resource_binding != PassContextResourceBinding::RendererResources) return nullptr;
            return static_cast<const RendererResources*>(resources);
        }

        // --- Debug knobs (optional) ---
        int debug_view = 0;              // 0=final, 1=shadow, 2=depth, ... гэх мэт
    };
} // namespace shs
