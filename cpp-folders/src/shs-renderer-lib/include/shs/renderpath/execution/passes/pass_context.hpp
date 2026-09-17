/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_context.hpp
    МОДУЛЬ: passes
    ЗОРИЛГО: Рендеринг пассууд хооронд дамжих ерөнхий контекст. 
            Камерын тогтмол хувьсагчид болон глобал (жинхэнэ) төлөвүүдийг агуулна.
*/


#pragma once

#include <cstdint>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include "shs/app/context.hpp"
#include "shs/render/targets/rt_types.hpp"
#include "shs/render/targets/resource_handles.hpp"
#include "shs/resources/storage/resource_registry.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
    {
    // Cross-module types are owner-namespaced since the step-7 cutover;
    // hoist them so unqualified/old-root spellings still resolve.
    using ::shs::scene::Scene;
    using ::shs::resources::ResourceRegistry;
    using ::shs::render::RendererResources;

    struct PassContext
    {
        // --- App / Engine context ---
        Context* ctx = nullptr;          // Цонх, гаралт/оролт (input), цаг хэмжигч (timing), төхөөрөмж (device) гэх мэт (demo-гийн үндсэн context)

        // Нэг кадрын үндсэн render target.
        DefaultRT* rt = nullptr;

        // --- Frame timing ---
        uint64_t frame_index = 0;        // Кадрын дугаар
        float    dt          = 0.0f;     // delta time (сек)

        // --- Camera ---
        const glm::mat4* view = nullptr;
        const glm::mat4* proj = nullptr;
        const glm::mat4* viewproj = nullptr;
        const glm::mat4* prev_viewproj = nullptr;

        const glm::vec3* cam_pos_ws = nullptr;
        const glm::vec3* sun_dir_ws = nullptr;

        // --- Common post params ---
        float exposure = 1.0f;           // Tonemap exposure
        float gamma    = 2.2f;           // Display gamma

        // --- Shared resource hubs ---
        Scene* scene = nullptr;
        ResourceRegistry* resources = nullptr;
        RendererResources* renderer_resources_hub = nullptr;

        inline void bind_scene(Scene* s)
        {
            scene = s;
        }

        [[nodiscard]]
        inline Scene* modern_scene()
        {
            return scene;
        }

        [[nodiscard]]
        inline const Scene* modern_scene() const
        {
            return scene;
        }

        inline void bind_resource_registry(ResourceRegistry* r)
        {
            resources = r;
            if (r) renderer_resources_hub = nullptr;
        }

        inline void bind_renderer_resources(RendererResources* r)
        {
            renderer_resources_hub = r;
            if (r) resources = nullptr;
        }

        [[nodiscard]]
        inline ResourceRegistry* resource_registry()
        {
            return resources;
        }

        [[nodiscard]]
        inline const ResourceRegistry* resource_registry() const
        {
            return resources;
        }

        [[nodiscard]]
        inline RendererResources* renderer_resources()
        {
            return renderer_resources_hub;
        }

        [[nodiscard]]
        inline const RendererResources* renderer_resources() const
        {
            return renderer_resources_hub;
        }

        // --- Debug тохиргоо (сонголттой) ---
        int debug_view = 0;              // 0=сүүлчийн гаралтын зураг, 1=сүүдэр, 2=гүн, ... гэх мэт
    };

    } // inline namespace renderpath
} // namespace shs
