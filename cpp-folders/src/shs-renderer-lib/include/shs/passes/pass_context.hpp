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
        void* resources = nullptr;       // (shs::RendererResources*) - resource_handles.hpp дээр тодорхойлно
        void* scene     = nullptr;       // (shs::SceneData*)         - scene_data.hpp дээр тодорхойлно

        // --- Debug knobs (optional) ---
        int debug_view = 0;              // 0=final, 1=shadow, 2=depth, ... гэх мэт
    };
} // namespace shs
